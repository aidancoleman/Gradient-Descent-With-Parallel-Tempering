// forest_model.c
#include "forest_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Gradient buffers (shared across calls)
//statics so there is exactly one copy of each buffer so we don't have overhead
//at start of forward_backward_batch call, all six buffers are zeroed out and then during the backward pass we compute the contributions from each example in the minibatch and add them into these arrays. Each parameter will then be updated during sgld_step (after whole batch processed we have the summed gradients over the minibatch)
static float z1_buf[BATCH_SIZE][HIDDEN_DIM], h1_buf[BATCH_SIZE][HIDDEN_DIM], z2_buf[BATCH_SIZE][HIDDEN2_DIM], h2_buf[BATCH_SIZE][HIDDEN2_DIM];
static float grad_W1[INPUT_DIM][HIDDEN_DIM];
static float grad_b1[HIDDEN_DIM];
static float grad_W2[HIDDEN_DIM][HIDDEN2_DIM];
static float grad_b2[HIDDEN2_DIM];
static float grad_W3[HIDDEN2_DIM][OUTPUT_DIM]; //3 represents output layer (gradients)
static float grad_b3[OUTPUT_DIM];

static float gaussian_sample(void);

void burn_normals(size_t N) {
    for (size_t i = 0; i < N; ++i) (void)gaussian_sample();
}

// Box–Muller Gaussian sampler
static float gaussian_sample(void) {
    static int has_spare = 0;
    static float spare; //did previous call produce a spare normal random variable?
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    has_spare = 1;
    float u, v, s;
    //draw two uniform points in unit circle
    do {
        u = (rand() / (float)RAND_MAX)*2.0f - 1.0f;
        v = (rand() / (float)RAND_MAX)*2.0f - 1.0f;
        s = u*u + v*v;
    } while (s >= 1.0f || s == 0.0f); //draw the two points until u^2 + v^2 <1
    float m = sqrtf(-2.0f * logf(s) / s);
    spare = v * m;
    return u * m;
}


/* Dropout mask generator
//p = probability we want to zero out a neuron on a given example
static void sample_dropout_mask(uint8_t mask[BATCH_SIZE][HIDDEN_DIM], float p) {
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int j = 0; j < HIDDEN_DIM; j++)
            mask[b][j] = (rand() / (float)RAND_MAX) > p; //rhs gives a uniform random number in [0,1), if this is greater than p we set the mask to 1 (keep the neuron so no dropout, if <p drop the neuron)
}*/

// Initialize model weights & biases
void init_model(Model *m) { //pointer to Model struct and fill in weights/biases arrays
    //keep variance of activations roughly constant across layers when using ReLu by using sqrt(2/ number of inputs in layer) as scale factor (keeps signal magnitudes stable through ReLus)
    float scale1 = sqrtf(2.0f/INPUT_DIM);
    float scale2 = sqrtf(2.0f/HIDDEN_DIM);
    float scale3 = sqrtf(2.0f/HIDDEN2_DIM);
    for (int i = 0; i < INPUT_DIM; ++i)
     for (int j = 0; j < HIDDEN_DIM; ++j)
       m->W1[i][j] = scale1 * gaussian_sample(); //ech weight connecting input i to hidden unit j is filled with a random draw from a standard normal multiplied by scale1
    memset(m->b1,0,sizeof m->b1); //memset every bias in first hidden layer to 0
    for (int i = 0; i < HIDDEN_DIM; ++i)
     for (int j = 0; j < HIDDEN2_DIM; ++j)
       m->W2[i][j] = scale2 * gaussian_sample(); //set every weight in second layer W2[i][j] equal to a standard normal scaled by scale2 to keep variance constant across layers
    memset(m->b2,0,sizeof m->b2); //second layer biases are zeroed out
    for (int i = 0; i < HIDDEN2_DIM; ++i)
     for (int j = 0; j < OUTPUT_DIM; ++j)
       m->W3[i][j] = scale3 * gaussian_sample(); //same case for output layer
    memset(m->b3,0,sizeof m->b3);
}

// single sample forward to softmax used by MPI evaluation
void model_predict(const Model *m, const float *x, float *prob_out)
{
    float h1[HIDDEN_DIM], h2[HIDDEN2_DIM], logits[OUTPUT_DIM];
    // layer1
    for(int j=0; j<HIDDEN_DIM; ++j){
        float s = m->b1[j];
        for(int d=0; d<INPUT_DIM; ++d) s += x[d]*m->W1[d][j];
        h1[j] = (s>0.0f ? s : 0.0f);
    }
    // layer2
    for(int k=0; k<HIDDEN2_DIM; ++k){
        float s = m->b2[k];
        for(int j=0; j<HIDDEN_DIM; ++j) s += h1[j]*m->W2[j][k];
        h2[k] = (s>0.0f ? s : 0.0f);
    }
    // logits + softmax
    float mx = -INFINITY, sumexp = 0.0f;
    for(int o=0; o<OUTPUT_DIM; ++o){
        float s = m->b3[o];
        for(int k=0; k<HIDDEN2_DIM; ++k) s += h2[k]*m->W3[k][o];
        logits[o] = s;
        if (s > mx) mx = s;
    }
    for(int o=0; o<OUTPUT_DIM; ++o){
        float e = expf(logits[o] - mx); //logits and softmax
        prob_out[o] = e;
        sumexp += e;
    }
    for(int o=0; o<OUTPUT_DIM; ++o)
        prob_out[o] /= sumexp;
}
// Forward and backward pass returning average batch loss
double forward_backward_batch(Model *m, float inputs[BATCH_SIZE][INPUT_DIM], int labels[BATCH_SIZE] /*labels = true class index for each example in our batch*/, int bsz) {

    // stack allocated scratch
    /*z1 and z2 are pre and post ReLu activations for layer 1*/
    float (*z1)[HIDDEN_DIM] = z1_buf;
    float (*h1)[HIDDEN_DIM] = h1_buf;
    float (*z2)[HIDDEN2_DIM] = z2_buf;
    float (*h2)[HIDDEN2_DIM] = h2_buf;
    float logits[BATCH_SIZE][OUTPUT_DIM], soft[BATCH_SIZE][OUTPUT_DIM]; // logits = raw output scores before softmax for each example, soft = normalised (softmax) probabilities for each example
    double total_loss = 0.0;
    float dlog[OUTPUT_DIM]; //holds gradient of loss wrt logits for an example during back propagation

    // zero global grads from any previous batch
    memset(grad_W1,0,sizeof grad_W1);
    memset(grad_b1,0,sizeof grad_b1);
    memset(grad_W2,0,sizeof grad_W2);
    memset(grad_b2,0,sizeof grad_b2);
    memset(grad_W3,0,sizeof grad_W3);
    memset(grad_b3,0,sizeof grad_b3);

    /*uint8_t drop_mask[BATCH_SIZE][HIDDEN_DIM];
    sample_dropout_mask(drop_mask, DROPOUT_RATE);*/

    // Forward pass
    // Layer 1: parallelize over neurons j
    #pragma omp parallel for schedule(static)
    for(int j=0; j<HIDDEN_DIM; ++j){
        float bias = m->b1[j]; //load each neuron's bias
        for(int b=0; b<bsz; ++b){ //loop over each minibatch example
            float s = bias;
            for(int i=0; i<INPUT_DIM; ++i)
                s += inputs[b][i] * m->W1[i][j]; //to the bias add the dot product of the example's INPUT_DIM features with the weights in first layer
            z1[b][j] = s; //store this pre-activated sum
            h1[b][j] = (s > 0.0f ? s : 0.0f); //now apply ReLu to the sum : ReLU(s) = max{0, s}
        }
    }
    //parallelising over neurons during our feed forward so different threads work on the computation of ReLu outputs for different subsets of Neurons simultaneously
    // Layer 2: parallelize over neurons k
    #pragma omp parallel for schedule(static) //divide up neuron computations evenly
    for(int k=0; k<HIDDEN2_DIM; ++k){
        float bias = m->b2[k]; //load bias for neuron k
        for(int b=0; b<bsz; ++b){
            float s = bias;
            for(int j=0; j<HIDDEN_DIM; ++j)
                s += h1[b][j] * m->W2[j][k]; //to bias add the dot product of first hidden layer output with the neuron's weights
            z2[b][k] = s; //store sum
            h2[b][k] = (s > 0.0f ? s : 0.0f); //apply ReLu
        }
    }

    // Layer 3 => softmax + loss: parallel over batch
    #pragma omp parallel for reduction(+:total_loss)/*safely accumulate each thread's contributions to total_loss into a single sum*/ schedule(static)
    for(int b=0; b<bsz; ++b){
        // compute logits
        float mx = -INFINITY;
        for (int o=0; o<OUTPUT_DIM; ++o){
            float s = m->b3[o];
            for (int k=0; k<HIDDEN2_DIM; ++k)
                s += h2[b][k] * m->W3[k][o];
            logits[b][o] = s;
            if (s > mx) mx = s;
        } //track the maximum logit for numerical stability: logits are the raw scores before softmax and if we call exp(logit) for large logit values this can lead to overflow (or underflow when logit is very negative), so subtract the maximum logit from every logit before we exponentiate (manner similar to log-sum-exp trick in floating point arithmetic)
        // softmax
        float sumexp = 0.0f;
        for(int o=0; o<OUTPUT_DIM; ++o){
            float e = expf(logits[b][o] - mx); //first pass = compute this for each class
            soft[b][o] = e; //store here
            sumexp += e; //accumulate sum
        }
        for(int o=0; o<OUTPUT_DIM; ++o)
            soft[b][o] /= sumexp; //second pass => normalise so the outputs form a valid probability distribution

        // cross entropy: how well a model's predicted distribution matches the true empirical distribution => penalises "confident but wrong" predictions
        int y = labels[b];
        total_loss += - (logits[b][y] - mx) + logf(sumexp); //compute cross-entropy loss in numerically stable form and accumulate here
    }

    // --- Backward pass (serial reduction) ---
    for(int b=0; b<bsz; ++b){ //looping over each minibatch example
        // dL/dlogits => gradient wrt logits
        for(int o=0; o<OUTPUT_DIM; ++o){
            dlog[o] = soft[b][o] /*<<predicted probability for class o*/ - (o==labels[b] ? 1.0f : 0.0f); //the actual true label (the difference between these is the derivative of the cross-entropy loss wrt the logit for class o)
        }
        // layer3 grads = output gradients
        for(int o=0; o<OUTPUT_DIM; ++o){
            grad_b3[o] += dlog[o]; //sum of dlog over all examples is just each output bias gradient
            for(int k=0; k<HIDDEN2_DIM; ++k)
                grad_W3[k][o] += h2[b][k] * dlog[o]; //gradient of weight connecting unit k to output o
        }
        // backprop error into h2
        //we first compute pre ReLu gradient for each second-layer neuron k
        float dh2[HIDDEN2_DIM];
        for(int k=0; k<HIDDEN2_DIM; ++k){
            float sum = 0.0f;
            for(int o=0; o<OUTPUT_DIM; ++o)
                sum += m->W3[k][o] * dlog[o]; //sum over all output classes multiplying by derivative of ReLu
            dh2[k] = (z2[b][k] > 0.0f ? sum : 0.0f); //if pre activation was positive pass the sum as it is but otherwise zero it out as ReLu shut this neuron off => there is no sensitivity to this input so multiply it by zero, back propagting throigh a ReLu means multiplying the incoming gradient by the derivative of ReLu wrt its preactivation
        }
        // layer2 grads
        for(int k=0; k<HIDDEN2_DIM; ++k){
            grad_b2[k] += dh2[k]; //add bias gradients from layer 2 into hidden layer 2 gradient vector
            for(int j=0; j<HIDDEN_DIM; ++j)
                grad_W2[j][k] += h1[b][j] * dh2[k]; //each weight from input (first hidden unit) j to second hidden unit k gets gradient on LHS => accumulate this
        }
        // backprop into h1
        float dh1[HIDDEN_DIM];
        for(int j=0; j<HIDDEN_DIM; ++j){
            float sum = 0.0f;
            for(int k=0; k<HIDDEN2_DIM; ++k)
                sum += m->W2[j][k] * dh2[k]; //gradient flowing backwards into pre-ReLu activations of layer 1, using chain rule as model for how each layer two neuron k depends on each layer 1 unit j
            dh1[j] = (z1[b][j] > 0.0f ? sum : 0.0f); //original pre-ReLu activation > 0 => gradient passes through unchanged, else it is zeroed out
        }
        // layer1 grads (accumulate bias and weight gradients)
        for(int j=0; j<HIDDEN_DIM; ++j){
            grad_b1[j] += dh1[j];
            for(int i=0; i<INPUT_DIM; ++i)
                grad_W1[i][j] += inputs[b][i] * dh1[j];
        }
    }
    
    // scale grads by batch size (use batch-mean, this is standard in both SGD/SGLD)
    const float invB = 1.0f / (float)bsz;
    for (int i=0;i<INPUT_DIM;i++) for (int j=0;j<HIDDEN_DIM;j++) grad_W1[i][j] *= invB;
    for (int j=0;j<HIDDEN_DIM;j++) grad_b1[j] *= invB;
    for (int i=0;i<HIDDEN_DIM;i++) for (int j=0;j<HIDDEN2_DIM;j++) grad_W2[i][j] *= invB;
    for (int j=0;j<HIDDEN2_DIM;j++) grad_b2[j] *= invB;
    for (int i=0;i<HIDDEN2_DIM;i++) for (int j=0;j<OUTPUT_DIM;j++) grad_W3[i][j] *= invB;
    for (int j=0;j<OUTPUT_DIM;j++) grad_b3[j] *= invB;
    
    return total_loss / (double)bsz; //avg cross entropy loss across the batch
}
double model_l2_norm(const Model* m) {
  double s=0.0;
  for (int i=0;i<INPUT_DIM;i++) for (int j=0;j<HIDDEN_DIM;j++) { double v=m->W1[i][j]; s+=v*v; }
  for (int j=0;j<HIDDEN_DIM;j++) { double v=m->b1[j]; s+=v*v; }
  for (int j=0;j<HIDDEN_DIM;j++) for (int k=0;k<HIDDEN2_DIM;k++) { double v=m->W2[j][k]; s+=v*v; }
  for (int k=0;k<HIDDEN2_DIM;k++) { double v=m->b2[k]; s+=v*v; }
  for (int k=0;k<HIDDEN2_DIM;k++) for (int o=0;o<OUTPUT_DIM;o++) { double v=m->W3[k][o]; s+=v*v; }
  for (int o=0;o<OUTPUT_DIM;o++) { double v=m->b3[o]; s+=v*v; }
  return sqrt(s);
}

double model_diff_l2(const Model* a, const Model* b) {
  double s=0.0, v;
  for (int i=0;i<INPUT_DIM;i++) for (int j=0;j<HIDDEN_DIM;j++){ v=(double)a->W1[i][j]-b->W1[i][j]; s+=v*v; }
  for (int j=0;j<HIDDEN_DIM;j++){ v=(double)a->b1[j]-b->b1[j]; s+=v*v; }
  for (int j=0;j<HIDDEN_DIM;j++) for (int k=0;k<HIDDEN2_DIM;k++){ v=(double)a->W2[j][k]-b->W2[j][k]; s+=v*v; }
  for (int k=0;k<HIDDEN2_DIM;k++){ v=(double)a->b2[k]-b->b2[k]; s+=v*v; }
  for (int k=0;k<HIDDEN2_DIM;k++) for (int o=0;o<OUTPUT_DIM;o++){ v=(double)a->W3[k][o]-b->W3[k][o]; s+=v*v; }
  for (int o=0;o<OUTPUT_DIM;o++){ v=(double)a->b3[o]-b->b3[o]; s+=v*v; }
  return sqrt(s);
}

// grad_* arrays are already static in this file and filled by forward_backward_batch()
// expose their L2 norm so we don't add extra passes in the driver
double grad_l2_norm(void) {
  double s=0.0, v;
  for (int i=0;i<INPUT_DIM;i++) for (int j=0;j<HIDDEN_DIM;j++){ v=grad_W1[i][j]; s+=v*(double)v; }
  for (int j=0;j<HIDDEN_DIM;j++){ v=grad_b1[j]; s+=v*(double)v; }
  for (int j=0;j<HIDDEN_DIM;j++) for (int k=0;k<HIDDEN2_DIM;k++){ v=grad_W2[j][k]; s+=v*(double)v; }
  for (int k=0;k<HIDDEN2_DIM;k++){ v=grad_b2[k]; s+=v*(double)v; }
  for (int k=0;k<HIDDEN2_DIM;k++) for (int o=0;o<OUTPUT_DIM;o++){ v=grad_W3[k][o]; s+=v*(double)v; }
  for (int o=0;o<OUTPUT_DIM;o++){ v=grad_b3[o]; s+=v*(double)v; }
  return sqrt(s);
}

// high-precision loss on a batch (no grads), mirrors single-precision loss
double batch_loss_only_fp64(const Model* m, const float (*X)[INPUT_DIM], const int *y, int bsz) {
  double total=0.0;
  for (int b=0;b<bsz;++b){
    double h1[HIDDEN_DIM], h2[HIDDEN2_DIM], logits[OUTPUT_DIM];
    for (int j=0;j<HIDDEN_DIM;++j){
      double s=m->b1[j];
      for (int i=0;i<INPUT_DIM;++i) s += (double)X[b][i]*(double)m->W1[i][j];
      h1[j] = s>0.0 ? s : 0.0;
    }
    for (int k=0;k<HIDDEN2_DIM;++k){
      double s=m->b2[k];
      for (int j=0;j<HIDDEN_DIM;++j) s += h1[j]*(double)m->W2[j][k];
      h2[k] = s>0.0 ? s : 0.0;
    }
    double mx=-INFINITY;
    for (int o=0;o<OUTPUT_DIM;++o){
      double s=m->b3[o];
      for (int k=0;k<HIDDEN2_DIM;++k) s += h2[k]*(double)m->W3[k][o];
      logits[o]=s; if (s>mx) mx=s;
    }
    double sumexp=0.0;
    for (int o=0;o<OUTPUT_DIM;++o) sumexp += exp(logits[o]-mx);
    total += -(logits[y[b]]-mx) + log(sumexp);
  }
  return total/(double)bsz;
}

// Save model to a binary file
void forest_save(const Model *m, const char *fname) {
    FILE *f = fopen(fname,"wb"); //open file "fname" for writing in binary mode
    if (!f) { perror("fopen"); exit(1); }
    fwrite(m, sizeof *m, 1, f); //write the Model struct with associated weights and biases to the disk
    fclose(f);
}
//how much parameters in total in model:
size_t forest_param_count(void) {
    return INPUT_DIM*HIDDEN_DIM
         + HIDDEN_DIM
         + HIDDEN_DIM*HIDDEN2_DIM
         + HIDDEN2_DIM
         + HIDDEN2_DIM*OUTPUT_DIM
         + OUTPUT_DIM;
}
/*THE NEXT TWO ROUTINES ALLOW US TO SEND AND RECEIVE ENTIRE MODEL IN ONE OPERATION DURING REPLICA EXCHANGE STEP*/
//MPI only sends flattened contiguous arrays easily
//flatten every weight and bias into a 1d float array
void forest_serialize_f(const Model *m, float *buf) {
    size_t idx=0;
    for(int i=0;i<INPUT_DIM;i++)for(int j=0;j<HIDDEN_DIM;j++) buf[idx++]=m->W1[i][j];
    for(int j=0;j<HIDDEN_DIM;j++) buf[idx++]=m->b1[j];
    for(int i=0;i<HIDDEN_DIM;i++)for(int j=0;j<HIDDEN2_DIM;j++) buf[idx++]=m->W2[i][j];
    for(int j=0;j<HIDDEN2_DIM;j++) buf[idx++]=m->b2[j];
    for(int i=0;i<HIDDEN2_DIM;i++)for(int j=0;j<OUTPUT_DIM;j++) buf[idx++]=m->W3[i][j];
    for(int j=0;j<OUTPUT_DIM;j++) buf[idx++]=m->b3[j];
    //single incremeent counter so parameters are in a fixed order that can be reproduced
}
//reconstruct Model back from serial flattened version back into Model struct exactly in same order
void forest_deserialize_f(Model *m, const float *buf) {
    size_t idx=0;
    for(int i=0;i<INPUT_DIM;i++)for(int j=0;j<HIDDEN_DIM;j++) m->W1[i][j]=buf[idx++];
    for(int j=0;j<HIDDEN_DIM;j++) m->b1[j]=buf[idx++];
    for(int i=0;i<HIDDEN_DIM;i++)for(int j=0;j<HIDDEN2_DIM;j++) m->W2[i][j]=buf[idx++];
    for(int j=0;j<HIDDEN2_DIM;j++) m->b2[j]=buf[idx++];
    for(int i=0;i<HIDDEN2_DIM;i++)for(int j=0;j<OUTPUT_DIM;j++) m->W3[i][j]=buf[idx++];
    for(int j=0;j<OUTPUT_DIM;j++) m->b3[j]=buf[idx++];
}

static void sgld_update(float *param, float *grad, size_t N, float eta) {
    float sigma = sqrtf(2.0f * eta) * NOISE_SCALE;
    if (sigma == 0.0f) {
        // No Langevin noise just do SGD and return (avoid gaussian_sample cost)
        for (size_t i = 0; i < N; ++i) {
            param[i] -= eta * grad[i];
        }
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        float noise = sigma * gaussian_sample();
        param[i] -= eta * grad[i] + noise;
    }
}

void sgld_step(Model *m, float eta) {
    // layer 1 weights
        sgld_update(&m->W1[0][0], &grad_W1[0][0], INPUT_DIM * HIDDEN_DIM, eta);
        // layer 1 biases
        sgld_update(m->b1, grad_b1, HIDDEN_DIM, eta);

        // layer 2 weights
        sgld_update(&m->W2[0][0], &grad_W2[0][0], HIDDEN_DIM * HIDDEN2_DIM, eta);
        // layer 2 biases
        sgld_update(m->b2, grad_b2, HIDDEN2_DIM, eta);

        // output layer weights
        sgld_update(&m->W3[0][0], &grad_W3[0][0], HIDDEN2_DIM * OUTPUT_DIM, eta);
        // output layer biases
        sgld_update(m->b3, grad_b3, OUTPUT_DIM, eta);
}
//no Langevin noise SGD -> according to "Non-Reversible Parallel Tempering" only chain with temp = 1 (true posterior) uses noise (sgld_step does this -> on the "exploitation" replica)
static void sgd_update(float *param, float *grad, size_t N, float eta) {
    for (size_t i = 0; i < N; ++i) param[i] -= eta * grad[i];
}

void sgd_step(Model *m, float eta) {
    sgd_update(&m->W1[0][0], &grad_W1[0][0], INPUT_DIM * HIDDEN_DIM, eta);
    sgd_update(m->b1, grad_b1, HIDDEN_DIM, eta);
    sgd_update(&m->W2[0][0], &grad_W2[0][0], HIDDEN_DIM * HIDDEN2_DIM, eta);
    sgd_update(m->b2, grad_b2, HIDDEN2_DIM, eta);
    sgd_update(&m->W3[0][0], &grad_W3[0][0], HIDDEN2_DIM * OUTPUT_DIM, eta);
    sgd_update(m->b3, grad_b3, OUTPUT_DIM, eta);
}
//for only computing batch loss without dropout -> no gradient comp either
double batch_loss_only(const Model *m, const float inputs[BATCH_SIZE][INPUT_DIM], const int labels[BATCH_SIZE], int bsz) {
    float h1[HIDDEN_DIM], h2[HIDDEN2_DIM], logits[OUTPUT_DIM];
    double total = 0.0;
    for (int b = 0; b < bsz; ++b) {
        // layer 1
        for (int j=0;j<HIDDEN_DIM;++j){
            float s = m->b1[j];
            for (int i=0;i<INPUT_DIM;++i) s += inputs[b][i]*m->W1[i][j];
            h1[j] = (s>0.f? s:0.f);
        }
        // layer 2
        for (int k=0;k<HIDDEN2_DIM;++k){
            float s = m->b2[k];
            for (int j=0;j<HIDDEN_DIM;++j) s += h1[j]*m->W2[j][k];
            h2[k] = (s>0.f? s:0.f);
        }
        // logits + softmax xent
        float mx = -INFINITY;
        for (int o=0;o<OUTPUT_DIM;++o){
            float s = m->b3[o];
            for (int k=0;k<HIDDEN2_DIM;++k) s += h2[k]*m->W3[k][o];
            logits[o] = s; if (s>mx) mx=s;
        }
        float sumexp = 0.f;
        for (int o=0;o<OUTPUT_DIM;++o) sumexp += expf(logits[o]-mx);
        int y = labels[b];
        total += - (logits[y] - mx) + logf(sumexp);
    }
    return total / (double)bsz;
}
//evaluation accuracy
double eval_accuracy(const Model *m, const float (*X)[INPUT_DIM], const int32_t *y, int N){
    int correct = 0;
    float prob[OUTPUT_DIM];
    for(int i=0;i<N;++i){
        model_predict(m, X[i], prob);
        int argmax=0;
        for(int o=1;o<OUTPUT_DIM;++o) if(prob[o]>prob[argmax]) argmax=o;
        if (argmax == y[i]) ++correct;
    }
    return (double)correct / (double)N;
}
