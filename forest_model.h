// forest_model.h
#pragma once
#ifndef FOREST_MODEL_H
#define FOREST_MODEL_H
#include <stddef.h>
#include <stdint.h>

#define INPUT_DIM     54
#define HIDDEN_DIM    512
#define HIDDEN2_DIM   256
#define OUTPUT_DIM    7
#define BATCH_SIZE    256
#define LEARNING_RATE 0.001f
#define DROPOUT_RATE  0.2f
#define NOISE_SCALE   0.01f //Langevin  noise

typedef struct {
    float W1[INPUT_DIM][HIDDEN_DIM];
    float b1[HIDDEN_DIM];
    float W2[HIDDEN_DIM][HIDDEN2_DIM];
    float b2[HIDDEN2_DIM];
    float W3[HIDDEN2_DIM][OUTPUT_DIM];
    float b3[OUTPUT_DIM];
} Model;

// Core routines
void init_model(Model *m);
// analysis helpers 
double model_l2_norm(const Model* m);
double model_diff_l2(const Model* a, const Model* b);      // L2 distance between models
double grad_l2_norm(void);                                  // norm of last computed grads
double batch_loss_only_fp64(const Model* m,
                            const float (*X)[INPUT_DIM],
                            const int *y, int bsz);
void burn_normals(size_t N);
double forward_backward_batch(Model *m, float inputs[BATCH_SIZE][INPUT_DIM], int labels[BATCH_SIZE], int bsz);
void sgld_step(Model *m, float eta);
void forest_save(const Model *m, const char *fname);
void model_predict(const Model *m, const float *x, float *prob_out);
// PT helpers
size_t forest_param_count(void);
void forest_serialize_f(const Model *m, float *buf);
void forest_deserialize_f(Model *m, const float *buf);

double batch_loss_only(const Model *m, const float inputs[BATCH_SIZE][INPUT_DIM], const int labels[BATCH_SIZE], int bsz);

void sgd_step(Model *m, float eta); // same as sgld_step without injected noise

double eval_accuracy(const Model *m, const float (*X)[INPUT_DIM], const int32_t *y, int N);
#endif
