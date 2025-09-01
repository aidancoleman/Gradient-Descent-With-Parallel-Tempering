
//INVESTIGATING WEAK SCALING WITH GROWING DATASET
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "forest_model.h"
#include "load_csv.h"

//command line parser:
//feed in parameters like gamma (adaptation rate), eta_lo, eta_hi (lowest and highest learning rates), S = target swap rate between adjacent chains, K = total iters, W= window size we use to calculate S
static const char* arg_str(int argc,char**argv,const char*flag,const char*def){
  for (int i=1;i<argc-1;++i) {
    if (!strcmp(argv[i],flag)) return argv[i+1];
  }
  return def;
}
static int arg_int (int argc,char**argv,const char*flag,int   def){ const char* s=arg_str(argc,argv,flag,NULL); return s?atoi(s):def; }
static double arg_double(int argc,char**argv,const char*flag,double def){ const char* s=arg_str(argc,argv,flag,NULL); return s?atof(s):def; }

//helpers
static inline int window_size_opt(int P, double S){
    if (P <= 2) return 1; //don't divide by 0
  // W = ceil((log P + log log P)/(-log(1-S))) DEO window
  double num = log((double)P) + log(log((double)P));
  double den = -log(fmax(1e-12, 1.0 - S));
  int W = (int)ceil(num / fmax(1e-12, den));
  return (W<1)?1:W;
} //W is the ceiling of: (log(P) + log(log(P)))/(-log(1-S)), the optimal window size for quickest round trip from rank 0 to rank P given target swap acceptance rate = S
static inline double gamma_k(double g0,double gdecay,int k){
  return g0 / (1.0 + gdecay * (double)k);
} //stoch approximation step size schedule for algo adaptation -> used to update learning rates of interior ranks and correction buffer C
static void enforce_monotone(double* eta, int P){
  for(int p=1;p<P;++p) if(eta[p]<eta[p-1]) eta[p]=eta[p-1];
} //make learning rate ladder non-decreasing

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank,size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);
    const int stab_stride  = arg_int(argc, argv, "--stab_stride", 0); // 0 = off
    const int trace_stride = arg_int(argc, argv, "--trace_stride", 0);
    if (trace_stride > 0 && rank == 0) setvbuf(stdout, NULL, _IOLBF, 0);
    if (rank == 0) setvbuf(stdout, NULL, _IOLBF, 0);
  //compute partners of each rank for pairwise energy exchanges and model swaps
  int left  = (rank > 0) ? rank-1 : MPI_PROC_NULL;
  int right = (rank < size-1) ? rank+1 : MPI_PROC_NULL;
    //return errors rather than exiting
  MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

  const char* csv = arg_str(argc,argv,"--csv",NULL); //path to data
  if(!csv){ if(rank==0) fprintf(stderr,"--csv <path to covtype.csv> is required\n"); MPI_Abort(comm,1); }
  const int K = arg_int (argc,argv,"--iters", 20000); //iters
  const double S = arg_double(argc,argv,"--S",      0.4);     // target swap rate
  const double eta_lo = arg_double(argc,argv,"--eta_lo", 3e-3); //lowest temp chain learning rate
  const double eta_hi = arg_double(argc,argv,"--eta_hi", 2e-2); //highest temp. chain learning rate
  const double gamma0 = arg_double(argc,argv,"--gamma0", 0.05);
  const double gdecay = arg_double(argc,argv,"--gdecay", 1e-4);
  const double val_frac = arg_double(argc,argv,"--val", 0.2);// test split, how much of data to hold out
  const int snap_stride = arg_int(argc,argv,"--snap_stride", 1000); //take a snapshot of model's at snap_stride iterations to determine ensemble prediciton accuracy
  const unsigned seed = (unsigned)arg_int(argc,argv,"--seed", 42); //reproducibility for shuffles between chains
  const int balance_sgld = arg_int(argc, argv, "--balance_sgld", 0); //optional flag at run time so non target ranks do the same amount of Gaussian draws (Langevin) as the target rank
    
  const int P = size; // 1 replica (chain) per MPI process
  const int W = window_size_opt(P, S);// window for swapping scheme (dictates when adjacent pairs of replicas are allowed to swap models and energy) that minimizes round trip time of model from target chain to exploratory chain (highest temp.)
    const int per_rank = arg_int(argc, argv, "--per_rank", 0); // 0 = disabled
  if(rank==0) printf("[P=%d] W=%d  S=%.3f  K=%d  B=%d\n",P,W,S,K,BATCH_SIZE); //rank 0 will start off run by printing the chosen parameters

  //Load all the data on rank0, split, broadcast to other ranks to begin
  float (*Xall)[INPUT_DIM] = NULL;
  int32_t* yall = NULL;
  int Nall = 0; //number of rows initialised at 0
  if(rank==0){
    load_csv(csv, &Xall, &yall, &Nall); //rank 0 will load the data, fill Nall with actual number of rows in data
    if(Nall<=0){ fprintf(stderr,"Empty dataset\n"); MPI_Abort(comm,2); }
  }
    if (rank==0){
        long cnt[OUTPUT_DIM]={0};
        for (int i=0;i<Nall;++i) {
            int yv = yall[i];
            if (yv < 0 || yv >= OUTPUT_DIM) fprintf(stderr,"BAD LABEL row %d: %d\n", i, yv);
            else cnt[yv]++;
        }
        //label counts as sanity check to see if we imported data correctly
        printf("label counts:"); for(int c=0;c<OUTPUT_DIM;++c) printf(" %ld", cnt[c]); printf("\n");
        fflush(stdout);
    }
  MPI_Bcast(&Nall,1,MPI_INT,0,comm); //broadcast the size of the dataset to all other ranks -> everyone else will know how much memory to allocate
    
    // decide effective dataset size at this P
    int Nuse = Nall;
    if (per_rank > 0) {
        long want = (long)per_rank * (long)P;  // total subset size grows with P
        if (want < (long)Nuse) Nuse = (int)want;
    }
    // sanity: ensure we have at least a batch
    if (Nuse < BATCH_SIZE) {
        if (rank==0) fprintf(stderr, "Nuse=%d < BATCH_SIZE=%d; increase --per_rank or reduce BATCH_SIZE\n", Nuse, BATCH_SIZE);
        MPI_Abort(comm, 3);
    }

    if (rank==0) {
        printf("[data] Nall=%d  per_rank=%d  P=%d  Nuse=%d (%.1f%% of full)\n",
               Nall, per_rank, P, Nuse, 100.0*(double)Nuse/(double)Nall);
        fflush(stdout);
    }
  // allocate on all ranks and broadcast 1D
  float *Xflat = (float*)malloc((size_t)Nall*INPUT_DIM*sizeof(float)); //allocating buffer to hold all features on every rank
  int32_t *y   = (int32_t*)malloc((size_t)Nall*sizeof(int32_t)); //we are allocating buffers to hold the labels on every rank
    //rank 0 then copies Xall and yall (original features in 2D and labels 1D) into the flat buffers, then we free the originals to avoid memory leaks
  if(rank==0){
    memcpy(Xflat, &Xall[0][0], (size_t)Nall*INPUT_DIM*sizeof(float));
    memcpy(y, yall, (size_t)Nall*sizeof(int32_t));
    free(Xall); free(yall);
  }
  MPI_Bcast(Xflat, (int) (Nall*INPUT_DIM), MPI_FLOAT, 0, comm); //broadcast features from rank 0 to all others
  MPI_Bcast(y, Nall, MPI_INT32_T, 0, comm); //broadcast labels from rank 0 to all others
  float (*X)[INPUT_DIM] = (float (*)[INPUT_DIM])Xflat; //now tur flat buffer back into 2D, no COPY just a CAST for performance

  //train/test split with a shared permutation
    int *perm_split = (int*)malloc((size_t)Nall*sizeof(int));
    if(rank==0){
        //random permutation of row indices with a  shuffle, seed is fixed so we can reproduce results across runs
      srand(seed);
      for(int i=0;i<Nall;++i) perm_split[i]=i;
      for(int i=Nall-1;i>0;--i){ int j=rand()%(i+1); int t=perm_split[i]; perm_split[i]=perm_split[j]; perm_split[j]=t; }
    }
    MPI_Bcast(perm_split, Nall, MPI_INT, 0, comm); //broadcast this permutation so every rank uses same split

    //number of test and train samples based on validation fraction
    const int Ntest  = (int)lrint((double)Nuse * val_frac);
    const int Ntrain = Nuse - Ntest; //the first Ntrain indices that were permuted before will go to the training set
    
    // epoch math for the outer loop
  const int iters_per_epoch = (Ntrain + BATCH_SIZE - 1) / BATCH_SIZE; //the number of mini batches making one pass over our train set
  const int epochs = (K + iters_per_epoch - 1) / iters_per_epoch; //take ceiling so we get how many epochs needed so the inner loop can run at least K update steps

  // views using perm : 2D buffers for the training and testing features and labels -> memory is contiguous for nice cache access
  float (*Xtr)[INPUT_DIM] = (float (*)[INPUT_DIM])malloc((size_t)Ntrain*INPUT_DIM*sizeof(float));
  int32_t* ytr = (int32_t*)malloc((size_t)Ntrain*sizeof(int32_t));
  float (*Xte)[INPUT_DIM] = (float (*)[INPUT_DIM])malloc((size_t)Ntest*INPUT_DIM*sizeof(float));
  int32_t* yte = (int32_t*)malloc((size_t)Ntest*sizeof(int32_t));
    
    // copy split views: perm_split is the shared permutation of data across ranks, and we use it to copy rows into arrays of tresting and training data that are contiguous
    // copy split views from first Nuse permuted indices
    for (int i=0; i<Ntrain; ++i) {
        int idx = perm_split[i];
        memcpy(Xtr[i], X[idx], sizeof(float)*INPUT_DIM);
        ytr[i] = y[idx];
    }
    for (int i=0; i<Ntest; ++i) {
        int idx = perm_split[Ntrain + i];
        memcpy(Xte[i], X[idx], sizeof(float)*INPUT_DIM);
        yte[i] = y[idx];
    }
    free(perm_split); free(Xflat); free(y);
    //we get the normalised stats for the training set on rank 0
    double mean[INPUT_DIM], invstd[INPUT_DIM];
    if (rank==0){
      for (int d=0; d<INPUT_DIM; ++d) mean[d]=0.0;
      for (int i=0; i<Ntrain; ++i)
        for (int d=0; d<INPUT_DIM; ++d) mean[d] += Xtr[i][d];
      for (int d=0; d<INPUT_DIM; ++d) mean[d] /= (double)Ntrain;

      double sq[INPUT_DIM] /*accumulate E{X^2}*/; for (int d=0; d<INPUT_DIM; ++d) sq[d]=0.0;
      for (int i=0; i<Ntrain; ++i)
        for (int d=0; d<INPUT_DIM; ++d){ double v=Xtr[i][d]; sq[d]+=v*v; }
      for (int d=0; d<INPUT_DIM; ++d){
        double var = sq[d]/(double)Ntrain - mean[d]*mean[d]; //E{X^2} - E{X}^2
        invstd[d] = 1.0 / sqrt(var + 1e-8); //add epsilon to avoid division by zero for stability
      }//per-feature variance and mean with only the training set
    }
    //now broadcast these stats to the others
    MPI_Bcast(mean, INPUT_DIM, MPI_DOUBLE, 0, comm);
    MPI_Bcast(invstd,INPUT_DIM, MPI_DOUBLE, 0, comm);
    //now we standardise both the training and test sets with the training sets -> this applies zero mean/unit variance scaling to both sets using train mean/std (correct leakage-free normalization)
    for (int i=0; i<Ntrain; ++i)
      for (int d=0; d<INPUT_DIM; ++d)
        Xtr[i][d] = (float)((Xtr[i][d] - mean[d]) * invstd[d]);

    for (int i=0; i<Ntest; ++i)
      for (int d=0; d<INPUT_DIM; ++d)
        Xte[i][d] = (float)((Xte[i][d] - mean[d]) * invstd[d]); //cast back to float after we computed with double for accuracy

    // Shared train permutation for minibatches
    int *perm_tr = (int*)malloc((size_t)Ntrain*sizeof(int));
    if (!perm_tr) { perror("malloc perm_tr"); MPI_Abort(comm,1); }
    if (rank==0){
        //Rank 0 fills perm_tr with a Fisher–Yates shuffle (seeded with seed+1 so it’s reproducible and distinct from our previous split shuffle!).
      srand(seed+1);
      for (int i=0;i<Ntrain;++i) perm_tr[i]=i;
      for (int i=Ntrain-1;i>0;--i){ int j=rand()%(i+1); int t=perm_tr[i]; perm_tr[i]=perm_tr[j]; perm_tr[j]=t; }
    }
    MPI_Bcast(perm_tr, Ntrain, MPI_INT, 0, comm); //broadcast this sp all ranks have same order for minibatches every epoch -> need this to make swap energies comparable across chains
    

  //init model + learning rate ladder
  Model model; init_model(&model); //give random weights and biases to initialise
    double *eta = (double*)malloc(sizeof(double)*P);
    if (P == 1) {
      eta[0] = eta_lo; // <- single chain: just use eta_lo
    } else {
      for (int p=0; p<P; ++p) {
        double t = (double)p / (double)(P-1);
        eta[p] = eta_lo * pow(eta_hi/eta_lo, t); //geometric ladder from lowest to highest temperature this is our temperature "schedule"
      }
    }
    double my_eta = eta[rank]; //the current ranks initial step size which can be updated later during parallel tempering
  double C = 0.0; // correction buffer, this accumulates a correction to the global swap rate between replicas to our observed average swap rate approaches our target swap rate

  // swap gates: allow at most one swap per window for neighbouring replicas (p,p+1)
  int gate_right = (rank < P-1) ? 1 : 0;

  // scratch for serialization during swaps to allow us to swap entire models between replicas
  const size_t nparam /*total number of weights and biases in network*/ = forest_param_count();
  float *buf = (float*)malloc(nparam*sizeof(float));

  // wall timer timing accumulators:
    /*t_compute: timing of forward/backward passes and update of parameterss
     t_energy_comm: neighbour energy exchange timing
     t_swap_comm: model swapping timing
     t_allreduce: timing for global reductions for swap rate correction and avgA (average swap rate across ranks)
     */
  double t_compute=0, t_energy_comm=0, t_swap_comm=0, t_allreduce=0;
    
  int have_prev = 0; Model prev_model; double prev_norm = 0.0;
    // deltas for [trace]
  double t_compute_prev=0, t_energy_comm_prev=0, t_swap_comm_prev=0, t_allreduce_prev=0;
  // snapshot storage of models on rank0 (optional ensemble) -> some papers suggest this leads to better accuracy results -> perhaps not indicative of how well training went
  Model* snaps = NULL; int nsnaps=0;
  if(rank==0 && snap_stride>0) snaps = (Model*)malloc(((K/snap_stride)+2)*sizeof(Model));
    
    srand(seed + 777*rank); //each rank has its own RNG stream for any local randomness and this keeps runs reproducible but not identical across ranks
  //training loop
  int global_iter = 0;
  for(int e=0; e<epochs; ++e){
    for(int b0=0; b0<Ntrain; b0+=BATCH_SIZE){
      if (++global_iter > K) break;
      const int k = global_iter;
      const int phase = ((k-1)/W) % 2; // Deterministic Even-Odd (DEO) swap scheme parity, only even or odd pairs will attempt swaps
        if (((k-1) % W) == 0) gate_right = (right != MPI_PROC_NULL); //at start of swap window we reopen the gate for the neighbouring rank on the right so each pair can perform at most one swap in a given window

        // training batch for the kernel step
        float Xbatch[BATCH_SIZE][INPUT_DIM];
        int ybatch[BATCH_SIZE]; //stack buffers for current minibatch

        int start = ((k - 1) * BATCH_SIZE) % Ntrain; //where this training step's batch will begin in our permutation (shared -> perm_tr) and this wraps around the set of training examples
        int bsz = (start + BATCH_SIZE <= Ntrain) ? BATCH_SIZE : (Ntrain - start); //actual number of examples this batch has (last batch in an epoch may be smaller)
        for (int b = 0; b < bsz; ++b) {
            int idx = perm_tr[(start + b) % Ntrain];
            memcpy(Xbatch[b], Xtr[idx], sizeof(float) * INPUT_DIM);
            ybatch[b] = ytr[idx];
        } //fill first bsz rows from our training set
        for (int i = bsz; i < BATCH_SIZE; ++i) {
            memset(Xbatch[i], 0, sizeof(float)*INPUT_DIM);
            ybatch[i] = 0; //pad with zeros if batch is short -> we always assume buffers are equal length
        }

        // a separate, shared evaluation batch for swap energies (offset by Ntrain/2) -> computes the energy in swap decisions
        float Xswap[BATCH_SIZE][INPUT_DIM];
        int   yswap[BATCH_SIZE];

        int start_swap = (start + Ntrain/2) % Ntrain; //eval batch will be different from the training batch with this offset of indices but ranks still share it
        int bsz_swap   = (start_swap + BATCH_SIZE <= Ntrain) ? BATCH_SIZE : (Ntrain - start_swap);
        for (int b = 0; b < bsz_swap; ++b) {
            int idx = perm_tr[(start_swap + b) % Ntrain];
            memcpy(Xswap[b], Xtr[idx], sizeof(float)*INPUT_DIM);
            yswap[b] = ytr[idx];
        }
        for (int i = bsz_swap; i < BATCH_SIZE; ++i) {
            memset(Xswap[i], 0, sizeof(float)*INPUT_DIM);
            yswap[i] = 0;
        }

      // Exploration (hot) /exploitation (cold) kernel
      double t0 = MPI_Wtime(); //time compute time
      (void)forward_backward_batch(&model, Xbatch, ybatch, bsz); //minibatch forward and backward pass -> fills the global gradient buffers
      if (rank==0){
        // exploitation with SGLD noise on the target chain
        sgld_step(&model, (float)my_eta);
      } else {
        // exploration chains use pure SGD (no injected noise)
        sgd_step(&model, (float)my_eta);
        if (balance_sgld) burn_normals(nparam);
      }
      t_compute += MPI_Wtime() - t0;

        // post update stochastic energy (the cross entropy) on a SHARED eval batch which we only use for swapping and not gradient computation
        double myE = batch_loss_only(
            &model,
            (const float (*)[INPUT_DIM])Xswap,
            yswap,
            bsz_swap);
        
        if (!isfinite(myE)) {
            // Treat non-finite as bad (no swaps done safely)
            myE = INFINITY;
        }
        if (stab_stride > 0 && (k % stab_stride) == 0) {
          double theta_norm = model_l2_norm(&model);
          double step_norm  = have_prev ? model_diff_l2(&model, &prev_model) : 0.0;
          double growth     = have_prev && prev_norm>0.0 ? (theta_norm / prev_norm) : 1.0;

          double gnorm = grad_l2_norm();

          double L32 = myE;
          double L64 = batch_loss_only_fp64(&model, (const float (*)[INPUT_DIM])Xswap, yswap, bsz_swap);
          double rel_gap = fabs(L64 - L32) / fmax(1.0, fabs(L64));

          struct { double sum, max; } agg_theta={0,0}, agg_step={0,0}, agg_gn={0,0}, agg_gap={0,0};
          double v;

          v = theta_norm; MPI_Reduce(&v,&agg_theta.sum,1,MPI_DOUBLE,MPI_SUM,0,comm);
                          MPI_Reduce(&v,&agg_theta.max,1,MPI_DOUBLE,MPI_MAX,0,comm);
          v = step_norm;  MPI_Reduce(&v,&agg_step.sum, 1,MPI_DOUBLE,MPI_SUM,0,comm);
                          MPI_Reduce(&v,&agg_step.max, 1,MPI_DOUBLE,MPI_MAX,0,comm);
          v = gnorm;      MPI_Reduce(&v,&agg_gn.sum,   1,MPI_DOUBLE,MPI_SUM,0,comm);
                          MPI_Reduce(&v,&agg_gn.max,   1,MPI_DOUBLE,MPI_MAX,0,comm);
          v = rel_gap;    MPI_Reduce(&v,&agg_gap.sum,  1,MPI_DOUBLE,MPI_SUM,0,comm);
                          MPI_Reduce(&v,&agg_gap.max,  1,MPI_DOUBLE,MPI_MAX,0,comm);

          if (rank==0) {
            double Pd = (double)size; // == P
            printf("[stab] k=%d mean||θ||=%.6e max||θ||=%.6e "
                   "mean_step=%.6e max_step=%.6e "
                   "mean|∇|=%.6e max|∇|=%.6e "
                   "mean_rel_gap=%.3e max_rel_gap=%.3e "
                   "growth≈%.6e\n",
                   k,
                   agg_theta.sum/Pd, agg_theta.max,
                   agg_step.sum/Pd,  agg_step.max,
                   agg_gn.sum/Pd,    agg_gn.max,
                   agg_gap.sum/Pd,   agg_gap.max,
                   growth);
            fflush(stdout);
          }

          // snapshot for next stride
          prev_model = model;
          prev_norm  = theta_norm;
          have_prev  = 1;
        }
        //exchange energies with neighbors; build A^(p)
        t0 = MPI_Wtime();
        double E_left=0.0, E_right=0.0;
        //get your neighbours energies here (one left and one to the right)
        MPI_Sendrecv(&myE,1,MPI_DOUBLE,left,  10, &E_right,1,MPI_DOUBLE,right,10, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&myE,1,MPI_DOUBLE,right, 11, &E_left, 1,MPI_DOUBLE,left, 11, comm, MPI_STATUS_IGNORE);
        t_energy_comm += MPI_Wtime() - t0;

        // check if finite -> sanitize once and use consistently
        double myE_s = isfinite(myE) ? myE : INFINITY;
        double E_left_s = isfinite(E_left) ? E_left : INFINITY;
        double E_right_s = isfinite(E_right) ? E_right : INFINITY;

        if (rank==0 && (k%100)==0 && right != MPI_PROC_NULL) {
          printf("dbg: E0=%.4f  E1=%.4f  C=%.3f  diff=%.4f\n", myE_s, E_right_s, C, myE_s - E_right_s);
          fflush(stdout);
        }
        //build swap indicators used in calculation of dynamic swap correction value C
        int A_left = (rank>0 && (myE_s + C < E_left_s )) ? 1 : 0; //if model prefers left neighbour
        int A_right = (rank<P-1 && (E_right_s + C < myE_s)) ? 1 : 0; //if right preferable

      //Non-reversible swaps->DEO parity + gates
        t0 = MPI_Wtime();
        if ((rank % 2)==phase && right != MPI_PROC_NULL){
            int do_swap = (gate_right && A_right) ? 1 : 0;
            MPI_Send(&do_swap,1,MPI_INT,right,20,comm);
            if (do_swap){
                forest_serialize_f(&model, buf);
                MPI_Sendrecv_replace(buf, (int)nparam, MPI_FLOAT, right,21, right,21, comm, MPI_STATUS_IGNORE);
                forest_deserialize_f(&model, buf);
                gate_right = 0;
                
            }
        } else if (left != MPI_PROC_NULL && ((left % 2)==phase)){
            int do_swap_from_left=0;
            MPI_Recv(&do_swap_from_left,1,MPI_INT,left,20,comm,MPI_STATUS_IGNORE);
            if (do_swap_from_left){
                forest_serialize_f(&model, buf);
                MPI_Sendrecv_replace(buf, (int)nparam, MPI_FLOAT, left,21, left,21, comm, MPI_STATUS_IGNORE);
                forest_deserialize_f(&model, buf);
            }
        }
        t_swap_comm += MPI_Wtime() - t0; //add up the swap communication time

      // learning-rate update for interior replicas
      // exchange neighbor etas first
        if (P > 1) {
            double eta_left = my_eta, eta_right = my_eta;
            MPI_Sendrecv(&my_eta,1,MPI_DOUBLE,left,  30, &eta_right,1,MPI_DOUBLE,right,30, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&my_eta,1,MPI_DOUBLE,right, 31, &eta_left, 1,MPI_DOUBLE,left, 31, comm, MPI_STATUS_IGNORE);
            
            // share A values to get A^(p-1) and A^(p)
            int A_from_left=0, A_from_right=0;
            MPI_Sendrecv(&A_right,1,MPI_INT,right,32, &A_from_left, 1,MPI_INT,left,32, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&A_left, 1,MPI_INT,left, 33, &A_from_right,1,MPI_INT,right,33, comm, MPI_STATUS_IGNORE);
            
            double gk = gamma_k(gamma0, gdecay, k);
            double eta_prop = my_eta;
            if (rank>0 && rank<P-1){
                double uprev = my_eta - eta_left;
                double unext = eta_right- my_eta;
                double Hp = ((double)A_right - S);
                double Hm = ((double)A_from_left - S);
                double forward  = fmax(0.0, uprev) * exp(gk * Hm);
                double backward = fmax(0.0, unext) * exp(gk * Hp);
                eta_prop = 0.5*(eta_left + eta_right) + 0.5*(forward - backward);
            }
            // gather, enforce non-decreasing ladder, scatter back to others
            double* all_eta = NULL;
            if(rank==0) all_eta = (double*)malloc(sizeof(double)*P);
            MPI_Gather(&eta_prop,1,MPI_DOUBLE,all_eta,1,MPI_DOUBLE,0,comm);
            if(rank==0){ enforce_monotone(all_eta,P); }
            MPI_Scatter(all_eta,1,MPI_DOUBLE,&my_eta,1,MPI_DOUBLE,0,comm);
            if(rank==0) free(all_eta);
        }
      // correction buffer update (Allreduce on a bunch of ints)
        t0 = MPI_Wtime();
        // Count the indicator for our (rank, rank+1) pair
        int myA = (rank < P-1) ? A_right : 0;
        int sumA = 0;
        if (P > 1) MPI_Allreduce(&myA, &sumA, 1, MPI_INT, MPI_SUM, comm);
        double avgA = (P > 1) ? ((double)sumA / (double)(P - 1)) : 0.0;
        t_allreduce += MPI_Wtime() - t0;
        
        if (trace_stride > 0 && (k % trace_stride) == 0) {
          double dcomp = t_compute     - t_compute_prev;
          double de    = t_energy_comm - t_energy_comm_prev;
          double dsw   = t_swap_comm   - t_swap_comm_prev;
          double dar   = t_allreduce   - t_allreduce_prev;
          printf("[trace] r=%d k=%d d_compute=%.6f d_ecomm=%.6f d_swap=%.6f d_allred=%.6f "
                 "cum_compute=%.6f cum_ecomm=%.6f cum_swap=%.6f cum_allred=%.6f\n",
                 rank, k, dcomp, de, dsw, dar,
                 t_compute, t_energy_comm, t_swap_comm, t_allreduce);
          fflush(stdout);
          t_compute_prev     = t_compute;
          t_energy_comm_prev = t_energy_comm;
          t_swap_comm_prev   = t_swap_comm;
          t_allreduce_prev   = t_allreduce;
        }
        
        double gk_c = gamma_k(gamma0, gdecay, k);
        if (P > 1) C += gk_c * (avgA - S);
        
        if (rank == 0 && (k % 100) == 0) {
            printf("iter %d/%d  avgA=%.3f  S=%.2f  C=%.4f  eta(1)=%.5f\n",
                   k, K, avgA, S, C, my_eta);
            fflush(stdout);
        }

      // ---- snapshot of target chain ----
      if(rank==0 && snap_stride>0 && (k%snap_stride)==0){
        snaps[nsnaps++] = model;
      }
    }
    if (global_iter >= K) break;
  }
    
    double max_compute=0, max_ecomm=0, max_swap=0, max_allred=0;
    double sum_compute=0, sum_ecomm=0, sum_swap=0, sum_allred=0;

    MPI_Reduce(&t_compute,     &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_energy_comm, &max_ecomm,   1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_swap_comm,   &max_swap,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_allreduce,   &max_allred,  1, MPI_DOUBLE, MPI_MAX, 0, comm);

    MPI_Reduce(&t_compute,     &sum_compute, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&t_energy_comm, &sum_ecomm,   1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&t_swap_comm,   &sum_swap,    1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&t_allreduce,   &sum_allred,  1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank==0) {
      printf("[timings] compute=%.3fs  energy_comm=%.3fs  swap_comm=%.3fs  allreduce=%.3fs\n",
             t_compute,t_energy_comm,t_swap_comm,t_allreduce);
      printf("[timings:max] compute=%.3fs  energy_comm=%.3fs  swap_comm=%.3fs  allreduce=%.3fs\n",
             max_compute,max_ecomm,max_swap,max_allred);
      printf("[timings:sum] compute=%.3fs  energy_comm=%.3fs  swap_comm=%.3fs  allreduce=%.3fs\n",
             sum_compute,sum_ecomm,sum_swap,sum_allred);
    }
    
  // Evaluation on rank 0
  if(rank==0){
      double acc = eval_accuracy(&model, (const float (*)[INPUT_DIM])Xte, yte, Ntest);
    printf("[rank0] Point accuracy (last β^(1)) = %.4f\n", acc);

    if (nsnaps>0){
      // simple ensemble over snapshots (uniform)
      double correct = 0.0;
      float prob[OUTPUT_DIM];
      for(int i=0;i<Ntest;++i){
        double avg[OUTPUT_DIM]={0};
        for(int s=0;s<nsnaps;++s){
          model_predict(&snaps[s], Xte[i], prob);
          for(int o=0;o<OUTPUT_DIM;++o) avg[o]+=prob[o];
        }
        int argmax=0; double best=avg[0];
        for(int o=1;o<OUTPUT_DIM;++o) if(avg[o]>best){best=avg[o]; argmax=o;}
        if (argmax==yte[i]) correct+=1.0;
      }
      printf("[rank0] Ensemble accuracy (%d snaps) = %.4f\n", nsnaps, correct/(double)Ntest);
    }

    printf("[timings] compute=%.3fs  energy_comm=%.3fs  swap_comm=%.3fs  allreduce=%.3fs\n",
           t_compute,t_energy_comm,t_swap_comm,t_allreduce);
  }

  free(buf); free(eta); free(perm_tr);
  free(Xtr); free(ytr); free(Xte); free(yte);
    if (rank==0 && snaps) free(snaps);
  MPI_Finalize();
  return 0;
}
