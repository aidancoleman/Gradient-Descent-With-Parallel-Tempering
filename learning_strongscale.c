// learning_strongscale.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>

#include "forest_model.h"
#include "load_csv.h"
#include "load_bin_mpi.h"


// Minimal CLI (command line) helpers

static const char* arg_str(int argc,char**argv,const char*flag,const char*def){
  for (int i=1;i<argc-1;++i) if (!strcmp(argv[i],flag)) return argv[i+1];
  return def;
}
static int  arg_int(int argc,char**argv,const char*flag,int def){ const char* s=arg_str(argc,argv,flag,NULL); return s?atoi(s):def; }
static double arg_double(int argc,char**argv,const char*flag,double def){ const char* s=arg_str(argc,argv,flag,NULL); return s?atof(s):def; }


// Replica container

typedef struct {
  int replica_id; // 0..R-1
  double eta; // learning rate for this replica
  double last_loss; // minibatch loss => stochastic energy proxy
  Model model; // model state
} Replica;

// Block partition: contiguous chunks in mem
static inline int replica_to_rank_block(int rid, int size, int R) {
  int base = R / size;
  int rem = R % size;
  int edge = rem * (base + 1);
  if (rid < edge) return rid / (base + 1);
  return rem + (rid - edge) / base;
}
// Helper to see if CSC log file has exactly one header row, only rank 0 can write
static inline void write_header_if_needed(const char* path, int rank) {
  if (rank != 0) return;
  struct stat st;
  if (stat(path, &st) == 0) return; // exists => skip
  FILE* f = fopen(path, "a"); //open path in append mode
  if (!f) return;
  fprintf(f,
    "P,R,K_total,swap_stride,W,N,Ntrain,Ntest,eta_lo,eta_hi,S,gamma0,gdecay,"
    "time_sec,swap_attempts,swap_accepts,swap_rate,"
    "avg_replica_acc,ensemble_acc,mean_entropy,mean_var_ratio\n");
  fclose(f);
}

static inline double gamma_k(double g0, double gdecay, int k) {
  return g0 / (1.0 + gdecay * (double)k);
}

// DEO Window (guarded numerically)
static int compute_W_deostar(int P, double S){
  if (P < 4) return 1; //for less ranks and replicas use a single DEO block as small values of P in window calculation make the W formula noisy
  double r = 1.0 - S; //target rejection rate
  if (r <= 1e-12) r = 1e-12; //clamp r from pathological cases of boundary values preventing division by 0
  if (r >= 1.0 - 1e-12) r = 1.0 - 1e-12;
  double lp  = (P>1) ? log((double)P) : 1.0;
  double llp = log(lp);
  if (!isfinite(llp)) llp = 0.0;
  double num = log((double)P) + (llp > 0.0 ? llp : 0.0); //only add loglogP when its positive
  double den = -log(r); //denominator for window calculation
  int W = (int)ceil(num / den); //round up to an integer
  if (W < 1) W = 1; //avoid pathological window size cases
  if (W > 1000000) W = 1000000;
  return W;
}

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank,size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);
  if (rank==0) setvbuf(stdout, NULL, _IOLBF, 0);
  
    // CLI
  const int R = arg_int (argc,argv,"--replicas", 8);
  const int K_total = arg_int (argc,argv,"--K_total",  80000);
  const int swap_stride = arg_int (argc,argv,"--swap_stride", 100);
  const int print_stride = arg_int (argc,argv,"--print_stride", 1000);
  const double eta_lo = arg_double(argc,argv,"--eta_lo", 3e-3);
  const double eta_hi = arg_double(argc,argv,"--eta_hi", 2e-2);
  const double val_frac = arg_double(argc,argv,"--val", 0.2);
  const unsigned seed = (unsigned)arg_int(argc,argv,"--seed", 42);
  const char* csv = arg_str (argc,argv,"--csv", NULL);
  const char* bin = arg_str (argc,argv,"--bin", NULL);
  const char* log_csv_path = arg_str (argc,argv,"--log_csv", "results_strongscale.csv");

  // Adaptive correction buffer
  const double S = arg_double(argc,argv,"--S", 0.4);
  const double gamma0 = arg_double(argc,argv,"--gamma0", 0.05);
  const double gdecay = arg_double(argc,argv,"--gdecay", 1e-4);
  double C = 0.0; // correction buffer
  int win_index = 0; // swap-window counter

  if (R <= 0) {
    if (rank==0) fprintf(stderr, "R must be >= 1\n");
    MPI_Abort(comm,1);
  }
  if (!csv && !bin) {
    if (rank==0) fprintf(stderr,"pass --csv <file> or --bin <file>\n");
    MPI_Abort(comm,1);
  }

  // Compute DEO window size W
  const int W = compute_W_deostar(size, S);

  if (rank==0) {
    printf("[config] P=%d R=%d K_total=%d swap_stride=%d W=%d eta_lo=%.4g eta_hi=%.4g val=%.2f seed=%u print_stride=%d S=%.3f gamma0=%.3f gdecay=%.2e\n",
           size, R, K_total, swap_stride, W, eta_lo, eta_hi, val_frac, seed, print_stride, S, gamma0, gdecay);
  }

  const double t0 = MPI_Wtime();

  // Early CSV existence check
  
  if (csv) {
    int ok = 1; //csv provided, proceed
    if (rank == 0) {
      FILE* tf = fopen(csv, "rb"); // open in read binary mode
      if (!tf) { fprintf(stderr, "[io] cannot open %s: %s\n", csv, strerror(errno)); ok = 0; }
      else fclose(tf);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm); // broadcast result to all ranks
    if (!ok) MPI_Abort(comm,77); // abort whole job if rank 0 couldn't open the csv
  }

  // Data load (all-replica)

  float (*X)[INPUT_DIM] = NULL;
  int32_t* y = NULL;
  int Nall = 0;

  if (bin) {
    if (load_bin_mpi_allrep(bin, &X, &y, &Nall, comm) != 0) {
      if (rank==0) fprintf(stderr,"[io] failed to load %s via MPI-IO\n", bin);
      MPI_Abort(comm, 99);
    }
    if (rank==0) printf("[io] bin OK N=%d dim=%d\n", Nall, INPUT_DIM);
  } else {
    float (*Xall)[INPUT_DIM] = NULL;
    int32_t* yall = NULL;
    if (rank==0) {
      load_csv(csv, &Xall, &yall, &Nall);
      if (Nall<=0) { fprintf(stderr,"[io] empty dataset\n"); MPI_Abort(comm,2); }
      long cnt[OUTPUT_DIM]={0};
      for (int i=0;i<Nall;++i){ int yy=yall[i]; if (0<=yy && yy<OUTPUT_DIM) cnt[yy]++; }
      printf("[io] label counts:"); for (int c=0;c<OUTPUT_DIM;++c) printf(" %ld",cnt[c]); printf("\n");
    }
    MPI_Bcast(&Nall,1,MPI_INT,0,comm);
    float* Xflat = (float*)malloc((size_t)Nall*INPUT_DIM*sizeof(float));
    y = (int32_t*)malloc((size_t)Nall*sizeof(int32_t));
    if (rank==0){
      memcpy(Xflat, &Xall[0][0], (size_t)Nall*INPUT_DIM*sizeof(float));
      memcpy(y, yall, (size_t)Nall*sizeof(int32_t));
      free(Xall); free(yall);
    }
    MPI_Bcast(Xflat, (int)(Nall*INPUT_DIM), MPI_FLOAT, 0, comm);
    MPI_Bcast(y, Nall, MPI_INT, 0, comm);
    X = (float (*)[INPUT_DIM])Xflat;
    if (rank==0) printf("[io] csv OK N=%d dim=%d\n", Nall, INPUT_DIM);
  }
 
  // Shared split & normalize
  int *perm_split = (int*)malloc((size_t)Nall*sizeof(int)); // allocate index array of length Nall
  if (rank==0){
    srand(seed);
    for (int i=0;i<Nall;++i) perm_split[i]=i;
    for (int i=Nall-1;i>0;--i){ int j=rand()%(i+1); int t=perm_split[i]; perm_split[i]=perm_split[j]; perm_split[j]=t; } //in place index shuffle for uniform random permutation
  }
  MPI_Bcast(perm_split, Nall, MPI_INT, 0, comm);
    // Deciding sizes of train and test set :
  const int Ntest  = (int)lrint((double)Nall * val_frac);
  const int Ntrain = Nall - Ntest;
    //New arrays allocated for split and copy rows:
  float (*Xtr)[INPUT_DIM] = (float (*)[INPUT_DIM])malloc((size_t)Ntrain*INPUT_DIM*sizeof(float));
  int32_t* ytr = (int32_t*)malloc((size_t)Ntrain*sizeof(int32_t));
  float (*Xte)[INPUT_DIM] = (float (*)[INPUT_DIM])malloc((size_t)Ntest*INPUT_DIM*sizeof(float));
  int32_t* yte = (int32_t*)malloc((size_t)Ntest*sizeof(int32_t));
  for (int i=0;i<Ntrain;++i){ int idx=perm_split[i]; memcpy(Xtr[i], X[idx], sizeof(float)*INPUT_DIM); ytr[i]=y[idx]; }
  for (int i=0;i<Ntest; ++i){ int idx=perm_split[Ntrain+i]; memcpy(Xte[i], X[idx], sizeof(float)*INPUT_DIM); yte[i]=y[idx]; }
  free(perm_split); free(X); free(y);

  // train stats (training set per-feature, done by rank 0)
  double mean[INPUT_DIM], invstd[INPUT_DIM];
  if (rank==0){
    for (int d=0; d<INPUT_DIM; ++d) mean[d]=0.0;
    for (int i=0; i<Ntrain; ++i)
      for (int d=0; d<INPUT_DIM; ++d) mean[d]+=Xtr[i][d];
    for (int d=0; d<INPUT_DIM; ++d) mean[d]/=(double)Ntrain;
    double sq[INPUT_DIM]; for (int d=0; d<INPUT_DIM; ++d) sq[d]=0.0;
    for (int i=0; i<Ntrain; ++i)
      for (int d=0; d<INPUT_DIM; ++d){ double v=Xtr[i][d]; sq[d]+=v*v; }
    for (int d=0; d<INPUT_DIM; ++d){
      double var = sq[d]/(double)Ntrain - mean[d]*mean[d];
      invstd[d] = 1.0 / sqrt(var + 1e-8);
    }
  }
  MPI_Bcast(mean,  INPUT_DIM, MPI_DOUBLE, 0, comm);
  MPI_Bcast(invstd,INPUT_DIM, MPI_DOUBLE, 0, comm);
    //standardise train and test sets with these statistics:
  for (int i=0; i<Ntrain; ++i)
    for (int d=0; d<INPUT_DIM; ++d)
      Xtr[i][d] = (float)((Xtr[i][d]-mean[d])*invstd[d]);
  for (int i=0; i<Ntest; ++i)
    for (int d=0; d<INPUT_DIM; ++d)
      Xte[i][d] = (float)((Xte[i][d]-mean[d])*invstd[d]);

  // Shared minibatch order
  int *perm_tr = (int*)malloc((size_t)Ntrain*sizeof(int));
  if (!perm_tr){ perror("perm_tr"); MPI_Abort(comm,3); }
  if (rank==0){
    srand(seed+1);
    for (int i=0;i<Ntrain;++i) perm_tr[i]=i;
    for (int i=Ntrain-1;i>0;--i){ int j=rand()%(i+1); int t=perm_tr[i]; perm_tr[i]=perm_tr[j]; perm_tr[j]=t; }
  }
  MPI_Bcast(perm_tr, Ntrain, MPI_INT, 0, comm);

  
  // Build LR ladder and assign replicas (block)
 
  double *eta = (double*)malloc(sizeof(double)*R);
  if (R==1) eta[0]=eta_lo;
  else for (int r=0;r<R;++r){
    double t=(double)r/(double)(R-1);
    eta[r]=eta_lo*pow(eta_hi/eta_lo,t);
  }

  // Local replica IDs for this rank
  int rid_lo, rid_hi; // inclusive
  {
    int base = R/size, rem=R%size;
    if (rank < rem) { rid_lo = rank*(base+1); rid_hi = rid_lo + (base+1) - 1; }
    else { rid_lo = rem*(base+1) + (rank-rem)*base; rid_hi = rid_lo + base - 1; }
  }
  int R_local = (rid_hi>=rid_lo) ? (rid_hi-rid_lo+1) : 0;

  Replica* reps = (Replica*)malloc((size_t)R_local*sizeof(Replica));
  for (int i=0;i<R_local;++i){
    int rid = rid_lo + i;
    reps[i].replica_id = rid;
    reps[i].eta = eta[rid];
    init_model(&reps[i].model);
    reps[i].last_loss = 0.0;
  }

  if (rank==0) {
    printf("[assign] block mapping over ranks (P=%d):\n", size);
    for (int r=0;r<size;++r){
      int lo,hi;
      int base = R/size, rem=R%size;
      if (r < rem) { lo = r*(base+1); hi = lo + (base+1) - 1; }
      else { lo = rem*(base+1) + (r-rem)*base; hi = lo + base - 1; }
      printf("  rank %d: replicas [%d..%d]\n", r, lo, hi);
    }
  }

  
  // Training (global-k outer loop) + DEO with symmetric MPI
 
  const int iters_per_epoch   = (Ntrain + BATCH_SIZE - 1) / BATCH_SIZE;
  const int steps_per_replica = K_total / R;

  // DEO gating per adjacent pair (p=0..R-2), owned by initiator (lower id) rank
  unsigned char *gate = (unsigned char*)malloc((size_t)(R>1 ? R-1 : 1));
  const int include_wrap = 1; // include wrap-around edge {R-1,0}
  unsigned char gate_wrap = 0;
  if (gate) memset(gate, 0, (size_t)(R>1 ? R-1 : 1));

  int swap_attempts_local_total = 0;
  int swap_accepts_local_total  = 0;

  for (int k = 0; k < steps_per_replica; ++k) {

    // Step ALL local replicas once at global step k
    for (int i = 0; i < R_local; ++i) {
      Replica* rep = &reps[i];
      int ib = (k % iters_per_epoch) * BATCH_SIZE;
      int bsz = (ib + BATCH_SIZE <= Ntrain) ? BATCH_SIZE : (Ntrain - ib);

      float batch[BATCH_SIZE][INPUT_DIM];
      int   blabel[BATCH_SIZE];
      for (int j=0;j<bsz;++j){
        int idx = perm_tr[ib+j];
        memcpy(batch[j], Xtr[idx], sizeof(float)*INPUT_DIM);
        blabel[j] = ytr[idx];
      }
      rep->last_loss = forward_backward_batch(&rep->model, batch, blabel, bsz);
      sgld_step(&rep->model, rep->eta);
    }

    if (print_stride>0 && rank==0 && (k % print_stride)==0) {
      double show_loss = (R_local>0) ? reps[0].last_loss : 0.0;
      printf("[train] k=%d / %d  loss0=%.4f  C=%.5f\n", k, steps_per_replica, show_loss, C);
      fflush(stdout);
    }

    // DEO swap window (deadlock-proof via symmetric Sendrecv)
      if (swap_stride > 0 && k > 0 && (k % swap_stride) == 0) {
        const int even_block = ((win_index / W) % 2 == 0);
        const int new_block  = (win_index % W == 0);

        // Open gates at the start of each parity block for active edges only.
        // (We keep gates for p=0..R-2 in 'gate[p]' and a separate 'gate_wrap' for {R-1,0}.)
        if (new_block && R > 1) {
          // Non-wrap edges
          if (gate) memset(gate, 0, (size_t)(R-1));
          for (int p = 0; p < R-1; ++p) {
            const int is_even_edge = (p % 2 == 0);
            if ( (even_block && is_even_edge) || (!even_block && !is_even_edge) ) {
              if (replica_to_rank_block(p, size, R) == rank) gate[p] = 1;
            }
          }
          // Wrap edge {R-1,0} => choose initiator == (R-1)
          const int p_wrap = R - 1;
          const int is_even_edge_wrap = (p_wrap % 2 == 0);
          gate_wrap = 0;
          if ( (even_block && is_even_edge_wrap) || (!even_block && !is_even_edge_wrap) ) {
            if (replica_to_rank_block(p_wrap, size, R) == rank) gate_wrap = 1;
          }
        }

        int attempts_win_local = 0, accepts_win_local = 0;

        //  Non-wrap adjacent pairs: (p, p+1) for p=0..R-2
        for (int p = 0; p < R-1; ++p) {
          const int active_edge = even_block ? (p % 2 == 0) : (p % 2 == 1);
          if (!active_edge) continue;

          const int q = p + 1;
          const int p_rank = replica_to_rank_block(p, size, R);
          const int q_rank = replica_to_rank_block(q, size, R);

          // Local indices if present
          int ip = -1, iq = -1;
          if (p_rank == rank) for (int i=0;i<R_local;++i) if (reps[i].replica_id == p) { ip = i; break; }
          if (q_rank == rank) for (int i=0;i<R_local;++i) if (reps[i].replica_id == q) { iq = i; break; }

          // Case A: both local => in-memory swap
          if (p_rank == rank && q_rank == rank) {
            if (ip < 0 || iq < 0) continue;
            const int gate_open = gate ? gate[p] : 1;
            if (gate_open) {
              attempts_win_local++;
              const double loss_p = reps[ip].last_loss;
              const double loss_q = reps[iq].last_loss;
              const int do_swap = (loss_q + C < loss_p);
              if (do_swap) {
                Model tmp = reps[ip].model; reps[ip].model = reps[iq].model; reps[iq].model = tmp;
                double tl = reps[ip].last_loss; reps[ip].last_loss = reps[iq].last_loss; reps[iq].last_loss = tl;
                accepts_win_local++;
                if (gate) gate[p] = 0; // freeze after a swap
              }
            }
            continue;
          }

          // Case B: cross-rank edge => symmetric Sendrecv (deadlock-proof)
          if (rank == p_rank || rank == q_rank) {
            const int am_p = (rank == p_rank);      // initiator owner
            const int peer = am_p ? q_rank : p_rank;

            struct Payload { double loss; int gate_flag; } sendv, recvv;
            if (am_p) {
              if (ip < 0) continue;
              sendv.loss = reps[ip].last_loss;
              sendv.gate_flag = gate ? gate[p] : 1;
            } else {
              if (iq < 0) continue;
              sendv.loss = reps[iq].last_loss;
              sendv.gate_flag = 0; // only initiator chooses to open gate
            }

            const int tag_base = 10000 + p;
            MPI_Sendrecv(&sendv, sizeof sendv, MPI_BYTE, peer, tag_base+0,
                         &recvv, sizeof recvv, MPI_BYTE, peer, tag_base+0,
                         comm, MPI_STATUS_IGNORE);

            const double loss_p = am_p ? sendv.loss : recvv.loss;
            const double loss_q = am_p ? recvv.loss : sendv.loss;
            const int gate_open_from_p = am_p ? sendv.gate_flag : recvv.gate_flag;

            if (am_p && gate_open_from_p) attempts_win_local++;
            const int do_swap = gate_open_from_p && ((loss_q + C) < loss_p);

            if (do_swap) {
              if (am_p) {
                MPI_Sendrecv(&reps[ip].model, sizeof(Model), MPI_BYTE, peer, tag_base+2,
                             &reps[ip].model, sizeof(Model), MPI_BYTE, peer, tag_base+3,
                             comm, MPI_STATUS_IGNORE);
                reps[ip].last_loss = loss_q;
                accepts_win_local++;
                if (gate) gate[p] = 0; // freeze after a swap
              } else {
                MPI_Sendrecv(&reps[iq].model, sizeof(Model), MPI_BYTE, peer, tag_base+3,
                             &reps[iq].model, sizeof(Model), MPI_BYTE, peer, tag_base+2,
                             comm, MPI_STATUS_IGNORE);
                reps[iq].last_loss = loss_p;
              }
            }
          } // cross-rank
        } // for p in 0..R-2

        //Wrap-around adjacent pair: (R-1, 0)
        if (R > 1) {
          const int p = R - 1, q = 0; // initiator == p (R-1)
          const int active_edge = even_block ? (p % 2 == 0) : (p % 2 == 1);
          if (active_edge) {
            const int p_rank = replica_to_rank_block(p, size, R);
            const int q_rank = replica_to_rank_block(q, size, R);

            int ip = -1, iq = -1;
            if (p_rank == rank) for (int i=0;i<R_local;++i) if (reps[i].replica_id == p) { ip = i; break; }
            if (q_rank == rank) for (int i=0;i<R_local;++i) if (reps[i].replica_id == q) { iq = i; break; }

            if (p_rank == rank && q_rank == rank) {
              // local swap
              if (gate_wrap && ip >= 0 && iq >= 0) {
                attempts_win_local++;
                const double loss_p = reps[ip].last_loss;
                const double loss_q = reps[iq].last_loss;
                const int do_swap = (loss_q + C < loss_p);
                if (do_swap) {
                  Model tmp = reps[ip].model; reps[ip].model = reps[iq].model; reps[iq].model = tmp;
                  double tl = reps[ip].last_loss; reps[ip].last_loss = reps[iq].last_loss; reps[iq].last_loss = tl;
                  accepts_win_local++; gate_wrap = 0;
                }
              }
            } else if (rank == p_rank || rank == q_rank) {
              // cross-rank swap with symmetric Sendrecv
              const int am_p = (rank == p_rank); // initiator is p==R-1
              const int peer = am_p ? q_rank : p_rank;
              struct Payload { double loss; int gate_flag; } sendv, recvv;
              if (am_p) { sendv.loss = reps[ip].last_loss; sendv.gate_flag = gate_wrap; }
              else      { sendv.loss = reps[iq].last_loss; sendv.gate_flag = 0; }

              const int tag_base = 20000 + p;
              MPI_Sendrecv(&sendv, sizeof sendv, MPI_BYTE, peer, tag_base+0,
                           &recvv, sizeof recvv, MPI_BYTE, peer, tag_base+0,
                           comm, MPI_STATUS_IGNORE);

              const double loss_p = am_p ? sendv.loss : recvv.loss;
              const double loss_q = am_p ? recvv.loss : sendv.loss;
              const int gate_open_from_p = am_p ? sendv.gate_flag : recvv.gate_flag;

              if (am_p && gate_open_from_p) attempts_win_local++;
              const int do_swap = gate_open_from_p && ((loss_q + C) < loss_p);

              if (do_swap) {
                if (am_p) {
                  MPI_Sendrecv(&reps[ip].model, sizeof(Model), MPI_BYTE, peer, tag_base+2,
                               &reps[ip].model, sizeof(Model), MPI_BYTE, peer, tag_base+3,
                               comm, MPI_STATUS_IGNORE);
                  reps[ip].last_loss = loss_q;
                  accepts_win_local++; gate_wrap = 0;
                } else {
                  MPI_Sendrecv(&reps[iq].model, sizeof(Model), MPI_BYTE, peer, tag_base+3,
                               &reps[iq].model, sizeof(Model), MPI_BYTE, peer, tag_base+2,
                               comm, MPI_STATUS_IGNORE);
                  reps[iq].last_loss = loss_p;
                }
              }
            }
          }
        }

        // Aggregate window attempts/accepts and update C
        int attempts_win=0, accepts_win=0;
        MPI_Allreduce(&attempts_win_local, &attempts_win, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&accepts_win_local,  &accepts_win,  1, MPI_INT, MPI_SUM, comm);

        swap_attempts_local_total += attempts_win_local;
        swap_accepts_local_total  += accepts_win_local;

        if (rank == 0) {
          // Include wrap edge in the average
          const double avg_A = (double)accepts_win / (double)(R);
          const double g = gamma_k(gamma0, gdecay, win_index);
          C += g * (avg_A - S);
        }
        win_index++;
        MPI_Bcast(&C, 1, MPI_DOUBLE, 0, comm);
      }// end DEO window
  } // end global k loop

  // Save local replicas

  for (int i=0;i<R_local;++i){
    char fn[128];
    snprintf(fn,sizeof fn,"model.replica%d.rank%d.bin", reps[i].replica_id, rank);
    FILE* f=fopen(fn,"wb");
    if (f){ fwrite(&reps[i].model, sizeof(Model), 1, f); fclose(f); }
  }

  // Evaluation: replica accuracies + ensemble + stability

  int correct_local = 0;
  for (int i=0;i<R_local;++i){
    Replica* rep = &reps[i];
    int corr=0;
    for (int t=0;t<Ntest;++t){
      float probs[OUTPUT_DIM];
      model_predict(&rep->model, Xte[t], probs);
      int pred=0;
      for (int c=1;c<OUTPUT_DIM;++c) if (probs[c]>probs[pred]) pred=c;
      if (pred == yte[t]) corr++;
    }
    correct_local += corr;
  }
  int correct_global = 0;
  MPI_Reduce(&correct_local, &correct_global, 1, MPI_INT, MPI_SUM, 0, comm);
  double avg_replica_acc = 0.0;
  if (rank==0) avg_replica_acc = (double)correct_global / (double)(Ntest*R);

  // Ensemble: sum probs across all replicas
  double *sum_probs_local = (double*)calloc((size_t)Ntest*OUTPUT_DIM, sizeof(double));
  int *votes_local = (int*)calloc((size_t)Ntest*OUTPUT_DIM, sizeof(int));
  for (int i=0;i<R_local;++i){
    for (int t=0;t<Ntest;++t){
      float probs[OUTPUT_DIM];
      model_predict(&reps[i].model, Xte[t], probs);
      int pred=0; for (int c=1;c<OUTPUT_DIM;++c) if(probs[c]>probs[pred]) pred=c;
      votes_local[t*OUTPUT_DIM + pred] += 1;
      for (int c=0;c<OUTPUT_DIM;++c)
        sum_probs_local[t*OUTPUT_DIM + c] += (double)probs[c];
    }
  }

  double *sum_probs_global = NULL;
  int *votes_global = NULL;
  if (rank==0){
    sum_probs_global = (double*)calloc((size_t)Ntest*OUTPUT_DIM, sizeof(double));
    votes_global = (int*) calloc((size_t)Ntest*OUTPUT_DIM, sizeof(int));
  }
  MPI_Reduce(sum_probs_local, sum_probs_global, Ntest*OUTPUT_DIM, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(votes_local, votes_global, Ntest*OUTPUT_DIM, MPI_INT, MPI_SUM, 0, comm);

  double ensemble_acc = 0.0, mean_entropy = 0.0, mean_varratio = 0.0;
  if (rank==0){
    int ens_correct = 0;
    for (int t=0;t<Ntest;++t){
      // ensemble probs = average across R
      double maxp = -1.0; int pred=0;
      for (int c=0;c<OUTPUT_DIM;++c){
        double p = sum_probs_global[t*OUTPUT_DIM + c] / (double)R;
        if (p > maxp){ maxp=p; pred=c; }
      }
      if (pred == yte[t]) ens_correct++;

      // entropy
      double H=0.0;
      for (int c=0;c<OUTPUT_DIM;++c){
        double p = sum_probs_global[t*OUTPUT_DIM + c] / (double)R;
        if (p>0) H += -p*log(p + 1e-12);
      }
      mean_entropy += H;

      // variation ratio: 1 - (mode_votes / R)
      int mode_votes = 0;
      for (int c=0;c<OUTPUT_DIM;++c){
        int v = votes_global[t*OUTPUT_DIM + c];
        if (v > mode_votes) mode_votes = v;
      }
      double vr = 1.0 - (double)mode_votes / (double)R;
      mean_varratio += vr;
    }
    ensemble_acc = (double)ens_correct / (double)Ntest;
    mean_entropy /= (double)Ntest;
    mean_varratio /= (double)Ntest;
  }

  // Timing & swap metrics (totals)

  const double t1 = MPI_Wtime();
  double local_time = t1 - t0, time_sec = 0.0;
  MPI_Reduce(&local_time,&time_sec,1,MPI_DOUBLE,MPI_MAX,0,comm);

  int swap_attempts = 0, swap_accepts = 0;
  MPI_Reduce(&swap_attempts_local_total,&swap_attempts,1,MPI_INT,MPI_SUM,0,comm);
  MPI_Reduce(&swap_accepts_local_total,&swap_accepts, 1,MPI_INT,MPI_SUM,0,comm);
  double swap_rate = (swap_attempts>0)? ((double)swap_accepts/(double)swap_attempts) : 0.0;

  if (rank==0){
    printf("[timing] max wall time: %.3fs\n", time_sec);
    printf("[swaps] attempts=%d accepts=%d rate=%.2f%% (totals)\n", swap_attempts, swap_accepts, 100.0*swap_rate);
    printf("[acc] avg_replica=%.4f ensemble=%.4f\n", avg_replica_acc, ensemble_acc);
    printf("[stab] mean_entropy=%.4f mean_var_ratio=%.4f\n", mean_entropy, mean_varratio);
  }


  // CSV logging (rank 0)

  if (log_csv_path && rank==0){
    write_header_if_needed(log_csv_path, 0);
    FILE* f = fopen(log_csv_path, "a");
    if (f){
      fprintf(f,
        "%d,%d,%d,%d,%d,%d,%d,%d,%.8g,%.8g,%.4f,%.4f,%.2e,"
        "%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        size, R, K_total, swap_stride, W, Nall, Ntrain, Ntest,
        eta_lo, eta_hi, S, gamma0, gdecay,
        time_sec, swap_attempts, swap_accepts, swap_rate,
        avg_replica_acc, ensemble_acc, mean_entropy, mean_varratio);
      fclose(f);
    }
  }

  // cleanup
  free(eta);
  free(reps);
  free(gate);
  free(Xtr); free(ytr);
  free(Xte); free(yte);
  free(perm_tr);
  free(sum_probs_local);
  free(votes_local);
  if (rank==0){
    free(sum_probs_global);
    free(votes_global);
  }

  MPI_Finalize();
  return 0;
}
