
Parallel Tempering for Deep Learning (MPI + OpenMP)

This project trains a neural network using Parallel Tempering across MPI replicas with OpenMP for intra-replica calculation acceleration. It provides:

-> strong-scaling driver (train_strongscale, derived from learning_strongscale.c) that holds total work (data, replicas, training iterations) fixed and varies P.

-> a data weak-scaling driver (data_weak_scaling.c) that grows the data used with MPI ranks P.

-> and a classic PT driver (train_pt_mpi, from train_pt_mpi.c) used by the replica-weak-scaling script.

=================
Requirements
===========

-> MPI toolchain (mpicc, mpiexec): Open MPI or MPICH/Intel MPI (Hydra)

-> C11 compiler with OpenMP

-> POSIX shell

=============
Build
========

Use the provided Makefile (builds all executables and links common sources).

# from repo root
make

# clean
make clean


This produces:

train_strongscale from learning_strongscale.c

data_weak_scaling from data_weak_scaling.c

train_pt_mpi from train_pt_mpi.c

All link against: forest_model.c (network training functions), load_csv.c (load data as csv), load_bin_mpi.c (load data in binary format).

==================
Clean and Make
============
make clean
make MPICC=mpicc CFLAGS='-O3 -march=native -std=c11 -fopenmp -Wall -Wextra -MMD -MP'

============
Dataset
========
Find at: https://huggingface.co/datasets/mstz/covertype

Binary: covtype.bin — used by train_strongscale (--bin …) for fast I/O.

CSV: covtype.csv — used by data_weak_scaling (--csv …). Must have 55 comma-separated numbers per row (54 features + class label 1..7), no header. Can use "sed -i" commands to get rid of extra row if needed.

=====================
Helpful sanity checks
=================

# verify each row has 55 fields (OK message at end)
awk -F, 'NF!=55{printf "bad row %d has %d fields\n",NR,NF; exit 1} END{print "OK: all rows have 55 fields"}' "$HOME/covtype.csv"

# strip a UTF-8 BOM and any blank first line if present
awk 'NR==1{sub(/^\xef\xbb\xbf/,"")}1' "$HOME/covtype.csv" > "$HOME/.tmp" && mv "$HOME/.tmp" "$HOME/covtype.csv"
sed -i '1{/^[[:space:]]*$/d;}' "$HOME/covtype.csv"

===================
Threads and binding 
=============
OpenMP threads are per MPI rank.

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores


MPICH/Intel MPI: pass env with -genv on mpiexec

Open MPI: pass env with -x on mpirun

Avoid oversubscription: ensure P * OMP_NUM_THREADS is less than or equal to physical cores per node.

Strong scaling (fixed problem; vary P)
Quick smoke test:

Keeps replicas and total steps fixed; sweeps P = 1,2,4,8. Finishes fast and prints live progress.

MPICH/Intel MPI:

export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
R=8; K=800; SWAP=20; PRINT=20; CSV_OUT=/tmp/strongscale_quick.csv

for P in 1 2 4 8; do
  echo "==== P=$P (R=$R, K_total=$K) ===="
  mpiexec -np $P \
    -genv OMP_NUM_THREADS $OMP_NUM_THREADS \
    -genv OMP_PROC_BIND  $OMP_PROC_BIND \
    -genv OMP_PLACES     $OMP_PLACES \
    ./train_strongscale \
      --bin "$HOME/covtype.bin" \
      --replicas $R \
      --K_total $K \
      --swap_stride $SWAP \
      --S 0.4 --gamma0 0.05 --gdecay 1e-4 \
      --print_stride $PRINT \
      --log_csv "$CSV_OUT"
done


Open MPI: replace the three -genv ... with -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES on mpirun.

================
Overnight log to store strong scaling results if overnight job, use nohup
==============

Full sweep via script (background)
chmod +x learning_strong_scaling.sh
OMP_NUM_THREADS=4 nohup bash learning_strong_scaling.sh \
  --bin "$HOME/covtype.bin" \
  --replicas 8 \
  --K_total 24000 \
  --swap_stride 100 \
  --S 0.4 --gamma0 0.05 --gdecay 1e-4 \
  --print_stride 1000 \
  --log_csv results_strongscale.csv \
  > overnight_strong.log 2>&1 &
tail -f overnight_strong.log

======================================
Data weak scaling (grow replicas and data)
=======================

Executable: data_weak_scaling (CSV input). Keep threads/rank constant; increase P. Use --per_rank so N_use ≈ per_rank * P (capped by dataset). Leave algorithm knobs fixed across the sweep.

ITERS=2000
PER_RANK=20000
STAB=200
TRACE=500
OUT=/tmp/weak_quick.csv
echo "P,N_use,time_sec" > "$OUT"

for P in 1 2 4 8; do
  echo "==== P=$P (N_use ≈ $((PER_RANK*P))) ===="
  /usr/bin/time -f "%e" -o /tmp/t.$$ \
  mpiexec -np $P \
    -genv OMP_NUM_THREADS $OMP_NUM_THREADS \
    -genv OMP_PROC_BIND  $OMP_PROC_BIND \
    -genv OMP_PLACES     $OMP_PLACES \
    ./data_weak_scaling \
      --csv "$CSV" \
      --iters $ITERS \
      --per_rank $PER_RANK \
      --S 0.4 --eta_lo 3e-3 --eta_hi 2e-2 \
      --gamma0 0.05 --gdecay 1e-4 \
      --stab_stride $STAB --trace_stride $TRACE
  ELAPSED=$(cat /tmp/t.$$); rm -f /tmp/t.$$
  printf "%d,%d,%.3f\n" "$P" "$((PER_RANK*P))" "$ELAPSED" | tee -a "$OUT"
done

# Weak-scaling efficiency (~1.0 is ideal: flat time across P)
awk -F, 'NR==2{t1=$3} NR>1{printf "P=%s  time=%.3fs  weak_eff=%.2f\n",$1,$3,t1/$3}' "$OUT"

Full sweep via script

Edit the P-list inside data_weak_scaling.sh (e.g., 1 2 4 8), then:

chmod +x data_weak_scaling.sh
bash -x ./data_weak_scaling.sh \
  --csv "$HOME/covtype.csv" \
  --iters 20000 \
  --per_rank 30000 \
  --S 0.4 --eta_lo 3e-3 --eta_hi 2e-2 \
  --gamma0 0.05 --gdecay 1e-4 \
  --stab_stride 500 --trace_stride 1000

Replica weak scaling (fixed data, more replicas)

Script: paper_weak_scaling.sh runs train_pt_mpi with one replica per rank while keeping dataset size fixed.

chmod +x paper_weak_scaling.sh
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
bash -x ./paper_weak_scaling.sh \
  --csv "$HOME/covtype.csv" \
  --iters 20000 \
  --S 0.4 --eta_lo 3e-3 --eta_hi 2e-2 \
  --gamma0 0.05 --gdecay 1e-4 \
  --stab_stride 500 --trace_stride 1000

Reading results

Strong scaling: results_strongscale.csv (header written once by rank 0; one row per run).

Weak scaling (quick loop above): /tmp/weak_quick.csv (P,N_use,time_sec table).

Speedup/efficiency from strong CSV (time column time_sec):

awk -F, 'NR==1{next} NR==2{t1=$14} {printf "P=%s  time=%.3fs  speedup=%.3f  eff=%.1f%%\n",$1,$14,t1/$14,(t1/$14)/$1*100}' results_strongscale.csv

==============
Troubleshooting
==========

CSV parse error with data_weak_scaling: ensure no header/BOM; each row has 55 fields (see Datasets fixes).

-x not recognized on mpiexec: you’re on MPICH/Intel MPI; use -genv instead.

Oversubscription: reduce OMP_NUM_THREADS or the P sweep so P*threads is less than or equal to total cores.
