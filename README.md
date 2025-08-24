
# Parallel Tempering for Deep Learning (MPI + OpenMP)

This project trains a neural network using **Parallel Tempering** across MPI replicas (processes) with OpenMP for intra-replica acceleration. It provides:
-> a classic/weak-scaling driver (per-rank `--iters`), and
-> a strong-scaling driver (global `--K_total`, optional pinned swap window `--W`).

=======================

## Requirements

- MPI toolchain (`mpicc`, `mpirun`) -> Open MPI or MPICH
- C11 compiler with OpenMP support
- POSIX shell environment

-==========================

## Build

Use the provided Makefile.

```bash
# Classic driver (train_pt_mpi)
make

# Strong-scaling variant (also copies to train_pt_mpi for scripts)
make strong

# Clean artifacts
make clean
