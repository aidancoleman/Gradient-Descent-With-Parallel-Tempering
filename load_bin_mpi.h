// load_bin_mpi.h
#ifndef LOAD_BIN_MPI_H
#define LOAD_BIN_MPI_H
#include <stdint.h>
#include <mpi.h>
#include "load_csv.h" // for INPUT_DIM/OUTPUT_DIM from csv loader fallback
int load_bin_mpi_allrep(const char* path, float (**X_out)[INPUT_DIM], int32_t **y_out, int *N_out, MPI_Comm comm);
#endif
