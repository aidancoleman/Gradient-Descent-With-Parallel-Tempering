// load_bin_mpi.c
#include "load_bin_mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#pragma pack(push,1)
typedef struct { int32_t N, input_dim, output_dim, reserved; } BinHeader;
#pragma pack(pop)

int load_bin_mpi_allrep(const char* path, float (**X_out)[INPUT_DIM], int32_t **y_out, int *N_out, MPI_Comm comm)
{
  int rank; MPI_Comm_rank(comm,&rank);
  MPI_File fh;
  MPI_Info info = MPI_INFO_NULL;

  // open Data collectively
  if (MPI_File_open(comm, (char*)path, MPI_MODE_RDONLY, info, &fh) != MPI_SUCCESS){
    if (rank==0) fprintf(stderr,"MPI_File_open failed for %s\n", path);
    return -1;
  }

  // read header collectively
  BinHeader h;
  MPI_Status st;
  if (MPI_File_read_at_all(fh, 0, &h, (int)sizeof h, MPI_BYTE, &st) != MPI_SUCCESS){
    if (rank==0) fprintf(stderr,"MPI_File_read_at_all(header) failed\n");
    MPI_File_close(&fh);
    return -2;
  }
  if (h.input_dim != INPUT_DIM || h.output_dim != OUTPUT_DIM){
    if (rank==0) fprintf(stderr,"[bin] dim mismatch: file (%d,%d) vs build (%d,%d)\n", h.input_dim,h.output_dim,INPUT_DIM,OUTPUT_DIM);
    MPI_File_close(&fh);
    return -3;
  }
  const int N = h.N;
  *N_out = N;

  // allocate local full replicas
  float *Xflat = (float*)malloc((size_t)N*INPUT_DIM*sizeof(float));
  int32_t *y = (int32_t*)malloc((size_t)N*sizeof(int32_t));
  if (!Xflat || !y){ perror("malloc"); MPI_Abort(comm,1); }

// displacements
  MPI_Offset disp = (MPI_Offset)sizeof(BinHeader);
  // read features
  if (MPI_File_read_at_all(fh, disp, Xflat, (MPI_Count)((size_t)N*INPUT_DIM), MPI_FLOAT, &st) != MPI_SUCCESS){
    if (rank==0) fprintf(stderr,"MPI_File_read_at_all(features) failed\n");
    MPI_File_close(&fh);
    return -4;
  }
  disp += (MPI_Offset)((size_t)N*INPUT_DIM*sizeof(float));
  // read labels
  if (MPI_File_read_at_all(fh, disp, y, (MPI_Count)N, MPI_INT32_T, &st) != MPI_SUCCESS){
    if (rank==0) fprintf(stderr,"MPI_File_read_at_all(labels) failed\n");
    MPI_File_close(&fh);
    return -5;
  }

  MPI_File_close(&fh);
  *X_out = (float (*)[INPUT_DIM])Xflat;
  *y_out = y;
  return 0;
}
                                    
