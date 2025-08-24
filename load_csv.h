// load_csv.h
#ifndef LOAD_CSV_H
#define LOAD_CSV_H

#include <stdint.h>

#define INPUT_DIM  54
#define OUTPUT_DIM 7

/**
 * loads `fn` into:
 *   *inputs → pointer to an array [N][INPUT_DIM]
 *   *labels → pointer to an int32_t[N] (zero‐based, 0…OUTPUT_DIM-1)
 *   *N      → number of rows
 */
void load_csv(const char *fn,
              float (* *inputs)[INPUT_DIM],
              int32_t **labels,
              int *N);

#endif // LOAD_CSV_H
