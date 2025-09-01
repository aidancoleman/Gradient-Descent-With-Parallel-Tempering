#include "load_csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

struct BinHeader {
  int32_t N; // number of rows
  int32_t input_dim; // should be 54 for our covertype dataset make sure it matches whatever data you want to look at
  int32_t output_dim; // should be 7
  int32_t reserved; // 0
};

void load_csv(const char *fn /*file name we read "covtype.csv"*/, float (* *inputs)[INPUT_DIM], int32_t **labels, int *N /*output number of rows we read*/)
{
    //open fn to read
    FILE *f = fopen(fn, "r");
    if (!f) {
        perror(fn);
        exit(1);
    }

    // Buffer for one line => 4 kb for each line
    char line[4096];
    // Count all lines (no header to skip)
    int count = 0; //tell me how many data rows there are in this csv
    while (fgets(line, sizeof line, f)) {
    count++;
    }
    // rewind to start of file and parse all lines
    rewind(f);

    // malloc arrays
    *N = count;
    *inputs = malloc(count * sizeof **inputs);
    *labels = malloc(count * sizeof **labels);
    if (!*inputs || !*labels) {
        perror("malloc");
        exit(1);
    }

    // parse each line
    const int nf = INPUT_DIM;
    int idx = 0;
    while (fgets(line, sizeof line, f)) {
        char *p = line, *end; //end marks where parsing stops, p = start of parsing on each line

        // parse nf floats
        for (int j = 0; j < nf; j++) { //loop over each of the j feature columns
            float v = strtof(p, &end); //convert substring located at p to a float, where end is pointing just after the numeric value
            if (end == p) {
                fprintf(stderr, "parse error on row %d, col %d\n", idx, j);
                exit(1);
            }
            (*inputs)[idx][j] = v;
            p = end + 1; // skip comma
        }

        // parse label, subtract 1 to zero-base
        long L = strtol(p, &end, 10); //after last comma we parse int label with rhs
        if (end == p) {
            fprintf(stderr, "parse error on label at row %d\n", idx); //if no conversion => error
            exit(1);
        }
        (*labels)[idx] = (int32_t)(L - 1); //convert to indexing 0-6

        idx++; //move to next row index
    }
    
    
    fclose(f);
}
