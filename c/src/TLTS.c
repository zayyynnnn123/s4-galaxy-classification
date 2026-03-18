#include <stdio.h>
#include "../include/nn.h"

/*
  take_last_timestep: extracts the final row from a (L, HIDDEN_DIM) sequence
  operation: last[d] = sequence[(L-1) * HIDDEN_DIM + d]
  pure indexing, no arithmetic -- must match Python bit-for-bit
  tolerance: MSE < 1e-12 (exact match)
*/
void take_last_timestep(const float *sequence, float *last, int L)
{
    const float *last_row = sequence + (L - 1) * HIDDEN_DIM;

    for (int d = 0; d < HIDDEN_DIM; d++) {
        last[d] = last_row[d];
    }
}