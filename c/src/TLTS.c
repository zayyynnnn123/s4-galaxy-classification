#include <stddef.h>
#include <math.h>

/* 
    Given X of shape (L, D), extract Y[d] = X[L-1, d]
 
  In memory (row-major C layout), element [row][col] is at flat_index = row * D + col
 
  So the last row (row = L-1) starts at offset (L-1) * D.
  We copy D floats from that offset into the output array.
 
  This is pure indexing so it must match
  PyTorch exactly (MSE < 1e-12).
 
  Parameters:
    input  - pointer to 2D array of shape (L, D) stored row-major
    output - pointer to 1D output array of size D
    L      - sequence length (number of timesteps)
    D      - feature dimension (d_model, 64 in this case)
 */
void take_last_timestep(const float *input, float *output, size_t L, size_t D)
{   
    /* The last timestep starts at index (L-1) * D in the flat array.
      We copy D consecutive floats from there into output. */
    const float *last_row = input + (L - 1) * D;
 
    for (size_t d = 0; d < D; d++) {
        output[d] = last_row[d];
    }
}