#include <stdio.h>
#include <stdint.h>
#include "../include/nn.h"

/*
  hilbert_scan: reorders image pixels into a 1D sequence
  following a precomputed hilbert curve index order

  input:  (H, W, C) channel-last, pixel (row,col,c) = image[(row*W + col)*C + c]
  output: (H*W, C) channel-last, position (i,c)     = sequence[i*C + c]

  pure indexing -- no arithmetic on pixel values
  output should match python bit-for-bit (MSE < 1e-12)
*/
void hilbert_scan(float* image, float* sequence, int32_t* indices,
                  int channels, int height, int width) {

    int seq_len = height * width;

    for (int i = 0; i < seq_len; i++) {

        /* unpack flattened index into row and col */
        int idx = indices[i];
        int row = idx / width;
        int col = idx % width;

        /* copy all channels for this pixel into the output sequence */
        for (int c = 0; c < channels; c++) {
            sequence[i * channels + c] = image[(row * width + col) * channels + c];
        }
    }
}