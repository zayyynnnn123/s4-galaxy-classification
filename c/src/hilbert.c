
#ifndef NN_H
#define NN_H

// c/src/hilbert.c
#include <stdio.h>
#include <stdint.h>
#include "../include/nn.h"

#define IMG_SIZE  4096
#define IMG_DIM   64
/*
  hilbert_scan: reorders image pixels into a 1D sequence
  folowing a precomputed hilbert curve index order
 
  the hilbert curve is a space filling curve that preserves
  spatial locality better than simple rowmajor scanning
  pixels that are close in 2D space stay close in the 1D sequence
 
  input image layout: (H, W, C) channellast, row major
    pixel at (row,col), channel c =  image[(row*W + col)*C + c]
 
  output sequence layout: (H*W, C) channel-last, row major
    hilbert position i, channel c =  sequence[i*C + c]
 
  this is pure indexing no arithmetic on the pixel values at all.
  in assembly terms this will just be loads and stores in a loop,
  which is why validaton should be bit-for-bit exact with python
 */
void hilbert_scan(float* image, float* sequence, int32_t* indices, 
                  int channels, int height, int width) {

    /* total number of pixels, for a 64x64 image this is 4096 */
    int seq_len = height * width;
    
    /* iterate over every position in the hilbert curve order */
    for (int i = 0; i < seq_len; i++) {

        /*
          indices[i] is the flattened pixel position that maps to
          hilbert distance i, its just row*width + col packed into one int.
          we unpack it back into row and col to address the image array.
          later in assembly this will be an integer divide and modulo
          which are expensive might want to precompute row/col then
         */
        int idx = indices[i];
        int row = idx / width;
        int col = idx % width;
        
        /* copy all channels for this pixel into the output sequence */
        for (int c = 0; c < channels; c++) {

            /*
              source address: image is (H, W, C) so pixel (row,col,c)
              lives at offset (row*W + col)*C + c from the base pointer.
              in assembly this is: base_addr + (row*W + col)*C*4 + c*4
              where *4 is because each float is 4 bytes
             
              dest address: sequence is (seq_len, C) so position (i,c)
              lives at offset i*C + c from the base pointer
             */
            sequence[i * channels + c] = image[(row * width + col) * channels + c];
        }
    }
}
#endif /* NN_H */