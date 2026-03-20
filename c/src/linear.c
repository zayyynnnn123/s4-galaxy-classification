#include <stdio.h>
#include "../include/nn.h"

/*
  linear_forward: Y = X * W^T + b
  used twice -- input projection (4096,C)->(4096,64) and classifier (64,)->(4,)

  weights stored as (out_features, in_features) row-major
  so W^T is implicit -- weight row j is the coefficients for output neuron j

  three nested loops: i (rows), j (out_features), k (in_features)
  innermost loop is a multiply-accumulate -- maps cleanly to RISC-V fmadd
*/
void linear_forward(const float* input, const float* weights,
                    const float* bias, float* output,
                    int rows, int in_features, int out_features)
{
    int i, j, k;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < out_features; j++) {

            /* initialise accumulator with bias, then dot product */
            float acc = bias[j];
            for (k = 0; k < in_features; k++) {
                acc += input[i * in_features + k] * weights[j * in_features + k];
            }
            output[i * out_features + j] = acc;
        }
    }
}