/*
 * implements the linear (fully connected) layer
 * used twice in the model:
 *   1. input projection: (4096, C) --> (4096, 64)
 *   2. final classification: (64,) --> (4,)
 *
 * the operation is Y = X * W^T + b
 * weights are stored as (out_features, in_features) in row-major order
 * so W^T is never explicitly computed -- we just swap the indexing */

#include <stdio.h>
#include "../include/nn.h"

/*
 * linear_forward
 *
 * this is a matrix-vector multiply repeated for each row of input.
 * for each output row i and each output feature j:
 *   output[i][j] = bias[j] + sum over k of (input[i][k] * weights[j][k])
 *
 * notice weights[j][k] not weights[k][j] -- this is the W^T part.
 * since weights are stored as (out, in), row j of weights IS column j of W^T
 * so we just index weights[j * in_features + k] directly
 *
 * in assembly terms this will be:
 *   three nested loops: i (rows), j (out_features), k (in_features)
 *   innermost loop is a multiply-accumulate: output += input * weight
 *   this is the classic GEMV (general matrix vector multiply) pattern
 *   which maps very cleanly to RISC-V instructions later */
void linear_forward(const float* input,
                    const float* weights,
                    const float* bias,
                    float*       output,
                    int          rows,
                    int          in_features,
                    int          out_features)
{
    int i, j, k;

    /* iterate over each row of the input sequence
     * for the fc layer rows=1, for projection rows=4096 */
    for (i = 0; i < rows; i++) {

        /* iterate over each output feature (neuron) */
        for (j = 0; j < out_features; j++) {

            /* start with the bias value for this output neuron
             * in assembly this is just a load from the bias array
             * we accumulate into this variable in the inner loop */
            float acc = bias[j];

            /* dot product of input row i with weight row j
             * weight row j corresponds to output neuron j
             * weights are (out_features, in_features) row-major
             * so weight[j][k] = weights[j * in_features + k] */
            for (k = 0; k < in_features; k++) {

                /* input[i][k]   = input[i * in_features + k]
                 * weights[j][k] = weights[j * in_features + k]
                 * one multiply-accumulate per iteration
                 * in RISC-V this becomes fmadd or fmul + fadd */
                acc += input[i * in_features + k] * weights[j * in_features + k];
            }

            /* write accumulated result to output[i][j]
             * output is (rows, out_features) row-major */
            output[i * out_features + j] = acc;
        }
    }
}