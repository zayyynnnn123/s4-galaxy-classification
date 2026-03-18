#include <stdio.h>
#include <math.h>
#include "../include/nn.h"
/*
  softmax_forward: converts logits to probabilities over NUM_CLASSES classes
  formula: softmax(x_i) = exp(x_i) / sum_j( exp(x_j) )
  subtracts max before exp() for numerical stability
  tolerance: MSE < 1e-8, MAE < 1e-4
*/
void softmax_forward(const float *logits, float *probs)
{
    /* find max for numerical stability */
    float max_val = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] > max_val)
            max_val = logits[i];
    }
    /* compute exp(x_i - max) and accumulate sum */
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum     += probs[i];
    }
    /* normalise so probabilities sum to 1.0 */
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] /= sum;
    }
}