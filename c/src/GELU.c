#include <stdio.h>
#include <math.h>
#include "../include/nn.h"

/*
  gelu_forward: applies GELU activation in-place across n floats
  formula (tanh approximation, matches PyTorch exactly):
    GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
  tolerance: MSE < 1e-7, MAE < 1e-4
*/
void gelu_forward(float *x, int n)
{
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF          = 0.044715f;

    for (int i = 0; i < n; i++) {
        float xi    = x[i];
        float inner = SQRT_2_OVER_PI * (xi + COEFF * xi * xi * xi);
        x[i]        = 0.5f * xi * (1.0f + tanhf(inner));
    }
}