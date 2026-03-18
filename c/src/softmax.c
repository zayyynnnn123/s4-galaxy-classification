# include <stddef.h>
#include <math.h>

/* 
  Formula: softmax(x_i) = exp(x_i) / sum_j( exp(x_j) )
 
  Numerically stable version:
    1. Find max value in the array
    2. Subtract max from every element before exp()
    3. Sum all exp() values
    4. Divide each exp() value by the sum
 
  Output probabilities will sum to 1.0 (within floating-point tolerance).
 
  Parameters:
    x       - pointer to input array  (the raw logits)
    out     - pointer to output array (the probabilities, can be same as x)
    n_class - number of classes (4 for this model)
 */
void softmax(const float *x, float *out, size_t n_class)
{
    /* Step 1: Find the maximum value for numerical stability */
    float max_val = x[0];
    for (size_t i = 1; i < n_class; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
 
    /* Step 2: Compute exp(x_i - max) for each element */
    float sum = 0.0f;
    for (size_t i = 0; i < n_class; i++) {
        out[i] = expf(x[i] - max_val);
        sum   += out[i];
    }
 
    /* Step 3: Normalise by dividing each value by the sum */
    for (size_t i = 0; i < n_class; i++) {
        out[i] /= sum;
    }
 
    /*
     * After this, out[0] + out[1] + out[2] + out[3] == 1.0
     * (within floating-point rounding error, < 1e-6)
     */
}