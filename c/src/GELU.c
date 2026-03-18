#include <math.h>   
#include <stddef.h> 
 
/* 
  Formula (tanh approximation, matches PyTorch exactly):
 
    GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
 
  Applied element-wise to every value in the array.

  Parameters:
    x   - pointer to input array  (read from)
    out - pointer to output array (written to, can be same as x)
    n   - total number of elements in the array
 */
void gelu(const float *x, float *out, size_t n)
{
    /*
     * Precompute the constant sqrt(2/pi) once.
    */
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF          = 0.044715f;
 
    for (size_t i = 0; i < n; i++) {
        float xi    = x[i];
        float inner = SQRT_2_OVER_PI * (xi + COEFF * xi * xi * xi);
        out[i]      = 0.5f * xi * (1.0f + tanhf(inner));
    }
}