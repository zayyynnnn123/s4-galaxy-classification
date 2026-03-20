/* c/src/s4d.c
 implements the S4D (diagonal structured state space) layer
 each channel is processed independently -- no cross-channel mixing
 complex numbers stored as float pairs, no complex.h needed

 math steps per channel:
   1. dt      = exp(log_dt)
   2. A       = -exp(log_A_real) + j*A_imag
   3. A_bar   = exp(dt * A)
   4. C_tilde = C * (A_bar - 1) / A
   5. K[t]    = 2 * real(sum_n(C_tilde[n] * A_bar[n]^t))
   6. y       = causal_conv(u, K) + D * u
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"

/* (a_r + j*a_i) * (b_r + j*b_i)
   out_r and out_i must not alias the input pointers */
static void complex_mul(float a_r, float a_i, float b_r, float b_i,
                        float* out_r, float* out_i)
{
    *out_r = a_r * b_r - a_i * b_i;
    *out_i = a_r * b_i + a_i * b_r;
}

/* exp(a_r + j*a_i) via euler's formula:
   = exp(a_r) * (cos(a_i) + j*sin(a_i))
   most expensive operation in kernel computation */
static void complex_exp(float a_r, float a_i, float* out_r, float* out_i)
{
    float mag = expf(a_r);
    *out_r = mag * cosf(a_i);
    *out_i = mag * sinf(a_i);
}

/* (a_r + j*a_i) / (b_r + j*b_i)
   multiply top and bottom by conjugate of denominator */
static void complex_div(float a_r, float a_i, float b_r, float b_i,
                        float* out_r, float* out_i)
{
    float denom = b_r * b_r + b_i * b_i;
    *out_r = (a_r * b_r + a_i * b_i) / denom;
    *out_i = (a_i * b_r - a_r * b_i) / denom;
}

void s4d_forward(const float* input, float* output,
                 const S4DParams* params, int L)
{
    int h, n, t, j;

    /* reused for each channel -- avoids allocating L floats 64 times */
    float K[IMG_SIZE];

    for (h = 0; h < HIDDEN_DIM; h++) {

        float dt = expf(params->log_dt[h]);
        memset(K, 0, L * sizeof(float));

        for (n = 0; n < S4D_N; n++) {

            /* A = -exp(log_A_real) + j*A_imag
               negative real part guarantees SSM stability */
            float A_r = -expf(params->log_A_real[h * S4D_N + n]);
            float A_i =  params->A_imag[h * S4D_N + n];

            /* A_bar = exp(dt * A) -- discrete transition factor */
            float A_bar_r, A_bar_i;
            complex_exp(dt * A_r, dt * A_i, &A_bar_r, &A_bar_i);

            /* C_tilde = C * (A_bar - 1) / A */
            float C_r = params->C[h * S4D_N * 2 + n * 2 + 0];
            float C_i = params->C[h * S4D_N * 2 + n * 2 + 1];
            float tmp_r, tmp_i, Ct_r, Ct_i;
            complex_div(A_bar_r - 1.0f, A_bar_i, A_r, A_i, &tmp_r, &tmp_i);
            complex_mul(C_r, C_i, tmp_r, tmp_i, &Ct_r, &Ct_i);

            /* K[t] += 2 * real(C_tilde * A_bar^t)
               A_bar^t updated iteratively -- avoids calling complex_exp
               inside the loop which would be very expensive */
            float pow_r = 1.0f, pow_i = 0.0f;
            for (t = 0; t < L; t++) {
                K[t] += 2.0f * (Ct_r * pow_r - Ct_i * pow_i);
                float new_pow_r, new_pow_i;
                complex_mul(pow_r, pow_i, A_bar_r, A_bar_i,
                            &new_pow_r, &new_pow_i);
                pow_r = new_pow_r;
                pow_i = new_pow_i;
            }
        }

        /* causal convolution: y[t] = D*u[t] + sum_{j=0}^{t} K[L-1-j]*u[t-j]
           K indexed as K[L-1-j] not K[j] to match pytorch conv1d behaviour
           with padding=L-1 which applies the kernel in reverse order */
        float D_h = params->D[h];
        for (t = 0; t < L; t++) {
            float acc = D_h * input[t * HIDDEN_DIM + h];
            for (j = 0; j <= t; j++) {
                acc += K[L - 1 - j] * input[(t - j) * HIDDEN_DIM + h];
            }
            output[t * HIDDEN_DIM + h] = acc;
        }
    }
}