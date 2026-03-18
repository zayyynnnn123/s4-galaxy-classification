/* c/src/s4d.c
 * implements the S4D (diagonal structured state space) layer
 * this is the core sequence modeling component of the network
 *
 * the math follows exactly what the python code does:
 *   1. dt       = exp(log_dt)                 -- discretization timestep
 *   2. A        = -exp(log_A_real) + j*A_imag -- complex eigenvalues
 *   3. dtA      = dt * A                      -- scaled eigenvalues
 *   4. C_tilde  = C * (exp(dtA) - 1) / A     -- modified output projection
 *   5. A_bar    = exp(dtA)                    -- discrete transition
 *   6. K[t]     = 2 * real(sum_n(C_tilde[n] * A_bar[n]^t)) -- kernel
 *   7. y        = causal_conv(u, K) + D * u   -- output
 *
 * all complex numbers are stored as separate real and imag floats
 * no complex.h needed -- we implement everything manually
 * this makes the assembly translation in milestone 3 straightforward */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"

/* ------------------------------------------------------------------ */
/* complex number arithmetic                                           */
/* we store complex numbers as two floats: real and imag              */
/* all operations take pointers to output and inputs                  */
/* this maps directly to RISC-V float register pairs later            */
/* ------------------------------------------------------------------ */

/*
 * complex_mul -- multiplies two complex numbers
 * (a_r + j*a_i) * (b_r + j*b_i)
 *   = (a_r*b_r - a_i*b_i) + j*(a_r*b_i + a_i*b_r)
 * out_r and out_i must not alias a_r/a_i or b_r/b_i */
static void complex_mul(float  a_r, float  a_i,
                        float  b_r, float  b_i,
                        float* out_r, float* out_i)
{
    *out_r = a_r * b_r - a_i * b_i;
    *out_i = a_r * b_i + a_i * b_r;
}

/*
 * complex_exp -- computes exp(a_r + j*a_i)
 * using eulers formula: exp(a_r + j*a_i) = exp(a_r) * (cos(a_i) + j*sin(a_i))
 * this is the most expensive operation in the kernel computation
 * in assembly this will require a software implementaton of exp/cos/sin */
static void complex_exp(float  a_r, float  a_i,
                        float* out_r, float* out_i)
{
    float mag = expf(a_r);   /* magnitude = exp(real part) */
    *out_r = mag * cosf(a_i);
    *out_i = mag * sinf(a_i);
}

/*
 * complex_div -- divides two complex numbers
 * (a_r + j*a_i) / (b_r + j*b_i)
 * multiply numerator and denominator by conjugate of denominator:
 *   = (a_r*b_r + a_i*b_i) / (b_r^2 + b_i^2)
 *   + j*(a_i*b_r - a_r*b_i) / (b_r^2 + b_i^2) */
static void complex_div(float  a_r, float  a_i,
                        float  b_r, float  b_i,
                        float* out_r, float* out_i)
{
    float denom = b_r * b_r + b_i * b_i;
    *out_r = (a_r * b_r + a_i * b_i) / denom;
    *out_i = (a_i * b_r - a_r * b_i) / denom;
}

/* ------------------------------------------------------------------ */
/* S4D forward pass                                                    */
/* ------------------------------------------------------------------ */

/*
 * s4d_forward
 *
 * processes one sequence through one S4D layer
 * input and output are both (L, HIDDEN_DIM) in row-major layout
 * meaning input[t * HIDDEN_DIM + h] is timestep t, channel h
 *
 * the outer loop is over channels h -- each channel is independent
 * this is the diagonal structure of S4D: no cross-channel mixing
 * so we compute a separate kernel for each of the 64 channels
 *
 * the convolution uses kernel in reverse order because pytorchs conv1d
 * with padding=L-1 applies K[L-1] at t=0, K[L-2] at t=1, etc
 * so we index K[L-1-j] instead of K[j] in the convolution loop
 *
 * inner loop is the causal convolution:
 *   y[t] = D*u[t] + sum_{j=0}^{t} K[L-1-j] * u[t-j]
 * this is O(L^2) per channel, O(L^2 * H) total
 * for L=4096 and H=64 this is about 1 billion operations -- slow but correct
 * milestone 3 will optimize this */
void s4d_forward(const float*     input,
                 float*           output,
                 const S4DParams* params,
                 int              L)
{
    /* loop index variables declared at top for C89 compatibility
     * and to make the assembly translation more obvious later */
    int h, n, t, j;

    /* per-channel kernel buffer -- reused for each channel
     * K[t] is the kernel value at position t
     * maximum size is IMG_SIZE = 4096 */
    float K[IMG_SIZE];

    /* ---- outer loop: process each channel independently ---- */
    for (h = 0; h < HIDDEN_DIM; h++) {

        /* -------------------------------------------------- */
        /* step 1: compute dt for this channel                */
        /* dt = exp(log_dt[h])                                */
        /* -------------------------------------------------- */
        float dt = expf(params->log_dt[h]);

        /* -------------------------------------------------- */
        /* step 2: compute kernel K for this channel          */
        /* zero the kernel buffer before accumulating         */
        /* -------------------------------------------------- */
        memset(K, 0, L * sizeof(float));

        for (n = 0; n < S4D_N; n++) {

            /* compute A[n] = -exp(log_A_real) + j*A_imag
             * the negative sign ensures eigenvalues have negative
             * real part which guarantees stability of the SSM */
            float A_r = -expf(params->log_A_real[h * S4D_N + n]);
            float A_i =  params->A_imag[h * S4D_N + n];

            /* compute dtA[n] = dt * A[n]
             * dt is real so just scale both components */
            float dtA_r = dt * A_r;
            float dtA_i = dt * A_i;

            /* compute A_bar[n] = exp(dtA[n])
             * this is the discrete transition factor per step
             * we raise it to power t iteratively below */
            float A_bar_r, A_bar_i;
            complex_exp(dtA_r, dtA_i, &A_bar_r, &A_bar_i);

            /* compute exp(dtA) - 1
             * subtract 1 from real part only, imag stays the same */
            float expm1_r = A_bar_r - 1.0f;
            float expm1_i = A_bar_i;

            /* load C[h,n] stored as real/imag pair
             * flat index: h * S4D_N * 2 + n * 2 + 0/1 */
            float C_r = params->C[h * S4D_N * 2 + n * 2 + 0];
            float C_i = params->C[h * S4D_N * 2 + n * 2 + 1];

            /* compute C_tilde = C * (exp(dtA) - 1) / A
             * first divide (exp(dtA)-1) by A, then multiply by C */
            float tmp_r, tmp_i;
            complex_div(expm1_r, expm1_i, A_r, A_i, &tmp_r, &tmp_i);

            float Ct_r, Ct_i;
            complex_mul(C_r, C_i, tmp_r, tmp_i, &Ct_r, &Ct_i);

            /* accumulate kernel contribution from state n
             * K[t] += 2 * real(C_tilde * A_bar^t)
             * A_bar^t is computed iteratively -- much cheaper than
             * recomputing exp(dtA * t) from scratch each iteration */
            float pow_r = 1.0f;   /* real part of A_bar^t, t=0 -> 1+0j */
            float pow_i = 0.0f;   /* imag part of A_bar^t */

            for (t = 0; t < L; t++) {

                /* real part of (C_tilde * A_bar^t) is:
                 * Ct_r*pow_r - Ct_i*pow_i */
                K[t] += 2.0f * (Ct_r * pow_r - Ct_i * pow_i);

                /* advance: pow = pow * A_bar
                 * use temp vars to avoid overwriting pow_r
                 * before it is used to compute new pow_i */
                float new_pow_r, new_pow_i;
                complex_mul(pow_r, pow_i, A_bar_r, A_bar_i,
                            &new_pow_r, &new_pow_i);
                pow_r = new_pow_r;
                pow_i = new_pow_i;
            }
        } /* end loop over state dimensions n */

        /* -------------------------------------------------- */
        /* step 3: causal convolution                         */
        /* y[t] = D*u[t] + sum_{j=0}^{t} K[L-1-j] * u[t-j]  */
        /*                                                    */
        /* the kernel is indexed as K[L-1-j] not K[j] because */
        /* pytorchs conv1d with padding=L-1 applies the kernel */
        /* in reverse -- K[L-1] is used at t=0, K[L-2] at t=1 */
        /* -------------------------------------------------- */
        float D_h = params->D[h];

        for (t = 0; t < L; t++) {

            /* skip connection: D * u[t]
             * input is (L, H) so input[t,h] = input[t*HIDDEN_DIM + h] */
            float acc = D_h * input[t * HIDDEN_DIM + h];

            /* causal convolution sum
             * j=0 uses most recent input u[t] with kernel K[L-1]
             * j=t uses oldest input u[0] with kernel K[L-1-t] */
            for (j = 0; j <= t; j++) {
                acc += K[L - 1 - j] * input[(t - j) * HIDDEN_DIM + h];
            }

            /* write to output, same (L, H) layout as input */
            output[t * HIDDEN_DIM + h] = acc;
        }

    } /* end loop over channels h */
}