#ifndef NN_H
#define NN_H

/* nn.h -- declarations for all layers in the S4D galaxy classifier
 * this is the single source of truth for all function signatures
 * every .c file and every test file includes this one header
 * we add declarations here as we implement each layer one by one */

#include <stdint.h>   /* for int32_t */
#include <stddef.h>   /* for size_t  */

/* ------------------------------------------------------------------ */
/* model dimensions -- all fixed at compile time                       */
/* ------------------------------------------------------------------ */

/* total pixels in one image (64 * 64) */
#define IMG_SIZE     4096

/* side length of the square image in pixels */
#define IMG_DIM      64

/* number of hidden dimensions in the model */
#define HIDDEN_DIM   64

/* number of output classes */
#define NUM_CLASSES  4

/* number of S4D state dimensions per channel (N/2 complex pairs)
 * each channel has S4D_N complex eigenvalues stored as real/imag pairs
 * so C and A_imag arrays have length HIDDEN_DIM * S4D_N */
#define S4D_N        32

/* ------------------------------------------------------------------ */
/* S4D parameter structure                                             */
/* groups all parameters for one S4D layer into one struct            */
/* this makes it easy to pass around and load from the weight file    */
/* in assembly terms this is just a block of memory with known layout */
/* ------------------------------------------------------------------ */
typedef struct {
    /* log of the time step delta, one value per hidden channel
     * shape: (HIDDEN_DIM,)
     * actual dt = exp(log_dt) applied per channel */
    float log_dt[HIDDEN_DIM];

    /* log of real part of diagonal A eigenvalues
     * shape: (HIDDEN_DIM, S4D_N)
     * stored flat as log_A_real[h * S4D_N + n] for channel h, state n
     * actual A_real = -exp(log_A_real) to ensure stability (must be negative) */
    float log_A_real[HIDDEN_DIM * S4D_N];

    /* imaginary part of diagonal A eigenvalues
     * shape: (HIDDEN_DIM, S4D_N)
     * stored flat as A_imag[h * S4D_N + n]
     * not logged, used directly as the imaginary component */
    float A_imag[HIDDEN_DIM * S4D_N];

    /* output projection C, complex so stored as real/imag interleaved
     * shape: (HIDDEN_DIM, S4D_N, 2) where last dim is [real, imag]
     * stored flat as C[h * S4D_N * 2 + n * 2 + 0] for real
     *                  C[h * S4D_N * 2 + n * 2 + 1] for imag */
    float C[HIDDEN_DIM * S4D_N * 2];

    /* skip connection D, one scalar per hidden channel
     * shape: (HIDDEN_DIM,)
     * output += D * input at each timestep */
    float D[HIDDEN_DIM];

} S4DParams;

/* ------------------------------------------------------------------ */
/* layer 1 -- hilbert scan                                             */
/* reorders image pixels into a 1D sequence following hilbert curve   */
/* ------------------------------------------------------------------ */

/*
 * hilbert_scan
 *
 * image    : input image, shape (H, W, C), channel-last, row-major
 *            pixel at (row, col, c) = image[(row*W + col)*C + c]
 * sequence : output sequence, shape (H*W, C), channel-last
 *            position (i, c) = sequence[i*C + c]
 * indices  : precomputed hilbert curve indices, array of H*W int32s
 *            each value is a flattened pixel position = row*W + col
 * channels : 1 for grayscale, 3 for RGB
 * height   : image height in pixels (64)
 * width    : image width in pixels (64)
 */
void hilbert_scan(float*   image,
                  float*   sequence,
                  int32_t* indices,
                  int      channels,
                  int      height,
                  int      width);

/* ------------------------------------------------------------------ */
/* layer 2 -- linear (fully connected)                                */
/* used for input projection and final classification                 */
/* ------------------------------------------------------------------ */

/*
 * linear_forward
 *
 * implements Y = X * W^T + b
 * weights are stored row-major as (out_features, in_features)
 * so W^T is implicit -- we just swap the loop order
 *
 * input        : shape (rows, in_features), row-major
 * weights      : shape (out_features, in_features), row-major
 * bias         : shape (out_features,)
 * output       : shape (rows, out_features), row-major
 * rows         : sequence length for projection layer, 1 for fc layer
 * in_features  : input dimension
 * out_features : output dimension
 */
void linear_forward(const float* input,
                    const float* weights,
                    const float* bias,
                    float*       output,
                    int          rows,
                    int          in_features,
                    int          out_features);

/* ------------------------------------------------------------------ */
/* layer 3 -- S4D (diagonal state space model)                        */
/* the core sequence modeling component of the network                */
/* ------------------------------------------------------------------ */

/*
 * s4d_forward
 *
 * input  : input sequence, shape (L, HIDDEN_DIM), channel-last
 * output : output sequence, same shape as input
 * params : pointer to S4DParams struct with all layer weights
 * L      : sequence length (IMG_SIZE = 4096)
 */
void s4d_forward(const float*     input,
                 float*           output,
                 const S4DParams* params,
                 int              L);

/* ------------------------------------------------------------------ */
/* activation functions                                               */
/* ------------------------------------------------------------------ */

/*
 * gelu_forward
 *
 * gaussian error linear unit, applied element-wise in-place
 * uses the tanh approximation to match pytorch exactly:
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * x : array of n floats, modified in-place
 * n : number of elements
 */
void gelu_forward(float* x, int n);

/*
 * softmax_forward
 *
 * converts logits to probabilities over NUM_CLASSES classes
 * uses the numerically stable version: subtract max before exp
 * to avoid overflow when logits are large
 *
 * logits : input array of length NUM_CLASSES
 * probs  : output array of length NUM_CLASSES, sums to 1.0
 */
void softmax_forward(const float* logits, float* probs);

/* ------------------------------------------------------------------ */
/* take last timestep                                                 */
/* ------------------------------------------------------------------ */

/*
 * take_last_timestep
 *
 * extracts the final position from a sequence
 * analogous to taking the last hidden state of an RNN
 * the last position has seen all previous inputs through S4D dynamics
 *
 * sequence : input, shape (L, HIDDEN_DIM), row-major
 * last     : output, shape (HIDDEN_DIM,)
 *            = sequence[(L-1) * HIDDEN_DIM + 0 .. HIDDEN_DIM-1]
 * L        : sequence length
 */
void take_last_timestep(const float* sequence, float* last, int L);

/* ------------------------------------------------------------------ */
/* full model forward pass and weight loader                          */
/* implemented in nn.c                                                */
/* ------------------------------------------------------------------ */

/*
 * load_model_weights -- reads all parameters from model_weights.bin
 * must be called once before model_forward
 * returns 0 on success, -1 on failure
 * path : path to model_weights.bin */
int load_model_weights(const char* path);

/*
 * model_forward -- runs complete inference pipeline on one image
 * input_image  : flat (H,W,C) float array channel-last layout
 * C            : number of channels, must be 3 for this model
 * output_probs : output array of NUM_CLASSES probabilities summing to 1 */
void model_forward(float* input_image, int C, float* output_probs);

/* ------------------------------------------------------------------ */
/* weight accessors for benchmarking                                  */
/* ------------------------------------------------------------------ */
int32_t*   get_hilbert_indices(void);
float*     get_uproject_weight(void);
float*     get_uproject_bias(void);
S4DParams* get_s4_1_params(void);
S4DParams* get_s4_2_params(void);
S4DParams* get_s4_3_params(void);
float*     get_fc_weight(void);
float*     get_fc_bias(void);


#endif /* NN_H */