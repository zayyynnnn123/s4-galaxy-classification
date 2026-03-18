/* validates the linear layer C implementation against pytorch reference
 * tests the uproject layer: (4096, 3) --> (4096, 64)
 * run export_test_data.py first to generate the reference files */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"

/* ------------------------------------------------------------------ */
/* file loading helpers                                                */
/* ------------------------------------------------------------------ */

/*
 * load_floats -- loads float32 binary data from file into buffer
 * size is number of floats to read not bytes
 * exits immediately if file cant be opened or has wrong size
 * we exit instead of return so we never compare garbage data */
void load_floats(const char* filename, float* buffer, size_t size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("FATAL: Cannot open %s\n", filename);
        printf("       Run export_test_data.py first!\n");
        exit(1);
    }
    size_t n = fread(buffer, sizeof(float), size, fp);
    fclose(fp);
    if (n != size) {
        printf("FATAL: Expected %zu floats but got %zu from %s\n",
               size, n, filename);
        exit(1);
    }
}

/* ------------------------------------------------------------------ */
/* error metrics                                                       */
/* ------------------------------------------------------------------ */

/*
 * compute_mse -- mean squared error between two float arrays
 * this is sensitive to large deviations, penalizes outliers heavily
 * for linear layer we expect MSE < 1e-8 */
double compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return sum / n;
}

/*
 * compute_mae -- mean absolute error between two float arrays
 * more intuitive than MSE, gives average deviation magnitude
 * for linear layer we expect MAE < 1e-6 */
double compute_mae(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs((double)a[i] - (double)b[i]);
    }
    return sum / n;
}

/*
 * compute_max_err -- largest single element difference
 * useful for finding worst case errors in the output */
float compute_max_err(const float* a, const float* b, int n) {
    float max = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max) max = diff;
    }
    return max;
}

/* ------------------------------------------------------------------ */
/* main test                                                           */
/* ------------------------------------------------------------------ */

int main() {
    printf("Testing Linear Layer Implementation\n");
    printf("=====================================\n");

    /* uproject layer dimensions
     * input:  (4096, 3)  -- hilbert scan output for RGB image
     * output: (4096, 64) -- projected to hidden dimension
     * weight: (64, 3)    -- stored as (out_features, in_features)
     * bias:   (64,)      -- one bias per output feature */
    int rows         = IMG_SIZE;   /* 4096 */
    int in_features  = 3;          /* RGB channels */
    int out_features = HIDDEN_DIM; /* 64 */

    /* total element counts for each array */
    int input_size   = rows * in_features;           /* 4096 * 3  = 12288  */
    int weight_size  = out_features * in_features;   /* 64   * 3  = 192    */
    int bias_size    = out_features;                 /* 64              */
    int output_size  = rows * out_features;          /* 4096 * 64 = 262144 */

    /* allocate all arrays on the stack
     * sizes are known at compile time so no malloc needed
     * this matches the assembly mindset -- fixed size buffers */
    float input[IMG_SIZE * 3];
    float weights[HIDDEN_DIM * 3];
    float bias[HIDDEN_DIM];
    float output[IMG_SIZE * HIDDEN_DIM];
    float expected[IMG_SIZE * HIDDEN_DIM];

    /* zero everything before loading to avoid stale data */
    memset(input,    0, sizeof(input));
    memset(weights,  0, sizeof(weights));
    memset(bias,     0, sizeof(bias));
    memset(output,   0, sizeof(output));
    memset(expected, 0, sizeof(expected));

    /* load all test data files
     * paths are relative to c/ directory where make runs from */
    printf("\nLoading test data...\n");
    load_floats("../data/samples/linear_input.bin",   input,    input_size);
    printf("  Loaded input    (%d floats)\n", input_size);
    load_floats("../data/samples/uproject_weight.bin", weights, weight_size);
    printf("  Loaded weights  (%d floats)\n", weight_size);
    load_floats("../data/samples/uproject_bias.bin",   bias,    bias_size);
    printf("  Loaded bias     (%d floats)\n", bias_size);
    load_floats("../data/samples/linear_output.bin",   expected, output_size);
    printf("  Loaded expected (%d floats)\n", output_size);

    /* run our C linear layer implementation */
    printf("\nRunning linear_forward...\n");
    linear_forward(input, weights, bias, output,
                   rows, in_features, out_features);

    /* print first few values so we can sanity check direction
     * if signs are wrong or values are wildly off we catch it here
     * before looking at the error metrics */
    printf("\nFirst 5 output values (ours vs expected):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] got: %f   expected: %f\n",
               i, output[i], expected[i]);
    }

    /* compute error metrics across all output elements */
    double mse     = compute_mse(output, expected, output_size);
    double mae     = compute_mae(output, expected, output_size);
    float  max_err = compute_max_err(output, expected, output_size);

    /* count elements that exceed tolerance
     * tolerance is 1e-5 which is tighter than the spec requires
     * spec requires MSE < 1e-8 and MAE < 1e-6 */
    int errors = 0;
    for (int i = 0; i < output_size; i++) {
        if (fabsf(output[i] - expected[i]) > 1e-5f) {
            if (errors < 5) {
                printf("  Mismatch at [%d]: got %f  expected %f\n",
                       i, output[i], expected[i]);
            }
            errors++;
        }
    }

    /* print full summary */
    printf("\nResults:\n");
    printf("  Output elements : %d\n",  output_size);
    printf("  Errors (>1e-5)  : %d\n",  errors);
    printf("  MSE             : %e\n",  mse);
    printf("  MAE             : %e\n",  mae);
    printf("  Max error       : %e\n",  max_err);
    printf("  Target MSE      : < 1e-8\n");
    printf("  Target MAE      : < 1e-6\n");

    /* pass criteria from the spec
     * MSE < 1e-8 and MAE < 1e-6 for linear layers */
    if (mse < 1e-8 && mae < 1e-6) {
        printf("\nTEST PASSED! Linear layer matches PyTorch reference.\n");
        return 0;
    } else {
        printf("\nTEST FAILED!\n");
        if (mse >= 1e-8) printf("  MSE too high: %e (need < 1e-8)\n", mse);
        if (mae >= 1e-6) printf("  MAE too high: %e (need < 1e-6)\n", mae);
        return 1;
    }
}