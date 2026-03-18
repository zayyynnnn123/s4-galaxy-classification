/* validates the S4D layer C implementation against pytorch reference
 * run export_test_data.py first to generate the reference files */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"

/*
 * load_floats -- loads float32 binary data from file
 * exits if file cannot be opened or has wrong number of elements */
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

/* compute mean squared error between two float arrays */
double compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return sum / n;
}

/* compute mean absolute error between two float arrays */
double compute_mae(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs((double)a[i] - (double)b[i]);
    }
    return sum / n;
}

/* find largest single element difference */
float compute_max_err(const float* a, const float* b, int n) {
    float max = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max) max = diff;
    }
    return max;
}

int main() {
    printf("Testing S4D Layer Implementation\n");
    printf("==================================\n");

    /* S4D layer dimensions
     * input/output: (4096, 64) = (L, H)
     * log_dt:       (64,)
     * log_A_real:   (64, 32)
     * A_imag:       (64, 32)
     * C:            (64, 32, 2)
     * D:            (64,) */
    int L           = IMG_SIZE;              /* 4096 */
    int output_size = IMG_SIZE * HIDDEN_DIM; /* 262144 */

    /* declare S4DParams struct -- holds all layer weights
     * defined in nn.h with exact field sizes */
    S4DParams params;

    /* allocate input and output buffers */
    static float input[IMG_SIZE * HIDDEN_DIM];
    static float output[IMG_SIZE * HIDDEN_DIM];
    static float expected[IMG_SIZE * HIDDEN_DIM];

    /* zero everything before loading */
    memset(&params,  0, sizeof(S4DParams));
    memset(input,    0, sizeof(input));
    memset(output,   0, sizeof(output));
    memset(expected, 0, sizeof(expected));

    /* load all parameters and test data */
    printf("\nLoading S4D parameters...\n");
    load_floats("../data/samples/s4d_log_dt.bin",
                params.log_dt, HIDDEN_DIM);
    printf("  Loaded log_dt     (%d floats)\n", HIDDEN_DIM);

    load_floats("../data/samples/s4d_log_A_real.bin",
                params.log_A_real, HIDDEN_DIM * S4D_N);
    printf("  Loaded log_A_real (%d floats)\n", HIDDEN_DIM * S4D_N);

    load_floats("../data/samples/s4d_A_imag.bin",
                params.A_imag, HIDDEN_DIM * S4D_N);
    printf("  Loaded A_imag     (%d floats)\n", HIDDEN_DIM * S4D_N);

    load_floats("../data/samples/s4d_C.bin",
                params.C, HIDDEN_DIM * S4D_N * 2);
    printf("  Loaded C          (%d floats)\n", HIDDEN_DIM * S4D_N * 2);

    load_floats("../data/samples/s4d_D.bin",
                params.D, HIDDEN_DIM);
    printf("  Loaded D          (%d floats)\n", HIDDEN_DIM);

    load_floats("../data/samples/s4d_input.bin",
                input, output_size);
    printf("  Loaded input      (%d floats)\n", output_size);

    load_floats("../data/samples/s4d_output.bin",
                expected, output_size);
    printf("  Loaded expected   (%d floats)\n", output_size);

    /* run S4D forward pass -- this will take a few seconds
     * O(L^2 * H) = O(4096^2 * 64) ~ 1 billion operations */
    printf("\nRunning s4d_forward (this may take a moment)...\n");
    s4d_forward(input, output, &params, L);
    printf("Done!\n");

    /* print first few values for sanity check */
    printf("\nFirst 5 output values (ours vs expected):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] got: %f   expected: %f\n",
               i, output[i], expected[i]);
    }

    /* compute error metrics */
    double mse     = compute_mse(output, expected, output_size);
    double mae     = compute_mae(output, expected, output_size);
    float  max_err = compute_max_err(output, expected, output_size);

    /* count elements exceeding tolerance */
    int errors = 0;
    for (int i = 0; i < output_size; i++) {
        if (fabsf(output[i] - expected[i]) > 1e-3f) {
            if (errors < 5) {
                printf("  Mismatch at [%d]: got %f  expected %f\n",
                       i, output[i], expected[i]);
            }
            errors++;
        }
    }

    /* print summary */
    printf("\nResults:\n");
    printf("  Output elements : %d\n",  output_size);
    printf("  Errors (>1e-3)  : %d\n",  errors);
    printf("  MSE             : %e\n",  mse);
    printf("  MAE             : %e\n",  mae);
    printf("  Max error       : %e\n",  max_err);
    printf("  Target MSE      : < 1e-7\n");
    printf("  Target MAE      : < 1e-4\n");

    /* pass criteria from spec for S4D layer */
    if (mse < 1e-7 && mae < 1e-4) {
        printf("\nTEST PASSED! S4D layer matches PyTorch reference.\n");
        return 0;
    } else {
        printf("\nTEST FAILED!\n");
        if (mse >= 1e-7) printf("  MSE too high: %e (need < 1e-7)\n", mse);
        if (mae >= 1e-4) printf("  MAE too high: %e (need < 1e-4)\n", mae);
        return 1;
    }
}