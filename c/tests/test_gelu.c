/* validates the gelu C implementation against pytorch reference
 * run export_test_data.py first to generate the reference files */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"

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

double compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return sum / n;
}

double compute_mae(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs((double)a[i] - (double)b[i]);
    }
    return sum / n;
}

float compute_max_err(const float* a, const float* b, int n) {
    float max = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max) max = diff;
    }
    return max;
}

int main() {
    printf("Testing GELU Implementation\n");
    printf("============================\n");

    /* gelu is applied to the full s4d output: (4096, 64) = 262144 elements */
    int total = IMG_SIZE * HIDDEN_DIM;

    static float input[IMG_SIZE * HIDDEN_DIM];
    static float output[IMG_SIZE * HIDDEN_DIM];
    static float expected[IMG_SIZE * HIDDEN_DIM];

    memset(input,    0, sizeof(input));
    memset(output,   0, sizeof(output));
    memset(expected, 0, sizeof(expected));

    printf("\nLoading test data...\n");
    load_floats("../data/samples/gelu_input.bin",  input,    total);
    printf("  Loaded input    (%d floats)\n", total);
    load_floats("../data/samples/gelu_output.bin", expected, total);
    printf("  Loaded expected (%d floats)\n", total);

    /* copy input into output then apply gelu in-place */
    memcpy(output, input, sizeof(float) * total);
    printf("\nRunning gelu_forward...\n");
    gelu_forward(output, total);

    printf("\nFirst 5 output values (ours vs expected):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] got: %f   expected: %f\n",
               i, output[i], expected[i]);
    }

    double mse     = compute_mse(output, expected, total);
    double mae     = compute_mae(output, expected, total);
    float  max_err = compute_max_err(output, expected, total);

    int errors = 0;
    for (int i = 0; i < total; i++) {
        if (fabsf(output[i] - expected[i]) > 1e-4f) {
            if (errors < 5) {
                printf("  Mismatch at [%d]: got %f  expected %f\n",
                       i, output[i], expected[i]);
            }
            errors++;
        }
    }

    printf("\nResults:\n");
    printf("  Output elements : %d\n",  total);
    printf("  Errors (>1e-4)  : %d\n",  errors);
    printf("  MSE             : %e\n",  mse);
    printf("  MAE             : %e\n",  mae);
    printf("  Max error       : %e\n",  max_err);
    printf("  Target MSE      : < 1e-7\n");
    printf("  Target MAE      : < 1e-4\n");

    if (mse < 1e-7 && mae < 1e-4) {
        printf("\nTEST PASSED! GELU matches PyTorch reference.\n");
        return 0;
    } else {
        printf("\nTEST FAILED!\n");
        if (mse >= 1e-7) printf("  MSE too high: %e (need < 1e-7)\n", mse);
        if (mae >= 1e-4) printf("  MAE too high: %e (need < 1e-4)\n", mae);
        return 1;
    }
}
