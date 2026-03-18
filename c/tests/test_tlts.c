/* validates the take_last_timestep C implementation against pytorch reference
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
    printf("Testing TakeLastTimestep Implementation\n");
    printf("========================================\n");

    /* input is (4096, 64), output is last row (64,) */
    static float input[IMG_SIZE * HIDDEN_DIM];
    float output[HIDDEN_DIM];
    float expected[HIDDEN_DIM];

    memset(input,    0, sizeof(input));
    memset(output,   0, sizeof(output));
    memset(expected, 0, sizeof(expected));

    printf("\nLoading test data...\n");
    load_floats("../data/samples/tlts_input.bin",  input,    IMG_SIZE * HIDDEN_DIM);
    printf("  Loaded input    (%d floats)\n", IMG_SIZE * HIDDEN_DIM);
    load_floats("../data/samples/tlts_output.bin", expected, HIDDEN_DIM);
    printf("  Loaded expected (%d floats)\n", HIDDEN_DIM);

    printf("\nRunning take_last_timestep...\n");
    take_last_timestep(input, output, IMG_SIZE);

    printf("\nFirst 5 output values (ours vs expected):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] got: %f   expected: %f\n",
               i, output[i], expected[i]);
    }

    double mse     = compute_mse(output, expected, HIDDEN_DIM);
    double mae     = compute_mae(output, expected, HIDDEN_DIM);
    float  max_err = compute_max_err(output, expected, HIDDEN_DIM);

    /* for pure indexing we expect zero errors -- any mismatch is a bug */
    int errors = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        if (output[i] != expected[i]) {
            if (errors < 5) {
                printf("  Mismatch at [%d]: got %f  expected %f\n",
                       i, output[i], expected[i]);
            }
            errors++;
        }
    }

    printf("\nResults:\n");
    printf("  Output elements : %d\n",  HIDDEN_DIM);
    printf("  Errors (exact)  : %d\n",  errors);
    printf("  MSE             : %e\n",  mse);
    printf("  MAE             : %e\n",  mae);
    printf("  Max error       : %e\n",  max_err);
    printf("  Target MSE      : < 1e-12 (pure indexing, should be exact)\n");

    if (errors == 0 && mse < 1e-12) {
        printf("\nTEST PASSED! TakeLastTimestep matches PyTorch reference exactly.\n");
        return 0;
    } else {
        printf("\nTEST FAILED! Check tlts.c index calculations.\n");
        return 1;
    }
}
