/* validates the softmax C implementation against pytorch reference
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
    printf("Testing Softmax Implementation\n");
    printf("================================\n");

    /* softmax input and output are both (NUM_CLASSES,) = 4 elements */
    float input[NUM_CLASSES];
    float output[NUM_CLASSES];
    float expected[NUM_CLASSES];

    memset(input,    0, sizeof(input));
    memset(output,   0, sizeof(output));
    memset(expected, 0, sizeof(expected));

    printf("\nLoading test data...\n");
    load_floats("../data/samples/softmax_input.bin",  input,    NUM_CLASSES);
    printf("  Loaded input    (%d floats)\n", NUM_CLASSES);
    load_floats("../data/samples/softmax_output.bin", expected, NUM_CLASSES);
    printf("  Loaded expected (%d floats)\n", NUM_CLASSES);

    printf("\nRunning softmax_forward...\n");
    softmax_forward(input, output);

    /* print all 4 values since the array is small */
    printf("\nAll output values (ours vs expected):\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("  class [%d] got: %f   expected: %f\n",
               i, output[i], expected[i]);
    }

    double mse     = compute_mse(output, expected, NUM_CLASSES);
    double mae     = compute_mae(output, expected, NUM_CLASSES);
    float  max_err = compute_max_err(output, expected, NUM_CLASSES);

    /* verify probabilities sum to 1.0 */
    float prob_sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) prob_sum += output[i];

    int errors = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (fabsf(output[i] - expected[i]) > 1e-4f) errors++;
    }

    printf("\nResults:\n");
    printf("  Output elements : %d\n",  NUM_CLASSES);
    printf("  Errors (>1e-4)  : %d\n",  errors);
    printf("  MSE             : %e\n",  mse);
    printf("  MAE             : %e\n",  mae);
    printf("  Max error       : %e\n",  max_err);
    printf("  Prob sum        : %.8f  (should be 1.0)\n", prob_sum);
    printf("  Target MSE      : < 1e-8\n");
    printf("  Target MAE      : < 1e-4\n");

    if (mse < 1e-8 && mae < 1e-4 && fabsf(prob_sum - 1.0f) < 1e-6f) {
        printf("\nTEST PASSED! Softmax matches PyTorch reference.\n");
        return 0;
    } else {
        printf("\nTEST FAILED!\n");
        if (mse >= 1e-8) printf("  MSE too high: %e (need < 1e-8)\n", mse);
        if (mae >= 1e-4) printf("  MAE too high: %e (need < 1e-4)\n", mae);
        if (fabsf(prob_sum - 1.0f) >= 1e-6f)
            printf("  Prob sum wrong: %f (need 1.0)\n", prob_sum);
        return 1;
    }
}
