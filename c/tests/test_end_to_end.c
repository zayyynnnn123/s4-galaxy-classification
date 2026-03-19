/* c/tests/test_end_to_end.c
 * validates the full end-to-end forward pass on a single test sample
 * compares C output probabilities against pytorch reference
 *
 * usage:
 *   ./test_end_to_end <weights.bin> <input.bin> <ref_probs.bin> <true_label>
 *
 * exits with 0 if predictions match, 1 if they differ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"

/* class names for human readable output */
static const char* CLASS_NAMES[4] = {
    "Smooth", "Disk", "Edge-on", "Irregular"
};

/* load float binary file into buffer
 * exits if file cannot be opened or has wrong size */
static void load_floats(const char* path, float* buf, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("FATAL: Cannot open %s\n", path);
        exit(1);
    }
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) {
        printf("FATAL: Expected %zu floats but got %zu from %s\n",
               n, got, path);
        exit(1);
    }
}

/* compute mse between two float arrays */
static double compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum / n;
}

/* compute mae between two float arrays */
static double compute_mae(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += fabs((double)a[i] - (double)b[i]);
    return sum / n;
}

/* find index of largest value */
static int argmax(const float* arr, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[best]) best = i;
    return best;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <weights.bin> <input.bin> <ref_probs.bin> <true_label>\n",
               argv[0]);
        return 2;
    }

    const char* weights_path   = argv[1];
    const char* input_path     = argv[2];
    const char* ref_probs_path = argv[3];
    int         true_label     = atoi(argv[4]);

    /* load model weights -- only load once */
    static int weights_loaded_flag = 0;
    if (!weights_loaded_flag) {
        if (load_model_weights(weights_path) != 0) {
            printf("FATAL: Failed to load weights\n");
            return 3;
        }
        weights_loaded_flag = 1;
    }

    /* load input image (64,64,3) = 12288 floats */
    static float image[IMG_SIZE * 3];
    memset(image, 0, sizeof(image));
    load_floats(input_path, image, IMG_SIZE * 3);

    /* load reference probabilities from pytorch */
    float ref_probs[NUM_CLASSES];
    load_floats(ref_probs_path, ref_probs, NUM_CLASSES);

    /* run C forward pass */
    float c_probs[NUM_CLASSES];
    model_forward(image, 3, c_probs);

    /* compute error metrics */
    double mse     = compute_mse(c_probs, ref_probs, NUM_CLASSES);
    double mae     = compute_mae(c_probs, ref_probs, NUM_CLASSES);
    float  max_err = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float d = fabsf(c_probs[i] - ref_probs[i]);
        if (d > max_err) max_err = d;
    }

    /* get predictions */
    int c_pred  = argmax(c_probs,   NUM_CLASSES);
    int py_pred = argmax(ref_probs, NUM_CLASSES);

    /* print results */
    printf("True label   : %d (%s)\n", true_label, CLASS_NAMES[true_label]);
    printf("Python pred  : %d (%s)\n", py_pred,    CLASS_NAMES[py_pred]);
    printf("C pred       : %d (%s)\n", c_pred,     CLASS_NAMES[c_pred]);
    printf("\nProbabilities (C vs Python):\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("  Class %d (%s): C=%.6f  Py=%.6f  diff=%.2e\n",
               i, CLASS_NAMES[i], c_probs[i], ref_probs[i],
               fabsf(c_probs[i] - ref_probs[i]));
    }
    printf("\nError metrics:\n");
    printf("  MSE     : %e  (target: prediction match)\n", mse);
    printf("  MAE     : %e  (target: prediction match)\n", mae);
    printf("  Max err : %e\n", max_err);
    printf("  Prob sum: %.8f (should be 1.0)\n",
           c_probs[0]+c_probs[1]+c_probs[2]+c_probs[3]);

/* pass criteria for end-to-end test:
 * the doc requires 100% prediction agreement between C and Python
 * MSE/MAE tolerances are PER-LAYER metrics NOT end-to-end
 * softmax output differences accumulate across all layers so we
 * use a relaxed tolerance here and rely on prediction match as
 * the primary validation criterion */
    int pred_match = (c_pred == py_pred);
    int mse_pass   = (mse < 1e-1);   /* relaxed -- softmax accumulates error */
    int mae_pass   = (mae < 5e-1);   /* relaxed -- 4 class output vector */
    int pass       = pred_match && mse_pass && mae_pass;

    if (pass) {
        printf("\nPASS\n");
        return 0;
    } else {
        printf("\nFAIL\n");
        if (!pred_match)
            printf("  prediction mismatch: C=%d Python=%d\n", c_pred, py_pred);
        if (!mse_pass)
            printf("  MSE too high: %e\n", mse);
        if (!mae_pass)
            printf("  MAE too high: %e\n", mae);
        return 1;
    }
}