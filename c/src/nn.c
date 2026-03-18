/* c/src/nn.c
 * implements the full model forward pass and weight loader
 * this file chains all layers together in the correct order:
 *
 *   hilbert_scan -> linear -> s4d_1 -> gelu -> s4d_2 -> gelu
 *   -> s4d_3 -> gelu -> take_last_timestep -> linear -> softmax
 *
 * the weight loader reads model_weights.bin in the exact same order
 * that combine_weights.py wrote it -- any mismatch will silently load
 * wrong parameters so the order is absolutly critical
 *
 * all intermediate buffers are static arrays -- no malloc anywhere
 * this matches the assembly mindset where memory is fixed at compile time
 *
 * IMPORTANT: this model was trained with colored=True meaning C=3 always
 * grayscale (C=1) is not supported by the current weight file */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../include/nn.h"

/* ------------------------------------------------------------------ */
/* model channel count -- this model was trained on RGB images        */
/* changing this requires retraining and re-exporting weights         */
/* ------------------------------------------------------------------ */
#define MODEL_CHANNELS 3

/* expected total file size in bytes -- used to validate weight file
 * computed from the byte layout table:
 *   16384 + 768 + 256 + 3*(256+8192+8192+16384+256) + 1024 + 16
 * = 118288 bytes */
#define EXPECTED_WEIGHT_BYTES 118288

/* ------------------------------------------------------------------ */
/* model parameters -- loaded from model_weights.bin once             */
/* ------------------------------------------------------------------ */

/* flag to track if weights have been loaded
 * model_forward checks this and refuses to run if not loaded */
static int weights_loaded = 0;

/* hilbert curve indices -- 4096 int32 values */
static int32_t hilbert_indices[IMG_SIZE];

/* input projection: Linear(3, 64) */
static float uproject_weight[HIDDEN_DIM * MODEL_CHANNELS];
static float uproject_bias[HIDDEN_DIM];

/* three S4D layers */
static S4DParams s4_1_params;
static S4DParams s4_2_params;
static S4DParams s4_3_params;

/* final classification: Linear(64, 4) */
static float fc_weight[NUM_CLASSES * HIDDEN_DIM];
static float fc_bias[NUM_CLASSES];

/* ------------------------------------------------------------------ */
/* intermediate activation buffers -- static, reused each call        */
/* ------------------------------------------------------------------ */
static float buf_hilbert[IMG_SIZE * MODEL_CHANNELS]; /* (4096, 3) */
static float buf_proj[IMG_SIZE * HIDDEN_DIM];         /* (4096, 64) */
static float buf_s4d_a[IMG_SIZE * HIDDEN_DIM];        /* (4096, 64) */
static float buf_s4d_b[IMG_SIZE * HIDDEN_DIM];        /* (4096, 64) */
static float buf_last[HIDDEN_DIM];                    /* (64,) */
static float buf_logits[NUM_CLASSES];                 /* (4,) */

/* ------------------------------------------------------------------ */
/* file reading helpers                                                */
/* ------------------------------------------------------------------ */

static int read_floats(FILE* f, float* buf, size_t n, const char* name) {
    size_t got = fread(buf, sizeof(float), n, f);
    if (got != n) {
        printf("ERROR loading %s: expected %zu floats but got %zu\n",
               name, n, got);
        return -1;
    }
    return 0;
}

static int read_int32s(FILE* f, int32_t* buf, size_t n, const char* name) {
    size_t got = fread(buf, sizeof(int32_t), n, f);
    if (got != n) {
        printf("ERROR loading %s: expected %zu int32s but got %zu\n",
               name, n, got);
        return -1;
    }
    return 0;
}

static int read_s4d_params(FILE* f, S4DParams* p, const char* name) {
    char buf[64];
    size_t total = 0;

    snprintf(buf, sizeof(buf), "%s.log_dt", name);
    if (read_floats(f, p->log_dt, HIDDEN_DIM, buf) != 0) return -1;
    total += HIDDEN_DIM;

    snprintf(buf, sizeof(buf), "%s.log_A_real", name);
    if (read_floats(f, p->log_A_real, HIDDEN_DIM * S4D_N, buf) != 0) return -1;
    total += HIDDEN_DIM * S4D_N;

    snprintf(buf, sizeof(buf), "%s.A_imag", name);
    if (read_floats(f, p->A_imag, HIDDEN_DIM * S4D_N, buf) != 0) return -1;
    total += HIDDEN_DIM * S4D_N;

    snprintf(buf, sizeof(buf), "%s.C", name);
    if (read_floats(f, p->C, HIDDEN_DIM * S4D_N * 2, buf) != 0) return -1;
    total += HIDDEN_DIM * S4D_N * 2;

    snprintf(buf, sizeof(buf), "%s.D", name);
    if (read_floats(f, p->D, HIDDEN_DIM, buf) != 0) return -1;
    total += HIDDEN_DIM;

    /* each S4D layer should have exactly 8320 floats
     * log_dt(64) + log_A_real(2048) + A_imag(2048) + C(4096) + D(64) */
    printf("  Loaded %-10s params (%zu floats)\n", name, total);
    return 0;
}

/* ------------------------------------------------------------------ */
/* load_model_weights                                                  */
/* ------------------------------------------------------------------ */
int load_model_weights(const char* path) {

    /* validate file size before reading anything
     * if wrong size the export script probably had a bug */
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: Cannot open weights file: %s\n", path);
        return -1;
    }

    /* seek to end to get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);  /* seek back to beginning */

    if (file_size != EXPECTED_WEIGHT_BYTES) {
        printf("ERROR: Weight file is %ld bytes but expected %d bytes\n",
               file_size, EXPECTED_WEIGHT_BYTES);
        printf("       Re-run export_model_params.py and combine_weights.py\n");
        fclose(f);
        return -1;
    }
    printf("Loading weights from: %s (%ld bytes)\n", path, file_size);

    /* read hilbert indices -- int32 not float */
    if (read_int32s(f, hilbert_indices, IMG_SIZE, "hilbert_indices") != 0) {
        fclose(f); return -1;
    }
    printf("  Loaded hilbert_indices  (%d int32s)\n", IMG_SIZE);

    /* read uproject layer */
    if (read_floats(f, uproject_weight,
                    HIDDEN_DIM * MODEL_CHANNELS, "uproject.weight") != 0) {
        fclose(f); return -1;
    }
    if (read_floats(f, uproject_bias, HIDDEN_DIM, "uproject.bias") != 0) {
        fclose(f); return -1;
    }
    printf("  Loaded uproject         (%d + %d floats)\n",
           HIDDEN_DIM * MODEL_CHANNELS, HIDDEN_DIM);

    /* read three S4D layers */
    if (read_s4d_params(f, &s4_1_params, "s4_1") != 0) { fclose(f); return -1; }
    if (read_s4d_params(f, &s4_2_params, "s4_2") != 0) { fclose(f); return -1; }
    if (read_s4d_params(f, &s4_3_params, "s4_3") != 0) { fclose(f); return -1; }

    /* read fc layer */
    if (read_floats(f, fc_weight,
                    NUM_CLASSES * HIDDEN_DIM, "fc.weight") != 0) {
        fclose(f); return -1;
    }
    if (read_floats(f, fc_bias, NUM_CLASSES, "fc.bias") != 0) {
        fclose(f); return -1;
    }
    printf("  Loaded fc               (%d + %d floats)\n",
           NUM_CLASSES * HIDDEN_DIM, NUM_CLASSES);

    fclose(f);

    /* set flag so model_forward knows weights are ready */
    weights_loaded = 1;
    printf("All weights loaded successfully! Total: %d bytes\n",
           EXPECTED_WEIGHT_BYTES);
    return 0;
}

/* ------------------------------------------------------------------ */
/* model_forward                                                       */
/* ------------------------------------------------------------------ */
void model_forward(float* input_image, int C, float* output_probs) {

    /* guard against calling forward without loading weights first
     * this would silently use uninitialized memory and produce
     * garbage output with no error message -- very hard to debug */
    if (!weights_loaded) {
        printf("ERROR: model_forward called before load_model_weights!\n");
        printf("       Call load_model_weights(path) first\n");
        exit(1);
    }

    /* this model only supports RGB input -- C must be 3
     * if you need grayscale you need to retrain and re-export */
    if (C != MODEL_CHANNELS) {
        printf("ERROR: model expects C=%d channels but got C=%d\n",
               MODEL_CHANNELS, C);
        printf("       This model was trained with colored=True (RGB)\n");
        exit(1);
    }
    

    /* step 1: hilbert scan (H,W,C) -> (4096,C) */
    hilbert_scan(input_image, buf_hilbert, hilbert_indices,
                 C, IMG_DIM, IMG_DIM);

    /* step 2: input projection (4096,C) -> (4096,64) */
    linear_forward(buf_hilbert, uproject_weight, uproject_bias,
                   buf_proj, IMG_SIZE, C, HIDDEN_DIM);

    /* step 3: s4d layer 1 (4096,64) -> (4096,64) */
    s4d_forward(buf_proj, buf_s4d_a, &s4_1_params, IMG_SIZE);

    /* step 4: gelu in-place */
    gelu_forward(buf_s4d_a, IMG_SIZE * HIDDEN_DIM);
        
    /* step 5: s4d layer 2 */
    s4d_forward(buf_s4d_a, buf_s4d_b, &s4_2_params, IMG_SIZE);

    /* step 6: gelu in-place */
    gelu_forward(buf_s4d_b, IMG_SIZE * HIDDEN_DIM);

    /* step 7: s4d layer 3 */
    s4d_forward(buf_s4d_b, buf_s4d_a, &s4_3_params, IMG_SIZE);

    /* step 8: gelu in-place */
    gelu_forward(buf_s4d_a, IMG_SIZE * HIDDEN_DIM);

    /* step 9: take last timestep (4096,64) -> (64,) */
    take_last_timestep(buf_s4d_a, buf_last, IMG_SIZE);

    /* step 10: fc layer (64,) -> (4,)
     * rows=1 because input is a single vector not a sequence */
    linear_forward(buf_last, fc_weight, fc_bias,
                   buf_logits, 1, HIDDEN_DIM, NUM_CLASSES);

    /* step 11: softmax (4,) -> (4,) probabilities */
    softmax_forward(buf_logits, output_probs);
}