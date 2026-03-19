/* c/src/benchmark.c
 * performance benchmarking for the S4D galaxy classifier
 * measures per-layer timing averaged over multiple iterations
 * and total inference time
 *
 * usage:
 *   ./benchmark <weights.bin> <image.bin>
 *
 * output:
 *   per-layer timing breakdown
 *   total inference time
 *   throughput in images per second */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../include/nn.h"

/* number of iterations to average timing over
 * more iterations = more stable results but longer runtime
 * 100 is sufficient for stable measurements on most hardware */
#define NUM_ITERATIONS 100

/* number of warmup iterations before timing starts
 * warmup ensures caches are populated and cpu is at steady state
 * results without warmup can be misleadingly slow */
#define WARMUP_ITERATIONS 5

/* get current time in seconds as a double
 * uses CLOCK_MONOTONIC which is not affected by system time changes
 * resolution is typically nanoseconds on Linux */
static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* load float binary file into buffer */
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

/* ------------------------------------------------------------------ */
/* per-layer timing buffers                                            */
/* declared static at file scope to avoid stack overflow              */
/* ------------------------------------------------------------------ */
static float bench_image[IMG_SIZE * 3];
static float bench_hilbert[IMG_SIZE * 3];
static float bench_proj[IMG_SIZE * HIDDEN_DIM];
static float bench_s4d_a[IMG_SIZE * HIDDEN_DIM];
static float bench_s4d_b[IMG_SIZE * HIDDEN_DIM];
static float bench_last[HIDDEN_DIM];
static float bench_logits[NUM_CLASSES];
static float bench_probs[NUM_CLASSES];

/* timing arrays -- one entry per iteration */
static double times_hilbert[NUM_ITERATIONS];
static double times_linear1[NUM_ITERATIONS];
static double times_s4d1[NUM_ITERATIONS];
static double times_gelu1[NUM_ITERATIONS];
static double times_s4d2[NUM_ITERATIONS];
static double times_gelu2[NUM_ITERATIONS];
static double times_s4d3[NUM_ITERATIONS];
static double times_gelu3[NUM_ITERATIONS];
static double times_tlts[NUM_ITERATIONS];
static double times_linear2[NUM_ITERATIONS];
static double times_softmax[NUM_ITERATIONS];
static double times_total[NUM_ITERATIONS];

/* compute mean of an array of doubles */
static double mean_of(const double* arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum / n;
}

/* compute standard deviation of an array of doubles */
static double std_of(const double* arr, int n, double mean) {
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = arr[i] - mean;
        var += d * d;
    }
    return sqrt(var / n);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <weights.bin> <image.bin>\n", argv[0]);
        return 1;
    }

    /* load weights first -- must happen before calling getters */
    if (load_model_weights(argv[1]) != 0) {
        printf("FATAL: Failed to load weights\n");
        return 2;
    }

    /* get pointers to weights loaded by nn.c
     * these must be called AFTER load_model_weights()
     * the getters return pointers to the static arrays inside nn.c */
    int32_t*   hidx    = get_hilbert_indices();
    float*     uproj_w = get_uproject_weight();
    float*     uproj_b = get_uproject_bias();
    S4DParams* s4p1    = get_s4_1_params();
    S4DParams* s4p2    = get_s4_2_params();
    S4DParams* s4p3    = get_s4_3_params();
    float*     fc_w    = get_fc_weight();
    float*     fc_b    = get_fc_bias();

    /* load test image */
    load_floats(argv[2], bench_image, IMG_SIZE * 3);
    printf("Weights and image loaded\n\n");

    /* warmup -- run a few full forward passes to populate caches
     * and get cpu to steady-state frequency before timing starts */
    printf("Warming up (%d iterations)...\n", WARMUP_ITERATIONS);
    for (int w = 0; w < WARMUP_ITERATIONS; w++) {
        float probs[NUM_CLASSES];
        model_forward(bench_image, 3, probs);
    }

    /* timed iterations -- time each layer individually */
    printf("Benchmarking (%d iterations)...\n\n", NUM_ITERATIONS);

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double t0, t1;

        /* record total start */
        double t_start = get_time_seconds();

        /* step 1: hilbert scan */
        t0 = get_time_seconds();
        hilbert_scan(bench_image, bench_hilbert,
                     hidx, 3, IMG_DIM, IMG_DIM);
        t1 = get_time_seconds();
        times_hilbert[iter] = t1 - t0;

        /* step 2: input projection (4096,3) -> (4096,64) */
        t0 = get_time_seconds();
        linear_forward(bench_hilbert, uproj_w, uproj_b,
                       bench_proj, IMG_SIZE, 3, HIDDEN_DIM);
        t1 = get_time_seconds();
        times_linear1[iter] = t1 - t0;

        /* step 3: s4d layer 1 */
        t0 = get_time_seconds();
        s4d_forward(bench_proj, bench_s4d_a, s4p1, IMG_SIZE);
        t1 = get_time_seconds();
        times_s4d1[iter] = t1 - t0;

        /* step 4: gelu 1 in-place */
        t0 = get_time_seconds();
        gelu_forward(bench_s4d_a, IMG_SIZE * HIDDEN_DIM);
        t1 = get_time_seconds();
        times_gelu1[iter] = t1 - t0;

        /* step 5: s4d layer 2 */
        t0 = get_time_seconds();
        s4d_forward(bench_s4d_a, bench_s4d_b, s4p2, IMG_SIZE);
        t1 = get_time_seconds();
        times_s4d2[iter] = t1 - t0;

        /* step 6: gelu 2 in-place */
        t0 = get_time_seconds();
        gelu_forward(bench_s4d_b, IMG_SIZE * HIDDEN_DIM);
        t1 = get_time_seconds();
        times_gelu2[iter] = t1 - t0;

        /* step 7: s4d layer 3 */
        t0 = get_time_seconds();
        s4d_forward(bench_s4d_b, bench_s4d_a, s4p3, IMG_SIZE);
        t1 = get_time_seconds();
        times_s4d3[iter] = t1 - t0;

        /* step 8: gelu 3 in-place */
        t0 = get_time_seconds();
        gelu_forward(bench_s4d_a, IMG_SIZE * HIDDEN_DIM);
        t1 = get_time_seconds();
        times_gelu3[iter] = t1 - t0;

        /* step 9: take last timestep */
        t0 = get_time_seconds();
        take_last_timestep(bench_s4d_a, bench_last, IMG_SIZE);
        t1 = get_time_seconds();
        times_tlts[iter] = t1 - t0;

        /* step 10: fc layer (64,) -> (4,) */
        t0 = get_time_seconds();
        linear_forward(bench_last, fc_w, fc_b,
                       bench_logits, 1, HIDDEN_DIM, NUM_CLASSES);
        t1 = get_time_seconds();
        times_linear2[iter] = t1 - t0;

        /* step 11: softmax */
        t0 = get_time_seconds();
        softmax_forward(bench_logits, bench_probs);
        t1 = get_time_seconds();
        times_softmax[iter] = t1 - t0;

        times_total[iter] = get_time_seconds() - t_start;
    }

    /* compute mean and stddev for each layer */
    double mean_hilbert  = mean_of(times_hilbert,  NUM_ITERATIONS);
    double mean_linear1  = mean_of(times_linear1,  NUM_ITERATIONS);
    double mean_s4d1     = mean_of(times_s4d1,     NUM_ITERATIONS);
    double mean_gelu1    = mean_of(times_gelu1,    NUM_ITERATIONS);
    double mean_s4d2     = mean_of(times_s4d2,     NUM_ITERATIONS);
    double mean_gelu2    = mean_of(times_gelu2,    NUM_ITERATIONS);
    double mean_s4d3     = mean_of(times_s4d3,     NUM_ITERATIONS);
    double mean_gelu3    = mean_of(times_gelu3,    NUM_ITERATIONS);
    double mean_tlts     = mean_of(times_tlts,     NUM_ITERATIONS);
    double mean_linear2  = mean_of(times_linear2,  NUM_ITERATIONS);
    double mean_softmax  = mean_of(times_softmax,  NUM_ITERATIONS);
    double mean_total    = mean_of(times_total,    NUM_ITERATIONS);

    double std_hilbert   = std_of(times_hilbert,  NUM_ITERATIONS, mean_hilbert);
    double std_linear1   = std_of(times_linear1,  NUM_ITERATIONS, mean_linear1);
    double std_s4d1      = std_of(times_s4d1,     NUM_ITERATIONS, mean_s4d1);
    double std_gelu1     = std_of(times_gelu1,    NUM_ITERATIONS, mean_gelu1);
    double std_s4d2      = std_of(times_s4d2,     NUM_ITERATIONS, mean_s4d2);
    double std_gelu2     = std_of(times_gelu2,    NUM_ITERATIONS, mean_gelu2);
    double std_s4d3      = std_of(times_s4d3,     NUM_ITERATIONS, mean_s4d3);
    double std_gelu3     = std_of(times_gelu3,    NUM_ITERATIONS, mean_gelu3);
    double std_tlts      = std_of(times_tlts,     NUM_ITERATIONS, mean_tlts);
    double std_linear2   = std_of(times_linear2,  NUM_ITERATIONS, mean_linear2);
    double std_softmax   = std_of(times_softmax,  NUM_ITERATIONS, mean_softmax);
    double std_total     = std_of(times_total,    NUM_ITERATIONS, mean_total);

    /* sum of individual layer means for percentage calculation
     * this is slightly less than mean_total due to timing overhead */
    double sum_layers = mean_hilbert + mean_linear1 +
                        mean_s4d1    + mean_gelu1   +
                        mean_s4d2    + mean_gelu2   +
                        mean_s4d3    + mean_gelu3   +
                        mean_tlts    + mean_linear2  + mean_softmax;

    /* print results table */
    printf("=======================================================================\n");
    printf("Per-Layer Timing Breakdown (averaged over %d iterations)\n",
           NUM_ITERATIONS);
    printf("=======================================================================\n");
    printf("%-25s %10s %10s %8s\n",
           "Layer", "Mean(ms)", "Std(ms)", "Pct(%%)");
    printf("-----------------------------------------------------------------------\n");

    #define PRINT_ROW(name, mn, sd) \
        printf("%-25s %10.4f %10.4f %7.2f%%\n", \
               name, (mn)*1000.0, (sd)*1000.0, \
               (mn) / sum_layers * 100.0)

    PRINT_ROW("Hilbert Scan",      mean_hilbert, std_hilbert);
    PRINT_ROW("Linear (uproject)", mean_linear1, std_linear1);
    PRINT_ROW("S4D Layer 1",       mean_s4d1,    std_s4d1);
    PRINT_ROW("GELU 1",            mean_gelu1,   std_gelu1);
    PRINT_ROW("S4D Layer 2",       mean_s4d2,    std_s4d2);
    PRINT_ROW("GELU 2",            mean_gelu2,   std_gelu2);
    PRINT_ROW("S4D Layer 3",       mean_s4d3,    std_s4d3);
    PRINT_ROW("GELU 3",            mean_gelu3,   std_gelu3);
    PRINT_ROW("TakeLastTimestep",  mean_tlts,    std_tlts);
    PRINT_ROW("Linear (fc)",       mean_linear2, std_linear2);
    PRINT_ROW("Softmax",           mean_softmax, std_softmax);

    printf("-----------------------------------------------------------------------\n");
    printf("%-25s %10.4f %10.4f\n",
           "TOTAL", mean_total*1000.0, std_total*1000.0);
    printf("\nThroughput: %.4f images/second\n", 1.0 / mean_total);
    printf("Optimization level: -O2 (default)\n");

    return 0;
}