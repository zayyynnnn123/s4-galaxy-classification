// c/tests/test_hilbert.c
// validates the hilbert scan C implementation against python reference output
// run this after export_test_data.py has been executed from the project root

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>      /* needed for exit() */
#include "../include/nn.h"

/* hilbert_scan is declared in nn.h so no need to redeclare it here
 * if you get implicit declaration warnings it means nn.h is missing
 * the function signature -- go fix nn.h not this file */

/*
 * compare_float -- checks if two floats are close enough
 * returns 1 if they are within tolerance, 0 if they differ too much
 * we use fabsf for absolute value of a float (not double)
 */
int compare_float(float a, float b, float tolerance) {
    return fabsf(a - b) < tolerance;
}

/*
 * load_test_data -- loads float32 binary data from a file into buffer
 * size is the number of FLOATS to read, not the number of bytes
 * calls exit(1) if file cant be opened or has wrong number of elements
 * this is intentional -- if data doesnt load we should not continue
 * and compare garbage values, that just wastes time debuging fake errors
 */
void load_test_data(const char* filename, float* buffer, size_t size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("FATAL: Cannot open file: %s\n", filename);
        printf("       Make sure you ran export_test_data.py first!\n");
        exit(1);   /* stop everything, no point continuing with empty buffers */
    }
    size_t elements_read = fread(buffer, sizeof(float), size, fp);
    fclose(fp);
    if (elements_read != size) {
        printf("FATAL: Expected %zu floats but only got %zu from %s\n",
               size, elements_read, filename);
        printf("       The file exists but may be wrong size or corrupted\n");
        exit(1);
    }
}

/*
 * load_indices -- loads int32 hilbert indices from a binary file
 * same logic as load_test_data but reads int32_t instead of float
 * the indices are precomputed in python and saved as raw int32 bytes
 */
void load_indices(const char* filename, int32_t* buffer, size_t size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("FATAL: Cannot open file: %s\n", filename);
        printf("       Make sure you ran export_test_data.py first!\n");
        exit(1);
    }
    size_t elements_read = fread(buffer, sizeof(int32_t), size, fp);
    fclose(fp);
    if (elements_read != size) {
        printf("FATAL: Expected %zu indices but only got %zu from %s\n",
               size, elements_read, filename);
        exit(1);
    }
}

int main() {
    printf("Testing Hilbert Scan Implementation\n");
    printf("====================================\n");
    
    /* test parameters -- must match what export_test_data.py used */
    int height   = 64;
    int width    = 64;
    int channels = 3;
    int seq_len  = height * width;  /* 4096 total pixels */
    
    /*
     * declare all arrays as local stack variables
     * sizes are known at compile time so no malloc needed
     * this mirrors what we will do in assembly later -- fixed size buffers
     * image   : (64, 64, 3) channel-last layout
     * sequence: (4096, 3)   hilbert-ordered output from our C code
     * expected: (4096, 3)   reference output from python
     * indices : (4096,)     precomputed hilbert curve pixel order
     */
    float   image[64 * 64 * 3];    /* using 1D arrays is cleaner for fread */
    float   sequence[4096 * 3];
    float   expected[4096 * 3];
    int32_t indices[4096];
    
    /* zero everything out before loading
     * this way if something goes wrong we get zeros not garbage */
    memset(image,    0, sizeof(image));
    memset(sequence, 0, sizeof(sequence));
    memset(expected, 0, sizeof(expected));
    memset(indices,  0, sizeof(indices));
    
    /* load all three input files -- paths are relative to c/ folder
     * since make runs from the c/ directory
     * ../model_params/ goes up one level to the project root */
    printf("\nLoading test data...\n");
    load_indices("../model_params/hilbert_scan.indices.bin", indices, 4096);
    printf("  Loaded hilbert indices\n");
    load_test_data("../data/samples/test_image_rgb.bin", image, 64*64*3);
    printf("  Loaded test image\n");
    load_test_data("../data/samples/hilbert_output.bin", expected, 4096*3);
    printf("  Loaded expected output\n");
    
    /* run our C hilbert scan implementation
     * this is the function we are testing -- defined in src/hilbert.c */
    printf("\nRunning Hilbert scan...\n");
    hilbert_scan(image, sequence, indices, channels, height, width);
    
    /*
     * compare every element of our output against the python reference
     * we track:
     *   errors   -- count of elements that differ by more than 1e-5
     *   max_diff -- the single largest difference found
     *   mse      -- mean squared error across all elements
     * for pure indexing like hilbert scan we expect ZERO errors
     * any mismatch means the index math in hilbert.c is wrong
     */
    printf("\nComparing results...\n");
    int    errors   = 0;
    float  max_diff = 0.0f;
    double mse      = 0.0;
    
    for (int i = 0; i < seq_len; i++) {
        for (int c = 0; c < channels; c++) {
            /* index into flat 1D arrays -- (i,c) -> i*channels + c */
            float our_val = sequence[i * channels + c];
            float ref_val = expected[i * channels + c];
            float diff    = fabsf(our_val - ref_val);
            
            mse += (double)diff * (double)diff;
            if (diff > max_diff) max_diff = diff;
            
            /* print first 10 mismatches so we can see the pattern
             * if there are many errors they are usually all the same bug */
            if (diff > 1e-5f) {
                if (errors < 10) {
                    printf("  Mismatch at [%d][%d]: got %f  expected %f  diff=%e\n",
                           i, c, our_val, ref_val, diff);
                }
                errors++;
            }
        }
    }
    
    mse /= (double)(seq_len * channels);
    
    /* print summary statistics */
    printf("\nResults:\n");
    printf("  Total elements : %d\n", seq_len * channels);
    printf("  Errors found   : %d\n", errors);
    printf("  Max difference : %e\n", max_diff);
    printf("  MSE            : %e\n", mse);
    printf("  Target MSE     : < 1e-12 (pure indexing should be exact)\n");
    
    /* for hilbert scan specifically we expect bit-for-bit exact match
     * because it is pure indexing with no floating point arithmetic at all
     * any error here means the index calculation is wrong */
    if (errors == 0 && mse < 1e-12) {
        printf("\nTEST PASSED! Hilbert scan matches Python reference exactly.\n");
        return 0;
    } else {
        printf("\nTEST FAILED! Check hilbert.c index calculations.\n");
        return 1;
    }
}