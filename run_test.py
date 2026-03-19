# run_test.py
# runs test_end_to_end on all 12 test samples and reports aggregate results
# this is the automated batch testing script required by Task 2
# exits with code 0 only if all samples pass

import subprocess
import os
import numpy as np
import sys

# paths
WEIGHTS     = 'model_weights.bin'
TEST_DIR    = 'data/test_samples'
TEST_BIN    = 'c/test_end_to_end'
NUM_SAMPLES = 12

CLASS_NAMES = ['Smooth', 'Disk', 'Edge-on', 'Irregular']

def main():
    # first build the test binary
    print("Building test binary...")
    result = subprocess.run(
        ['make', 'test_end_to_end'],
        cwd='c',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("BUILD FAILED:")
        print(result.stderr)
        sys.exit(1)
    print("Build successful\n")

    # load true labels
    labels_path = os.path.join(TEST_DIR, 'true_labels.bin')
    true_labels = np.fromfile(labels_path, dtype=np.int32)

    # run each sample
    passed = 0
    failed = 0
    results = []

    print("=" * 60)
    print(f"Running {NUM_SAMPLES} test samples")
    print("=" * 60)

    for i in range(NUM_SAMPLES):
        input_path = os.path.join(TEST_DIR, f'sample_{i:02d}_input.bin')
        probs_path = os.path.join(TEST_DIR, f'sample_{i:02d}_probs.bin')
        true_label = int(true_labels[i])

        # run the C test binary
        result = subprocess.run(
            [f'./{TEST_BIN}', WEIGHTS, input_path, probs_path, str(true_label)],
            capture_output=True,
            text=True
        )

        output = result.stdout
        passed_sample = (result.returncode == 0)

        # parse output for key values
        c_pred  = None
        py_pred = None
        mse     = None
        mae     = None

        for line in output.split('\n'):
            if line.startswith('C pred'):
                c_pred = int(line.split(':')[1].strip().split()[0])
            elif line.startswith('Python pred'):
                py_pred = int(line.split(':')[1].strip().split()[0])
            elif 'MSE' in line and ':' in line:
                try:
                    mse = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'MAE' in line and ':' in line:
                try:
                    mae = float(line.split(':')[1].strip().split()[0])
                except:
                    pass

        status = "PASS" if passed_sample else "FAIL"
        print(f"Sample {i:02d} | true={true_label} ({CLASS_NAMES[true_label]:<10}) | "
              f"py={py_pred} c={c_pred} | "
              f"MSE={mse:.2e} MAE={mae:.2e} | {status}")

        results.append({
            'sample': i,
            'true':   true_label,
            'py_pred': py_pred,
            'c_pred':  c_pred,
            'mse':     mse,
            'mae':     mae,
            'pass':    passed_sample
        })

        if passed_sample:
            passed += 1
        else:
            failed += 1

    # print aggregate summary
    print("=" * 60)
    print(f"RESULTS: {passed}/{NUM_SAMPLES} passed, {failed}/{NUM_SAMPLES} failed")
    print("=" * 60)

    # per class summary
    print("\nPer-class breakdown:")
    for cls in range(4):
        cls_results = [r for r in results if r['true'] == cls]
        cls_passed  = sum(1 for r in cls_results if r['pass'])
        print(f"  Class {cls} ({CLASS_NAMES[cls]:<10}): "
              f"{cls_passed}/{len(cls_results)} passed")

    # overall mse and mae
    mse_vals = [r['mse'] for r in results if r['mse'] is not None]
    mae_vals = [r['mae'] for r in results if r['mae'] is not None]
    if mse_vals:
        print(f"\nAggregate MSE: mean={np.mean(mse_vals):.2e} "
              f"max={np.max(mse_vals):.2e}")
    if mae_vals:
        print(f"Aggregate MAE: mean={np.mean(mae_vals):.2e} "
              f"max={np.max(mae_vals):.2e}")

    # prediction agreement between C and Python
    agree = sum(1 for r in results
                if r['c_pred'] is not None and r['py_pred'] is not None
                and r['c_pred'] == r['py_pred'])
    print(f"\nC vs Python prediction agreement: {agree}/{NUM_SAMPLES}")

    if failed == 0:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n{failed} TESTS FAILED")
        sys.exit(1)

if __name__ == '__main__':
    main()