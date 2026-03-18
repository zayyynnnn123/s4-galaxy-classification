# export_test_data.py
# run this from the project root directory before running any C tests
# it exports the hilbert indices, a test image, and the expected
# hilbert output so the C test has real data to compare against

import torch
import numpy as np
import os
from model import GalaxyClassifierS4D
from model.hilbert import HilbertScan

# make sure output directories exist before we try to write to them
os.makedirs('data/samples', exist_ok=True)
os.makedirs('model_params', exist_ok=True)

# load your trained model weights
# use the correct path to your model file
model_path = 'model_params/galaxys4-colored-31771.pth'
print(f"Loading model from: {model_path}")

# we need eval mode so dropout and batchnorm behave correctly
model = GalaxyClassifierS4D(colored=True)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ Model loaded successfully!")

# create a random test image with fixed seed so results are
# reproducible every time we run this script
torch.manual_seed(42)
test_image = torch.randn(1, 3, 64, 64)  # shape: (batch, C, H, W)

# get the hilbert scan module and extract its precomputed indices
hilbert = HilbertScan()
indices = hilbert.indices.numpy().astype(np.int32)
print(f"Hilbert indices shape: {indices.shape}")  # should be (4096,)

# save hilbert indices -- these are the same ones the C code
# will load from model_weights.bin so they must match exactly
indices.tofile('model_params/hilbert_scan.indices.bin')
print("✅ Saved hilbert indices --> model_params/hilbert_scan.indices.bin")

# save the test image in channel-last (H, W, C) layout
# because that is what the C code expects
# pytorch default is channel-first (C, H, W) so we must permute
test_image_hwc = test_image.squeeze(0).permute(1, 2, 0)  # (64, 64, 3)
print(f"Test image shape after permute: {test_image_hwc.shape}")  # (64,64,3)
test_image_hwc.numpy().astype(np.float32).tofile('data/samples/test_image_rgb.bin')
print("✅ Saved test image --> data/samples/test_image_rgb.bin")

# run the hilbert scan through pytorch to get the reference output
# the C code must match this output exactly (MSE < 1e-12)
with torch.no_grad():
    hilbert_output = hilbert(test_image).numpy()

# print the shape so we can verify it matches what C expects
print(f"Hilbert output shape: {hilbert_output.shape}")  # should be (1, 4096, 3) or (4096, 3)

# if there is a batch dimension, squeeze it out before saving
if hilbert_output.ndim == 3 and hilbert_output.shape[0] == 1:
    hilbert_output = hilbert_output.squeeze(0)  # (4096, 3)

print(f"Hilbert output shape after squeeze: {hilbert_output.shape}")  # (4096, 3)
hilbert_output.astype(np.float32).tofile('data/samples/hilbert_output.bin')
print("✅ Saved expected output --> data/samples/hilbert_output.bin")

print("\n✅ All test data exported successfully!")
print("\nFiles created:")
print("  - model_params/hilbert_scan.indices.bin")
print("  - data/samples/test_image_rgb.bin")
print("  - data/samples/hilbert_output.bin")
print("\nNow run: cd c && make test_hilbert")