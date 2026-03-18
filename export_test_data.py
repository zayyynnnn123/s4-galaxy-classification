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
# ---- linear layer test data ----
# we test the input projection layer (uproject)
# input shape:  (4096, 3)  -- hilbert scan output
# output shape: (4096, 64) -- projected to hidden dim

# get the uproject weights and bias from the model
uproject_weight = model.uproject.weight.detach().numpy().astype(np.float32)
uproject_bias   = model.uproject.bias.detach().numpy().astype(np.float32)

print(f"UProject weight shape: {uproject_weight.shape}")  # should be (64, 3)
print(f"UProject bias shape:   {uproject_bias.shape}")    # should be (64,)

# save weights and bias
uproject_weight.tofile('data/samples/uproject_weight.bin')
uproject_bias.tofile('data/samples/uproject_bias.bin')

# compute expected output using pytorch
# input to linear is the hilbert output we already computed
hilbert_tensor = torch.tensor(hilbert_output).unsqueeze(0)  # (1, 4096, 3)
with torch.no_grad():
    linear_output = model.uproject(hilbert_tensor).numpy()

if linear_output.ndim == 3:
    linear_output = linear_output.squeeze(0)  # (4096, 64)

print(f"Linear output shape: {linear_output.shape}")  # should be (4096, 64)
linear_output.astype(np.float32).tofile('data/samples/linear_output.bin')

# also save the hilbert output as linear input so C test can load it directly
hilbert_output.astype(np.float32).tofile('data/samples/linear_input.bin')

print("Saved linear layer test data")

# ---- S4D layer test data ----
# we test s4_1 only, s4_2 and s4_3 follow the exact same pattern
# input shape:  (4096, 64) -- output of uproject linear layer
# output shape: (4096, 64) -- same shape, sequence transformed by SSM

# get S4D layer 1 parameters -- attribute is s4_1 not s4_layers[0]
s4d_layer = model.s4_1

# extract each parameter as numpy float32
s4d_log_dt     = s4d_layer.log_dt.detach().numpy().astype(np.float32)
s4d_log_A_real = s4d_layer.log_A_real.detach().numpy().astype(np.float32)
s4d_A_imag     = s4d_layer.A_imag.detach().numpy().astype(np.float32)
s4d_C          = s4d_layer.C.detach().numpy().astype(np.float32)
s4d_D          = s4d_layer.D.detach().numpy().astype(np.float32)

print(f"S4D log_dt shape:     {s4d_log_dt.shape}")      # should be (64,)
print(f"S4D log_A_real shape: {s4d_log_A_real.shape}")  # should be (64, 32)
print(f"S4D A_imag shape:     {s4d_A_imag.shape}")      # should be (64, 32)
print(f"S4D C shape:          {s4d_C.shape}")            # should be (64, 32, 2)
print(f"S4D D shape:          {s4d_D.shape}")            # should be (64,)

# save all parameters to binary files
s4d_log_dt.tofile('data/samples/s4d_log_dt.bin')
s4d_log_A_real.tofile('data/samples/s4d_log_A_real.bin')
s4d_A_imag.tofile('data/samples/s4d_A_imag.bin')
s4d_C.tofile('data/samples/s4d_C.bin')
s4d_D.tofile('data/samples/s4d_D.bin')
print("Saved S4D parameters")

# the input to s4_1 is the output of uproject (linear layer)
# linear_output is (4096, 64) meaning (L, H) layout
# pytorch S4D expects (B, H, L) so we need to:
#   1. convert to tensor
#   2. transpose L and H dimensions
#   3. add batch dimension
linear_tensor  = torch.tensor(linear_output)                # (4096, 64) = (L, H)
print(f"linear_tensor shape: {linear_tensor.shape}")

# transposed=False means S4D expects (B, L, H) so just add batch dim
s4d_input_BLH  = linear_tensor.unsqueeze(0)                 # (1, 4096, 64) = (B, L, H)
print(f"S4D input to pytorch: {s4d_input_BLH.shape}")       # should be (1, 4096, 64)

# save s4d input in (L, H) layout for C code
linear_output.astype(np.float32).tofile('data/samples/s4d_input.bin')
print(f"S4D input shape for C: {linear_output.shape}")      # (4096, 64)

# run through pytorch s4_1 to get the reference output
with torch.no_grad():
    s4d_output_BLH, _ = s4d_layer(s4d_input_BLH)           # (1, 4096, 64)

# squeeze batch dimension, output is already (L, H) because transposed=False
s4d_output_LH = s4d_output_BLH.squeeze(0).numpy()          # (4096, 64)
print(f"S4D output shape for C: {s4d_output_LH.shape}")     # (4096, 64)
s4d_output_LH.astype(np.float32).tofile('data/samples/s4d_output.bin')

print("Saved S4D test data:")
print("  - data/samples/s4d_log_dt.bin")
print("  - data/samples/s4d_log_A_real.bin")
print("  - data/samples/s4d_A_imag.bin")
print("  - data/samples/s4d_C.bin")
print("  - data/samples/s4d_D.bin")
print("  - data/samples/s4d_input.bin")
print("  - data/samples/s4d_output.bin")