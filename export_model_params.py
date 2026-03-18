# export_model_params.py
# exports all model parameters as individual binary files
# these get combined into model_weights.bin by combine_weights.py

import torch
import numpy as np
import os
from model import GalaxyClassifierS4D

os.makedirs('model_params', exist_ok=True)

model = GalaxyClassifierS4D(colored=True)
model.load_state_dict(torch.load('model_params/galaxys4-colored-31771.pth'))
model.eval()
print("Model loaded")

# hilbert indices -- already exists but export again to be sure
from model.hilbert import HilbertScan
hilbert = HilbertScan()
indices = hilbert.indices.numpy().astype(np.int32)
indices.tofile('model_params/hilbert_scan.indices.bin')
print(f"hilbert_scan.indices: {indices.shape}")

# uproject linear layer
model.uproject.weight.detach().numpy().astype(np.float32).tofile('model_params/uproject.weight.bin')
model.uproject.bias.detach().numpy().astype(np.float32).tofile('model_params/uproject.bias.bin')
print(f"uproject.weight: {model.uproject.weight.shape}")
print(f"uproject.bias:   {model.uproject.bias.shape}")

# s4 layers
for name, layer in [('s4_1', model.s4_1), ('s4_2', model.s4_2), ('s4_3', model.s4_3)]:
    layer.log_dt.detach().numpy().astype(np.float32).tofile(f'model_params/{name}.log_dt.bin')
    layer.log_A_real.detach().numpy().astype(np.float32).tofile(f'model_params/{name}.log_A_real.bin')
    layer.A_imag.detach().numpy().astype(np.float32).tofile(f'model_params/{name}.A_imag.bin')
    layer.C.detach().numpy().astype(np.float32).tofile(f'model_params/{name}.C.bin')
    layer.D.detach().numpy().astype(np.float32).tofile(f'model_params/{name}.D.bin')
    print(f"{name}.log_dt:     {layer.log_dt.shape}")
    print(f"{name}.log_A_real: {layer.log_A_real.shape}")
    print(f"{name}.A_imag:     {layer.A_imag.shape}")
    print(f"{name}.C:          {layer.C.shape}")
    print(f"{name}.D:          {layer.D.shape}")

# fc layer
model.fc.weight.detach().numpy().astype(np.float32).tofile('model_params/fc.weight.bin')
model.fc.bias.detach().numpy().astype(np.float32).tofile('model_params/fc.bias.bin')
print(f"fc.weight: {model.fc.weight.shape}")
print(f"fc.bias:   {model.fc.bias.shape}")

print("\nAll parameters exported successfully!")
print("Now run: python combine_weights.py")