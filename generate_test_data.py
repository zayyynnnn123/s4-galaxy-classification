# generate_test_data.py
# exports test images and reference outputs from pytorch for Task 2 validation
# selects at least 10 samples covering all 4 galaxy classes
# saves each sample as a separate binary file for the C test program

import torch
import numpy as np
import h5py
import os
from model import GalaxyClassifierS4D
from model.hilbert import HilbertScan

# create output directory for test samples
os.makedirs('data/test_samples', exist_ok=True)

# load model
model = GalaxyClassifierS4D(colored=True)
model.load_state_dict(torch.load('model_params/galaxys4-colored-31771.pth'))
model.eval()
print("Model loaded")

# load test dataset
with h5py.File('data/GalaxyMNIST/raw/test_dataset.hdf5', 'r') as f:
    images = f['images'][:]   # (2000, 64, 64, 3) uint8
    labels = f['labels'][:]   # (2000,) uint8
print(f"Dataset: {images.shape}, labels: {labels.shape}")
print(f"Class distribution: {np.bincount(labels)}")

# select samples -- at least 3 per class, 12 total
# find indices for each class
selected_indices = []
samples_per_class = 3
for cls in range(4):
    cls_indices = np.where(labels == cls)[0]
    print(f"Class {cls}: {len(cls_indices)} samples available")
    # pick evenly spaced samples from this class for diversity
    chosen = cls_indices[:samples_per_class]
    selected_indices.extend(chosen.tolist())

print(f"\nSelected {len(selected_indices)} samples: {selected_indices}")
print(f"Their labels: {labels[selected_indices]}")

# now export each sample
hilbert = HilbertScan()

for i, idx in enumerate(selected_indices):
    true_label = int(labels[idx])

    # get image -- convert uint8 to float32 in range [0,1]
    # GalaxyMNIST images are uint8 (0-255), normalize to [0,1] to match training
    img_uint8 = images[idx]                          # (64, 64, 3) uint8
    img_float = img_uint8.astype(np.float32) / 255.0 # (64, 64, 3) float32

    # save input image in (H, W, C) layout for C code
    img_float.tofile(f'data/test_samples/sample_{i:02d}_input.bin')

    # run through pytorch model to get reference output
    # pytorch expects (B, C, H, W) so permute from (H, W, C)
    img_tensor = torch.tensor(img_float).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 64, 64)

    with torch.no_grad():
        # get raw logits then softmax
        logits = model(img_tensor)                              # (1, 4)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)      # (4,)
        pred   = probs.argmax().item()

    # save reference probabilities
    probs.numpy().astype(np.float32).tofile(
        f'data/test_samples/sample_{i:02d}_probs.bin')

    # save true label as single int32
    np.array([true_label], dtype=np.int32).tofile(
        f'data/test_samples/sample_{i:02d}_label.bin')

    print(f"Sample {i:02d} (dataset idx {idx}): "
          f"true={true_label} pred={pred} "
          f"probs=[{probs[0]:.3f} {probs[1]:.3f} {probs[2]:.3f} {probs[3]:.3f}]")

# save the list of selected indices and labels for reference
np.array(selected_indices, dtype=np.int32).tofile('data/test_samples/selected_indices.bin')
np.array([labels[i] for i in selected_indices], dtype=np.int32).tofile('data/test_samples/true_labels.bin')

print(f"\nExported {len(selected_indices)} test samples to data/test_samples/")
print("Files per sample:")
print("  sample_XX_input.bin  -- (64,64,3) float32 input image")
print("  sample_XX_probs.bin  -- (4,) float32 reference probabilities")
print("  sample_XX_label.bin  -- int32 true class label")
print("\nNow run: cd c && make test_batch")