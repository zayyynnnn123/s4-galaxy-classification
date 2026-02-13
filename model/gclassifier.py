#--------------------------------------
#RUN THIS FILE IN UBUNTU WITH THE COMMAND: python -m model.gclassifier
#there is an error in the test_classifier function, it is not able to find the HilbertScan and S4D classes, make sure to implement those classes in the same directory as this file or adjust the import statements accordingly.
#---------------------------------------

import torch
import torch.nn as nn

from .hilbert import HilbertScan
from .tlts import TakeLastTimestep
from .s4d import S4D

class GalaxyClassifierS4D(nn.Module):
    """
    Galaxy classifier using Hilbert Scan and S4 sequence modeling.
    
    This model scans 2D galaxy images into a 1D Hilbert sequence, projects
    the multi-channel pixel values to a higher-dimensional feature space,
    processes the sequence with stacked S4 layers with GELU activations, 
    takes the final timestep as a summary representation, and applies a 
    linear classifier to predict galaxy types.
    
    Parameters
    ----------
    s4_state : int, optional
        Hidden state dimension for the S4 layers (default is 64).
    d_model : int, optional
        Output feature dimension of the S4 layers (default is 64).
    num_classes : int, optional
        Number of output classes (default is 4).
    colored : bool, optional
        If True, expects RGB input images (3 channels); if False, expects
        grayscale images (1 channel) (default is True).
    
    Attributes
    ----------
    seq_len : int
        Sequence length after Hilbert scan (64*64 = 4096).
    d_model : int
        Dimension of the S4 output features.
    hilbert_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    hilbert_scan : HilbertScan
        Layer that converts 2D images into 1D sequences using a Hilbert scan.
    uproject : nn.Linear
        Linear projection mapping hilbert_channels to d_model dimensions.
    s4_1 : S4D
        First S4 layer.
    act1 : nn.GELU
        GELU activation after the first S4 layer.
    s4_2 : S4D
        Second S4 layer.
    act2 : nn.GELU
        GELU activation after the second S4 layer.
    take_last : TakeLastTimestep
        Layer that extracts the last timestep from the sequence.
    fc : nn.Linear
        Linear classifier mapping S4 features to output classes.
    softmax : nn.Softmax
        Softmax layer for output probabilities.
    """
    def __init__(self, s4_state=64, d_model=64, num_classes=4, colored=True):
        super().__init__()
        self.seq_len = 64 * 64 
        self.d_model = d_model

        # Hilbert Scan layer
        self.hilbert_scan = HilbertScan()
        self.hilbert_channels = 1 if not colored else 3

        self.uproject = nn.Linear(self.hilbert_channels, d_model)

        # S4 layers
        self.s4_1 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act1 = nn.GELU()

        self.s4_2 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act2 = nn.GELU()

        # Take last timestep
        self.take_last = TakeLastTimestep()

        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_logits=False):
        #first we convert the 2D image to a 1D sequence using the Hilbert scan
        B, C, H, W = x.shape
        x = self.hilbert_scan(x)  # (B, seq_len, hilbert_channels)

        #now we will project the hilbert scan output to the d_model dimension
        x = self.uproject(x)  # (B, seq_len, d_model)

        #first S4 layer 
        x, _ = self.s4_1(x)  # (B, seq_len, d_model)
        x = self.act1(x)

        #now for second S4 layer
        x, _ = self.s4_2(x)  # (B, seq_len, d_model)
        x = self.act2(x)

        #take the last timestep as the summary 
        x = self.take_last(x)  # (B, d_model)
        #mapp the summery to the class logits

        logits = self.fc(x)  # (B, num_classes)
        if return_logits:
            return logits
        else:
            probs = self.softmax(logits)  # (B, num_classes)
            return probs
        

def test_classifier():
    """Test the GalaxyClassifierS4D implementation with GPU support."""
    print("=" * 60)
    print("Testing GalaxyClassifierS4D (Task 8)")
    print("=" * 60)
    
    # --- GPU Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   ⚠️  Running on CPU - this will be SLOW!")
        print("   💡 To enable GPU, install PyTorch with CUDA:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Test with grayscale (colored=False)
    print("\n📋 Test 1: Grayscale input (colored=False)")
    model_gray = GalaxyClassifierS4D(colored=False).to(device)
    x_gray = torch.randn(2, 1, 64, 64).to(device)
    
    # Warmup (for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        for _ in range(5):
            _ = model_gray(x_gray, return_logits=True)
        torch.cuda.synchronize()
    
    # Time the forward pass
    import time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    logits = model_gray(x_gray, return_logits=True)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.time() - start) * 1000  # ms
    
    print(f"   Logits shape: {logits.shape}")
    print(f"   Forward time: {elapsed:.2f} ms")
    assert logits.shape == (2, 4), f"Shape mismatch: {logits.shape}"
    
    probs = model_gray(x_gray, return_logits=False)
    print(f"   Probs shape:  {probs.shape}")
    print(f"   Sum of probs: {probs[0].sum().item():.4f}")
    print(f"   ✅ Passed!")
    
    # Test with RGB (colored=True)
    print("\n📋 Test 2: RGB input (colored=True)")
    model_rgb = GalaxyClassifierS4D(colored=True).to(device)
    x_rgb = torch.randn(2, 3, 64, 64).to(device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    logits = model_rgb(x_rgb, return_logits=True)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.time() - start) * 1000
    
    print(f"   Logits shape: {logits.shape}")
    print(f"   Forward time: {elapsed:.2f} ms")
    assert logits.shape == (2, 4), f"Shape mismatch: {logits.shape}"
    print(f"   ✅ Passed!")
    
    # Parameter count
    print("\n📋 Parameter Count:")
    total_params = sum(p.numel() for p in model_rgb.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Optional: Benchmark different batch sizes
    print("\n📋 GPU Speed Test:")
    batch_sizes = [1, 4, 16, 32]
    print(f"{'Batch':<10} {'Time (ms)':>15} {'Images/sec':>15}")
    print("-" * 42)
    
    for batch_size in batch_sizes:
        x_bench = torch.randn(batch_size, 3, 64, 64).to(device)
        
        # Warmup
        for _ in range(5):
            _ = model_rgb(x_bench)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            _ = model_rgb(x_bench)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.time() - start) * 1000 / 10  # ms per iteration
        
        images_per_sec = batch_size * (1000 / elapsed)
        print(f"{batch_size:<10} {elapsed:>15.2f} {images_per_sec:>15.1f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! GalaxyClassifierS4D ready for Task 8!")
    print("=" * 60)
    
    return model_rgb

#--------------------------------------
#RUN THIS FILE IN UBUNTU WITH THE COMMAND: python -m model.gclassifier
#there is an error in the test_classifier function, it is not able to find the HilbertScan and S4D classes, make sure to implement those classes in the same directory as this file or adjust the import statements accordingly.
#---------------------------------------

"""
================================
If Nividia GPU is available, this test will run on GPU and report the forward pass time in milliseconds. If no GPU is detected, it will run on CPU and warn about the slower performance.
use this command in terminal to run the test function with GPU support:python -c "from model.gclassifier import test_classifier; test_classifier()"
"""