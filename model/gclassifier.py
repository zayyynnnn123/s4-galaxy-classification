#--------------------------------------
# RUN THIS FILE IN UBUNTU WITH THE COMMAND: python -m model.gclassifier
#---------------------------------------

import torch
import torch.nn as nn
import sys
import os
import math

# Fix the import path - add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now use absolute imports instead of relative imports
from model.hilbert import HilbertScan
from model.tlts import TakeLastTimestep
from model.s4d import S4D


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
    """
    def __init__(self, s4_state=64, d_model=64, num_classes=4, colored=True):
        super().__init__()
        self.seq_len = 64 * 64 
        self.d_model = d_model
        self.s4_state = s4_state
        self.num_classes = num_classes

        # Hilbert Scan layer
        self.hilbert_scan = HilbertScan()
        self.hilbert_channels = 1 if not colored else 3

        self.uproject = nn.Linear(self.hilbert_channels, d_model)

        # S4 layers - NOW WITH 3 LAYERS!
        self.s4_1 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act1 = nn.GELU()

        self.s4_2 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act2 = nn.GELU()
        
        # NEW THIRD S4D LAYER
        self.s4_3 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act3 = nn.GELU()

        # Take last timestep
        self.take_last = TakeLastTimestep()

        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_logits=False):
        # First we convert the 2D image to a 1D sequence using the Hilbert scan
        B, C, H, W = x.shape
        x = self.hilbert_scan(x)  # (B, seq_len, hilbert_channels)

        # Now we will project the hilbert scan output to the d_model dimension
        x = self.uproject(x)  # (B, seq_len, d_model)

        # First S4 layer 
        x, _ = self.s4_1(x)  # (B, seq_len, d_model)
        x = self.act1(x)

        # Second S4 layer
        x, _ = self.s4_2(x)  # (B, seq_len, d_model)
        x = self.act2(x)
        
        # THIRD S4 LAYER - NEW!
        x, _ = self.s4_3(x)  # (B, seq_len, d_model)
        x = self.act3(x)

        # Take the last timestep as the summary 
        x = self.take_last(x)  # (B, d_model)
        
        # Map the summary to the class logits
        logits = self.fc(x)  # (B, num_classes)
        
        if return_logits:
            return logits
        else:
            probs = self.softmax(logits)  # (B, num_classes)
            return probs
    
    # ============================================================
    # FLOPs Estimation (Section 8.5) - UPDATED for 3 layers
    # ============================================================
    def get_flops_estimate(self, batch_size=1):
        """
        Estimate FLOPs for one forward pass (Section 8.5).
        
        Derivation:
        - Hilbert scanning: O(1) - just indexing, negligible
        - Input projection: L × C × d_model  (L=4096, C=3, d_model=64)
        - S4D layer: O(L log L · d_model) each
        - Classification head: d_model × num_classes
        
        Returns:
            Dictionary with FLOPs breakdown
        """
        L = self.seq_len  # 4096
        C = self.hilbert_channels  # 1 or 3
        d = self.d_model  # 64
        N = self.s4_state  # 64
        classes = self.num_classes  # 4
        
        # Input projection: (L * C * d) multiply-adds
        input_flops = L * C * d
        
        # S4D layers: O(L log L · d) each
        log2_L = math.log2(L)
        s4_flops_per_layer = int(L * log2_L * d)
        s4_flops = 3 * s4_flops_per_layer  # THREE S4D layers (updated)
        
        # Classification head: d * classes
        head_flops = d * classes
        
        # Total
        total_flops = input_flops + s4_flops + head_flops
        
        # Return breakdown
        return {
            'input_projection': input_flops,
            's4d_layers': s4_flops,
            'classification_head': head_flops,
            'total_flops': total_flops,
            'total_gflops': total_flops / 1e9  # GigaFLOPs
        }
    
    # ============================================================
    # Forward Pass Trace (Section 8.7) - UPDATED for 3 layers
    # ============================================================
    def trace_forward_pass(self, device='cpu'):
        """
        Show tensor shapes at each layer for one RGB image.
        
        This creates an example trace for documentation purposes.
        
        Args:
            device: 'cpu' or 'cuda' for demonstration
        
        Returns:
            Dictionary with shapes at each layer
        """
        print("\n" + "=" * 60)
        print("GalaxyClassifierS4D Forward Pass Trace (3 Layers)")
        print("=" * 60)
        print(f"Device: {device}")
        
        # Create dummy input (1 RGB image)
        x = torch.randn(1, 3, 64, 64).to(device)
        print(f"\nInput: {x.shape}")
        
        shapes = {}
        
        # HilbertScan
        x = self.hilbert_scan(x)
        shapes['hilbert'] = tuple(x.shape)
        print(f"After HilbertScan: {x.shape}")
        
        # Input projection
        x = self.uproject(x)
        shapes['projection'] = tuple(x.shape)
        print(f"After uProject: {x.shape}")
        
        # First S4D layer
        x, _ = self.s4_1(x)
        shapes['s4d_1'] = tuple(x.shape)
        print(f"After S4D-1: {x.shape}")
        
        # First GELU
        x = self.act1(x)
        shapes['gelu_1'] = tuple(x.shape)
        print(f"After GELU-1: {x.shape}")
        
        # Second S4D layer
        x, _ = self.s4_2(x)
        shapes['s4d_2'] = tuple(x.shape)
        print(f"After S4D-2: {x.shape}")
        
        # Second GELU
        x = self.act2(x)
        shapes['gelu_2'] = tuple(x.shape)
        print(f"After GELU-2: {x.shape}")
        
        # THIRD S4D LAYER - NEW!
        x, _ = self.s4_3(x)
        shapes['s4d_3'] = tuple(x.shape)
        print(f"After S4D-3: {x.shape}")
        
        # Third GELU
        x = self.act3(x)
        shapes['gelu_3'] = tuple(x.shape)
        print(f"After GELU-3: {x.shape}")
        
        # TakeLastTimestep
        x = self.take_last(x)
        shapes['take_last'] = tuple(x.shape)
        print(f"After TakeLastTimestep: {x.shape}")
        
        # Classification head
        logits = self.fc(x)
        shapes['classifier'] = tuple(logits.shape)
        print(f"After FC: {logits.shape}")
        
        print("\n" + "=" * 60)
        print("Trace complete!")
        print("=" * 60)
        
        return shapes
    
    # ============================================================
    # Parameter Count Analysis (Section 8.4) - UPDATED for 3 layers
    # ============================================================
    def get_parameter_count(self):
        """Calculate number of trainable parameters by component."""
        print("\n" + "=" * 60)
        print("Parameter Count Analysis (3 S4D Layers)")
        print("=" * 60)
        
        total = 0
        params = {}
        
        # HilbertScan (0 params)
        hilbert_params = sum(p.numel() for p in self.hilbert_scan.parameters())
        params['HilbertScan'] = hilbert_params
        total += hilbert_params
        
        # Input projection
        input_params = sum(p.numel() for p in self.uproject.parameters())
        params['Input Projection'] = input_params
        total += input_params
        
        # S4D Layer 1
        s4d1_params = sum(p.numel() for p in self.s4_1.parameters())
        params['S4D Layer 1'] = s4d1_params
        total += s4d1_params
        
        # S4D Layer 2
        s4d2_params = sum(p.numel() for p in self.s4_2.parameters())
        params['S4D Layer 2'] = s4d2_params
        total += s4d2_params
        
        # S4D Layer 3 - NEW!
        s4d3_params = sum(p.numel() for p in self.s4_3.parameters())
        params['S4D Layer 3'] = s4d3_params
        total += s4d3_params
        
        # GELU activations (0 params)
        params['GELU Activations'] = 0
        
        # TakeLastTimestep (0 params)
        take_last_params = sum(p.numel() for p in self.take_last.parameters())
        params['TakeLastTimestep'] = take_last_params
        total += take_last_params
        
        # Classification head
        fc_params = sum(p.numel() for p in self.fc.parameters())
        params['Classification Head'] = fc_params
        total += fc_params
        
        # Print table
        print(f"\n{'Component':<25} {'Parameters':>15}")
        print("-" * 42)
        for component, count in params.items():
            print(f"{component:<25} {count:>15,d}")
        print("-" * 42)
        print(f"{'TOTAL':<25} {total:>15,d}")
        print("=" * 60)
        
        return params, total


def test_classifier():
    """Test the GalaxyClassifierS4D implementation with GPU support."""
    print("=" * 60)
    print("Testing GalaxyClassifierS4D (Task 8) - 3 Layer Version")
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
    
    # Create model
    model_rgb = GalaxyClassifierS4D(colored=True).to(device)
    
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
    print("\n📊 Parameter Count:")
    model_rgb.get_parameter_count()
    
    # FLOPs estimation
    print("\n📊 FLOPs Estimation:")
    flops = model_rgb.get_flops_estimate()
    print(f"   Input projection: {flops['input_projection']:,} FLOPs")
    print(f"   S4D layers (3x):  {flops['s4d_layers']:,} FLOPs")
    print(f"   Classification:   {flops['classification_head']:,} FLOPs")
    print(f"   TOTAL:            {flops['total_flops']:,} FLOPs ({flops['total_gflops']:.2f} GFLOPs)")
    
    # Forward pass trace
    print("\n📋 Forward Pass Trace:")
    model_rgb.trace_forward_pass(device)
    
    # GPU Speed Test
    print("\n📊 GPU Speed Test:")
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
        
        with torch.no_grad():  # Disable gradients for speed
            for _ in range(10):
                _ = model_rgb(x_bench)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.time() - start) * 1000 / 10  # ms per iteration
        
        images_per_sec = batch_size * (1000 / elapsed)
        print(f"{batch_size:<10} {elapsed:>15.2f} {images_per_sec:>15.1f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! GalaxyClassifierS4D (3-layer) ready for Task 8!")
    print("=" * 60)
    
    return model_rgb


# This allows running the test directly
if __name__ == "__main__":
    # If running as script, set up package context
    if __package__ is None:
        __package__ = "model"
    test_classifier()