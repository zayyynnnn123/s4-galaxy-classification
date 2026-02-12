"""
S4 Convolutional Implementation
Task 5.2: Implement convolutional formulation of S4 with HiPPO initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try multiple import strategies
try:
    from s4_recurrent import S4Recurrent
    from hilbert import HilbertScan
except ImportError:
    print("Error: Ensure 's4_recurrent.py' and 'hilbert.py' are in the same folder!")

try:
    # When running as module: python -m model.s4_conv
    from model.s4_recurrent import S4Recurrent
    print("✓ Imported S4Recurrent (as module)")
except ImportError:
    try:
        # When running as script: python model/s4_conv.py
        from s4_recurrent import S4Recurrent
        print("✓ Imported S4Recurrent (as script)")
    except ImportError:
        try:
            # When running from project root
            from model.s4_recurrent import S4Recurrent
            print("✓ Imported S4Recurrent (with model prefix)")
        except ImportError:
            # Last resort - direct import
            sys.path.insert(0, project_root)
            from model.s4_recurrent import S4Recurrent
            print("✓ Imported S4Recurrent (with path fix)")

class S4Convolutional(nn.Module):
    """
    Convolutional formulation of Structured State Space (S4) model.
    Uses HiPPO-LegT initialization for optimal long-range memory.
    """
    
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # --- HiPPO-LegT Initialization for A matrix ---
        # This MUST match EXACTLY with recurrent implementation
        P = torch.sqrt(1 + 2 * torch.arange(d_state, dtype=torch.float32))
        A = P[:, None] * P[None, :]
        A = -torch.tril(A)  # Make ALL entries negative
        A = A + torch.diag(torch.arange(d_state, dtype=torch.float32))  # Add i to diagonal
        self.A = nn.Parameter(A)
        
        # --- B matrix: input projection ---
        self.B = nn.Parameter(torch.randn(d_state, 1) / math.sqrt(d_state))
        
        # --- C matrix: output projection ---
        self.C = nn.Parameter(torch.randn(1, d_state) / math.sqrt(d_state))
        
        # --- D scalar: skip connection ---
        self.D = nn.Parameter(torch.randn(1))
        
        # --- Step size Δ in log-space ---
        log_dt_init = (math.log(dt_min) + math.log(dt_max)) / 2.0
        self.log_dt = nn.Parameter(torch.tensor([log_dt_init]))
        
        # --- Input/output projections ---
        self.input_proj = nn.Linear(d_model, 1, bias=False)
        self.output_proj = nn.Linear(1, d_model, bias=False)
        
        # --- For storing discretized matrices ---
        self.register_buffer('A_bar', None)
        self.register_buffer('B_bar', None)
    
    def _discretize(self):
        """Compute discretized matrices Ā and B̄ using matrix exponential."""
        dt = torch.exp(self.log_dt).clamp(self.dt_min, self.dt_max)
        
        # Ā = exp(Δ A)
        delta_A = dt * self.A
        A_bar = torch.matrix_exp(delta_A)
        
        # B̄ = (Δ A)⁻¹ (exp(Δ A) - I) (Δ B)
        I = torch.eye(self.d_state, device=self.A.device)
        try:
            inv_delta_A = torch.inverse(delta_A)
        except:
            inv_delta_A = torch.pinverse(delta_A)
        
        delta_B = dt * self.B
        B_bar = inv_delta_A @ (A_bar - I) @ delta_B
        
        return A_bar, B_bar
    
    def compute_kernel(self, L):
        """
        Compute SSM convolution kernel of length L.
        
        Returns:
            kernel: (1, 1, L) - convolution kernel for conv1d (FLIPPED for proper convolution)
        """
        A_bar, B_bar = self._discretize()
        
        self.A_bar = A_bar
        self.B_bar = B_bar
        
        # Compute kernel elements: K[i] = C @ Ā^i @ B̄
        kernel = []
        
        # K[0] = C @ B̄
        kernel.append(self.C @ B_bar)
        
        # Compute powers of A_bar
        A_power = A_bar.clone()
        for i in range(1, L):
            k_i = self.C @ A_power @ B_bar
            kernel.append(k_i)
            A_power = A_power @ A_bar
        
        # Stack kernel elements
        kernel = torch.stack(kernel).squeeze()  # (L,)
        
        # --- CRITICAL: Flip kernel for proper convolution ---
        # PyTorch conv1d performs cross-correlation, not convolution
        kernel = torch.flip(kernel, dims=[0])
        
        # Reshape for conv1d: (1, 1, L)
        kernel = kernel.view(1, 1, L)
        
        return kernel
    
    def forward(self, u):
        """Forward pass using convolution."""
        batch_size, seq_len, _ = u.shape
        
        # Project input
        u_proj = self.input_proj(u)
        u_proj = u_proj.transpose(1, 2)  # (B, 1, L)
        
        # Compute kernel
        K = self.compute_kernel(seq_len)  # (1, 1, L)
        
        # Apply causal convolution
        y = F.conv1d(
            u_proj,
            K,
            padding=seq_len - 1
        )
        y = y[..., :seq_len]  # Take first L elements
        
        # Add skip connection
        y = y + self.D * u_proj
        
        # Project output
        y = y.transpose(1, 2)
        y = self.output_proj(y)
        
        return y
    
    def get_complexity(self, L):
        """Compute computational complexity comparison."""
        N = self.d_state
        return {
            'recurrent_O(L·N²)': L * N * N,
            'kernel_computation_O(L·N²)': L * N * N,
            'naive_convolution_O(L²·N)': L * L * N,
            'fft_convolution_O(L_logL·N)': int(L * math.log2(L) * N),
        }


def test_s4_convolutional():
    """Test function to verify S4Convolutional implementation."""
    print("=" * 60)
    print("Testing S4Convolutional Implementation with HiPPO-LegT")
    print("=" * 60)
    
    # Create model
    d_model = 64
    d_state = 32
    model = S4Convolutional(d_model=d_model, d_state=d_state)
    
    # Test input
    batch_size = 2
    seq_len = 100
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = model(test_input)
    kernel = model.compute_kernel(seq_len)
    
    print(f"\n Model Configuration:")
    print(f"   - d_model: {d_model}")
    print(f"   - d_state: {d_state}")
    print(f"   - A matrix shape: {model.A.shape}")
    print(f"   - A[0,0]: {model.A[0,0].item():.4f} (should be -1.0)")
    
    print(f"\n Shape Checks:")
    print(f"   - Input shape:  {test_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Kernel shape: {kernel.shape}")
    print(f"   - A_bar shape:  {model.A_bar.shape}")
    print(f"   - B_bar shape:  {model.B_bar.shape}")
    
    assert output.shape == test_input.shape
    assert kernel.shape == (1, 1, seq_len)
    print(f"\n Shape verification: PASSED")
    
    return model, output


def verify_equivalence():
    """Verify recurrent and convolutional implementations match EXACTLY."""
    print("=" * 60)
    print("Verifying Recurrent ↔ Convolutional Equivalence")
    print("=" * 60)
    
    try:
        from model.s4_recurrent import S4Recurrent
    except ImportError:
        print(" S4Recurrent not found")
        return
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create models with IDENTICAL initialization
    d_model = 64
    d_state = 32
    
    conv_model = S4Convolutional(d_model=d_model, d_state=d_state)
    rec_model = S4Recurrent(d_model=d_model, d_state=d_state)
    
    # --- CRITICAL: Force EXACT same parameters ---
    with torch.no_grad():
        # Copy ALL parameters
        rec_model.A.data = conv_model.A.data.clone()
        rec_model.B.data = conv_model.B.data.clone()
        rec_model.C.data = conv_model.C.data.clone()
        rec_model.D.data = conv_model.D.data.clone()
        rec_model.log_dt.data = conv_model.log_dt.data.clone()
        rec_model.input_proj.weight.data = conv_model.input_proj.weight.data.clone()
        rec_model.output_proj.weight.data = conv_model.output_proj.weight.data.clone()
    
    # Verify HiPPO matrices are IDENTICAL
    A_diff = torch.max(torch.abs(rec_model.A - conv_model.A)).item()
    print(f"\n HiPPO Verification:")
    print(f"   - A matrices identical: {A_diff < 1e-10}")
    print(f"   - Max A difference: {A_diff:.2e}")
    
    # Test with small sequence for speed
    batch_size = 2
    seq_len = 20
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Forward passes
    with torch.no_grad():
        conv_output = conv_model(test_input)
        rec_output = rec_model(test_input)
    
    # Compute differences
    diff = torch.abs(conv_output - rec_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n Equivalence Test Results:")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Max absolute difference: {max_diff:.2e}")
    print(f"   - Mean absolute difference: {mean_diff:.2e}")
    
    # Check discretized matrices
    A_bar_conv, B_bar_conv = conv_model._discretize()
    A_bar_rec, B_bar_rec = rec_model._discretize()
    
    A_bar_diff = torch.max(torch.abs(A_bar_conv - A_bar_rec)).item()
    B_bar_diff = torch.max(torch.abs(B_bar_conv - B_bar_rec)).item()
    
    print(f"\n Discretization Verification:")
    print(f"   - A_bar identical: {A_bar_diff < 1e-10}")
    print(f"   - B_bar identical: {B_bar_diff < 1e-10}")
    
    if max_diff < 1e-5:
        print(f"\n EQUIVALENCE PASSED! (diff < 1e-5)")
        print(f"   Recurrent and Convolutional implementations match!")
    else:
        print(f"\n EQUIVALENCE FAILED! (diff = {max_diff:.2e})")
        
        # Print first few values to diagnose
        print(f"\n📋 First 5 outputs comparison:")
        print(f"{'t':<4} {'Recurrent':>15} {'Conv':>15} {'Diff':>15}")
        for t in range(5):
            r = rec_output[0, t, 0].item()
            c = conv_output[0, t, 0].item()
            d = abs(r - c)
            print(f"{t:<4} {r:15.8f} {c:15.8f} {d:15.8f}")
    
    print("=" * 60)
    return conv_output, rec_output

def run_all_milestone_tests():
    """Performs the tests required for Milestone 1."""
    print("\n" + "="*50)
    print("STARTING MILESTONE 1 VALIDATION")
    print("="*50)

    # Settings
    d_model, d_state = 1, 32
    
    # 1. Setup Models
    conv_model = S4Convolutional(d_model, d_state)
    rec_model = S4Recurrent(d_model, d_state)
    scan = HilbertScan()
    
    # Sync weights so they are identical 'twins'
    rec_model.load_state_dict(conv_model.state_dict())
    
    # 2. Create a "Galaxy Image" (64x64)
    fake_galaxy = torch.randn(1, 1, 64, 64)
    
    # 3. Test Hilbert Path
    print(f"Step 1: Flattening galaxy using Hilbert Curve...")
    sequence = scan(fake_galaxy) # (1, 4096, 1)
    print(f"✓ Sequence created. Length: {sequence.shape[1]} pixels.")

    # 4. Test Brain Equivalence
    print(f"Step 2: Feeding sequence to Recurrent and Convolutional brains...")
    with torch.no_grad():
        out_conv = conv_model(sequence)
        out_rec = rec_model(sequence)
    
    # Calculate Difference
    diff = torch.abs(out_conv - out_rec).max().item()
    
    print("-" * 30)
    print(f"MAX NUMERICAL DIFFERENCE: {diff:.2e}")
    print("-" * 30)
    
    if diff < 1e-5:
        print("✅ TEST PASSED: Models are mathematically equivalent!")
        print("✅ TEST PASSED: Hilbert integration is successful!")
    else:
        print("❌ TEST FAILED: Differences are too high.")
    print("="*50 + "\n")



if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Run test
    model, output = test_s4_convolutional()
    
    # Verify equivalence
    print("\n")
    verify_equivalence()

    #test hilberts
    run_all_milestone_tests()