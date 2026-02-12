"""
S4 Convolutional Implementation
Task 5.2: Implement convolutional formulation of S4

Mathematical Derivation:
------------------------
From recurrence: y_k = C·Ā^k·B̄·u_0 + C·Ā^{k-1}·B̄·u_1 + ... + C·B̄·u_k + D·u_k

This is a convolution: y = K * u + D·u

Where kernel K = [C·B̄, C·Ā·B̄, C·Ā²·B̄, ..., C·Ā^{L-1}·B̄]

IMPORTANT: PyTorch's conv1d performs cross-correlation, not convolution.
           We must flip the kernel to achieve true convolution.

Computational complexity:
- Recurrent: O(L·N²)
- Convolution (naive): O(L²·N)
- Convolution (FFT): O(L log L·N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class S4Convolutional(nn.Module):
    """
    Convolutional formulation of Structured State Space (S4) model.
    
    Instead of recurrence, precompute convolution kernel and use F.conv1d:
        y = K * u + D·u
    
    where:
        K = [C·B̄, C·Ā·B̄, C·Ā²·B̄, ..., C·Ā^{L-1}·B̄]
        * denotes convolution
    
    This enables parallel training across sequence length.
    """
    
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # --- Initialize continuous parameters ---
        # A matrix: (d_state, d_state) - state transition
        # Initialize with negative real parts for stability
        A = torch.randn(d_state, d_state) / math.sqrt(d_state)
        A = A - torch.eye(d_state) * 1.5
        self.A = nn.Parameter(A)
        
        # B matrix: (d_state, 1) - input projection
        self.B = nn.Parameter(torch.randn(d_state, 1) / math.sqrt(d_state))
        
        # C matrix: (1, d_state) - output projection
        self.C = nn.Parameter(torch.randn(1, d_state) / math.sqrt(d_state))
        
        # D scalar: skip connection
        self.D = nn.Parameter(torch.randn(1))
        
        # Step size Δ in log-space (ensures positivity)
        log_dt_init = (math.log(dt_min) + math.log(dt_max)) / 2.0
        self.log_dt = nn.Parameter(torch.tensor([log_dt_init]))
        
        # Input/output projections (for feature dimensions)
        self.input_proj = nn.Linear(d_model, 1, bias=False)
        self.output_proj = nn.Linear(1, d_model, bias=False)
        
        # For storing discretized matrices (for analysis)
        self.register_buffer('A_bar', None)
        self.register_buffer('B_bar', None)
    
    def _discretize(self):
        """
        Compute discretized matrices Ā and B̄ using matrix exponential.
        
        Formulas:
            Ā = exp(Δ A)
            B̄ = (Δ A)⁻¹ (exp(Δ A) - I) (Δ B)
        
        Returns:
            A_bar: Discretized state matrix (d_state, d_state)
            B_bar: Discretized input matrix (d_state, 1)
        """
        dt = torch.exp(self.log_dt).clamp(self.dt_min, self.dt_max)
        
        # Ā = exp(Δ A)
        delta_A = dt * self.A
        A_bar = torch.matrix_exp(delta_A)
        
        # B̄ = (Δ A)⁻¹ (exp(Δ A) - I) (Δ B)
        I = torch.eye(self.d_state, device=self.A.device)
        try:
            inv_delta_A = torch.inverse(delta_A)
        except:
            # Use pseudo-inverse if matrix is singular
            inv_delta_A = torch.pinverse(delta_A)
        
        delta_B = dt * self.B
        B_bar = inv_delta_A @ (A_bar - I) @ delta_B
        
        return A_bar, B_bar
    
    def compute_kernel(self, L):
        """
        Compute SSM convolution kernel of length L.
        
        Mathematical definition:
            K[i] = C @ Ā^i @ B̄ for i = 0, 1, ..., L-1
        
        However, PyTorch's conv1d performs cross-correlation, not convolution.
        Therefore we need to FLIP the kernel to achieve true convolution.
        
        Args:
            L: Sequence length
            
        Returns:
            kernel: (1, 1, L) - convolution kernel for conv1d (ALREADY FLIPPED)
        """
        # Get discretized matrices
        A_bar, B_bar = self._discretize()
        
        # Store for analysis
        self.A_bar = A_bar
        self.B_bar = B_bar
        
        # Compute kernel elements: K[i] = C @ Ā^i @ B̄
        kernel = []
        
        # K[0] = C @ B̄
        kernel.append(self.C @ B_bar)  # (1, 1)
        
        # Compute powers of A_bar iteratively (O(L·N²))
        A_power = A_bar.clone()
        for i in range(1, L):
            # K[i] = C @ Ā^i @ B̄
            k_i = self.C @ A_power @ B_bar
            kernel.append(k_i)
            
            # Update A_power for next iteration: Ā^{i+1} = Ā^i @ Ā
            A_power = A_power @ A_bar
        
        # Stack kernel elements: (L, 1, 1) -> (L,)
        kernel = torch.stack(kernel).squeeze()
        
        # --- CRITICAL: Flip kernel for proper convolution ---
        # PyTorch conv1d performs cross-correlation: output[t] = sum_i kernel[i] * input[t+i]
        # True convolution requires: output[t] = sum_i kernel[i] * input[t-i]
        # Flipping the kernel converts cross-correlation to convolution
        kernel = torch.flip(kernel, dims=[0])
        
        # Reshape for conv1d: (out_channels, in_channels, kernel_size)
        kernel = kernel.view(1, 1, L)
        
        return kernel
    
    def forward(self, u):
        """
        Forward pass using convolution.
        
        Args:
            u: Input tensor of shape (batch, sequence_length, d_model)
            
        Returns:
            Output tensor of shape (batch, sequence_length, d_model)
        
        Computational steps:
            1. Project input from d_model → 1 dimension
            2. Compute convolution kernel of length L
            3. Apply conv1d: y = K * u + D·u
            4. Project output from 1 → d_model dimension
        """
        batch_size, seq_len, _ = u.shape
        
        # --- Step 1: Project input from d_model to 1 dimension ---
        # u_proj: (batch, seq_len, 1)
        u_proj = self.input_proj(u)
        
        # Reshape for conv1d: (batch, channels=1, sequence_length)
        u_proj = u_proj.transpose(1, 2)  # (batch, 1, seq_len)
        
        # --- Step 2: Compute convolution kernel of length seq_len ---
        # Kernel is already flipped for proper convolution
        K = self.compute_kernel(seq_len)  # (1, 1, seq_len)
        
        # --- Step 3: Apply causal convolution ---
        # For causal convolution, output at time t depends only on inputs at times ≤ t
        # Use padding = kernel_size - 1, then take first L elements
        y = F.conv1d(
            u_proj,              # (batch, 1, L)
            K,                   # (1, 1, L) - already flipped
            padding=seq_len - 1  # Full padding for causal
        )  # (batch, 1, 2L-1)
        
        # Take first L elements (causal)
        y = y[..., :seq_len]  # (batch, 1, L)
        
        # --- Step 4: Add skip connection D·u ---
        # D is scalar applied per timestep
        y = y + self.D * u_proj  # (batch, 1, L)
        
        # --- Step 5: Project output back to d_model dimensions ---
        # Reshape: (batch, 1, L) -> (batch, L, 1)
        y = y.transpose(1, 2)  # (batch, L, 1)
        
        # Project: (batch, L, 1) -> (batch, L, d_model)
        y = self.output_proj(y)
        
        return y
    
    def get_complexity(self, L):
        """
        Compute computational complexity comparison.
        
        Returns dictionary with operation counts for different implementations.
        """
        N = self.d_state
        
        # Recurrent: O(L·N²)
        recurrent = L * N * N
        
        # Kernel computation: O(L·N²)
        kernel_comp = L * N * N
        
        # Naive convolution: O(L²·N)
        naive_conv = L * L * N
        
        # FFT convolution: O(L log L·N)
        fft_conv = int(L * math.log2(L) * N)
        
        return {
            'recurrent_O(L·N²)': recurrent,
            'kernel_computation_O(L·N²)': kernel_comp,
            'naive_convolution_O(L²·N)': naive_conv,
            'fft_convolution_O(L_logL·N)': fft_conv,
            'total_naive': kernel_comp + naive_conv,
            'total_fft': kernel_comp + fft_conv
        }


def test_s4_convolutional():
    """Test function to verify S4Convolutional implementation."""
    print("=" * 60)
    print("Testing S4Convolutional Implementation (Task 5.2)")
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
    
    # Check kernel shape
    kernel = model.compute_kernel(seq_len)
    
    print(f"\n Model Configuration:")
    print(f"   - d_model: {d_model}")
    print(f"   - d_state: {d_state}")
    print(f"   - sequence length: {seq_len}")
    print(f"   - batch size: {batch_size}")
    
    print(f"\n Shape Checks:")
    print(f"   - Input shape:  {test_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Kernel shape: {kernel.shape}")
    print(f"   - A_bar shape:  {model.A_bar.shape}")
    print(f"   - B_bar shape:  {model.B_bar.shape}")
    
    # Verify shapes match
    assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} != {test_input.shape}"
    assert kernel.shape == (1, 1, seq_len), f"Kernel shape mismatch: {kernel.shape}"
    print(f"\n Shape verification: PASSED")
    
    # Test complexity analysis
    complexity = model.get_complexity(seq_len)
    print(f"\n Complexity Analysis (L={seq_len}, N={d_state}):")
    print(f"   - Recurrent:        O({complexity['recurrent_O(L·N²)']:>12,}) ≈ O(L·N²)")
    print(f"   - Kernel compute:   O({complexity['kernel_computation_O(L·N²)']:>12,}) ≈ O(L·N²)")
    print(f"   - Naive conv:       O({complexity['naive_convolution_O(L²·N)']:>12,}) ≈ O(L²·N)")
    print(f"   - FFT conv:         O({complexity['fft_convolution_O(L_logL·N)']:>12,}) ≈ O(L log L·N)")
    print(f"   - Total (naive):    O({complexity['total_naive']:>12,})")
    print(f"   - Total (FFT):      O({complexity['total_fft']:>12,})")
    
    print(f"\n Test passed: S4Convolutional implementation works!")
    print("=" * 60)
    
    return model, output


def verify_recurrent_conv_equivalence():
    """
    Verify that recurrent and convolutional implementations
    produce numerically identical outputs (for Task 5.3).
    """
    print("=" * 60)
    print("Verifying Recurrent ↔ Convolutional Equivalence (Task 5.3)")
    print("=" * 60)
    
    # Import recurrent implementation
    try:
        from model.s4_recurrent import S4Recurrent
    except ImportError:
        try:
            from s4_recurrent import S4Recurrent
        except ImportError:
            print(" S4Recurrent not found - skipping test")
            return None, None
    
    # Create models with SAME initialization
    d_model = 64
    d_state = 32
    
    conv_model = S4Convolutional(d_model=d_model, d_state=d_state)
    rec_model = S4Recurrent(d_model=d_model, d_state=d_state)
    
    # Copy ALL parameters to ensure identical initialization
    with torch.no_grad():
        rec_model.A.data = conv_model.A.data.clone()
        rec_model.B.data = conv_model.B.data.clone()
        rec_model.C.data = conv_model.C.data.clone()
        rec_model.D.data = conv_model.D.data.clone()
        rec_model.log_dt.data = conv_model.log_dt.data.clone()
        rec_model.input_proj.weight.data = conv_model.input_proj.weight.data.clone()
        rec_model.output_proj.weight.data = conv_model.output_proj.weight.data.clone()
    
    # Test input
    batch_size = 2
    seq_len = 50  # Shorter for faster testing
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Forward passes
    with torch.no_grad():
        conv_output = conv_model(test_input)
        rec_output = rec_model(test_input)
    
    # Compute difference
    diff = torch.abs(conv_output - rec_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n Equivalence Test Results:")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Max absolute difference: {max_diff:.2e}")
    print(f"   - Mean absolute difference: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print(f"\n EQUIVALENCE PASSED! (diff < 1e-5)")
        print(f"   Recurrent and Convolutional implementations match!")
    else:
        print(f"\n EQUIVALENCE FAILED! (diff = {max_diff:.2e})")
        print(f"\n   Debugging steps:")
        print(f"   1. Check if kernel is flipped correctly")
        print(f"   2. Verify causal convolution padding")
        print(f"   3. Ensure D·u skip connection is identical")
        print(f"   4. Run the diagnostic script for more details")
    
    print("=" * 60)
    return conv_output, rec_output


if __name__ == "__main__":
    # Run main test
    model, output = test_s4_convolutional()
    
    # Test equivalence with recurrent
    print("\n")
    verify_recurrent_conv_equivalence()
    
    print("\n S4Convolutional implementation ready for Task 5.2!")
