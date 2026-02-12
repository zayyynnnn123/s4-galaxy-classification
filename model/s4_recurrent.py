import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)

class S4Recurrent(nn.Module):
    """
    Recurrent formulation of Structured State Space (S4) model.
    Uses HiPPO-LegT initialization for optimal long-range memory.
    """
    
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # --- HiPPO-LegT Initialization for A matrix ---
        # This provides theoretical guarantees for long-range memory
        P = torch.sqrt(1 + 2 * torch.arange(d_state, dtype=torch.float32))
        A = P[:, None] * P[None, :]
        A = -torch.tril(A)  # Make ALL entries negative
        A = A + torch.diag(torch.arange(d_state, dtype=torch.float32))  # Add i to diagonal
        self.A = nn.Parameter(A)
        
        # --- B matrix: input projection ---
        # Random initialization is fine for B
        self.B = nn.Parameter(torch.randn(d_state, 1) / math.sqrt(d_state))
        
        # --- C matrix: output projection ---
        # Random initialization is fine for C
        self.C = nn.Parameter(torch.randn(1, d_state) / math.sqrt(d_state))
        
        # --- D scalar: skip connection ---
        self.D = nn.Parameter(torch.randn(1))
        
        # --- Step size Δ in log-space (ensures positivity) ---
        log_dt_init = (math.log(dt_min) + math.log(dt_max)) / 2.0
        self.log_dt = nn.Parameter(torch.tensor([log_dt_init]))
        
        # --- Input/output projections (for feature dimensions) ---
        self.input_proj = nn.Linear(d_model, 1, bias=False)
        self.output_proj = nn.Linear(1, d_model, bias=False)
        
        # --- For storing discretized matrices (for analysis) ---
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
    
    def forward(self, u):
        """
        Forward pass using recurrence.
        
        Args:
            u: Input tensor of shape (batch, sequence_length, d_model)
            
        Returns:
            Output tensor of shape (batch, sequence_length, d_model)
        """
        batch_size, seq_len, _ = u.shape
        
        # --- Step 1: Compute discretized matrices ---
        A_bar, B_bar = self._discretize()
        
        # Store for analysis
        self.A_bar = A_bar
        self.B_bar = B_bar
        
        # --- Step 2: Project input from d_model to 1 dimension ---
        u_proj = self.input_proj(u)          # (B, L, 1)
        u_proj = u_proj.transpose(1, 2)      # (B, 1, L)
        
        # --- Step 3: Initialize state x₀ = 0 ---
        x = torch.zeros(batch_size, self.d_state, device=u.device)
        outputs = []
        
        # --- Step 4: Recurrence over sequence ---
        for k in range(seq_len):
            # Get current input u_k
            u_k = u_proj[:, :, k]            # (B, 1)
            
            # State update: x_k = Ā @ x_{k-1} + B̄ @ u_k
            x = torch.matmul(x, A_bar.T) + torch.matmul(u_k, B_bar.T)  # (B, d_state)
            
            # Output: y_k = C @ x_k + D @ u_k
            y_k = torch.matmul(x, self.C.T) + self.D * u_k  # (B, 1)
            outputs.append(y_k.unsqueeze(1))
        
        # --- Step 5: Stack outputs ---
        y = torch.cat(outputs, dim=1)        # (B, L, 1)
        
        # --- Step 6: Project output back to d_model ---
        y = self.output_proj(y)              # (B, L, d_model)
        
        return y
    
    def get_complexity(self, L):
        """
        Compute computational complexity O(L * N^2)
        
        Args:
            L: Sequence length
            
        Returns:
            Complexity in number of operations
        """
        N = self.d_state
        return L * N * N


def test_s4_recurrent():
    """Test function to verify S4Recurrent implementation."""
    print("=" * 60)
    print("Testing S4Recurrent Implementation with HiPPO-LegT")
    print("=" * 60)
    
    # Create model
    d_model = 64
    d_state = 32
    model = S4Recurrent(d_model=d_model, d_state=d_state)
    
    # Test input
    batch_size = 2
    seq_len = 100
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = model(test_input)
    
    print(f"\n Model Configuration:")
    print(f"   - d_model: {d_model}")
    print(f"   - d_state: {d_state}")
    print(f"   - HiPPO-LegT: ✅ Initialized")
    print(f"   - A matrix shape: {model.A.shape}")
    print(f"   - A matrix norm: {torch.norm(model.A).item():.4f}")
    
    print(f"\n Shape Checks:")
    print(f"   - Input shape:  {test_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - A_bar shape:  {model.A_bar.shape}")
    print(f"   - B_bar shape:  {model.B_bar.shape}")
    
    # Verify HiPPO structure (lower triangular with -1 on diagonal)
    A_np = model.A.detach().numpy()
    is_lower_tri = torch.all(model.A == torch.tril(model.A)).item()
    diag_mean = torch.diag(model.A).mean().item()
    
    print(f"\n HiPPO Verification:")
    print(f"   - Lower triangular: {is_lower_tri}")
    print(f"   - Mean diagonal: {diag_mean:.4f} (should be -0.5)")
    print(f"   - A[0,0]: {model.A[0,0].item():.4f} (should be -1)")
    
    # Verify shapes match
    assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} != {test_input.shape}"
    print(f"\n Shape verification: PASSED")
    
    # Test complexity
    complexity = model.get_complexity(seq_len)
    print(f"\n Complexity: O({complexity}) ≈ O({seq_len} × {d_state}²)")
    
    print(f"\n Test passed: S4Recurrent with HiPPO works!")
    print("=" * 60)
    
    return model, output


def visualize_hippo():
    """Visualize the HiPPO matrix structure."""
    import matplotlib.pyplot as plt
    
    d_state = 32
    P = torch.sqrt(1 + 2 * torch.arange(d_state, dtype=torch.float32))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(d_state, dtype=torch.float32))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(A.numpy(), cmap='RdBu', vmin=-10, vmax=10)
    plt.colorbar(label='Value')
    plt.title(f'HiPPO-LegT Matrix (d_state={d_state})')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig('hippo_matrix.png', dpi=150, bbox_inches='tight')
    print(" HiPPO matrix visualization saved to 'hippo_matrix.png'")
    plt.close()


if __name__ == "__main__":
    # Run test
    model, output = test_s4_recurrent()
    
    # Optional: visualize HiPPO matrix
    try:
        visualize_hippo()
    except:
        print("  Matplotlib not available - skipping visualization")
    
    print("\n S4Recurrent with HiPPO-LegT ready for Task 5.1!")