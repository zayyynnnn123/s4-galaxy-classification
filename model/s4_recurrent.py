"""
S4 Recurrent Implementation
Task 5.1: Implement standard recurrent formulation of S4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class S4Recurrent(nn.Module):
    """
    Recurrent formulation of Structured State Space (S4) model.
    """
    
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # Parameters
        A = torch.randn(d_state, d_state) / math.sqrt(d_state)
        A = A - torch.eye(d_state) * 1.5
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(torch.randn(d_state, 1) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(1, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.randn(1))
        
        # Step size in log space
        log_dt_init = (math.log(dt_min) + math.log(dt_max)) / 2.0
        self.log_dt = nn.Parameter(torch.tensor([log_dt_init]))
        
        # Input/output projections
        self.input_proj = nn.Linear(d_model, 1, bias=False)
        self.output_proj = nn.Linear(1, d_model, bias=False)
        
        # For storing discretized matrices
        self.register_buffer('A_bar', None)
        self.register_buffer('B_bar', None)
        
    def _discretize(self):
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
    
    def forward(self, u):

        batch_size, seq_len, _ = u.shape
    
        # --- Use the SAME discretize method as convolutional ---
        A_bar, B_bar = self._discretize()
    
        # Store for analysis
        self.A_bar = A_bar
        self.B_bar = B_bar
    
        # --- Rest of forward pass ---
        u_proj = self.input_proj(u)
        u_proj = u_proj.transpose(1, 2)
    
        x = torch.zeros(batch_size, self.d_state, device=u.device)
        outputs = []
    
        for k in range(seq_len):
            u_k = u_proj[:, :, k]
            x = torch.matmul(x, A_bar.T) + torch.matmul(u_k, B_bar.T)
            y_k = torch.matmul(x, self.C.T) + self.D * u_k
            outputs.append(y_k.unsqueeze(1))
    
        y = torch.cat(outputs, dim=1)
        y = self.output_proj(y)
    
        return y
    
    def get_complexity(self, L):
        """Compute computational complexity O(L * N^2)"""
        N = self.d_state
        return L * N * N


def test_s4_recurrent():
    """Test function to verify S4Recurrent implementation."""
    print("=" * 50)
    print("Testing S4Recurrent Implementation...")
    print("=" * 50)
    
    # Create model
    model = S4Recurrent(d_model=64, d_state=32)
    
    # Test input
    batch_size = 2
    seq_len = 100
    d_model = 64
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = model(test_input)
    
    print(f"\n✓ Input shape:  {test_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ A_bar shape:  {model.A_bar.shape}")  # Now works!
    print(f"✓ B_bar shape:  {model.B_bar.shape}")  # Now works!
    
    # Test complexity
    complexity = model.get_complexity(seq_len)
    print(f"\n✓ Complexity: O({complexity}) ≈ O({seq_len} * {model.d_state}²)")
    
    # Verify shapes match
    assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} != {test_input.shape}"
    print("\n✓ Test passed: Input and output shapes match!")
    print("=" * 50)
    
    return model, output


if __name__ == "__main__":
    model, output = test_s4_recurrent()
    print("\n✅ S4Recurrent implementation ready for Task 5.1!")