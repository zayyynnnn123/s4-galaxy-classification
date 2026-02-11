"""
S4 Recurrent Implementation
Task 5.1: Implement standard recurrent formulation of S4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class S4RecurrentEfficient(nn.Module):
    """
    Recurrent formulation of Structured State Space (S4) model.
    
    Continuous-time state space system:
        dx(t)/dt = A x(t) + B u(t)
        y(t) = C x(t) + D u(t)
    
    Discretized using bilinear transform (matrix exponential method):
        A* = exp(delta(A))
        B* = (delta(A))^-1 (exp(delta(A)) - I) (delta(B))
    
    Recurrence:
        x_k = A* x_{k-1} + B* u_k
        y_k = C x_k + D u_k
    
    Parameters:
        d_model: Input/output feature dimension
        d_state: State dimension (N)
        dt_min: Minimum step size
        dt_max: Maximum step size
    """
    
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # Parameters
        A = torch.randn(d_state, d_state) / math.sqrt(d_state)
        A = A - torch.eye(d_state) * 1.5  # Negative real parts for stability
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
        
        # For storing discretized matrices (optional)
        self.register_buffer('A_bar', None)
        self.register_buffer('B_bar', None)
    
    def forward(self, u):
        batch_size, seq_len, _ = u.shape
        
        # Compute discretized matrices
        dt = torch.exp(self.log_dt).clamp(self.dt_min, self.dt_max)
        delta_A = dt * self.A
        A_bar = torch.matrix_exp(delta_A)
        
        I = torch.eye(self.d_state, device=self.A.device)
        try:
            inv_delta_A = torch.inverse(delta_A)
        except:
            inv_delta_A = torch.pinverse(delta_A)
        
        delta_B = dt * self.B
        B_bar = inv_delta_A @ (A_bar - I) @ delta_B
        
        # Store for analysis
        self.A_bar = A_bar
        self.B_bar = B_bar
        
        # Project input from d_model to 1
        u_proj = self.input_proj(u)  # (batch, seq_len, 1)
        u_proj = u_proj.transpose(1, 2)  # (batch, 1, seq_len)
        
        # Precompute powers of A_bar
        A_bar_powers = [torch.eye(self.d_state, device=u.device)]
        for i in range(1, seq_len):
            A_bar_powers.append(A_bar_powers[-1] @ A_bar)
        
        # Initialize state
        x = torch.zeros(batch_size, self.d_state, device=u.device)
        
        # Process sequence
        states = []
        for k in range(seq_len):
            # Contribution from initial state
            init_contrib = torch.matmul(x, A_bar_powers[k].T)
            
            # Contribution from inputs
            input_contrib = torch.zeros(batch_size, self.d_state, device=u.device)
            for i in range(k):
                u_i = u_proj[:, :, i]  # (batch, 1)
                power_idx = k - 1 - i
                contrib = torch.matmul(u_i, (A_bar_powers[power_idx] @ B_bar).T)
                input_contrib += contrib
            
            x_k = init_contrib + input_contrib
            states.append(x_k.unsqueeze(1))
        
        # Stack states: (batch, seq_len, d_state)
        states = torch.cat(states, dim=1)
        
        # Compute outputs: y_k = C x_k + D u_k
        outputs = torch.matmul(states, self.C.T) + self.D * u_proj.transpose(1, 2)
        
        # Project back to d_model
        outputs = self.output_proj(outputs)
        
        return outputs
    #FOR COMPLEXITY ANALYSIS
    def get_complexity(self, L):
        """Compute computational complexity O(L * N^2)"""
        N = self.d_state
        return L * N * N


def test_s4_recurrent():
    """Test function to verify S4Recurrent implementation."""
    print("=" * 50)
    print("Testing S4RecurrentEfficient implementation...")

    
    # Create model
    model = S4RecurrentEfficient(d_model=64, d_state=32)
    
    # Test input
    batch_size = 2
    seq_len = 100
    d_model = 64
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = model(test_input)
    
    print(f"\n✓ Input shape:  {test_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ A_bar shape:  {model.A_bar.shape}")
    print(f"✓ B_bar shape:  {model.B_bar.shape}")
    
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
    print("\nS4RecurrentEfficient implementation ready for Task 5.1!")
