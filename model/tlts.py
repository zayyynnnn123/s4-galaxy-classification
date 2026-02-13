import torch
import torch.nn as nn

class TakeLastTimestep(nn.Module):
    """
    Module that extracts the last timestep from a sequence.

    This layer is used to summarize sequence outputs from recurrent 
    or sequence models by taking only the final timestep as a feature vector.

    Parameters
    ----------
    None

    Input
    -----
    x : torch.Tensor
        Input tensor of shape (B, L, D), where
        B : batch size,
        L : sequence length,
        D : feature dimension.

    Returns
    -------
    out : torch.Tensor
        Output tensor of shape (B, D), corresponding to the last timestep
        of each sequence in the batch.
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        if self.dim == 1:
            return x[:, -1, :]  # (B, L, D) -> (B, D)
        elif self.dim == 2:
            return x[:, :, -1]  # (B, D, L) -> (B, D)
        else:
            raise ValueError(f"dim must be 1 or 2, got {self.dim}")
    
def test_take_last_timestep():
    """Test TakeLastTimestep implementation."""
    print("=" * 60)
    print("Testing TakeLastTimestep Layer (Task 7)")
    print("=" * 60)
    
    # Test 1: (B, L, D) format
    print("\n Test 1: (B, L, D) format")
    tl = TakeLastTimestep(dim=1)
    x1 = torch.randn(2, 100, 64)
    y1 = tl(x1)
    print(f"   Input shape:  {x1.shape}")
    print(f"   Output shape: {y1.shape}")
    print(f"   Expected:     (2, 64)")
    assert y1.shape == (2, 64), f"Shape mismatch: {y1.shape}"
    print(f"    Passed!")
    
    # Test 2: (B, D, L) format
    print("\n Test 2: (B, D, L) format")
    tl = TakeLastTimestep(dim=2)
    x2 = torch.randn(2, 64, 100)
    y2 = tl(x2)
    print(f"   Input shape:  {x2.shape}")
    print(f"   Output shape: {y2.shape}")
    print(f"   Expected:     (2, 64)")
    assert y2.shape == (2, 64), f"Shape mismatch: {y2.shape}"
    print(f"    Passed!")
    
    # Test 3: Verify values (last timestep extraction)
    print("\n Test 3: Value verification")
    x3 = torch.arange(10).reshape(1, 5, 2).float()  # (1, 5, 2)
    print(f"   Input tensor:\n{x3[0]}")
    tl = TakeLastTimestep(dim=1)
    y3 = tl(x3)
    print(f"   Last timestep: {y3[0]}")
    assert torch.all(y3[0] == x3[0, -1]), "Wrong values extracted!"
    print(f"    Passed!")
    
    # Test 4: Gradient flow
    print("\n Test 4: Gradient flow")
    x4 = torch.randn(2, 100, 64, requires_grad=True)
    y4 = tl(x4)
    loss = y4.sum()
    loss.backward()
    print(f"   Gradient norm: {x4.grad.norm():.2e}")
    assert x4.grad is not None, "No gradients!"
    print(f"   Passed!")
    
    print("\n" + "=" * 60)
    print(" All tests passed! TakeLastTimestep ready for Task 7!")
    print("=" * 60)
    
    return tl


def demonstrate_usage():
    """Show how TakeLastTimestep fits into the full model."""
    print("\n" + "=" * 60)
    print("TakeLastTimestep Usage Example")
    print("=" * 60)
    
    # Simulate S4 output
    batch_size = 2
    seq_len = 4096
    d_model = 64
    
    # S4 processes sequence and produces output at every timestep
    s4_output = torch.randn(batch_size, seq_len, d_model)
    print(f"\n S4 output shape: {s4_output.shape}")
    
    # Extract last timestep for classification
    tl = TakeLastTimestep(dim=1)
    sequence_summary = tl(s4_output)
    print(f" After TakeLastTimestep: {sequence_summary.shape}")
    
    # Classification head
    classifier = nn.Linear(d_model, 4)
    logits = classifier(sequence_summary)
    print(f" Classification logits: {logits.shape}")
    
    print(f"\n Complete pipeline ready for Task 8!")
    print("   (B, L, D) → (B, D) → (B, 4)")
    
    return sequence_summary, logits


if __name__ == "__main__":
    # Run tests
    tl = test_take_last_timestep()
    
    # Show usage example
    summary, logits = demonstrate_usage()
    
    print("\n" + "=" * 60)
    print(" TakeLastTimestep implementation ready for Task 7!")
    print("=" * 60)