import torch   
import torch.nn as nn

class HilbertScan(nn.Module):
    """
    Improved Hilbert Curve reordering.
    Preserves spatial locality for GalaxyMNIST images.
    """
    def __init__(self):
        super().__init__()
        # Precompute for 64x64 grid (Section 6.1 requirement)
        indices = self.get_hilbert_indices(64)
        self.register_buffer('indices', indices)

    def _rot(self, s, x, y, rx, ry):
        """Standard Hilbert rotation logic for quadrant transitions."""
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            return y, x
        return x, y

    def _d2xy(self, n, d):
        """Converts 1D distance 'd' to 2D coordinates (x, y)."""
        x = y = 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = self._rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    def get_hilbert_indices(self, n):
        """Generates the sequence lookup table."""
        indices = []
        for d in range(n * n):
            x, y = self._d2xy(n, d)
            # Map 2D coordinates to flattened 1D index
            indices.append(y * n + x)
        return torch.LongTensor(indices)

    def forward(self, x):
        """
        Transforms (Batch, Channels, Height, Width) 
        to (Batch, Sequence_Length, Channels).
        """
        B, C, H, W = x.shape
        # 1. Flatten spatial dimensions: (B, C, 4096)
        x = x.view(B, C, -1)           
        # 2. Reorder pixels based on Hilbert path
        x = x[:, :, self.indices]      
        # 3. Swap dimensions for S4: (B, 4096, C)
        return x.permute(0, 2, 1)