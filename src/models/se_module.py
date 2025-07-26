import torch
import torch.nn as nn

class SEModule(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C) or (B, C, N) or (B, N, C)
        - Nếu (B, N, C): tự động chuyển về (B, C, N)
        - Nếu (B, C): coi như (B, C, 1)
        """
        if x.dim() == 3:
            # Nếu input là (B, N, C) thì chuyển về (B, C, N)
            if x.shape[1] != self.fc[0].in_features:
                x = x.transpose(1, 2)
            # (B, C, N)
            y = self.avg_pool(x).squeeze(-1)  # (B, C)
            y = self.fc(y).unsqueeze(-1)      # (B, C, 1)
            return x * y
        elif x.dim() == 2:
            # (B, C)
            y = self.fc(x)
            return x * y
        else:
            raise ValueError(f"SEModule only supports 2D/3D input, got shape {x.shape}")
