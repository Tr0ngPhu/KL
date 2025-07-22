import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticDepth(nn.Module):
    """
    Stochastic Depth module.
    This layer randomly drops the main branch of a residual block during training.
    """
    def __init__(self, p: float = 0.1, mode: str = 'row'):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_().add_(keep_prob)
        return x.div(keep_prob) * random_tensor

class VisionTransformer(nn.Module):
    """
    A Vision Transformer model with Stochastic Depth for regularization.
    """
    def __init__(self, image_size=224, patch_size=16, num_classes=2, dim=768, 
                 depth=12, heads=12, mlp_dim=3072, dropout=0.1, 
                 emb_dropout=0.1, stochastic_depth_prob=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer Encoder with Stochastic Depth
        transformer_layers = []
        for i in range(depth):
            # Apply stochastic depth linearly increasing through layers
            sd_prob = stochastic_depth_prob * (i / (depth - 1))
            
            transformer_layers.append(nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ))
            transformer_layers.append(StochasticDepth(p=sd_prob))

        self.transformer = nn.Sequential(*transformer_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Patch embedding
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(x.size(0), -1, 3, p, p)
        x = x.view(x.size(0), -1, 3 * p * p)
        x = self.patch_to_embedding(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x += self.pos_embed
        x = self.dropout(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Get CLS token representation
        cls_token_final = x[:, 0]
        
        # MLP head
        return self.mlp_head(cls_token_final) 