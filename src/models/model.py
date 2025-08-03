# Kiến trúc ViT
import torch
import torch.nn as nn

import torch.nn.functional as F
from .se_module import SEModule
import timm

# PatchEmbedding: Chia ảnh thành các patch và embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x) # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        x = self.norm(x)
        return x

# MultiHeadSelfAttention: Cơ chế attention nhiều đầu
class EnhancedMultiHeadSelfAttention(nn.Module):
    """Attention nâng cao, trả về attention map khi cần"""
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_maps = None
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        self.attention_maps = attn.detach()
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if return_attention:
            return x, attn
        return x

# MLP: Mạng fully connected với GELU
class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# DropPath: Stochastic Depth
class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# TransformerBlock: Block transformer với residual
class EnhancedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EnhancedMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x



# VisionTransformer: ViT chuyên biệt cho fake/real

# Wrapper cho ViT pretrained từ timm
class PretrainedVisionTransformer(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class VisionTransformer(nn.Module):
    def get_attention_maps(self, x):
        """
        Returns a list of attention maps for each layer, compatible with API usage.
        The last map can be used for visualization as in the API.
        """
        with torch.no_grad():
            attention_maps, _ = self.get_advanced_attention_maps(x)
            return attention_maps
    """ViT với head đơn giản, hỗ trợ multi-scale fusion (tùy chọn)"""
    def __init__(self, img_size, patch_size, in_channels, num_classes,
                 embed_dim, depth, num_heads, mlp_ratio,
                 dropout, drop_path_rate, use_cls_token=True, with_multiscale=True, use_se=False):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.use_cls_token = use_cls_token
        self.with_multiscale = with_multiscale
        self.use_se = use_se
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = num_patches + 1
        else:
            self.num_tokens = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            EnhancedTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Head: LayerNorm + (SE block nếu có) + Linear
        head_in_dim = embed_dim + (3 * (embed_dim // 4) if with_multiscale else 0)
        head_layers = [nn.LayerNorm(head_in_dim)]
        if self.use_se:
            head_layers.append(SEModule(head_in_dim, reduction=8))
        head_layers.append(nn.Linear(head_in_dim, num_classes))
        self.head = nn.Sequential(*head_layers)
        # Multi-scale projection layers
        if with_multiscale:
            self.multiscale_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d(1)
                ) for _ in range(3)
            ])
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.pos_embed)
        if self.use_cls_token:
            nn.init.xavier_uniform_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _check_input(self, x):
        if x.dim() != 4:
            raise ValueError(f"Input phải có shape (B, C, H, W), nhận được {x.shape}")
        if x.shape[1] != 3:
            raise ValueError(f"Input phải có 3 channels RGB, nhận được {x.shape[1]}")
        if x.shape[2] != self.patch_embed.img_size or x.shape[3] != self.patch_embed.img_size:
            raise ValueError(f"Kích thước ảnh phải là {self.patch_embed.img_size}x{self.patch_embed.img_size}, nhận được {x.shape[2]}x{x.shape[3]}")
        if not x.dtype.is_floating_point:
            raise TypeError(f"Input phải là kiểu float (float32), nhận được {x.dtype}")
        if self.pos_embed.device != x.device:
            raise RuntimeError(f"Model và input phải cùng device. Model: {self.pos_embed.device}, Input: {x.device}")

    def forward_features(self, x):
        self._check_input(x)
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Multi-scale fusion
        if self.with_multiscale:
            # Luôn loại bỏ CLS token khi reshape spatial features
            if self.use_cls_token:
                spatial_features = x[:, 1:]
            else:
                spatial_features = x
            B, N, C = spatial_features.shape
            H = W = int(N ** 0.5)
            if H * W != N:
                raise RuntimeError(f"Số patch không phải hình vuông: N={N}, H={H}, W={W}")
            spatial = spatial_features.permute(0, 2, 1).contiguous().view(B, C, H, W)
            multiscale_feats = []
            for proj in self.multiscale_proj:
                scaled = proj(spatial).squeeze(-1).squeeze(-1)
                multiscale_feats.append(scaled)
            return torch.cat([x[:, 0]] + multiscale_feats, dim=-1)
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_advanced_attention_maps(self, x):
        """
        Trả về attention maps chi tiết và aggregated heatmap
        Returns:
            - attention_maps: List attention maps từ mỗi layer
            - aggregated_heatmap: Heatmap tổng hợp từ nhiều layer
        """
        self._check_input(x)
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        attention_maps = []
        layer_weights = []
        for i, block in enumerate(self.blocks):
            x, attn = block(x, return_attention=True)
            attention_maps.append(attn.detach().cpu())
            weight = (i + 1) / len(self.blocks)
            layer_weights.append(weight)
        aggregated_heatmap = self._create_aggregated_heatmap(attention_maps, layer_weights, x.shape[1])
        return attention_maps, aggregated_heatmap

    def _create_aggregated_heatmap(self, attention_maps, layer_weights, num_tokens):
        if not attention_maps:
            return None
        B = attention_maps[0].shape[0]
        patch_size = int((num_tokens - 1) ** 0.5) if self.use_cls_token else int(num_tokens ** 0.5)
        aggregated = torch.zeros(B, patch_size, patch_size)
        for i, (attn_map, weight) in enumerate(zip(attention_maps, layer_weights)):
            attn_avg = attn_map.mean(dim=1)
            if self.use_cls_token:
                cls_attn = attn_avg[:, 0, 1:]
            else:
                cls_attn = attn_avg.mean(dim=1)
            cls_attn = cls_attn.reshape(B, patch_size, patch_size)
            aggregated += cls_attn * weight
        return aggregated

    def get_interpretable_features(self, x):
        """
        Trả về attention maps và aggregated heatmap cho interpretability đơn giản.
        """
        with torch.no_grad():
            attention_maps, aggregated_heatmap = self.get_advanced_attention_maps(x)
            return {
                'attention_maps': attention_maps,
                'aggregated_heatmap': aggregated_heatmap
            }