"""
Vision Transformer (ViT) implementation from scratch
Implements ViT-Base/16 architecture with patch embedding, positional encoding, and transformer blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """
    Converts image into patches and embeds them

    Args:
        image_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        hidden_dim: Embedding dimension
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Convolutional layer for patch embedding
        self.projection = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [batch_size, channels, height, width]
        Returns:
            Patch embeddings [batch_size, num_patches, hidden_dim]
        """
        # x: [B, C, H, W]
        x = self.projection(x)  # [B, hidden_dim, H/P, W/P]
        x = x.flatten(2)  # [B, hidden_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, hidden_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism

    Args:
        hidden_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            return_attention: Whether to return attention weights
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
            Attention weights (optional) [batch_size, num_heads, seq_len, seq_len]
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn if return_attention else None
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, hidden_dim]
        x = self.proj(x)
        x = self.dropout(x)

        return x, attn_weights


class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation

    Args:
        hidden_dim: Input dimension
        mlp_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with self-attention and MLP

    Args:
        hidden_dim: Dimension of embeddings
        num_heads: Number of attention heads
        mlp_dim: Dimension of MLP hidden layer
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            return_attention: Whether to return attention weights
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
            Attention weights (optional)
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model
    Implements ViT-Base/16 architecture

    Args:
        image_size: Input image size
        patch_size: Size of image patches
        in_channels: Number of input channels
        hidden_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
        use_gradient_checkpointing: Whether to use gradient checkpointing
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, hidden_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim, num_heads, mlp_dim, dropout, attention_dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT paper"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize other parameters
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        """Initialize module weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input images [batch_size, channels, height, width]
            return_all_tokens: If True, return all patch tokens along with CLS
            return_attention: If True, return attention weights from last layer
        Returns:
            cls_token: Global image features [batch_size, hidden_dim]
            all_tokens (optional): All tokens including CLS [batch_size, num_patches+1, hidden_dim]
            attention_weights (optional): Attention weights from last layer
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, hidden_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, hidden_dim]

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply transformer blocks
        attn_weights = None
        for i, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x, _ = torch.utils.checkpoint.checkpoint(
                    block, x, False, use_reentrant=False
                )
            else:
                # Return attention only from last layer
                return_attn = return_attention and (i == len(self.blocks) - 1)
                x, attn_weights = block(x, return_attn)

        # Final layer norm
        x = self.norm(x)

        # Extract CLS token
        cls_token = x[:, 0]  # [B, hidden_dim]

        if return_all_tokens:
            return cls_token, x, attn_weights
        else:
            return cls_token, None, attn_weights


def create_vit_base_16(**kwargs) -> VisionTransformer:
    """Create ViT-Base/16 model"""
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def create_vit_large_16(**kwargs) -> VisionTransformer:
    """Create ViT-Large/16 model"""
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        mlp_dim=4096,
        **kwargs
    )


if __name__ == "__main__":
    # Test Vision Transformer
    model = create_vit_base_16()
    x = torch.randn(2, 3, 224, 224)

    # Test forward pass
    cls_token, all_tokens, attn = model(x, return_all_tokens=True, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"CLS token shape: {cls_token.shape}")
    print(f"All tokens shape: {all_tokens.shape}")
    print(f"Attention shape: {attn.shape if attn is not None else None}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
