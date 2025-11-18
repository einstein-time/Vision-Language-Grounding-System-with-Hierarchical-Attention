"""
Multi-modal Fusion Module
Implements contrastive learning (CLIP-style) and cross-attention for vision-language grounding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    Projects vision/text features to a common embedding space

    Args:
        input_dim: Input feature dimension
        projection_dim: Output projection dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, input_dim]
        Returns:
            Projected features [batch_size, projection_dim]
        """
        x = self.projection(x)
        x = self.norm(x)
        return x


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (InfoNCE) for vision-language alignment
    Similar to CLIP's contrastive objective

    Args:
        temperature: Temperature parameter for softmax
        learnable_temperature: Whether to make temperature learnable
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = False
    ):
        super().__init__()
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute contrastive loss

        Args:
            image_features: Normalized image features [batch_size, dim]
            text_features: Normalized text features [batch_size, dim]
        Returns:
            total_loss: Combined contrastive loss
            image_loss: Image-to-text contrastive loss
            text_loss: Text-to-image contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity matrix
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = text_features @ image_features.T / self.temperature

        # Create labels (diagonal elements are positive pairs)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # Average the two losses
        total_loss = (loss_i2t + loss_t2i) / 2

        return total_loss, loss_i2t, loss_t2i


class CrossAttention(nn.Module):
    """
    Cross-attention module for visual grounding
    Allows text to attend to image patches

    Args:
        query_dim: Dimension of query (text)
        key_dim: Dimension of key/value (image)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        query_dim: int = 512,
        key_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection (from text)
        self.q_proj = nn.Linear(query_dim, query_dim)

        # Key and Value projections (from image)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)

        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            query: Query tensor (text) [batch_size, text_len, query_dim]
            key: Key tensor (image) [batch_size, num_patches, key_dim]
            value: Value tensor (image) [batch_size, num_patches, key_dim]
            return_attention: Whether to return attention weights
        Returns:
            output: Cross-attended features [batch_size, text_len, query_dim]
            attention_weights (optional): [batch_size, num_heads, text_len, num_patches]
        """
        B, N_text, C_query = query.shape
        N_img = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query).reshape(B, N_text, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(B, N_img, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(B, N_img, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, N_text, head_dim]
        k = k.permute(0, 2, 1, 3)  # [B, num_heads, N_img, head_dim]
        v = v.permute(0, 2, 1, 3)  # [B, num_heads, N_img, head_dim]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_text, N_img]
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn if return_attention else None
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_text, C_query)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out, attn_weights


class VisualGroundingModule(nn.Module):
    """
    Visual grounding module for locating image regions corresponding to text

    Args:
        text_dim: Text feature dimension
        vision_dim: Vision feature dimension
        num_heads: Number of cross-attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        text_dim: int = 512,
        vision_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cross_attn = CrossAttention(text_dim, vision_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(text_dim)

        # Grounding head for predicting relevance scores
        self.grounding_head = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, 1)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_patch_features: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute visual grounding

        Args:
            text_features: Text token features [batch_size, text_len, text_dim]
            image_patch_features: Image patch features [batch_size, num_patches, vision_dim]
            return_attention: Whether to return attention weights
        Returns:
            grounded_features: Text features attended to image [batch_size, text_len, text_dim]
            attention_map: Cross-attention weights [batch_size, num_heads, text_len, num_patches]
        """
        # Cross-attention: text attends to image patches
        attn_out, attn_weights = self.cross_attn(
            text_features,
            image_patch_features,
            image_patch_features,
            return_attention
        )

        # Residual connection and normalization
        grounded_features = self.norm(text_features + attn_out)

        return grounded_features, attn_weights

    def compute_grounding_scores(
        self,
        text_features: torch.Tensor,
        image_patch_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute grounding scores for each patch

        Args:
            text_features: Text features [batch_size, text_dim]
            image_patch_features: Image patch features [batch_size, num_patches, vision_dim]
        Returns:
            Grounding scores [batch_size, num_patches]
        """
        # Expand text features to match patches
        B, N, D = image_patch_features.shape
        text_expanded = text_features.unsqueeze(1).expand(B, N, -1)

        # Concatenate or use cross-attention
        grounded, attn = self.forward(
            text_expanded,
            image_patch_features,
            return_attention=True
        )

        # Average attention across heads and text tokens to get patch scores
        # attn: [B, num_heads, text_len, num_patches]
        patch_scores = attn.mean(dim=(1, 2))  # [B, num_patches]

        return patch_scores


class VisionLanguageFusion(nn.Module):
    """
    Complete multi-modal fusion module combining contrastive learning and visual grounding

    Args:
        vision_dim: Vision feature dimension
        text_dim: Text feature dimension
        projection_dim: Projection dimension for contrastive learning
        temperature: Temperature for contrastive loss
        num_cross_attn_heads: Number of heads for cross-attention
        dropout: Dropout probability
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        projection_dim: int = 512,
        temperature: float = 0.07,
        num_cross_attn_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.projection_dim = projection_dim

        # Projection heads for contrastive learning
        self.vision_projection = ProjectionHead(vision_dim, projection_dim, dropout)
        self.text_projection = ProjectionHead(text_dim, projection_dim, dropout)

        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(temperature, learnable_temperature=True)

        # Visual grounding module
        self.grounding_module = VisualGroundingModule(
            text_dim, vision_dim, num_cross_attn_heads, dropout
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_patch_features: Optional[torch.Tensor] = None,
        text_token_features: Optional[torch.Tensor] = None,
        compute_grounding: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            image_features: Global image features (CLS) [batch_size, vision_dim]
            text_features: Global text features (CLS) [batch_size, text_dim]
            image_patch_features: Patch features [batch_size, num_patches, vision_dim]
            text_token_features: Token features [batch_size, text_len, text_dim]
            compute_grounding: Whether to compute visual grounding
            return_attention: Whether to return attention weights
        Returns:
            Dictionary containing:
                - image_embeddings: Projected image features
                - text_embeddings: Projected text features
                - similarity_matrix: Cosine similarity matrix
                - grounded_features (optional): Grounded text features
                - grounding_attention (optional): Cross-attention weights
        """
        # Project features for contrastive learning
        image_embeddings = self.vision_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Normalize embeddings
        image_embeddings_norm = F.normalize(image_embeddings, dim=-1)
        text_embeddings_norm = F.normalize(text_embeddings, dim=-1)

        # Compute similarity matrix
        similarity_matrix = image_embeddings_norm @ text_embeddings_norm.T

        outputs = {
            'image_embeddings': image_embeddings_norm,
            'text_embeddings': text_embeddings_norm,
            'similarity_matrix': similarity_matrix
        }

        # Visual grounding if requested
        if compute_grounding and image_patch_features is not None and text_token_features is not None:
            grounded_features, grounding_attn = self.grounding_module(
                text_token_features,
                image_patch_features,
                return_attention
            )
            outputs['grounded_features'] = grounded_features
            if return_attention:
                outputs['grounding_attention'] = grounding_attn

        return outputs

    def compute_contrastive_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute contrastive loss

        Args:
            image_features: Image features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
        Returns:
            total_loss, image_to_text_loss, text_to_image_loss
        """
        # Project features
        image_embeddings = self.vision_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Compute contrastive loss
        return self.contrastive_loss(image_embeddings, text_embeddings)

    def get_image_text_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between images and texts

        Args:
            image_features: Image features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
        Returns:
            Similarity scores [batch_size, batch_size]
        """
        # Project features
        image_embeddings = self.vision_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Normalize and compute similarity
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        return image_embeddings @ text_embeddings.T


if __name__ == "__main__":
    # Test fusion module
    batch_size = 4
    vision_dim = 768
    text_dim = 512
    num_patches = 196
    text_len = 77

    # Create fusion module
    fusion = VisionLanguageFusion(
        vision_dim=vision_dim,
        text_dim=text_dim,
        projection_dim=512
    )

    # Create dummy inputs
    image_features = torch.randn(batch_size, vision_dim)
    text_features = torch.randn(batch_size, text_dim)
    image_patch_features = torch.randn(batch_size, num_patches, vision_dim)
    text_token_features = torch.randn(batch_size, text_len, text_dim)

    # Test contrastive learning
    loss, loss_i2t, loss_t2i = fusion.compute_contrastive_loss(
        image_features, text_features
    )
    print(f"Contrastive loss: {loss.item():.4f}")
    print(f"Image-to-text loss: {loss_i2t.item():.4f}")
    print(f"Text-to-image loss: {loss_t2i.item():.4f}")

    # Test full forward pass with grounding
    outputs = fusion(
        image_features,
        text_features,
        image_patch_features,
        text_token_features,
        compute_grounding=True,
        return_attention=True
    )

    print(f"\nOutput keys: {outputs.keys()}")
    print(f"Similarity matrix shape: {outputs['similarity_matrix'].shape}")
    if 'grounded_features' in outputs:
        print(f"Grounded features shape: {outputs['grounded_features'].shape}")
    if 'grounding_attention' in outputs:
        print(f"Grounding attention shape: {outputs['grounding_attention'].shape}")

    print(f"\nNumber of parameters: {sum(p.numel() for p in fusion.parameters()) / 1e6:.2f}M")
