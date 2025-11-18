"""
Complete Vision-Language Grounding Model
Combines Vision Transformer, Text Encoder, and Fusion Module
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .vision_transformer import VisionTransformer, create_vit_base_16
from .text_encoder import TextEncoder, create_text_encoder_base
from .fusion_module import VisionLanguageFusion


class VisionLanguageGroundingModel(nn.Module):
    """
    Complete Vision-Language Grounding Model

    Integrates:
    - Vision Transformer for image encoding
    - Text Transformer for text encoding
    - Multi-modal fusion with contrastive learning
    - Visual grounding capabilities

    Args:
        vision_config: Configuration for vision encoder
        text_config: Configuration for text encoder
        fusion_config: Configuration for fusion module
        use_gradient_checkpointing: Whether to use gradient checkpointing
    """

    def __init__(
        self,
        vision_config: dict,
        text_config: dict,
        fusion_config: dict,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        # Vision encoder
        self.vision_encoder = VisionTransformer(
            image_size=vision_config.get('image_size', 224),
            patch_size=vision_config.get('patch_size', 16),
            in_channels=vision_config.get('in_channels', 3),
            hidden_dim=vision_config.get('hidden_dim', 768),
            num_layers=vision_config.get('num_layers', 12),
            num_heads=vision_config.get('num_heads', 12),
            mlp_dim=vision_config.get('mlp_dim', 3072),
            dropout=vision_config.get('dropout', 0.1),
            attention_dropout=vision_config.get('attention_dropout', 0.1),
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=text_config.get('vocab_size', 49408),
            max_length=text_config.get('max_length', 77),
            hidden_dim=text_config.get('hidden_dim', 512),
            num_layers=text_config.get('num_layers', 12),
            num_heads=text_config.get('num_heads', 8),
            mlp_dim=text_config.get('mlp_dim', 2048),
            dropout=text_config.get('dropout', 0.1),
            attention_dropout=text_config.get('attention_dropout', 0.1),
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        # Fusion module
        self.fusion = VisionLanguageFusion(
            vision_dim=fusion_config.get('vision_dim', 768),
            text_dim=fusion_config.get('text_dim', 512),
            projection_dim=fusion_config.get('projection_dim', 512),
            temperature=fusion_config.get('temperature', 0.07),
            num_cross_attn_heads=fusion_config.get('cross_attention_heads', 8),
            dropout=fusion_config.get('cross_attention_dropout', 0.1)
        )

        # Zero-shot classification head (optional)
        self.classification_head = nn.Linear(
            fusion_config.get('projection_dim', 512),
            1000  # Number of classes, can be adjusted
        )

    def encode_image(
        self,
        images: torch.Tensor,
        return_all_tokens: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode images

        Args:
            images: Input images [batch_size, 3, H, W]
            return_all_tokens: Whether to return patch tokens
        Returns:
            image_features: Global image features [batch_size, vision_dim]
            patch_features: Patch features [batch_size, num_patches+1, vision_dim]
        """
        image_features, patch_features, _ = self.vision_encoder(
            images,
            return_all_tokens=return_all_tokens
        )
        return image_features, patch_features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text

        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_all_tokens: Whether to return all token embeddings
        Returns:
            text_features: Sentence embeddings [batch_size, text_dim]
            token_features: Token embeddings [batch_size, seq_len, text_dim]
        """
        text_features, token_features, _ = self.text_encoder(
            input_ids,
            attention_mask,
            return_all_tokens=return_all_tokens
        )
        return text_features, token_features

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_grounding: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            images: Input images [batch_size, 3, H, W]
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            compute_grounding: Whether to compute visual grounding
            return_attention: Whether to return attention weights
        Returns:
            Dictionary containing:
                - image_features: Global image features
                - text_features: Global text features
                - image_embeddings: Projected image embeddings
                - text_embeddings: Projected text embeddings
                - similarity_matrix: Image-text similarity
                - grounded_features (optional): Grounded text features
                - grounding_attention (optional): Cross-attention weights
        """
        # Encode images
        image_features, patch_features = self.encode_image(images, return_all_tokens=True)

        # Encode text
        text_features, token_features = self.encode_text(
            input_ids, attention_mask, return_all_tokens=True
        )

        # Remove CLS token from patch features for grounding
        if patch_features is not None:
            patch_features_only = patch_features[:, 1:, :]  # Remove CLS token
        else:
            patch_features_only = None

        # Fusion
        fusion_outputs = self.fusion(
            image_features,
            text_features,
            patch_features_only,
            token_features,
            compute_grounding=compute_grounding,
            return_attention=return_attention
        )

        # Add original features to outputs
        fusion_outputs['image_features'] = image_features
        fusion_outputs['text_features'] = text_features
        fusion_outputs['patch_features'] = patch_features_only
        fusion_outputs['token_features'] = token_features

        return fusion_outputs

    def compute_loss(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        contrastive_weight: float = 1.0,
        classification_weight: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses

        Args:
            images: Input images [batch_size, 3, H, W]
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            labels: Classification labels (optional) [batch_size]
            contrastive_weight: Weight for contrastive loss
            classification_weight: Weight for classification loss
        Returns:
            Dictionary containing losses
        """
        # Forward pass
        outputs = self.forward(images, input_ids, attention_mask, compute_grounding=False)

        # Contrastive loss
        contrastive_loss, loss_i2t, loss_t2i = self.fusion.compute_contrastive_loss(
            outputs['image_features'],
            outputs['text_features']
        )

        losses = {
            'contrastive_loss': contrastive_loss,
            'loss_i2t': loss_i2t,
            'loss_t2i': loss_t2i,
            'total_loss': contrastive_weight * contrastive_loss
        }

        # Classification loss (optional)
        if labels is not None:
            logits = self.classification_head(outputs['image_embeddings'])
            classification_loss = nn.functional.cross_entropy(logits, labels)
            losses['classification_loss'] = classification_loss
            losses['total_loss'] += classification_weight * classification_loss

        return losses

    @torch.no_grad()
    def get_similarity(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute similarity between images and texts

        Args:
            images: Input images (optional if image_features provided)
            input_ids: Input token ids (optional if text_features provided)
            attention_mask: Text attention mask
            image_features: Precomputed image features
            text_features: Precomputed text features
        Returns:
            Similarity matrix [num_images, num_texts]
        """
        # Encode if features not provided
        if image_features is None:
            assert images is not None, "Either images or image_features must be provided"
            image_features, _ = self.encode_image(images)

        if text_features is None:
            assert input_ids is not None, "Either input_ids or text_features must be provided"
            text_features, _ = self.encode_text(input_ids, attention_mask)

        # Compute similarity
        return self.fusion.get_image_text_similarity(image_features, text_features)

    @torch.no_grad()
    def zero_shot_classify(
        self,
        images: torch.Tensor,
        text_descriptions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Zero-shot classification using text descriptions

        Args:
            images: Input images [batch_size, 3, H, W]
            text_descriptions: Text descriptions for classes [num_classes, seq_len]
            attention_mask: Text attention mask [num_classes, seq_len]
        Returns:
            Class predictions [batch_size]
        """
        # Encode images and texts
        image_features, _ = self.encode_image(images)
        text_features, _ = self.encode_text(text_descriptions, attention_mask)

        # Compute similarity
        similarity = self.fusion.get_image_text_similarity(image_features, text_features)

        # Get predictions (argmax over classes)
        predictions = similarity.argmax(dim=1)

        return predictions

    def get_attention_maps(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get attention maps from vision and text encoders

        Args:
            images: Input images [batch_size, 3, H, W]
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Text attention mask
        Returns:
            vision_attention: Vision self-attention [batch_size, num_heads, num_patches+1, num_patches+1]
            text_attention: Text self-attention [batch_size, num_heads, seq_len, seq_len]
        """
        # Vision attention
        _, _, vision_attn = self.vision_encoder(images, return_attention=True)

        # Text attention
        _, _, text_attn = self.text_encoder(
            input_ids, attention_mask, return_attention=True
        )

        return vision_attn, text_attn


def create_vlg_model(config) -> VisionLanguageGroundingModel:
    """
    Create VLG model from configuration

    Args:
        config: Configuration object
    Returns:
        VisionLanguageGroundingModel instance
    """
    vision_config = {
        'image_size': config.vision.image_size,
        'patch_size': config.vision.patch_size,
        'in_channels': config.vision.in_channels,
        'hidden_dim': config.vision.hidden_dim,
        'num_layers': config.vision.num_layers,
        'num_heads': config.vision.num_heads,
        'mlp_dim': config.vision.mlp_dim,
        'dropout': config.vision.dropout,
        'attention_dropout': config.vision.attention_dropout
    }

    text_config = {
        'vocab_size': config.text.vocab_size,
        'max_length': config.text.max_length,
        'hidden_dim': config.text.hidden_dim,
        'num_layers': config.text.num_layers,
        'num_heads': config.text.num_heads,
        'mlp_dim': config.text.mlp_dim,
        'dropout': config.text.dropout,
        'attention_dropout': config.text.attention_dropout
    }

    fusion_config = {
        'vision_dim': config.fusion.vision_dim,
        'text_dim': config.fusion.text_dim,
        'projection_dim': config.fusion.projection_dim,
        'temperature': config.fusion.temperature,
        'cross_attention_heads': config.fusion.cross_attention_heads,
        'cross_attention_dropout': config.fusion.cross_attention_dropout
    }

    return VisionLanguageGroundingModel(
        vision_config,
        text_config,
        fusion_config,
        use_gradient_checkpointing=config.training.use_gradient_checkpointing
    )


if __name__ == "__main__":
    from config import get_config

    # Create model
    config = get_config()
    model = create_vlg_model(config)

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 49408, (batch_size, 77))

    outputs = model(images, input_ids, compute_grounding=True, return_attention=True)

    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Test loss computation
    losses = model.compute_loss(images, input_ids)
    print("\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
