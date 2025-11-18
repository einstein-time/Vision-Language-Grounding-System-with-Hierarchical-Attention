"""
Unit tests for model components
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_transformer import VisionTransformer, create_vit_base_16
from models.text_encoder import TextEncoder, create_text_encoder_base, SimpleTokenizer
from models.fusion_module import VisionLanguageFusion, ContrastiveLoss
from models.vlg_model import VisionLanguageGroundingModel
from config import get_config


class TestVisionTransformer:
    """Test Vision Transformer"""

    def test_vit_forward(self):
        """Test forward pass"""
        model = create_vit_base_16()
        x = torch.randn(2, 3, 224, 224)

        cls_token, all_tokens, attn = model(
            x, return_all_tokens=True, return_attention=True
        )

        assert cls_token.shape == (2, 768)
        assert all_tokens.shape == (2, 197, 768)  # 196 patches + 1 CLS
        assert attn.shape == (2, 12, 197, 197)  # 12 heads

    def test_vit_patch_embedding(self):
        """Test patch embedding"""
        from models.vision_transformer import PatchEmbedding

        patch_embed = PatchEmbedding(224, 16, 3, 768)
        x = torch.randn(2, 3, 224, 224)
        patches = patch_embed(x)

        assert patches.shape == (2, 196, 768)  # (224/16)^2 = 196 patches

    def test_vit_attention(self):
        """Test multi-head self-attention"""
        from models.vision_transformer import MultiHeadSelfAttention

        attn = MultiHeadSelfAttention(768, 12, 0.1)
        x = torch.randn(2, 197, 768)

        out, weights = attn(x, return_attention=True)

        assert out.shape == (2, 197, 768)
        assert weights.shape == (2, 12, 197, 197)


class TestTextEncoder:
    """Test Text Encoder"""

    def test_text_encoder_forward(self):
        """Test forward pass"""
        model = create_text_encoder_base()
        input_ids = torch.randint(0, 49408, (2, 77))

        sent_emb, all_tokens, attn = model(
            input_ids, return_all_tokens=True, return_attention=True
        )

        assert sent_emb.shape == (2, 512)
        assert all_tokens.shape == (2, 77, 512)
        assert attn.shape == (2, 8, 77, 77)  # 8 heads

    def test_tokenizer(self):
        """Test simple tokenizer"""
        tokenizer = SimpleTokenizer()
        texts = ["hello world", "test"]

        input_ids = tokenizer.encode(texts)

        assert input_ids.shape == (2, 77)
        assert input_ids.dtype == torch.long


class TestFusionModule:
    """Test Fusion Module"""

    def test_contrastive_loss(self):
        """Test contrastive loss"""
        loss_fn = ContrastiveLoss(temperature=0.07)

        image_features = torch.randn(4, 512)
        text_features = torch.randn(4, 512)

        total_loss, loss_i2t, loss_t2i = loss_fn(image_features, text_features)

        assert total_loss.item() >= 0
        assert loss_i2t.item() >= 0
        assert loss_t2i.item() >= 0

    def test_cross_attention(self):
        """Test cross-attention"""
        from models.fusion_module import CrossAttention

        cross_attn = CrossAttention(query_dim=512, key_dim=768, num_heads=8)

        query = torch.randn(2, 77, 512)  # text
        key = torch.randn(2, 196, 768)  # image patches
        value = key

        out, attn = cross_attn(query, key, value, return_attention=True)

        assert out.shape == (2, 77, 512)
        assert attn.shape == (2, 8, 77, 196)

    def test_fusion_forward(self):
        """Test fusion module forward pass"""
        fusion = VisionLanguageFusion(
            vision_dim=768,
            text_dim=512,
            projection_dim=512
        )

        image_features = torch.randn(4, 768)
        text_features = torch.randn(4, 512)
        image_patch_features = torch.randn(4, 196, 768)
        text_token_features = torch.randn(4, 77, 512)

        outputs = fusion(
            image_features,
            text_features,
            image_patch_features,
            text_token_features,
            compute_grounding=True,
            return_attention=True
        )

        assert 'image_embeddings' in outputs
        assert 'text_embeddings' in outputs
        assert 'similarity_matrix' in outputs
        assert 'grounded_features' in outputs
        assert outputs['similarity_matrix'].shape == (4, 4)


class TestVLGModel:
    """Test complete VLG model"""

    def test_vlg_model_forward(self):
        """Test VLG model forward pass"""
        config = get_config()
        from models.vlg_model import create_vlg_model

        model = create_vlg_model(config)

        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (2, 77))

        outputs = model(images, input_ids, compute_grounding=True)

        assert 'image_features' in outputs
        assert 'text_features' in outputs
        assert 'similarity_matrix' in outputs
        assert outputs['image_features'].shape == (2, 768)
        assert outputs['text_features'].shape == (2, 512)

    def test_vlg_model_loss(self):
        """Test loss computation"""
        config = get_config()
        from models.vlg_model import create_vlg_model

        model = create_vlg_model(config)

        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (2, 77))

        losses = model.compute_loss(images, input_ids)

        assert 'total_loss' in losses
        assert 'contrastive_loss' in losses
        assert losses['total_loss'].item() >= 0

    def test_vlg_model_similarity(self):
        """Test similarity computation"""
        config = get_config()
        from models.vlg_model import create_vlg_model

        model = create_vlg_model(config)
        model.eval()

        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (3, 77))

        with torch.no_grad():
            similarity = model.get_similarity(images=images, input_ids=input_ids)

        assert similarity.shape == (2, 3)


def test_parameter_count():
    """Test that parameter counts are reasonable"""
    config = get_config()
    from models.vlg_model import create_vlg_model

    model = create_vlg_model(config)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6

    # ViT-Base should have around 85M params total
    assert 50 < num_params < 150, f"Parameter count {num_params:.2f}M seems unusual"

    print(f"Total parameters: {num_params:.2f}M")


def test_gradient_flow():
    """Test that gradients flow properly"""
    config = get_config()
    from models.vlg_model import create_vlg_model

    model = create_vlg_model(config)

    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 49408, (2, 77))

    losses = model.compute_loss(images, input_ids)
    loss = losses['total_loss']

    loss.backward()

    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break

    assert has_grad, "No gradients found in model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
