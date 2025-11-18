"""
Text Transformer Encoder implementation
Implements a BERT/CLIP-style text encoder for processing text inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with learned embeddings

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Embedding dimension
    """

    def __init__(self, vocab_size: int = 49408, hidden_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch_size, seq_len]
        Returns:
            Token embeddings [batch_size, seq_len, hidden_dim]
        """
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding

    Args:
        max_length: Maximum sequence length
        hidden_dim: Embedding dimension
    """

    def __init__(self, max_length: int = 77, hidden_dim: int = 512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, hidden_dim]
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, hidden_dim]
        """
        return x + self.pos_embedding[:, :x.size(1), :]


class TextMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention for text

    Args:
        hidden_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
        causal: Whether to use causal masking
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        # Query, Key, Value projections
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Mask for padding [batch_size, seq_len]
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

        # Apply causal mask if needed
        if self.causal:
            mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(mask, float('-inf'))

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [B, N] -> [B, 1, 1, N]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attention_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn_weights = attn if return_attention else None
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, hidden_dim]
        x = self.proj(x)
        x = self.dropout(x)

        return x, attn_weights


class TextMLP(nn.Module):
    """
    MLP for text transformer with GELU activation

    Args:
        hidden_dim: Input dimension
        mlp_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        mlp_dim: int = 2048,
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


class TextTransformerBlock(nn.Module):
    """
    Transformer encoder block for text

    Args:
        hidden_dim: Dimension of embeddings
        num_heads: Number of attention heads
        mlp_dim: Dimension of MLP hidden layer
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
        causal: Whether to use causal attention
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = TextMultiHeadAttention(
            hidden_dim, num_heads, attention_dropout, causal
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = TextMLP(hidden_dim, mlp_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
            Attention weights (optional)
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(
            self.norm1(x), attention_mask, return_attention
        )
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class TextEncoder(nn.Module):
    """
    Text Transformer Encoder

    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
        hidden_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
        causal: Whether to use causal attention
        use_gradient_checkpointing: Whether to use gradient checkpointing
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        max_length: int = 77,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        causal: bool = False,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token embedding
        self.token_embed = TokenEmbedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_encode = PositionalEncoding(max_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TextTransformerBlock(
                hidden_dim, num_heads, mlp_dim, dropout, attention_dropout, causal
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_encode.pos_embedding, std=0.02)

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
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def create_attention_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Create attention mask from input ids

        Args:
            input_ids: Input token ids [batch_size, seq_len]
            pad_token_id: Padding token id
        Returns:
            Attention mask [batch_size, seq_len]
        """
        return (input_ids != pad_token_id).to(torch.bool)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            input_ids: Input token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_all_tokens: If True, return all token embeddings
            return_attention: If True, return attention weights from last layer
        Returns:
            cls_token: Sentence embedding from first token [batch_size, hidden_dim]
            all_tokens (optional): All token embeddings [batch_size, seq_len, hidden_dim]
            attention_weights (optional): Attention weights from last layer
        """
        # Token embedding
        x = self.token_embed(input_ids)  # [B, seq_len, hidden_dim]

        # Add positional encoding
        x = self.pos_encode(x)
        x = self.dropout(x)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)

        # Apply transformer blocks
        attn_weights = None
        for i, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x, _ = torch.utils.checkpoint.checkpoint(
                    block, x, attention_mask, False, use_reentrant=False
                )
            else:
                # Return attention only from last layer
                return_attn = return_attention and (i == len(self.blocks) - 1)
                x, attn_weights = block(x, attention_mask, return_attn)

        # Final layer norm
        x = self.norm(x)

        # Extract sentence embedding from first token (EOS token in CLIP)
        # In CLIP, the EOS token position is used as the sentence representation
        # For simplicity, we use the first token like BERT's [CLS]
        sentence_embedding = x[:, 0]  # [B, hidden_dim]

        if return_all_tokens:
            return sentence_embedding, x, attn_weights
        else:
            return sentence_embedding, None, attn_weights


class SimpleTokenizer:
    """
    Simple tokenizer for text processing
    In production, use a proper tokenizer like CLIP's or BERT's
    """

    def __init__(self, vocab_size: int = 49408, max_length: int = 77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2

    def encode(self, texts: list, device: str = "cpu") -> torch.Tensor:
        """
        Simple character-level encoding (placeholder)
        In production, replace with proper tokenizer

        Args:
            texts: List of text strings
            device: Device to place tensors on
        Returns:
            Token ids [batch_size, max_length]
        """
        batch_size = len(texts)
        input_ids = torch.full(
            (batch_size, self.max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )

        for i, text in enumerate(texts):
            # Simple character-level encoding
            tokens = [self.sos_token_id]
            for char in text[:self.max_length - 2]:
                # Map character to token id (simple ASCII mapping)
                token_id = min(ord(char), self.vocab_size - 1)
                tokens.append(token_id)
            tokens.append(self.eos_token_id)

            # Fill input_ids
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        return input_ids


def create_text_encoder_base(**kwargs) -> TextEncoder:
    """Create base text encoder"""
    return TextEncoder(
        vocab_size=49408,
        max_length=77,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


if __name__ == "__main__":
    # Test Text Encoder
    model = create_text_encoder_base()
    tokenizer = SimpleTokenizer()

    texts = ["A cat sitting on a mat", "A dog playing in the park"]
    input_ids = tokenizer.encode(texts)

    # Test forward pass
    sentence_emb, all_tokens, attn = model(
        input_ids, return_all_tokens=True, return_attention=True
    )

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Sentence embedding shape: {sentence_emb.shape}")
    print(f"All tokens shape: {all_tokens.shape}")
    print(f"Attention shape: {attn.shape if attn is not None else None}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
