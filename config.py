"""
Configuration file for Vision-Language Grounding System
Contains all hyperparameters and settings for training and inference
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class VisionConfig:
    """Vision Transformer configuration"""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    num_patches: int = (224 // 16) ** 2  # 196 patches


@dataclass
class TextConfig:
    """Text Transformer configuration"""
    vocab_size: int = 49408  # CLIP-like vocabulary
    max_length: int = 77
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    mlp_dim: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1


@dataclass
class FusionConfig:
    """Multi-modal fusion configuration"""
    vision_dim: int = 768
    text_dim: int = 512
    projection_dim: int = 512
    temperature: float = 0.07
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True

    # Optimization
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Scheduling
    scheduler_type: str = "cosine"  # cosine, linear, constant
    min_lr: float = 1e-6

    # Loss weights
    contrastive_weight: float = 1.0
    classification_weight: float = 0.5
    grounding_weight: float = 0.5

    # Logging
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 5000

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "vision-language-grounding"
    wandb_entity: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "coco"  # coco, flickr30k
    data_root: str = "data"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # Image augmentation
    image_size: int = 224
    random_resized_crop: bool = True
    color_jitter: bool = True
    random_horizontal_flip: bool = True
    normalize_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    normalize_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    # Text processing
    max_text_length: int = 77
    tokenizer_type: str = "clip"  # clip, bert


@dataclass
class InferenceConfig:
    """Inference configuration"""
    checkpoint_path: str = "checkpoints/best_model.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Retrieval
    retrieval_top_k: int = 5

    # Visual grounding
    grounding_threshold: float = 0.5

    # Visualization
    attention_head: int = 0  # Which attention head to visualize (-1 for average)
    save_visualizations: bool = True
    visualization_dir: str = "results/visualizations"


@dataclass
class Config:
    """Master configuration"""
    vision: VisionConfig = VisionConfig()
    text: TextConfig = TextConfig()
    fusion: FusionConfig = FusionConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    inference: InferenceConfig = InferenceConfig()

    def __post_init__(self):
        """Validate configuration consistency"""
        assert self.vision.hidden_dim == self.fusion.vision_dim, \
            "Vision hidden dim must match fusion vision dim"
        assert self.text.hidden_dim == self.fusion.text_dim, \
            "Text hidden dim must match fusion text dim"
        assert self.data.image_size == self.vision.image_size, \
            "Data image size must match vision image size"
        assert self.data.max_text_length == self.text.max_length, \
            "Data max text length must match text max length"


def get_config() -> Config:
    """Get default configuration"""
    return Config()
