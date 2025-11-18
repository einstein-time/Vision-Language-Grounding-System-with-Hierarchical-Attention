# Vision-Language Grounding System with Hierarchical Attention

A production-quality PyTorch implementation of a multi-task vision-language model that combines Vision Transformers with text encoders for cross-modal understanding.

## Features

- **Vision Transformer (ViT)** implemented from scratch (ViT-Base/16)
- **Text Transformer Encoder** for processing text inputs
- **Multi-modal Fusion** with contrastive learning (CLIP-style)
- **Visual Grounding** with cross-attention mechanisms
- **Multiple Downstream Tasks**:
  - Zero-shot image classification
  - Image-to-text retrieval
  - Text-to-image retrieval
  - Visual grounding with attention visualization
- **Training Infrastructure**:
  - Mixed precision training (AMP)
  - Gradient checkpointing for memory efficiency
  - Cosine learning rate scheduling with warmup
  - WandB and TensorBoard logging
- **Production Features**:
  - Comprehensive evaluation metrics
  - Attention visualization tools
  - Unit tests
  - Type hints and docstrings

## Architecture

### Vision Encoder
- **Model**: Vision Transformer (ViT-Base/16)
- **Parameters**:
  - 12 transformer layers
  - 768 hidden dimensions
  - 12 attention heads
  - Image size: 224×224
  - Patch size: 16×16 (196 patches)

### Text Encoder
- **Model**: Transformer encoder (BERT/CLIP-style)
- **Parameters**:
  - 12 transformer layers
  - 512 hidden dimensions
  - 8 attention heads
  - Max sequence length: 77 tokens
  - Vocabulary size: 49,408

### Fusion Module
- **Contrastive Learning**: InfoNCE loss for vision-language alignment
- **Cross-Attention**: Text-to-image grounding
- **Projection Heads**: Map features to common 512-dim space
- **Temperature**: 0.07 (learnable)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Vision-Language-Grounding-System-with-Hierarchical-Attention.git
cd Vision-Language-Grounding-System-with-Hierarchical-Attention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training

```bash
# Train with default configuration
python train.py

# Train with custom settings
python train.py --config custom_config.yaml

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pt

# Train without WandB
python train.py --no-wandb
```

### 2. Inference

```bash
# Run all inference tasks
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --task all \
    --output-dir results/inference

# Zero-shot classification only
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --task classify

# Visual grounding
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --task grounding

# Attention visualization
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --task attention
```

### 3. Evaluation

```bash
# Evaluate retrieval performance
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --task retrieval \
    --split val \
    --output results/evaluation.json

# Evaluate zero-shot classification
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --task classification \
    --split test

# Evaluate all tasks
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --task all \
    --split val
```

## Dataset Preparation

### COCO Captions

```bash
# Download COCO dataset
mkdir -p data/coco
cd data/coco

# Download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### Flickr30k

```bash
# Download Flickr30k from Kaggle
# https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

mkdir -p data/flickr30k
# Place images in data/flickr30k/flickr30k_images
# Place annotations in data/flickr30k/
```

## Configuration

All hyperparameters are defined in `config.py`. Key configurations:

```python
# Vision Configuration
image_size = 224
patch_size = 16
hidden_dim = 768
num_layers = 12

# Training Configuration
batch_size = 256
learning_rate = 1e-4
num_epochs = 50
warmup_epochs = 5

# Contrastive Learning
temperature = 0.07
contrastive_weight = 1.0
```

## Project Structure

```
Vision-Language-Grounding-System-with-Hierarchical-Attention/
├── models/
│   ├── __init__.py
│   ├── vision_transformer.py    # ViT implementation
│   ├── text_encoder.py          # Text transformer
│   ├── fusion_module.py         # Multi-modal fusion
│   └── vlg_model.py            # Complete model
├── data/
│   ├── __init__.py
│   └── dataset.py              # Dataset loading
├── utils/
│   ├── __init__.py
│   └── visualization.py        # Visualization tools
├── tests/
│   ├── __init__.py
│   └── test_models.py          # Unit tests
├── config.py                   # Configuration
├── train.py                    # Training script
├── inference.py                # Inference demo
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── demo.ipynb                  # Jupyter notebook demo
```

## Usage Examples

### Python API

```python
import torch
from config import get_config
from models import create_vlg_model
from inference import VLGInference

# Load model
inference_engine = VLGInference(
    checkpoint_path="checkpoints/best_model.pt",
    device="cuda"
)

# Zero-shot classification
predicted_class, rankings = inference_engine.zero_shot_classify(
    image_path="example.jpg",
    class_descriptions=["cat", "dog", "bird", "car"]
)
print(f"Predicted: {predicted_class}")

# Image-to-text retrieval
results = inference_engine.image_to_text_retrieval(
    query_image_path="query.jpg",
    candidate_texts=[
        "a cat on a mat",
        "a dog in a park",
        "a bird in the sky"
    ],
    top_k=3
)

# Visual grounding
grounding_scores = inference_engine.visual_grounding(
    image_path="image.jpg",
    text_query="the cat",
    save_path="results/grounding.png"
)

# Attention visualization
attention = inference_engine.visualize_attention(
    image_path="image.jpg",
    head_idx=0,
    save_path="results/attention.png"
)
```

## Training Details

### Loss Functions
1. **Contrastive Loss (InfoNCE)**: Aligns vision and language embeddings
   - Computed bidirectionally (image-to-text and text-to-image)
   - Temperature-scaled similarity

2. **Optional Classification Loss**: For supervised fine-tuning

### Optimization
- **Optimizer**: AdamW with weight decay (0.1)
- **Learning Rate**: 1e-4 with cosine annealing
- **Warmup**: 5 epochs linear warmup
- **Gradient Clipping**: Max norm 1.0
- **Mixed Precision**: Automatic Mixed Precision (AMP)

### Data Augmentation
- RandomResizedCrop (scale 0.8-1.0)
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness, contrast, saturation, hue)
- Normalization (CLIP statistics)

## Evaluation Metrics

### Retrieval
- **Recall@K** (K=1,5,10,20): Percentage of queries where correct item is in top-K
- **Mean Rank**: Average rank of correct item
- **Median Rank**: Median rank of correct item
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank

### Classification
- **Accuracy**: Top-1 accuracy
- **Top-5 Accuracy**: Top-5 accuracy
- **Per-class Accuracy**: Accuracy for each class

## Visualization

The system provides comprehensive visualization tools:

1. **Attention Maps**: Self-attention from vision encoder
2. **Cross-Modal Attention**: Text-to-image attention
3. **Similarity Matrices**: Image-text similarity heatmaps
4. **Grounding Maps**: Visual grounding results with highlighted regions
5. **Retrieval Results**: Top-K retrieval visualizations

## Testing

Run unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=models --cov-report=html
```

## Performance

Expected performance on COCO Captions (val split):

| Metric | Score |
|--------|-------|
| Image-to-Text R@1 | ~40-50% |
| Image-to-Text R@5 | ~65-75% |
| Text-to-Image R@1 | ~30-40% |
| Text-to-Image R@5 | ~55-65% |

*Note: Actual performance depends on training data size, epochs, and hyperparameters*

## Model Checkpoints

Checkpoints include:
- Model state dictionary
- Optimizer state
- Scheduler state
- Training configuration
- Evaluation metrics

## Tips for Training

1. **Memory Management**:
   - Use gradient checkpointing for large models
   - Enable mixed precision training
   - Adjust batch size based on GPU memory

2. **Learning Rate**:
   - Start with 1e-4 for ViT-Base
   - Use warmup for stable training
   - Monitor gradients for exploding/vanishing issues

3. **Data**:
   - More data = better performance
   - Data augmentation helps generalization
   - Balance image and text quality

4. **Monitoring**:
   - Watch contrastive loss convergence
   - Monitor retrieval metrics during training
   - Check attention maps for sanity

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vlg_system,
  title={Vision-Language Grounding System with Hierarchical Attention},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Vision-Language-Grounding-System-with-Hierarchical-Attention}
}
```

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model variant

2. **Slow Training**:
   - Use mixed precision (AMP)
   - Increase num_workers for data loading
   - Use torch.compile() (PyTorch 2.0+)

3. **Poor Convergence**:
   - Check learning rate
   - Verify data preprocessing
   - Inspect attention visualizations

## Contact

For questions or issues, please open a GitHub issue or contact [your.email@example.com]

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- CLIP paper authors for inspiration
- Open source community for valuable feedback
