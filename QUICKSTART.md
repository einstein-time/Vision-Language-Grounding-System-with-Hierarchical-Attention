# Quick Start Guide

Get up and running with the Vision-Language Grounding System in 5 minutes!

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Vision-Language-Grounding-System-with-Hierarchical-Attention.git
cd Vision-Language-Grounding-System-with-Hierarchical-Attention

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Test that everything works:

```bash
# Run unit tests
pytest tests/test_models.py -v

# Test model creation
python -c "from models import create_vlg_model; from config import get_config; model = create_vlg_model(get_config()); print('Model created successfully!')"
```

## Train a Model

### Option 1: Quick Training (Dummy Data)

```bash
# Train with dummy data (for testing)
python train.py --no-wandb
```

This will:
- Create dummy COCO-like data automatically
- Train for 50 epochs
- Save checkpoints to `checkpoints/`
- Log to TensorBoard

### Option 2: Train with Real Data

```bash
# 1. Download COCO Captions dataset
mkdir -p data/coco
cd data/coco
# Download from https://cocodataset.org/#download

# 2. Update config.py to point to your data
# Edit: config.data.data_root = "data/coco"

# 3. Start training
python train.py
```

## Run Inference

```bash
# After training, run inference on an image
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/your/image.jpg \
    --task all \
    --output-dir results/inference
```

This will perform:
- Zero-shot classification
- Image-text retrieval
- Visual grounding
- Attention visualization

Results saved to `results/inference/`

## Evaluate Model

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --task retrieval \
    --split val
```

## Interactive Demo

```bash
# Launch Jupyter notebook
jupyter notebook demo.ipynb
```

## Common Use Cases

### 1. Zero-Shot Classification

```python
from inference import VLGInference

inference = VLGInference("checkpoints/best_model.pt")
predicted_class, rankings = inference.zero_shot_classify(
    image_path="cat.jpg",
    class_descriptions=["cat", "dog", "bird"]
)
print(f"Predicted: {predicted_class}")
```

### 2. Image-Text Retrieval

```python
results = inference.image_to_text_retrieval(
    query_image_path="image.jpg",
    candidate_texts=[
        "a cat sitting",
        "a dog playing",
        "a bird flying"
    ],
    top_k=3
)
```

### 3. Visual Grounding

```python
grounding_scores = inference.visual_grounding(
    image_path="image.jpg",
    text_query="the cat",
    save_path="grounding_result.png"
)
```

## Project Structure

```
├── models/              # Model implementations
│   ├── vision_transformer.py
│   ├── text_encoder.py
│   └── fusion_module.py
├── data/               # Dataset loading
├── utils/              # Visualization tools
├── train.py           # Training script
├── inference.py       # Inference demo
├── evaluate.py        # Evaluation script
└── config.py          # Configuration
```

## Training Tips

1. **Start Small**: Use batch_size=32 for testing
2. **Monitor Training**: Watch contrastive loss - should decrease
3. **Checkpoints**: Save every 5 epochs by default
4. **Visualization**: Check attention maps to verify learning

## Troubleshooting

### CUDA Out of Memory
```python
# In config.py, reduce batch size:
config.training.batch_size = 32
config.training.use_gradient_checkpointing = True
```

### Slow Training
```python
# Enable optimizations:
config.training.use_amp = True  # Mixed precision
# Use DataParallel if multiple GPUs available
```

### Poor Results
- Check data quality
- Increase training epochs
- Verify preprocessing
- Inspect attention visualizations

## Next Steps

1. ✅ Train on full COCO dataset
2. ✅ Fine-tune on your specific task
3. ✅ Experiment with hyperparameters
4. ✅ Try different architectures
5. ✅ Evaluate on benchmarks

## Resources

- **Documentation**: See README.md
- **Examples**: Run demo.ipynb
- **Tests**: Check tests/test_models.py
- **Issues**: GitHub Issues

## Getting Help

- Check README.md for detailed documentation
- Run tests: `pytest tests/ -v`
- Open an issue on GitHub

Happy experimenting! 🚀
