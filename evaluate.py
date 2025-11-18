"""
Evaluation script for Vision-Language Grounding System
Evaluates model on various metrics: retrieval (R@K), zero-shot classification accuracy
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
from typing import Dict, List
import numpy as np

from config import get_config
from models import create_vlg_model, SimpleTokenizer
from data import create_dataloaders


@torch.no_grad()
def evaluate_retrieval(
    model,
    dataloader,
    device: str,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate image-text retrieval performance

    Args:
        model: VLG model
        dataloader: Evaluation dataloader
        device: Device to use
        k_values: K values for Recall@K metric
    Returns:
        Dictionary of retrieval metrics
    """
    model.eval()

    # Collect all embeddings
    all_image_embeddings = []
    all_text_embeddings = []
    all_image_ids = []

    print("Extracting embeddings...")
    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Get embeddings
        outputs = model(images, input_ids, attention_mask)

        all_image_embeddings.append(outputs['image_embeddings'].cpu())
        all_text_embeddings.append(outputs['text_embeddings'].cpu())
        all_image_ids.extend(batch['image_id'])

    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    print(f"Total samples: {len(all_image_embeddings)}")

    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = all_image_embeddings @ all_text_embeddings.T

    # Image-to-text retrieval
    print("Evaluating image-to-text retrieval...")
    i2t_ranks = []
    for i in range(len(similarity_matrix)):
        # Get ranking of correct text (diagonal element)
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)

    i2t_ranks = torch.tensor(i2t_ranks)

    # Text-to-image retrieval
    print("Evaluating text-to-image retrieval...")
    t2i_ranks = []
    for i in range(len(similarity_matrix)):
        sorted_indices = torch.argsort(similarity_matrix[:, i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)

    t2i_ranks = torch.tensor(t2i_ranks)

    # Compute Recall@K metrics
    metrics = {}
    for k in k_values:
        i2t_recall = (i2t_ranks < k).float().mean().item() * 100
        t2i_recall = (t2i_ranks < k).float().mean().item() * 100

        metrics[f'i2t_r{k}'] = i2t_recall
        metrics[f't2i_r{k}'] = t2i_recall

    # Mean rank
    metrics['i2t_mean_rank'] = i2t_ranks.float().mean().item()
    metrics['t2i_mean_rank'] = t2i_ranks.float().mean().item()

    # Median rank
    metrics['i2t_median_rank'] = i2t_ranks.float().median().item()
    metrics['t2i_median_rank'] = t2i_ranks.float().median().item()

    # Mean reciprocal rank
    metrics['i2t_mrr'] = (1.0 / (i2t_ranks.float() + 1)).mean().item()
    metrics['t2i_mrr'] = (1.0 / (t2i_ranks.float() + 1)).mean().item()

    return metrics


@torch.no_grad()
def evaluate_zero_shot_classification(
    model,
    dataloader,
    class_names: List[str],
    tokenizer,
    device: str,
    template: str = "a photo of a {}"
) -> Dict[str, float]:
    """
    Evaluate zero-shot classification accuracy

    Args:
        model: VLG model
        dataloader: Evaluation dataloader (must have 'label' field)
        class_names: List of class names
        tokenizer: Text tokenizer
        device: Device to use
        template: Template for class descriptions
    Returns:
        Dictionary of classification metrics
    """
    model.eval()

    # Encode class descriptions
    class_descriptions = [template.format(cls) for cls in class_names]
    class_input_ids = tokenizer.encode(class_descriptions, device=device)

    # Get text embeddings for classes
    class_features, _ = model.encode_text(class_input_ids)
    class_embeddings = model.fusion.text_projection(class_features)
    class_embeddings = F.normalize(class_embeddings, dim=-1)

    # Evaluate
    all_predictions = []
    all_labels = []
    all_scores = []

    print("Evaluating zero-shot classification...")
    for batch in tqdm(dataloader):
        images = batch['image'].to(device)

        # Get image embeddings
        image_features, _ = model.encode_image(images)
        image_embeddings = model.fusion.vision_projection(image_features)
        image_embeddings = F.normalize(image_embeddings, dim=-1)

        # Compute similarity with class embeddings
        similarity = image_embeddings @ class_embeddings.T

        # Get predictions
        predictions = similarity.argmax(dim=1).cpu().numpy()
        scores = F.softmax(similarity, dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_scores.append(scores)

        # Get labels if available
        if 'label' in batch:
            all_labels.extend(batch['label'].numpy())

    all_predictions = np.array(all_predictions)
    all_scores = np.concatenate(all_scores, axis=0)

    metrics = {}

    # Accuracy (if labels available)
    if len(all_labels) > 0:
        all_labels = np.array(all_labels)
        accuracy = (all_predictions == all_labels).mean() * 100
        metrics['accuracy'] = accuracy

        # Top-5 accuracy
        top5_preds = np.argsort(all_scores, axis=1)[:, -5:]
        top5_accuracy = np.mean([label in top5_preds[i] for i, label in enumerate(all_labels)]) * 100
        metrics['top5_accuracy'] = top5_accuracy

        # Per-class accuracy
        per_class_acc = {}
        for i, cls in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[cls] = (all_predictions[mask] == i).mean() * 100

        metrics['per_class_accuracy'] = per_class_acc

    # Confidence
    max_scores = all_scores.max(axis=1)
    metrics['mean_confidence'] = max_scores.mean()
    metrics['median_confidence'] = np.median(max_scores)

    return metrics


@torch.no_grad()
def evaluate_visual_grounding(
    model,
    dataloader,
    device: str,
    iou_thresholds: List[float] = [0.5, 0.7, 0.9]
) -> Dict[str, float]:
    """
    Evaluate visual grounding performance
    Note: Requires bounding box annotations in dataset

    Args:
        model: VLG model
        dataloader: Evaluation dataloader (must have bounding boxes)
        device: Device to use
        iou_thresholds: IoU thresholds for evaluation
    Returns:
        Dictionary of grounding metrics
    """
    model.eval()

    # This is a placeholder for visual grounding evaluation
    # In practice, you would need bounding box annotations

    print("Visual grounding evaluation requires bounding box annotations.")
    print("Skipping for now...")

    return {
        'grounding_accuracy@0.5': 0.0,
        'grounding_accuracy@0.7': 0.0,
        'grounding_accuracy@0.9': 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Vision-Language Grounding Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--task', type=str, default='retrieval',
                       choices=['retrieval', 'classification', 'grounding', 'all'],
                       help='Evaluation task')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output', type=str, default='results/evaluation.json',
                       help='Path to save evaluation results')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    args = parser.parse_args()

    # Load configuration
    config = get_config()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    if 'config' in checkpoint:
        config = checkpoint['config']

    model = create_vlg_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Create dataloader
    print(f"Loading {args.split} dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader

    # Run evaluation
    all_metrics = {}

    if args.task in ['retrieval', 'all']:
        print("\n" + "="*50)
        print("EVALUATING RETRIEVAL")
        print("="*50)
        retrieval_metrics = evaluate_retrieval(
            model, dataloader, args.device, k_values=[1, 5, 10, 20]
        )
        all_metrics.update(retrieval_metrics)

        print("\nRetrieval Results:")
        print(f"  Image-to-Text:")
        for k in [1, 5, 10, 20]:
            print(f"    R@{k}: {retrieval_metrics[f'i2t_r{k}']:.2f}%")
        print(f"  Text-to-Image:")
        for k in [1, 5, 10, 20]:
            print(f"    R@{k}: {retrieval_metrics[f't2i_r{k}']:.2f}%")
        print(f"  Mean Rank (I2T): {retrieval_metrics['i2t_mean_rank']:.2f}")
        print(f"  Mean Rank (T2I): {retrieval_metrics['t2i_mean_rank']:.2f}")

    if args.task in ['classification', 'all']:
        print("\n" + "="*50)
        print("EVALUATING ZERO-SHOT CLASSIFICATION")
        print("="*50)

        # Define some example classes (modify as needed)
        class_names = ["cat", "dog", "bird", "car", "person"]

        classification_metrics = evaluate_zero_shot_classification(
            model, dataloader, class_names, tokenizer, args.device
        )
        all_metrics.update(classification_metrics)

        print("\nClassification Results:")
        if 'accuracy' in classification_metrics:
            print(f"  Accuracy: {classification_metrics['accuracy']:.2f}%")
            print(f"  Top-5 Accuracy: {classification_metrics['top5_accuracy']:.2f}%")

    if args.task in ['grounding', 'all']:
        print("\n" + "="*50)
        print("EVALUATING VISUAL GROUNDING")
        print("="*50)
        grounding_metrics = evaluate_visual_grounding(model, dataloader, args.device)
        all_metrics.update(grounding_metrics)

    # Save results
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print("\n" + "="*50)
    print("EVALUATION COMPLETED!")
    print("="*50)


if __name__ == "__main__":
    main()
