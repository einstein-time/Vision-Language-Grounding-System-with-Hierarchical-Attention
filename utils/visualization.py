"""
Visualization utilities for attention maps and results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from typing import Optional, List, Tuple
import os


def visualize_attention_map(
    image: torch.Tensor,
    attention: torch.Tensor,
    head_idx: int = 0,
    patch_size: int = 16,
    save_path: Optional[str] = None,
    title: str = "Attention Map"
) -> np.ndarray:
    """
    Visualize attention map overlaid on image

    Args:
        image: Input image tensor [3, H, W] or [H, W, 3]
        attention: Attention weights [num_heads, num_patches+1, num_patches+1]
        head_idx: Which attention head to visualize (-1 for average)
        patch_size: Size of image patches
        save_path: Path to save visualization
        title: Plot title
    Returns:
        Visualization as numpy array
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    # Denormalize image (assuming CLIP normalization)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    # Get attention weights
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()

    # Select or average attention heads
    if head_idx == -1:
        attn_weights = attention.mean(axis=0)  # Average over heads
    else:
        attn_weights = attention[head_idx]

    # Get attention from CLS token to patches
    cls_attn = attn_weights[0, 1:]  # Remove CLS-to-CLS attention

    # Reshape to grid
    num_patches = int(np.sqrt(len(cls_attn)))
    attn_map = cls_attn.reshape(num_patches, num_patches)

    # Resize attention map to image size
    H, W = image.shape[:2]
    attn_map_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Normalize attention map
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / \
                       (attn_map_resized.max() - attn_map_resized.min() + 1e-8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Attention heatmap
    axes[1].imshow(attn_map_resized, cmap='jet')
    axes[1].set_title("Attention Heatmap")
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attn_map_resized, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    # Convert to numpy array
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return vis


def visualize_cross_attention(
    image: torch.Tensor,
    text: str,
    cross_attention: torch.Tensor,
    token_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize cross-attention between text and image patches

    Args:
        image: Input image tensor [3, H, W]
        text: Input text string
        cross_attention: Cross-attention weights [num_heads, text_len, num_patches]
        token_idx: Which text token to visualize
        head_idx: Which attention head to visualize
        save_path: Path to save visualization
    Returns:
        Visualization as numpy array
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    # Denormalize
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    # Get cross-attention weights
    if isinstance(cross_attention, torch.Tensor):
        cross_attention = cross_attention.cpu().numpy()

    # Select head and token
    attn_weights = cross_attention[head_idx, token_idx]  # [num_patches]

    # Reshape to grid
    num_patches = int(np.sqrt(len(attn_weights)))
    attn_map = attn_weights.reshape(num_patches, num_patches)

    # Resize to image size
    H, W = image.shape[:2]
    attn_map_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_CUBIC)
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / \
                       (attn_map_resized.max() - attn_map_resized.min() + 1e-8)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nText: '{text}'")
    axes[0].axis('off')

    axes[1].imshow(image)
    axes[1].imshow(attn_map_resized, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Cross-Attention for Token {token_idx}")
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return vis


def visualize_similarity_matrix(
    similarity_matrix: torch.Tensor,
    image_captions: Optional[List[str]] = None,
    text_queries: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Image-Text Similarity Matrix"
) -> np.ndarray:
    """
    Visualize similarity matrix between images and texts

    Args:
        similarity_matrix: Similarity scores [num_images, num_texts]
        image_captions: List of image captions
        text_queries: List of text queries
        save_path: Path to save visualization
        title: Plot title
    Returns:
        Visualization as numpy array
    """
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        xticklabels=text_queries if text_queries else range(similarity_matrix.shape[1]),
        yticklabels=image_captions if image_captions else range(similarity_matrix.shape[0]),
        cbar_kws={'label': 'Similarity Score'}
    )

    plt.title(title)
    plt.xlabel("Text Queries")
    plt.ylabel("Images")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def visualize_retrieval_results(
    query_image: torch.Tensor,
    retrieved_images: List[torch.Tensor],
    similarity_scores: List[float],
    captions: List[str],
    save_path: Optional[str] = None,
    mode: str = "image2text"
) -> None:
    """
    Visualize retrieval results

    Args:
        query_image: Query image
        retrieved_images: List of retrieved images
        similarity_scores: Similarity scores
        captions: Retrieved captions
        save_path: Path to save visualization
        mode: "image2text" or "text2image"
    """
    num_results = len(retrieved_images)
    fig, axes = plt.subplots(1, num_results + 1, figsize=(4 * (num_results + 1), 4))

    # Denormalize function
    def denorm(img):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = img * std + mean
        return np.clip(img, 0, 1)

    # Show query
    axes[0].imshow(denorm(query_image))
    axes[0].set_title("Query", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Show retrieved results
    for i in range(num_results):
        axes[i + 1].imshow(denorm(retrieved_images[i]))
        title = f"Rank {i+1}\nScore: {similarity_scores[i]:.3f}\n{captions[i][:30]}..."
        axes[i + 1].set_title(title, fontsize=10)
        axes[i + 1].axis('off')

    mode_str = "Image-to-Text" if mode == "image2text" else "Text-to-Image"
    plt.suptitle(f"{mode_str} Retrieval Results", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def visualize_grounding(
    image: torch.Tensor,
    text: str,
    grounding_scores: torch.Tensor,
    patch_size: int = 16,
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize visual grounding results

    Args:
        image: Input image [3, H, W]
        text: Query text
        grounding_scores: Grounding scores for each patch [num_patches]
        patch_size: Size of patches
        threshold: Threshold for grounding
        save_path: Path to save visualization
    Returns:
        Visualization as numpy array
    """
    # Convert image
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    # Denormalize
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    # Convert scores
    if isinstance(grounding_scores, torch.Tensor):
        grounding_scores = grounding_scores.cpu().numpy()

    # Reshape to grid
    num_patches = int(np.sqrt(len(grounding_scores)))
    score_map = grounding_scores.reshape(num_patches, num_patches)

    # Resize to image size
    H, W = image.shape[:2]
    score_map_resized = cv2.resize(score_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Normalize
    score_map_resized = (score_map_resized - score_map_resized.min()) / \
                        (score_map_resized.max() - score_map_resized.min() + 1e-8)

    # Create binary mask
    mask = score_map_resized > threshold

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Grounding heatmap
    axes[1].imshow(score_map_resized, cmap='hot')
    axes[1].set_title("Grounding Scores")
    axes[1].axis('off')

    # Overlay with mask
    axes[2].imshow(image)
    axes[2].imshow(score_map_resized, cmap='jet', alpha=0.5)
    axes[2].contour(mask, colors='lime', linewidths=2)
    axes[2].set_title(f"Grounded Regions\nText: '{text}'")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return vis


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training curves

    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_metrics: Additional training metrics
        val_metrics: Additional validation metrics
        save_path: Path to save plot
    """
    num_plots = 2 if (train_metrics is None and val_metrics is None) else 4
    fig, axes = plt.subplots(2, 2 if num_plots == 4 else 1, figsize=(12, 10))

    if num_plots == 2:
        axes = [axes[0], axes[1]]

    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metrics
    if train_metrics and val_metrics:
        # R@1 scores
        if 'i2t_r1' in val_metrics:
            axes[1].plot(epochs, val_metrics['i2t_r1'], 'g-', label='I2T R@1', linewidth=2)
            axes[1].plot(epochs, val_metrics['t2i_r1'], 'b-', label='T2I R@1', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Recall@1 (%)')
            axes[1].set_title('Retrieval Performance')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    # Create dummy data
    image = torch.randn(3, 224, 224)
    attention = torch.randn(12, 197, 197)  # 12 heads, 196 patches + 1 CLS
    attention = torch.softmax(attention, dim=-1)

    # Test attention visualization
    vis = visualize_attention_map(
        image,
        attention,
        head_idx=0,
        save_path="results/test_attention.png"
    )
    print(f"Attention visualization shape: {vis.shape}")

    # Test similarity matrix
    similarity = torch.randn(5, 5)
    visualize_similarity_matrix(
        similarity,
        save_path="results/test_similarity.png"
    )
    print("Similarity matrix visualized")

    print("Visualization tests completed!")
