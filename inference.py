"""
Inference script for Vision-Language Grounding System
Demonstrates all downstream tasks: zero-shot classification, retrieval, visual grounding, attention visualization
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import os
from typing import List, Optional, Tuple
import numpy as np

from config import get_config, InferenceConfig
from models import VisionLanguageGroundingModel, SimpleTokenizer
from data import get_transforms
from utils.visualization import (
    visualize_attention_map,
    visualize_cross_attention,
    visualize_similarity_matrix,
    visualize_retrieval_results,
    visualize_grounding
)


class VLGInference:
    """
    Inference wrapper for Vision-Language Grounding Model

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = SimpleTokenizer()

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model from config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Use default config
            from config import get_config
            config = get_config()

        self.config = config

        # Create model
        from models import create_vlg_model
        self.model = create_vlg_model(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Create image transform
        self.transform = get_transforms(
            config.data.image_size,
            is_train=False,
            mean=config.data.normalize_mean,
            std=config.data.normalize_std
        )

        print("Model loaded successfully!")
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model parameters: {num_params:.2f}M")

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text descriptions"""
        input_ids = self.tokenizer.encode(texts, device=self.device)
        return input_ids

    @torch.no_grad()
    def zero_shot_classify(
        self,
        image_path: str,
        class_descriptions: List[str],
        template: str = "a photo of a {}"
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Zero-shot image classification

        Args:
            image_path: Path to image
            class_descriptions: List of class names
            template: Template for class descriptions
        Returns:
            predicted_class: Predicted class name
            rankings: List of (class, score) tuples
        """
        # Load image
        image = self.load_image(image_path)

        # Create text descriptions
        texts = [template.format(cls) for cls in class_descriptions]
        input_ids = self.encode_text(texts)

        # Get features
        image_features, _ = self.model.encode_image(image)
        text_features, _ = self.model.encode_text(input_ids)

        # Compute similarity
        similarity = self.model.fusion.get_image_text_similarity(
            image_features, text_features
        )

        # Get rankings
        scores = F.softmax(similarity[0], dim=0).cpu().numpy()
        rankings = [(class_descriptions[i], float(scores[i]))
                   for i in np.argsort(scores)[::-1]]

        predicted_class = rankings[0][0]

        return predicted_class, rankings

    @torch.no_grad()
    def image_to_text_retrieval(
        self,
        query_image_path: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float, int]]:
        """
        Image-to-text retrieval

        Args:
            query_image_path: Path to query image
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
        Returns:
            List of (text, score, index) tuples
        """
        # Load image
        image = self.load_image(query_image_path)

        # Encode texts
        input_ids = self.encode_text(candidate_texts)

        # Get features
        image_features, _ = self.model.encode_image(image)
        text_features, _ = self.model.encode_text(input_ids)

        # Compute similarity
        similarity = self.model.fusion.get_image_text_similarity(
            image_features, text_features
        )

        # Get top-k
        scores = similarity[0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (candidate_texts[i], float(scores[i]), i)
            for i in top_indices
        ]

        return results

    @torch.no_grad()
    def text_to_image_retrieval(
        self,
        query_text: str,
        candidate_image_paths: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float, int]]:
        """
        Text-to-image retrieval

        Args:
            query_text: Query text
            candidate_image_paths: List of candidate image paths
            top_k: Number of top results to return
        Returns:
            List of (image_path, score, index) tuples
        """
        # Encode query text
        input_ids = self.encode_text([query_text])
        text_features, _ = self.model.encode_text(input_ids)

        # Encode all images
        all_image_features = []
        for img_path in candidate_image_paths:
            image = self.load_image(img_path)
            image_features, _ = self.model.encode_image(image)
            all_image_features.append(image_features)

        all_image_features = torch.cat(all_image_features, dim=0)

        # Compute similarity
        similarity = self.model.fusion.get_image_text_similarity(
            all_image_features, text_features
        )

        # Get top-k
        scores = similarity[:, 0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (candidate_image_paths[i], float(scores[i]), i)
            for i in top_indices
        ]

        return results

    @torch.no_grad()
    def visual_grounding(
        self,
        image_path: str,
        text_query: str,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Visual grounding - locate image regions corresponding to text

        Args:
            image_path: Path to image
            text_query: Text query
            save_path: Path to save visualization
        Returns:
            Grounding scores for each patch
        """
        # Load image
        image = self.load_image(image_path)

        # Encode text
        input_ids = self.encode_text([text_query])

        # Get features with all tokens
        _, patch_features = self.model.encode_image(image, return_all_tokens=True)
        _, token_features = self.model.encode_text(input_ids, return_all_tokens=True)

        # Remove CLS token from patches
        patch_features_only = patch_features[:, 1:, :]

        # Get grounding scores through cross-attention
        grounded_features, grounding_attn = self.model.fusion.grounding_module(
            token_features,
            patch_features_only,
            return_attention=True
        )

        # Average attention over heads and tokens to get patch scores
        grounding_scores = grounding_attn.mean(dim=(1, 2))[0]  # [num_patches]

        # Visualize if save path provided
        if save_path:
            # Load original image for visualization
            orig_image = Image.open(image_path).convert('RGB')
            orig_image_tensor = self.transform(orig_image)

            visualize_grounding(
                orig_image_tensor,
                text_query,
                grounding_scores,
                save_path=save_path
            )
            print(f"Saved grounding visualization to {save_path}")

        return grounding_scores

    @torch.no_grad()
    def visualize_attention(
        self,
        image_path: str,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Visualize self-attention maps from vision encoder

        Args:
            image_path: Path to image
            head_idx: Which attention head to visualize (-1 for average)
            save_path: Path to save visualization
        Returns:
            Attention weights
        """
        # Load image
        image = self.load_image(image_path)

        # Get attention from vision encoder
        _, _, vision_attn = self.model.vision_encoder(
            image,
            return_all_tokens=True,
            return_attention=True
        )

        # Visualize
        if save_path:
            orig_image = Image.open(image_path).convert('RGB')
            orig_image_tensor = self.transform(orig_image)

            visualize_attention_map(
                orig_image_tensor,
                vision_attn[0],  # First sample
                head_idx=head_idx,
                save_path=save_path,
                title=f"Vision Attention (Head {head_idx})"
            )
            print(f"Saved attention visualization to {save_path}")

        return vision_attn

    @torch.no_grad()
    def visualize_cross_modal_attention(
        self,
        image_path: str,
        text_query: str,
        token_idx: int = 1,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Visualize cross-modal attention between text and image

        Args:
            image_path: Path to image
            text_query: Text query
            token_idx: Which text token to visualize
            head_idx: Which attention head to visualize
            save_path: Path to save visualization
        Returns:
            Cross-attention weights
        """
        # Load image and encode text
        image = self.load_image(image_path)
        input_ids = self.encode_text([text_query])

        # Get features
        _, patch_features = self.model.encode_image(image, return_all_tokens=True)
        _, token_features = self.model.encode_text(input_ids, return_all_tokens=True)

        # Remove CLS from patches
        patch_features_only = patch_features[:, 1:, :]

        # Get cross-attention
        _, cross_attn = self.model.fusion.grounding_module(
            token_features,
            patch_features_only,
            return_attention=True
        )

        # Visualize
        if save_path:
            orig_image = Image.open(image_path).convert('RGB')
            orig_image_tensor = self.transform(orig_image)

            visualize_cross_attention(
                orig_image_tensor,
                text_query,
                cross_attn[0],
                token_idx=token_idx,
                head_idx=head_idx,
                save_path=save_path
            )
            print(f"Saved cross-attention visualization to {save_path}")

        return cross_attn


def demo_zero_shot_classification(inference_engine: VLGInference, image_path: str):
    """Demo zero-shot classification"""
    print("\n" + "="*50)
    print("ZERO-SHOT CLASSIFICATION DEMO")
    print("="*50)

    # Define classes
    classes = ["cat", "dog", "bird", "car", "person", "bicycle", "tree", "building"]

    # Classify
    predicted_class, rankings = inference_engine.zero_shot_classify(
        image_path,
        classes
    )

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print("\nTop-5 rankings:")
    for i, (cls, score) in enumerate(rankings[:5]):
        print(f"  {i+1}. {cls}: {score:.4f}")


def demo_image_to_text_retrieval(inference_engine: VLGInference, image_path: str):
    """Demo image-to-text retrieval"""
    print("\n" + "="*50)
    print("IMAGE-TO-TEXT RETRIEVAL DEMO")
    print("="*50)

    # Define candidate texts
    candidate_texts = [
        "a cat sitting on a mat",
        "a dog playing in the park",
        "a bird flying in the sky",
        "a car on the street",
        "a person walking",
        "a bicycle near a tree",
        "a beautiful sunset",
        "a mountain landscape"
    ]

    # Retrieve
    results = inference_engine.image_to_text_retrieval(
        image_path,
        candidate_texts,
        top_k=5
    )

    print(f"\nQuery image: {image_path}")
    print("Top-5 matching texts:")
    for i, (text, score, idx) in enumerate(results):
        print(f"  {i+1}. [{score:.4f}] {text}")


def demo_visual_grounding(inference_engine: VLGInference, image_path: str, output_dir: str):
    """Demo visual grounding"""
    print("\n" + "="*50)
    print("VISUAL GROUNDING DEMO")
    print("="*50)

    # Define text queries
    queries = [
        "the cat",
        "the person",
        "the tree"
    ]

    for query in queries:
        save_path = os.path.join(output_dir, f"grounding_{query.replace(' ', '_')}.png")
        scores = inference_engine.visual_grounding(
            image_path,
            query,
            save_path=save_path
        )
        print(f"\nGrounding for '{query}':")
        print(f"  Max score: {scores.max().item():.4f}")
        print(f"  Min score: {scores.min().item():.4f}")


def demo_attention_visualization(inference_engine: VLGInference, image_path: str, output_dir: str):
    """Demo attention visualization"""
    print("\n" + "="*50)
    print("ATTENTION VISUALIZATION DEMO")
    print("="*50)

    # Visualize different attention heads
    for head_idx in [0, 3, 6, -1]:  # Different heads + average
        save_path = os.path.join(output_dir, f"attention_head_{head_idx}.png")
        inference_engine.visualize_attention(
            image_path,
            head_idx=head_idx,
            save_path=save_path
        )
        print(f"  Visualized head {head_idx}")


def main():
    parser = argparse.ArgumentParser(description='Vision-Language Grounding Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'classify', 'retrieval', 'grounding', 'attention'],
                       help='Task to perform')
    parser.add_argument('--output-dir', type=str, default='results/inference',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create inference engine
    print("Initializing inference engine...")
    inference_engine = VLGInference(args.checkpoint, device=args.device)

    # Run demos
    if args.task in ['all', 'classify']:
        demo_zero_shot_classification(inference_engine, args.image)

    if args.task in ['all', 'retrieval']:
        demo_image_to_text_retrieval(inference_engine, args.image)

    if args.task in ['all', 'grounding']:
        demo_visual_grounding(inference_engine, args.image, args.output_dir)

    if args.task in ['all', 'attention']:
        demo_attention_visualization(inference_engine, args.image, args.output_dir)

    print("\n" + "="*50)
    print("INFERENCE COMPLETED!")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
