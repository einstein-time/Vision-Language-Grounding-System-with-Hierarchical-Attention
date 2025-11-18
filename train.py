"""
Training script for Vision-Language Grounding System
Implements training loop with contrastive learning and optional classification objectives
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import argparse
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from typing import Dict, Optional

from config import get_config, Config
from models import VisionLanguageGroundingModel, create_vlg_model, SimpleTokenizer
from data import create_dataloaders


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, config: Config) -> AdamW:
    """
    Create optimizer with weight decay applied selectively

    Args:
        model: Model to optimize
        config: Configuration object
    Returns:
        AdamW optimizer
    """
    # Separate parameters for weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', 'norm.bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.training.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return optimizer


def create_scheduler(optimizer, config: Config, num_training_steps: int):
    """
    Create learning rate scheduler with warmup

    Args:
        optimizer: Optimizer
        config: Configuration object
        num_training_steps: Total number of training steps
    Returns:
        Scheduler
    """
    warmup_steps = int(num_training_steps * config.training.warmup_epochs / config.training.num_epochs)

    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # Main scheduler
    if config.training.scheduler_type == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=config.training.min_lr
        )
    else:
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.training.min_lr / config.training.learning_rate,
            total_iters=num_training_steps - warmup_steps
        )

    # Combine warmup and main scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )

    return scheduler


def train_epoch(
    model: VisionLanguageGroundingModel,
    dataloader,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    config: Config,
    epoch: int,
    logger,
    global_step: int
) -> Dict[str, float]:
    """
    Train for one epoch

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        config: Configuration
        epoch: Current epoch
        logger: Logger (wandb or tensorboard)
        global_step: Global training step
    Returns:
        Dictionary of average losses
    """
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_i2t = 0.0
    total_t2i = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(config.training.device)
        input_ids = batch['input_ids'].to(config.training.device)
        attention_mask = batch['attention_mask'].to(config.training.device)

        # Forward pass with mixed precision
        if config.training.use_amp:
            with autocast():
                losses = model.compute_loss(
                    images,
                    input_ids,
                    attention_mask,
                    contrastive_weight=config.training.contrastive_weight,
                    classification_weight=config.training.classification_weight
                )
                loss = losses['total_loss']
        else:
            losses = model.compute_loss(
                images,
                input_ids,
                attention_mask,
                contrastive_weight=config.training.contrastive_weight,
                classification_weight=config.training.classification_weight
            )
            loss = losses['total_loss']

        # Backward pass
        if config.training.use_amp:
            scaler.scale(loss).backward()

            # Gradient clipping
            if config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient_clip
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Gradient clipping
            if config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient_clip
                )

            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        total_contrastive += losses['contrastive_loss'].item()
        total_i2t += losses['loss_i2t'].item()
        total_t2i += losses['loss_t2i'].item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}"
        })

        # Log to wandb/tensorboard
        if (global_step + 1) % config.training.log_every == 0:
            log_dict = {
                'train/loss': loss.item(),
                'train/contrastive_loss': losses['contrastive_loss'].item(),
                'train/loss_i2t': losses['loss_i2t'].item(),
                'train/loss_t2i': losses['loss_t2i'].item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/epoch': epoch,
                'train/step': global_step
            }

            if config.training.use_wandb:
                wandb.log(log_dict, step=global_step)
            else:
                for key, value in log_dict.items():
                    logger.add_scalar(key, value, global_step)

        global_step += 1

    # Calculate average losses
    num_steps = len(dataloader)
    avg_losses = {
        'loss': total_loss / num_steps,
        'contrastive_loss': total_contrastive / num_steps,
        'loss_i2t': total_i2t / num_steps,
        'loss_t2i': total_t2i / num_steps
    }

    return avg_losses, global_step


@torch.no_grad()
def validate(
    model: VisionLanguageGroundingModel,
    dataloader,
    config: Config,
    epoch: int,
    logger
) -> Dict[str, float]:
    """
    Validate model

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        config: Configuration
        epoch: Current epoch
        logger: Logger
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_contrastive = 0.0

    # For retrieval metrics
    all_image_embeddings = []
    all_text_embeddings = []

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['image'].to(config.training.device)
        input_ids = batch['input_ids'].to(config.training.device)
        attention_mask = batch['attention_mask'].to(config.training.device)

        # Compute loss
        losses = model.compute_loss(
            images,
            input_ids,
            attention_mask,
            contrastive_weight=config.training.contrastive_weight
        )

        total_loss += losses['total_loss'].item()
        total_contrastive += losses['contrastive_loss'].item()

        # Get embeddings for retrieval
        outputs = model(images, input_ids, attention_mask)
        all_image_embeddings.append(outputs['image_embeddings'].cpu())
        all_text_embeddings.append(outputs['text_embeddings'].cpu())

    # Calculate average losses
    num_steps = len(dataloader)
    avg_loss = total_loss / num_steps
    avg_contrastive = total_contrastive / num_steps

    # Calculate retrieval metrics (R@1, R@5, R@10)
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    # Image-to-text retrieval
    similarity_i2t = all_image_embeddings @ all_text_embeddings.T
    i2t_ranks = []
    for i in range(len(similarity_i2t)):
        # Get ranking of correct text (diagonal element)
        sorted_indices = torch.argsort(similarity_i2t[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)

    i2t_ranks = torch.tensor(i2t_ranks)
    i2t_r1 = (i2t_ranks < 1).float().mean().item() * 100
    i2t_r5 = (i2t_ranks < 5).float().mean().item() * 100
    i2t_r10 = (i2t_ranks < 10).float().mean().item() * 100

    # Text-to-image retrieval
    similarity_t2i = all_text_embeddings @ all_image_embeddings.T
    t2i_ranks = []
    for i in range(len(similarity_t2i)):
        sorted_indices = torch.argsort(similarity_t2i[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)

    t2i_ranks = torch.tensor(t2i_ranks)
    t2i_r1 = (t2i_ranks < 1).float().mean().item() * 100
    t2i_r5 = (t2i_ranks < 5).float().mean().item() * 100
    t2i_r10 = (t2i_ranks < 10).float().mean().item() * 100

    metrics = {
        'val/loss': avg_loss,
        'val/contrastive_loss': avg_contrastive,
        'val/i2t_r1': i2t_r1,
        'val/i2t_r5': i2t_r5,
        'val/i2t_r10': i2t_r10,
        'val/t2i_r1': t2i_r1,
        'val/t2i_r5': t2i_r5,
        'val/t2i_r10': t2i_r10,
        'val/mean_recall': (i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10) / 6
    }

    # Log metrics
    if config.training.use_wandb:
        wandb.log(metrics, step=epoch)
    else:
        for key, value in metrics.items():
            logger.add_scalar(key, value, epoch)

    return metrics


def save_checkpoint(
    model: VisionLanguageGroundingModel,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    config: Config,
    is_best: bool = False
):
    """Save model checkpoint"""
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': config
    }

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Save latest checkpoint
    checkpoint_path = os.path.join(
        config.training.checkpoint_dir,
        f'checkpoint_epoch_{epoch}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(config.training.checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Vision-Language Grounding Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    # Load configuration
    config = get_config()
    if args.config:
        # Load custom config if provided
        pass

    if args.no_wandb:
        config.training.use_wandb = False

    if args.resume:
        config.training.resume_from = args.resume

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Initialize logger
    if config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            entity=config.training.wandb_entity,
            config=config.__dict__
        )
        logger = wandb
    else:
        logger = SummaryWriter(log_dir='runs/vlg_training')

    # Create model
    print("Creating model...")
    model = create_vlg_model(config)
    model = model.to(config.training.device)

    # Compile model for PyTorch 2.0+
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(config, tokenizer)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    num_training_steps = len(train_loader) * config.training.num_epochs
    scheduler = create_scheduler(optimizer, config, num_training_steps)

    # Mixed precision scaler
    scaler = GradScaler() if config.training.use_amp else None

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    best_metric = 0.0

    if config.training.resume_from and os.path.exists(config.training.resume_from):
        print(f"Resuming from checkpoint: {config.training.resume_from}")
        checkpoint = torch.load(config.training.resume_from, map_location=config.training.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_metric = checkpoint['metrics'].get('val/mean_recall', 0.0)

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")

        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch, logger, global_step
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Contrastive: {train_metrics['contrastive_loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, config, epoch, logger)
        print(f"Val - Loss: {val_metrics['val/loss']:.4f}, "
              f"I2T R@1: {val_metrics['val/i2t_r1']:.2f}, "
              f"T2I R@1: {val_metrics['val/t2i_r1']:.2f}, "
              f"Mean Recall: {val_metrics['val/mean_recall']:.2f}")

        # Save checkpoint
        is_best = val_metrics['val/mean_recall'] > best_metric
        if is_best:
            best_metric = val_metrics['val/mean_recall']

        if (epoch + 1) % 5 == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, val_metrics, config, is_best
            )

    print("\nTraining completed!")
    if config.training.use_wandb:
        wandb.finish()
    else:
        logger.close()


if __name__ == "__main__":
    main()
