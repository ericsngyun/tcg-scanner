#!/usr/bin/env python3
"""
FastViT Embedding Model Training Script

Enhanced training with:
- Heavy augmentation using Albumentations
- Multi-view dataset for single-sample-per-class metric learning
- Memory bank for cross-batch hard negative mining
- Recall@K validation metrics
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import albumentations as A
import lightning as L
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_heavy_augmentation(config: dict, split: str = "train") -> A.Compose:
    """
    Build heavy augmentation pipeline using Albumentations.
    Designed to simulate real-world camera conditions from single official images.
    """
    aug_config = config["augmentation"].get(split, {})
    transforms_list = []

    if split == "train":
        # Geometric transforms (simulate camera angles)
        if aug_config.get("perspective", {}).get("enabled", False):
            cfg = aug_config["perspective"]
            transforms_list.append(
                A.Perspective(
                    scale=tuple(cfg.get("scale", [0.05, 0.15])),
                    p=cfg.get("p", 0.5)
                )
            )

        if aug_config.get("affine", {}).get("enabled", False):
            cfg = aug_config["affine"]
            transforms_list.append(
                A.Affine(
                    rotate=tuple(cfg.get("rotate", [-20, 20])),
                    shear=tuple(cfg.get("shear", [-10, 10])),
                    scale=tuple(cfg.get("scale", [0.8, 1.2])),
                    p=cfg.get("p", 0.8)
                )
            )

        # Lighting simulation
        lighting_transforms = []
        if aug_config.get("brightness_contrast", {}).get("enabled", False):
            cfg = aug_config["brightness_contrast"]
            lighting_transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.get("brightness_limit", 0.4),
                    contrast_limit=cfg.get("contrast_limit", 0.4),
                    p=1.0
                )
            )
        if aug_config.get("hue_saturation", {}).get("enabled", False):
            cfg = aug_config["hue_saturation"]
            lighting_transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=cfg.get("hue_shift", 10),
                    sat_shift_limit=cfg.get("sat_shift", 30),
                    val_shift_limit=cfg.get("val_shift", 30),
                    p=1.0
                )
            )
        if aug_config.get("gamma", {}).get("enabled", False):
            cfg = aug_config["gamma"]
            lighting_transforms.append(
                A.RandomGamma(
                    gamma_limit=tuple(cfg.get("gamma_limit", [60, 140])),
                    p=1.0
                )
            )
        if lighting_transforms:
            transforms_list.append(
                A.OneOf(lighting_transforms, p=aug_config.get("brightness_contrast", {}).get("p", 0.7))
            )

        # Shadow simulation
        if aug_config.get("random_shadow", {}).get("enabled", False):
            cfg = aug_config["random_shadow"]
            transforms_list.append(
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=cfg.get("num_shadows", [1, 3])[0],
                    num_shadows_upper=cfg.get("num_shadows", [1, 3])[1],
                    p=cfg.get("p", 0.3)
                )
            )

        # Camera quality degradation (blur)
        blur_transforms = []
        if aug_config.get("blur", {}).get("enabled", False):
            cfg = aug_config["blur"]
            blur_limit = cfg.get("blur_limit", 7)
            blur_transforms.append(A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0))
            if cfg.get("motion_blur", True):
                blur_transforms.append(A.MotionBlur(blur_limit=blur_limit, p=1.0))
            blur_transforms.append(A.Defocus(radius=(3, 5), p=1.0))
            transforms_list.append(A.OneOf(blur_transforms, p=cfg.get("p", 0.4)))

        # Noise
        noise_transforms = []
        if aug_config.get("noise", {}).get("enabled", False):
            cfg = aug_config["noise"]
            var_limit = tuple(cfg.get("gauss_var", [10, 80]))
            noise_transforms.append(A.GaussNoise(var_limit=var_limit, p=1.0))
            if cfg.get("iso_noise", True):
                noise_transforms.append(
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.5),
                        p=1.0
                    )
                )
            noise_transforms.append(A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0))
            transforms_list.append(A.OneOf(noise_transforms, p=cfg.get("p", 0.5)))

        # JPEG compression
        if aug_config.get("jpeg_compression", {}).get("enabled", False):
            cfg = aug_config["jpeg_compression"]
            transforms_list.append(
                A.ImageCompression(
                    quality_lower=cfg.get("quality_lower", 60),
                    quality_upper=cfg.get("quality_upper", 100),
                    p=cfg.get("p", 0.5)
                )
            )

        # Occlusion simulation
        if aug_config.get("coarse_dropout", {}).get("enabled", False):
            cfg = aug_config["coarse_dropout"]
            transforms_list.append(
                A.CoarseDropout(
                    max_holes=cfg.get("max_holes", 3),
                    max_height=cfg.get("max_height", 40),
                    max_width=cfg.get("max_width", 40),
                    min_holes=cfg.get("min_holes", 1),
                    min_height=cfg.get("min_height", 10),
                    min_width=cfg.get("min_width", 10),
                    fill_value=0,
                    p=cfg.get("p", 0.3)
                )
            )

        # Color adjustments
        if aug_config.get("channel_dropout", {}).get("enabled", False):
            cfg = aug_config["channel_dropout"]
            transforms_list.append(
                A.ChannelDropout(
                    channel_drop_range=tuple(cfg.get("channel_drop_range", [1, 1])),
                    fill_value=0,
                    p=cfg.get("p", 0.1)
                )
            )

        if aug_config.get("rgb_shift", {}).get("enabled", False):
            cfg = aug_config["rgb_shift"]
            transforms_list.append(
                A.RGBShift(
                    r_shift_limit=cfg.get("r_shift", 20),
                    g_shift_limit=cfg.get("g_shift", 20),
                    b_shift_limit=cfg.get("b_shift", 20),
                    p=cfg.get("p", 0.3)
                )
            )

        # Resize and crop
        resize_size = aug_config.get("resize", 256)
        crop_size = aug_config.get("crop", 224)
        transforms_list.extend([
            A.Resize(resize_size, resize_size),
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=aug_config.get("horizontal_flip", 0.5)),
        ])

    else:  # Validation
        # Light augmentation for validation
        if aug_config.get("perspective", {}).get("enabled", False):
            cfg = aug_config["perspective"]
            transforms_list.append(
                A.Perspective(
                    scale=tuple(cfg.get("scale", [0.02, 0.08])),
                    p=cfg.get("p", 0.3)
                )
            )

        if aug_config.get("brightness_contrast", {}).get("enabled", False):
            cfg = aug_config["brightness_contrast"]
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.get("brightness_limit", 0.2),
                    contrast_limit=cfg.get("contrast_limit", 0.2),
                    p=cfg.get("p", 0.3)
                )
            )

        if aug_config.get("blur", {}).get("enabled", False):
            cfg = aug_config["blur"]
            transforms_list.append(
                A.GaussianBlur(
                    blur_limit=(3, cfg.get("blur_limit", 3)),
                    p=cfg.get("p", 0.2)
                )
            )

        # Resize and center crop
        resize_size = aug_config.get("resize", 256)
        crop_size = aug_config.get("center_crop", 224)
        transforms_list.extend([
            A.Resize(resize_size, resize_size),
            A.CenterCrop(crop_size, crop_size),
        ])

    # Normalize (always applied)
    normalize_cfg = aug_config.get("normalize", {})
    mean = normalize_cfg.get("mean", [0.485, 0.456, 0.406])
    std = normalize_cfg.get("std", [0.229, 0.224, 0.225])

    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


class MultiViewCardDataset(Dataset):
    """
    Dataset that generates multiple augmented views of each card.
    Essential for metric learning with single-sample-per-class.
    """

    def __init__(
        self,
        root_dir: str,
        transform: A.Compose,
        views_per_card: int = 4,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.views_per_card = views_per_card
        self.samples = []
        self.class_to_idx = {}

        # Load samples from directory structure: root/class_name/image.jpg
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.[jp][pn][g]"):
                    self.samples.append((img_path, idx))

        print(f"Loaded {len(self.samples)} cards, {len(self.class_to_idx)} classes")
        print(f"Generating {self.views_per_card} views per card = {len(self)} samples/epoch")

    def __len__(self) -> int:
        return len(self.samples) * self.views_per_card

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Map virtual index to real card
        card_idx = idx // self.views_per_card
        img_path, label = self.samples[card_idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply augmentation (different random transform each time)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


class MPerClassSampler(Sampler):
    """
    Sampler that ensures M samples per class in each batch.
    Essential for metric learning to have multiple views of same card in batch.
    """

    def __init__(
        self,
        dataset: MultiViewCardDataset,
        m_per_class: int = 4,
        batch_size: int = 64,
    ):
        self.dataset = dataset
        self.m_per_class = m_per_class
        self.batch_size = batch_size
        self.classes_per_batch = batch_size // m_per_class

        # Build class-to-indices mapping
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            # Add all view indices for this card
            for v in range(dataset.views_per_card):
                self.class_to_indices[label].append(idx * dataset.views_per_card + v)

        self.all_classes = list(self.class_to_indices.keys())

    def __iter__(self):
        # Shuffle classes
        classes = self.all_classes.copy()
        random.shuffle(classes)

        batch = []
        for cls in classes:
            indices = self.class_to_indices[cls].copy()
            random.shuffle(indices)

            # Take m samples from this class
            for idx in indices[:self.m_per_class]:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield from batch
                    batch = []

        # Yield remaining samples
        if batch:
            yield from batch

    def __len__(self):
        return len(self.dataset)


class MemoryBank:
    """
    Memory bank for storing embeddings across batches.
    Enables cross-batch hard negative mining.
    """

    def __init__(self, size: int, embedding_dim: int, momentum: float = 0.9):
        self.size = size
        self.embedding_dim = embedding_dim
        self.momentum = momentum

        # Initialize memory
        self.embeddings = torch.zeros(size, embedding_dim)
        self.labels = torch.zeros(size, dtype=torch.long)
        self.ptr = 0
        self.is_full = False

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Update memory bank with new embeddings."""
        batch_size = embeddings.size(0)
        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu()

        # Update with momentum or direct replacement
        end_ptr = self.ptr + batch_size

        if end_ptr <= self.size:
            if self.is_full:
                # Momentum update
                self.embeddings[self.ptr:end_ptr] = (
                    self.momentum * self.embeddings[self.ptr:end_ptr] +
                    (1 - self.momentum) * embeddings
                )
            else:
                self.embeddings[self.ptr:end_ptr] = embeddings
            self.labels[self.ptr:end_ptr] = labels
            self.ptr = end_ptr
        else:
            # Wrap around
            first_part = self.size - self.ptr
            self.embeddings[self.ptr:] = embeddings[:first_part]
            self.labels[self.ptr:] = labels[:first_part]
            self.embeddings[:batch_size - first_part] = embeddings[first_part:]
            self.labels[:batch_size - first_part] = labels[first_part:]
            self.ptr = batch_size - first_part
            self.is_full = True

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get all stored embeddings and labels."""
        if self.is_full:
            return self.embeddings, self.labels
        return self.embeddings[:self.ptr], self.labels[:self.ptr]


class CardEmbeddingModel(nn.Module):
    """FastViT-based embedding model for card recognition."""

    def __init__(
        self,
        backbone: str = "fastvit_t12",
        embedding_dim: int = 384,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feature_dim = features.shape[-1]

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class CardEmbeddingModule(L.LightningModule):
    """Lightning module for training card embedding model."""

    def __init__(self, config: dict, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Model
        model_config = config["model"]
        self.model = CardEmbeddingModel(
            backbone=model_config["backbone"],
            embedding_dim=model_config["embedding_dim"],
            pretrained=model_config["pretrained"],
            dropout=model_config["dropout"],
        )

        # Loss function (ArcFace)
        ml_config = config["metric_learning"]
        self.loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=model_config["embedding_dim"],
            margin=ml_config["arcface"]["margin"],
            scale=ml_config["arcface"]["scale"],
        )

        # Hard negative miner
        self.miner = miners.TripletMarginMiner(
            margin=ml_config["mining"]["margin"],
            type_of_triplets=ml_config["mining"]["type"],
        )

        # Memory bank for cross-batch mining
        mb_config = ml_config.get("memory_bank", {})
        if mb_config.get("enabled", False):
            self.memory_bank = MemoryBank(
                size=mb_config.get("size", 2048),
                embedding_dim=model_config["embedding_dim"],
                momentum=mb_config.get("momentum", 0.9),
            )
        else:
            self.memory_bank = None

        # Accuracy calculator for Recall@K
        self.accuracy_calculator = AccuracyCalculator(
            include=["precision_at_1", "r_precision", "mean_average_precision_at_r"],
            k=5,
        )

        # Validation storage
        self.val_embeddings = []
        self.val_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)

        # Mine hard triplets
        hard_pairs = self.miner(embeddings, labels)

        # Compute loss
        loss = self.loss_fn(embeddings, labels)

        # Update memory bank
        if self.memory_bank is not None:
            self.memory_bank.update(embeddings, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("hard_triplets", len(hard_pairs[0]) if hard_pairs else 0)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        loss = self.loss_fn(embeddings, labels)

        # Store for epoch-end metric calculation
        self.val_embeddings.append(embeddings.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"embeddings": embeddings, "labels": labels}

    def on_validation_epoch_end(self):
        if not self.val_embeddings:
            return

        # Concatenate all validation embeddings
        all_embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
        all_labels = torch.cat(self.val_labels, dim=0).numpy()

        # Calculate Recall@K metrics
        metrics = self._calculate_recall_at_k(all_embeddings, all_labels)

        self.log("val_recall_at_1", metrics["recall_at_1"], prog_bar=True)
        self.log("val_recall_at_5", metrics["recall_at_5"], prog_bar=True)
        self.log("val_map", metrics["map"], prog_bar=False)
        self.log("val_embedding_gap", metrics["embedding_gap"], prog_bar=False)

        # Clear storage
        self.val_embeddings = []
        self.val_labels = []

    def _calculate_recall_at_k(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        k_values: list = [1, 5]
    ) -> dict:
        """Calculate Recall@K metrics."""
        # Compute pairwise distances
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - embeddings[np.newaxis, :],
            axis=2
        )

        # Set diagonal to infinity (self-matches)
        np.fill_diagonal(distances, np.inf)

        results = {}

        for k in k_values:
            # Get k nearest neighbors
            nearest_indices = np.argsort(distances, axis=1)[:, :k]

            # Check if any of the k nearest neighbors have same label
            correct = 0
            for i in range(len(labels)):
                nearest_labels = labels[nearest_indices[i]]
                if labels[i] in nearest_labels:
                    correct += 1

            results[f"recall_at_{k}"] = correct / len(labels)

        # Calculate embedding gap (distance to nearest positive - distance to nearest negative)
        gaps = []
        for i in range(len(labels)):
            same_class_mask = labels == labels[i]
            same_class_mask[i] = False  # Exclude self

            if same_class_mask.sum() > 0:
                pos_dist = distances[i][same_class_mask].min()
            else:
                pos_dist = 0

            diff_class_mask = labels != labels[i]
            neg_dist = distances[i][diff_class_mask].min()

            gaps.append(neg_dist - pos_dist)

        results["embedding_gap"] = np.mean(gaps) if gaps else 0
        results["map"] = results.get("recall_at_1", 0)  # Simplified

        return results

    def configure_optimizers(self):
        train_config = self.config["training"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
        )

        # Cosine annealing with warmup
        warmup_epochs = train_config.get("warmup_epochs", 10)
        total_epochs = train_config["epochs"]

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=train_config["learning_rate"] * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Train FastViT embedding model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/embedding.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/embedding",
        help="Output directory",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled
    if args.wandb and WANDB_AVAILABLE:
        wandb_config = config.get("tracking", {}).get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "tcg-scanner"),
            tags=wandb_config.get("tags", []),
            config=config,
        )

    # Build transforms
    train_transform = build_heavy_augmentation(config, "train")
    val_transform = build_heavy_augmentation(config, "val")

    # Create datasets
    multi_view_config = config.get("multi_view", {})
    views_per_card = multi_view_config.get("views_per_card", 4) if multi_view_config.get("enabled", True) else 1

    train_dataset = MultiViewCardDataset(
        config["data"]["train_path"],
        transform=train_transform,
        views_per_card=views_per_card,
    )

    val_dataset = MultiViewCardDataset(
        config["data"]["val_path"],
        transform=val_transform,
        views_per_card=1,  # Single view for validation
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")

    # Create samplers
    classes_per_batch = multi_view_config.get("classes_per_batch", 16)
    batch_size = config["data"]["batch_size"]

    train_sampler = MPerClassSampler(
        train_dataset,
        m_per_class=views_per_card,
        batch_size=batch_size,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # Create model
    model = CardEmbeddingModule(config, num_classes=train_dataset.num_classes)

    # Callbacks
    checkpoint_config = config.get("tracking", {}).get("checkpointing", {})
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch}-{val_recall_at_1:.4f}",
            monitor=checkpoint_config.get("monitor", "val_recall_at_1"),
            mode=checkpoint_config.get("mode", "max"),
            save_top_k=checkpoint_config.get("save_top_k", 5),
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_recall_at_1",
            patience=15,
            mode="max",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if config["training"]["mixed_precision"] else 32,
        gradient_clip_val=config["training"]["gradient_clip"],
        callbacks=callbacks,
        default_root_dir=output_dir,
        log_every_n_steps=10,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    torch.save(model.model.state_dict(), output_dir / "final_model.pt")
    print(f"\nTraining complete! Model saved to {output_dir / 'final_model.pt'}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
