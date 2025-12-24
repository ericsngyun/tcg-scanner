#!/usr/bin/env python3
"""
FastViT Embedding Model Training Script

Trains a FastViT model with ArcFace loss for card recognition.
Outputs a model that generates embeddings for vector search.
"""

import argparse
from pathlib import Path
from typing import Tuple

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class CardDataset(Dataset):
    """Dataset for card images with labels."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Load samples from directory structure: root/class_name/image.jpg
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.[jp][pn][g]"):
                    self.samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class CardEmbeddingModel(nn.Module):
    """FastViT-based embedding model for card recognition."""

    def __init__(
        self,
        backbone: str = "fastvit_t8",
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
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
        # L2 normalize embeddings
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)

        # Mine hard triplets
        hard_pairs = self.miner(embeddings, labels)

        # Compute loss
        loss = self.loss_fn(embeddings, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        loss = self.loss_fn(embeddings, labels)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"embeddings": embeddings, "labels": labels}

    def configure_optimizers(self):
        train_config = self.config["training"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config["epochs"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def build_transforms(config: dict, split: str) -> transforms.Compose:
    """Build data transforms from config."""
    aug_config = config["augmentation"][split]
    transform_list = []

    for aug in aug_config:
        if "resize" in aug:
            transform_list.append(transforms.Resize(aug["resize"]["size"]))
        elif "random_crop" in aug:
            transform_list.append(transforms.RandomCrop(aug["random_crop"]["size"]))
        elif "center_crop" in aug:
            transform_list.append(transforms.CenterCrop(aug["center_crop"]["size"]))
        elif "horizontal_flip" in aug:
            transform_list.append(transforms.RandomHorizontalFlip(aug["horizontal_flip"]["p"]))
        elif "color_jitter" in aug:
            cj = aug["color_jitter"]
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0),
                    contrast=cj.get("contrast", 0),
                    saturation=cj.get("saturation", 0),
                )
            )
        elif "random_rotation" in aug:
            transform_list.append(transforms.RandomRotation(aug["random_rotation"]["degrees"]))
        elif "gaussian_blur" in aug:
            gb = aug["gaussian_blur"]
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(gb["kernel_size"])],
                    p=gb.get("p", 0.5),
                )
            )
        elif "normalize" in aug:
            transform_list.append(transforms.ToTensor())
            transform_list.append(
                transforms.Normalize(
                    mean=aug["normalize"]["mean"],
                    std=aug["normalize"]["std"],
                )
            )

    return transforms.Compose(transform_list)


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
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build transforms
    train_transform = build_transforms(config, "train")
    val_transform = build_transforms(config, "val")

    # Create datasets
    train_dataset = CardDataset(config["data"]["train_path"], transform=train_transform)
    val_dataset = CardDataset(config["data"]["val_path"], transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_to_idx)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # Create model
    model = CardEmbeddingModule(config, num_classes=len(train_dataset.class_to_idx))

    # Callbacks
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
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
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    torch.save(model.model.state_dict(), output_dir / "final_model.pt")
    print(f"\nTraining complete! Model saved to {output_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
