#!/usr/bin/env python3
"""
YOLOv8 Card Detection Training Script

Trains a YOLOv8-nano model to detect trading cards in images.
Outputs bounding boxes and orientation for downstream processing.
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict, output_dir: Path) -> None:
    """Train YOLOv8 detection model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model_name = config["model"]["name"]
    model = YOLO(f"{model_name}.pt")

    # Training parameters
    train_params = {
        "data": create_data_yaml(config, output_dir),
        "epochs": config["training"]["epochs"],
        "imgsz": config["data"]["image_size"],
        "batch": config["data"]["batch_size"],
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_epochs": config["training"]["warmup_epochs"],
        "patience": config["training"]["patience"],
        "project": str(output_dir),
        "name": "card_detector",
        "exist_ok": True,
        # Augmentation
        "hsv_h": config["augmentation"]["hsv_h"],
        "hsv_s": config["augmentation"]["hsv_s"],
        "hsv_v": config["augmentation"]["hsv_v"],
        "degrees": config["augmentation"]["degrees"],
        "translate": config["augmentation"]["translate"],
        "scale": config["augmentation"]["scale"],
        "shear": config["augmentation"]["shear"],
        "perspective": config["augmentation"]["perspective"],
        "flipud": config["augmentation"]["flipud"],
        "fliplr": config["augmentation"]["fliplr"],
        "mosaic": config["augmentation"]["mosaic"],
        "mixup": config["augmentation"]["mixup"],
    }

    # Train
    results = model.train(**train_params)

    # Validate
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return model


def create_data_yaml(config: dict, output_dir: Path) -> str:
    """Create YOLO data configuration file."""
    data_config = {
        "path": str(Path(config["data"]["train_path"]).parent),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "card"},
    }

    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f)

    return str(data_yaml_path)


def export_model(model: YOLO, config: dict, output_dir: Path) -> None:
    """Export model to mobile formats."""
    export_config = config["export"]

    for fmt in export_config["formats"]:
        print(f"\nExporting to {fmt}...")
        model.export(
            format=fmt,
            imgsz=export_config["input_size"],
            int8=export_config["quantization"] == "int8",
        )


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 card detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detection.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/detection",
        help="Output directory",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export model after training",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)

    print("Starting YOLOv8 card detection training...")
    model = train(config, output_dir)

    if args.export:
        export_model(model, config, output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
