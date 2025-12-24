#!/usr/bin/env python3
"""
Dataset Preparation Script

Organizes downloaded card images into train/val splits
with proper directory structure for ML training.

Usage:
    python prepare_dataset.py --manifest ml/data/processed/riftbound/training_manifest.json
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def load_manifest(manifest_path: str) -> List[dict]:
    """Load training manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def verify_images(
    manifest: List[dict],
    base_path: Path,
    min_size: Tuple[int, int] = (100, 100),
) -> List[dict]:
    """Verify images exist and meet minimum quality."""
    valid = []
    invalid = []

    for item in tqdm(manifest, desc="Verifying images"):
        img_path = base_path / item["image_path"]

        if not img_path.exists():
            invalid.append((item["product_id"], "missing"))
            continue

        try:
            with Image.open(img_path) as img:
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    invalid.append((item["product_id"], f"too small: {img.size}"))
                    continue

                # Verify image is valid
                img.verify()

            valid.append(item)
        except Exception as e:
            invalid.append((item["product_id"], str(e)))

    print(f"Valid images: {len(valid)}, Invalid: {len(invalid)}")

    if invalid:
        print("Sample invalid images:")
        for pid, reason in invalid[:5]:
            print(f"  {pid}: {reason}")

    return valid


def create_class_directories(
    manifest: List[dict],
    output_dir: Path,
    use_product_id: bool = True,
) -> Dict[int, Path]:
    """Create directory structure for classification training."""
    class_dirs = {}

    for item in manifest:
        if use_product_id:
            # Use product_id as class (unique per card)
            class_name = str(item["product_id"])
        else:
            # Use clean_name as class (might have duplicates)
            class_name = item["clean_name"].replace(" ", "_").replace("/", "_")

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_dirs[item["product_id"]] = class_dir

    return class_dirs


def split_dataset(
    manifest: List[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split manifest into train and validation sets."""
    random.seed(seed)

    # Group by product_id to ensure same card doesn't appear in both splits
    by_product = defaultdict(list)
    for item in manifest:
        by_product[item["product_id"]].append(item)

    product_ids = list(by_product.keys())
    random.shuffle(product_ids)

    split_idx = int(len(product_ids) * train_ratio)
    train_ids = set(product_ids[:split_idx])
    val_ids = set(product_ids[split_idx:])

    train = [item for item in manifest if item["product_id"] in train_ids]
    val = [item for item in manifest if item["product_id"] in val_ids]

    return train, val


def copy_images_to_split(
    items: List[dict],
    base_path: Path,
    output_dir: Path,
    augment_copies: int = 1,
) -> None:
    """Copy images to train/val directory structure."""
    for item in tqdm(items, desc=f"Copying to {output_dir.name}"):
        src_path = base_path / item["image_path"]
        class_dir = output_dir / str(item["product_id"])
        class_dir.mkdir(parents=True, exist_ok=True)

        # Copy original
        dst_path = class_dir / src_path.name
        shutil.copy2(src_path, dst_path)


def create_embedding_structure(
    manifest: List[dict],
    base_path: Path,
    output_dir: Path,
) -> None:
    """Create directory structure optimized for embedding training."""
    # For embedding/metric learning, we organize by product_id
    # Each product_id folder contains all images of that card

    output_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(manifest, desc="Organizing for embeddings"):
        src_path = base_path / item["image_path"]
        class_dir = output_dir / str(item["product_id"])
        class_dir.mkdir(exist_ok=True)

        dst_path = class_dir / src_path.name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)


def save_class_mapping(manifest: List[dict], output_path: Path) -> None:
    """Save product_id to card name mapping."""
    mapping = {}
    for item in manifest:
        mapping[str(item["product_id"])] = {
            "name": item["name"],
            "clean_name": item["clean_name"],
            "group_name": item["group_name"],
            "rarity": item.get("rarity"),
            "number": item.get("number"),
            "card_type": item.get("card_type"),
            "tags": item.get("tags", []),
        }

    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Class mapping saved to {output_path}")


def generate_stats(manifest: List[dict]) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"Total cards: {len(manifest)}")

    # By group
    by_group = defaultdict(int)
    for item in manifest:
        by_group[item["group_name"]] += 1

    print("\nBy set:")
    for group, count in sorted(by_group.items()):
        print(f"  {group}: {count}")

    # By rarity
    by_rarity = defaultdict(int)
    for item in manifest:
        rarity = item.get("rarity") or "Unknown"
        by_rarity[rarity] += 1

    print("\nBy rarity:")
    for rarity, count in sorted(by_rarity.items()):
        print(f"  {rarity}: {count}")

    # By card type
    by_type = defaultdict(int)
    for item in manifest:
        card_type = item.get("card_type") or "Unknown"
        by_type[card_type] += 1

    print("\nBy card type:")
    for card_type, count in sorted(by_type.items()):
        print(f"  {card_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for ML training")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to training manifest JSON",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="ml/data",
        help="Base path for image paths in manifest",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/data/dataset/riftbound",
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip image verification",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't copy files",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)
    output_dir = Path(args.output)

    # Load manifest
    print("Loading manifest...")
    manifest = load_manifest(args.manifest)
    print(f"Loaded {len(manifest)} items")

    # Verify images
    if not args.skip_verify:
        manifest = verify_images(manifest, base_path)

    # Print stats
    generate_stats(manifest)

    if args.stats_only:
        return

    # Split dataset
    print("\nSplitting dataset...")
    train_items, val_items = split_dataset(manifest, args.train_ratio)
    print(f"Train: {len(train_items)}, Val: {len(val_items)}")

    # Create directories and copy images
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    print("\nPreparing training set...")
    copy_images_to_split(train_items, base_path, train_dir)

    print("\nPreparing validation set...")
    copy_images_to_split(val_items, base_path, val_dir)

    # Save class mapping
    save_class_mapping(manifest, output_dir / "class_mapping.json")

    # Save split info
    split_info = {
        "total": len(manifest),
        "train": len(train_items),
        "val": len(val_items),
        "train_ratio": args.train_ratio,
        "train_product_ids": [item["product_id"] for item in train_items],
        "val_product_ids": [item["product_id"] for item in val_items],
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")


if __name__ == "__main__":
    main()
