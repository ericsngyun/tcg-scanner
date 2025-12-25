#!/usr/bin/env python3
"""
Synthetic Detection Data Generator

Generates YOLO-format detection training data by compositing card images
onto various backgrounds with realistic transforms and annotations.

Usage:
    python generate_synthetic_data.py --cards-dir data/images/riftbound \
                                       --backgrounds-dir data/backgrounds \
                                       --output-dir data/detection \
                                       --num-train 10000 \
                                       --num-val 2000
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class CardPlacement:
    """Represents a card placement in a scene."""
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    angle: float
    corners: np.ndarray  # 4 corners after transform


@dataclass
class SceneConfig:
    """Configuration for scene generation."""
    output_size: tuple[int, int] = (640, 640)
    min_cards: int = 1
    max_cards: int = 4
    min_card_scale: float = 0.15  # Relative to output size
    max_card_scale: float = 0.55
    min_rotation: float = -45.0
    max_rotation: float = 45.0
    perspective_intensity: float = 0.15
    shadow_enabled: bool = True
    shadow_offset: tuple[int, int] = (8, 8)
    shadow_blur: int = 15
    shadow_opacity: float = 0.4
    allow_overlap: bool = True
    max_overlap_ratio: float = 0.3
    negative_sample_ratio: float = 0.05  # 5% scenes with no cards


class SyntheticSceneGenerator:
    """Generates synthetic detection scenes with card compositing."""

    def __init__(
        self,
        cards_dir: Path,
        backgrounds_dir: Optional[Path] = None,
        config: Optional[SceneConfig] = None,
    ):
        self.cards_dir = Path(cards_dir)
        self.backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else None
        self.config = config or SceneConfig()

        # Load card images
        self.card_paths = self._load_card_paths()
        print(f"Loaded {len(self.card_paths)} card images")

        # Load background images (or generate solid colors if none provided)
        self.background_paths = self._load_background_paths()
        print(f"Loaded {len(self.background_paths)} backgrounds")

    def _load_card_paths(self) -> list[Path]:
        """Load all card image paths."""
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        paths = []
        for ext in extensions:
            paths.extend(self.cards_dir.rglob(f"*{ext}"))
        return sorted(paths)

    def _load_background_paths(self) -> list[Path]:
        """Load background image paths or return empty for generated backgrounds."""
        if self.backgrounds_dir is None or not self.backgrounds_dir.exists():
            return []

        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        paths = []
        for ext in extensions:
            paths.extend(self.backgrounds_dir.rglob(f"*{ext}"))
        return sorted(paths)

    def _get_background(self) -> np.ndarray:
        """Get a background image (loaded or generated)."""
        h, w = self.config.output_size

        if self.background_paths:
            # Load random background
            bg_path = random.choice(self.background_paths)
            bg = cv2.imread(str(bg_path))
            if bg is not None:
                bg = cv2.resize(bg, (w, h))
                return bg

        # Generate synthetic background
        bg_type = random.choice(["solid", "gradient", "noise", "texture"])

        if bg_type == "solid":
            # Random solid color (wood tones, playmats, tables)
            colors = [
                (139, 90, 43),    # Brown wood
                (169, 130, 91),   # Light wood
                (64, 64, 64),     # Dark gray
                (45, 45, 45),     # Charcoal
                (20, 60, 20),     # Dark green (playmat)
                (30, 30, 80),     # Dark blue (playmat)
                (80, 80, 80),     # Medium gray
                (200, 200, 200),  # Light gray
                (240, 230, 220),  # Off-white
            ]
            color = random.choice(colors)
            # Add slight variation
            color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
            bg = np.full((h, w, 3), color, dtype=np.uint8)

        elif bg_type == "gradient":
            # Vertical or horizontal gradient
            color1 = np.array([random.randint(30, 200) for _ in range(3)])
            color2 = np.array([random.randint(30, 200) for _ in range(3)])

            if random.random() > 0.5:
                # Vertical gradient
                gradient = np.linspace(0, 1, h).reshape(-1, 1, 1)
            else:
                # Horizontal gradient
                gradient = np.linspace(0, 1, w).reshape(1, -1, 1)

            bg = (color1 * (1 - gradient) + color2 * gradient).astype(np.uint8)
            bg = np.broadcast_to(bg, (h, w, 3)).copy()

        elif bg_type == "noise":
            # Noisy texture (simulates fabric/paper)
            base_color = np.array([random.randint(60, 200) for _ in range(3)])
            noise = np.random.normal(0, 15, (h, w, 3))
            bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)

        else:  # texture
            # Simple procedural texture
            base_color = random.randint(80, 180)
            bg = np.full((h, w, 3), base_color, dtype=np.uint8)

            # Add some random lines/scratches
            for _ in range(random.randint(5, 20)):
                pt1 = (random.randint(0, w), random.randint(0, h))
                pt2 = (random.randint(0, w), random.randint(0, h))
                color = base_color + random.randint(-30, 30)
                cv2.line(bg, pt1, pt2, (color, color, color), random.randint(1, 3))

        return bg

    def _load_card(self, card_path: Path) -> np.ndarray:
        """Load a card image with alpha channel."""
        img = cv2.imread(str(card_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to load card: {card_path}")

        # Ensure 4 channels (add alpha if needed)
        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            img = np.dstack([img, alpha])

        return img

    def _apply_perspective(
        self,
        image: np.ndarray,
        intensity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random perspective transform to image."""
        h, w = image.shape[:2]

        # Original corners
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # Randomly perturb corners
        max_shift = int(min(w, h) * intensity)
        dst_pts = src_pts.copy()
        for i in range(4):
            dst_pts[i, 0] += random.randint(-max_shift, max_shift)
            dst_pts[i, 1] += random.randint(-max_shift, max_shift)

        # Ensure we don't flip the card inside out
        # Keep corners in roughly the right positions
        dst_pts[0] = np.clip(dst_pts[0], [0, 0], [w//2, h//2])
        dst_pts[1] = np.clip(dst_pts[1], [w//2, 0], [w, h//2])
        dst_pts[2] = np.clip(dst_pts[2], [w//2, h//2], [w, h])
        dst_pts[3] = np.clip(dst_pts[3], [0, h//2], [w//2, h])

        # Compute transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply transform
        warped = cv2.warpPerspective(
            image, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return warped, dst_pts

    def _rotate_image(
        self,
        image: np.ndarray,
        angle: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate image and return rotated corners."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix for new size
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # Rotate image
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Calculate rotated corners
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])
        rotated_corners = (M @ corners_h.T).T

        return rotated, rotated_corners

    def _add_shadow(
        self,
        scene: np.ndarray,
        card_mask: np.ndarray,
        x: int,
        y: int
    ) -> np.ndarray:
        """Add drop shadow beneath card."""
        h, w = scene.shape[:2]
        offset_x, offset_y = self.config.shadow_offset
        blur = self.config.shadow_blur
        opacity = self.config.shadow_opacity

        # Create shadow mask
        shadow_mask = np.zeros((h, w), dtype=np.float32)

        # Place card mask at shadow position
        card_h, card_w = card_mask.shape[:2]
        sx, sy = x + offset_x, y + offset_y

        # Calculate valid regions
        src_x1 = max(0, -sx)
        src_y1 = max(0, -sy)
        src_x2 = min(card_w, w - sx)
        src_y2 = min(card_h, h - sy)

        dst_x1 = max(0, sx)
        dst_y1 = max(0, sy)
        dst_x2 = min(w, sx + card_w)
        dst_y2 = min(h, sy + card_h)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            shadow_mask[dst_y1:dst_y2, dst_x1:dst_x2] = \
                card_mask[src_y1:src_y2, src_x1:src_x2]

        # Blur the shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask, (blur * 2 + 1, blur * 2 + 1), 0)

        # Apply shadow to scene
        shadow_mask = (shadow_mask * opacity).reshape(h, w, 1)
        scene = (scene.astype(np.float32) * (1 - shadow_mask)).astype(np.uint8)

        return scene

    def _composite_card(
        self,
        scene: np.ndarray,
        card: np.ndarray,
        x: int,
        y: int
    ) -> np.ndarray:
        """Composite card onto scene with alpha blending."""
        h, w = scene.shape[:2]
        card_h, card_w = card.shape[:2]

        # Calculate valid regions
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(card_w, w - x)
        src_y2 = min(card_h, h - y)

        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = min(w, x + card_w)
        dst_y2 = min(h, y + card_h)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return scene

        # Extract regions
        card_region = card[src_y1:src_y2, src_x1:src_x2]
        scene_region = scene[dst_y1:dst_y2, dst_x1:dst_x2]

        # Alpha blend
        alpha = card_region[:, :, 3:4].astype(np.float32) / 255.0
        blended = (
            card_region[:, :, :3].astype(np.float32) * alpha +
            scene_region.astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

        scene[dst_y1:dst_y2, dst_x1:dst_x2] = blended

        return scene

    def _get_bounding_box(
        self,
        corners: np.ndarray,
        x_offset: int,
        y_offset: int,
        scene_size: tuple[int, int]
    ) -> tuple[float, float, float, float]:
        """Get YOLO-format bounding box from corners."""
        h, w = scene_size

        # Offset corners to scene coordinates
        corners = corners.copy()
        corners[:, 0] += x_offset
        corners[:, 1] += y_offset

        # Get axis-aligned bounding box
        x_min = max(0, corners[:, 0].min())
        x_max = min(w, corners[:, 0].max())
        y_min = max(0, corners[:, 1].min())
        y_max = min(h, corners[:, 1].max())

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        center_x = (x_min + x_max) / 2 / w
        center_y = (y_min + y_max) / 2 / h
        box_w = (x_max - x_min) / w
        box_h = (y_max - y_min) / h

        return center_x, center_y, box_w, box_h

    def _check_overlap(
        self,
        new_box: tuple[float, float, float, float],
        existing_boxes: list[tuple[float, float, float, float]]
    ) -> bool:
        """Check if new box overlaps too much with existing boxes."""
        if not self.config.allow_overlap:
            for box in existing_boxes:
                if self._compute_iou(new_box, box) > 0:
                    return True
            return False

        for box in existing_boxes:
            iou = self._compute_iou(new_box, box)
            if iou > self.config.max_overlap_ratio:
                return True
        return False

    def _compute_iou(
        self,
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float]
    ) -> float:
        """Compute IoU between two YOLO-format boxes."""
        # Convert to corner format
        def to_corners(box):
            cx, cy, w, h = box
            return cx - w/2, cy - h/2, cx + w/2, cy + h/2

        x1_1, y1_1, x2_1, y2_1 = to_corners(box1)
        x1_2, y1_2, x2_2, y2_2 = to_corners(box2)

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def generate_scene(self) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
        """
        Generate a single synthetic scene.

        Returns:
            image: The composited scene (BGR)
            annotations: List of (class_id, center_x, center_y, width, height) in YOLO format
        """
        h, w = self.config.output_size

        # Get background
        scene = self._get_background()
        annotations = []

        # Decide if this is a negative sample (no cards)
        if random.random() < self.config.negative_sample_ratio:
            return scene, annotations

        # Number of cards in this scene
        num_cards = random.randint(self.config.min_cards, self.config.max_cards)

        existing_boxes = []
        max_attempts = 50

        for _ in range(num_cards):
            # Select random card
            card_path = random.choice(self.card_paths)

            for attempt in range(max_attempts):
                try:
                    # Load and transform card
                    card = self._load_card(card_path)

                    # Random scale
                    scale = random.uniform(
                        self.config.min_card_scale,
                        self.config.max_card_scale
                    )
                    target_w = int(w * scale)
                    aspect = card.shape[0] / card.shape[1]
                    target_h = int(target_w * aspect)

                    card = cv2.resize(card, (target_w, target_h))

                    # Apply perspective
                    card, _ = self._apply_perspective(
                        card,
                        self.config.perspective_intensity
                    )

                    # Rotate
                    angle = random.uniform(
                        self.config.min_rotation,
                        self.config.max_rotation
                    )
                    card, corners = self._rotate_image(card, angle)

                    # Random position
                    card_h, card_w = card.shape[:2]
                    margin = int(min(card_w, card_h) * 0.1)
                    x = random.randint(-card_w // 4, w - card_w * 3 // 4)
                    y = random.randint(-card_h // 4, h - card_h * 3 // 4)

                    # Get bounding box
                    bbox = self._get_bounding_box(corners, x, y, (h, w))

                    # Check if box is valid (at least 10% visible)
                    if bbox[2] < 0.05 or bbox[3] < 0.05:
                        continue

                    # Check overlap with existing cards
                    if self._check_overlap(bbox, existing_boxes):
                        continue

                    # Add shadow
                    if self.config.shadow_enabled:
                        card_mask = card[:, :, 3].astype(np.float32) / 255.0
                        scene = self._add_shadow(scene, card_mask, x, y)

                    # Composite card
                    scene = self._composite_card(scene, card, x, y)

                    # Record annotation
                    annotations.append((0, *bbox))
                    existing_boxes.append(bbox)

                    break

                except Exception as e:
                    continue

        return scene, annotations

    def generate_dataset(
        self,
        output_dir: Path,
        num_train: int = 10000,
        num_val: int = 2000,
        seed: int = 42
    ) -> dict:
        """Generate complete YOLO-format dataset."""
        random.seed(seed)
        np.random.seed(seed)

        output_dir = Path(output_dir)
        stats = {
            "train_images": 0,
            "val_images": 0,
            "train_cards": 0,
            "val_cards": 0,
        }

        for split, num_images in [("train", num_train), ("val", num_val)]:
            images_dir = output_dir / split / "images"
            labels_dir = output_dir / split / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nGenerating {split} set ({num_images} images)...")

            for i in tqdm(range(num_images), desc=split):
                # Generate scene
                image, annotations = self.generate_scene()

                # Save image
                image_path = images_dir / f"{i:06d}.jpg"
                cv2.imwrite(str(image_path), image)

                # Save labels
                label_path = labels_dir / f"{i:06d}.txt"
                with open(label_path, "w") as f:
                    for ann in annotations:
                        class_id, cx, cy, bw, bh = ann
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                stats[f"{split}_images"] += 1
                stats[f"{split}_cards"] += len(annotations)

        # Create data.yaml for YOLO
        data_yaml = {
            "path": str(output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "names": {0: "card"},
            "nc": 1,
        }

        with open(output_dir / "data.yaml", "w") as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)

        # Save generation stats
        stats_path = output_dir / "generation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nDataset generation complete!")
        print(f"  Train: {stats['train_images']} images, {stats['train_cards']} card instances")
        print(f"  Val: {stats['val_images']} images, {stats['val_cards']} card instances")
        print(f"  Output: {output_dir}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic detection training data"
    )
    parser.add_argument(
        "--cards-dir",
        type=str,
        default="ml/data/images/riftbound",
        help="Directory containing card images",
    )
    parser.add_argument(
        "--backgrounds-dir",
        type=str,
        default=None,
        help="Directory containing background images (optional, will generate if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/data/detection",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=10000,
        help="Number of training images to generate",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=2000,
        help="Number of validation images to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min-cards",
        type=int,
        default=1,
        help="Minimum cards per scene",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=4,
        help="Maximum cards per scene",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.15,
        help="Minimum card scale (relative to output size)",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=0.55,
        help="Maximum card scale (relative to output size)",
    )
    args = parser.parse_args()

    # Configure scene generation
    config = SceneConfig(
        min_cards=args.min_cards,
        max_cards=args.max_cards,
        min_card_scale=args.min_scale,
        max_card_scale=args.max_scale,
    )

    # Create generator
    generator = SyntheticSceneGenerator(
        cards_dir=Path(args.cards_dir),
        backgrounds_dir=Path(args.backgrounds_dir) if args.backgrounds_dir else None,
        config=config,
    )

    # Generate dataset
    generator.generate_dataset(
        output_dir=Path(args.output_dir),
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
