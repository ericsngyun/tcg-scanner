#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Embedder

Generates robust embeddings by averaging multiple augmented views of the same image.
This improves accuracy by making predictions more stable across lighting/angle variations.
"""

from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image


class TTAAugmentations:
    """Pre-defined TTA augmentation sets."""

    @staticmethod
    def get_augmentation(name: str, image_size: int = 224) -> A.Compose:
        """Get a specific augmentation by name."""
        base_transforms = [
            A.Resize(256, 256),
            A.CenterCrop(image_size, image_size),
        ]

        normalize = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

        if name == "original":
            return A.Compose(base_transforms + normalize)

        elif name == "horizontal_flip":
            return A.Compose(
                base_transforms + [A.HorizontalFlip(p=1.0)] + normalize
            )

        elif name == "rotation_10":
            return A.Compose([
                A.Resize(256, 256),
                A.Rotate(limit=(10, 10), p=1.0),
                A.CenterCrop(image_size, image_size),
            ] + normalize)

        elif name == "rotation_neg10":
            return A.Compose([
                A.Resize(256, 256),
                A.Rotate(limit=(-10, -10), p=1.0),
                A.CenterCrop(image_size, image_size),
            ] + normalize)

        elif name == "color_jitter":
            return A.Compose(
                base_transforms + [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=1.0
                    )
                ] + normalize
            )

        elif name == "slight_blur":
            return A.Compose(
                base_transforms + [A.GaussianBlur(blur_limit=(3, 3), p=1.0)] + normalize
            )

        elif name == "scale_up":
            return A.Compose([
                A.Resize(288, 288),
                A.CenterCrop(image_size, image_size),
            ] + normalize)

        elif name == "scale_down":
            return A.Compose([
                A.Resize(200, 200),
                A.Resize(image_size, image_size),
            ] + normalize)

        elif name == "perspective":
            return A.Compose([
                A.Resize(256, 256),
                A.Perspective(scale=(0.02, 0.05), p=1.0),
                A.CenterCrop(image_size, image_size),
            ] + normalize)

        else:
            raise ValueError(f"Unknown augmentation: {name}")


class TTAEmbedder:
    """
    Test-Time Augmentation embedder for improved accuracy.
    Generates multiple augmented views and averages their embeddings.
    """

    DEFAULT_AUGMENTATIONS = [
        "original",
        "horizontal_flip",
        "rotation_10",
        "color_jitter",
        "slight_blur",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        augmentation_names: Optional[list[str]] = None,
        image_size: int = 224,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize TTA embedder.

        Args:
            model: The embedding model (should output normalized embeddings)
            augmentation_names: List of augmentation names to use
            image_size: Input image size for the model
            device: Device to run inference on
        """
        self.model = model
        self.image_size = image_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Build augmentation transforms
        self.augmentation_names = augmentation_names or self.DEFAULT_AUGMENTATIONS
        self.augmentations = [
            TTAAugmentations.get_augmentation(name, image_size)
            for name in self.augmentation_names
        ]

        print(f"TTA Embedder initialized with {len(self.augmentations)} augmentations:")
        for name in self.augmentation_names:
            print(f"  - {name}")

    def get_embedding(
        self,
        image: Image.Image | np.ndarray,
        return_all: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Get embedding for an image using TTA.

        Args:
            image: PIL Image or numpy array (RGB)
            return_all: If True, return all individual embeddings as well

        Returns:
            Averaged and normalized embedding (and optionally all individual embeddings)
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        embeddings = []

        with torch.no_grad():
            for aug in self.augmentations:
                # Apply augmentation
                augmented = aug(image=image)
                tensor = augmented["image"].unsqueeze(0).to(self.device)

                # Get embedding
                embedding = self.model(tensor)
                embeddings.append(embedding.cpu())

        # Stack and average
        all_embeddings = torch.cat(embeddings, dim=0)
        mean_embedding = all_embeddings.mean(dim=0, keepdim=True)

        # Re-normalize
        mean_embedding = F.normalize(mean_embedding, p=2, dim=1)

        if return_all:
            return mean_embedding.numpy()[0], all_embeddings.numpy()
        return mean_embedding.numpy()[0]

    def get_embeddings_batch(
        self,
        images: list[Image.Image | np.ndarray],
    ) -> np.ndarray:
        """
        Get embeddings for a batch of images using TTA.

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            Array of averaged and normalized embeddings (N x embedding_dim)
        """
        all_embeddings = []

        for image in images:
            embedding = self.get_embedding(image)
            all_embeddings.append(embedding)

        return np.stack(all_embeddings, axis=0)

    def get_confidence(
        self,
        distances: np.ndarray,
        k: int = 5,
    ) -> float:
        """
        Calculate calibrated confidence score from search distances.

        Args:
            distances: Array of distances to k nearest neighbors
            k: Number of neighbors to consider

        Returns:
            Confidence score between 0 and 1
        """
        if len(distances) == 0:
            return 0.0

        top1_dist = distances[0]
        gap = distances[1] - distances[0] if len(distances) > 1 else 0.5
        variance = np.var(distances[:k]) if len(distances) >= k else 0.1

        # Confidence components:
        # 1. Lower distance = higher confidence (exponential decay)
        dist_conf = np.exp(-top1_dist * 3)

        # 2. Larger gap = more confident (clear winner)
        gap_conf = min(gap / 0.2, 1.0)

        # 3. Low variance in top-k suggests similar matches (ambiguity)
        var_penalty = max(0, 1 - variance * 5)

        # Combine with weights
        confidence = dist_conf * 0.5 + gap_conf * 0.3 + var_penalty * 0.2

        return min(max(confidence, 0.0), 1.0)


class MultiScaleEmbedder:
    """
    Multi-scale embedder for capturing features at different resolutions.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scales: list[float] = [0.8, 1.0, 1.2],
        weights: Optional[list[float]] = None,
        image_size: int = 224,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize multi-scale embedder.

        Args:
            model: The embedding model
            scales: Scale factors for multi-scale inference
            weights: Weights for combining scale embeddings (default: [0.25, 0.5, 0.25])
            image_size: Base image size
            device: Device to run inference on
        """
        self.model = model
        self.scales = scales
        self.weights = weights or [0.25, 0.5, 0.25]
        self.image_size = image_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert len(self.scales) == len(self.weights), "Scales and weights must have same length"
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1"

        self.model = self.model.to(self.device)
        self.model.eval()

        # Build transforms for each scale
        self.transforms = []
        for scale in scales:
            size = int(256 * scale)
            self.transforms.append(
                A.Compose([
                    A.Resize(size, size),
                    A.CenterCrop(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            )

    def get_embedding(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Get multi-scale embedding for an image."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        embeddings = []

        with torch.no_grad():
            for transform in self.transforms:
                augmented = transform(image=image)
                tensor = augmented["image"].unsqueeze(0).to(self.device)
                embedding = self.model(tensor)
                embeddings.append(embedding.cpu())

        # Weighted average
        combined = sum(
            w * e for w, e in zip(self.weights, embeddings)
        )

        # Re-normalize
        combined = F.normalize(combined, p=2, dim=1)

        return combined.numpy()[0]


class EnsembleEmbedder:
    """
    Ensemble embedder combining multiple models or techniques.
    """

    def __init__(
        self,
        embedders: list[TTAEmbedder | MultiScaleEmbedder],
        weights: Optional[list[float]] = None,
    ):
        """
        Initialize ensemble embedder.

        Args:
            embedders: List of embedders to combine
            weights: Weights for each embedder (default: equal weights)
        """
        self.embedders = embedders
        self.weights = weights or [1.0 / len(embedders)] * len(embedders)

        assert len(self.embedders) == len(self.weights)
        assert abs(sum(self.weights) - 1.0) < 1e-6

    def get_embedding(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Get ensemble embedding for an image."""
        embeddings = []

        for embedder in self.embedders:
            embedding = embedder.get_embedding(image)
            embeddings.append(embedding)

        # Weighted average
        combined = sum(
            w * e for w, e in zip(self.weights, embeddings)
        )

        # Re-normalize
        combined = combined / np.linalg.norm(combined)

        return combined


def load_embedder(
    model_path: str,
    backbone: str = "fastvit_t12",
    embedding_dim: int = 384,
    use_tta: bool = True,
    device: Optional[torch.device] = None,
) -> TTAEmbedder | torch.nn.Module:
    """
    Load embedding model from checkpoint and wrap with TTA if requested.

    Args:
        model_path: Path to model checkpoint (.pt file)
        backbone: Model backbone name
        embedding_dim: Embedding dimension
        use_tta: Whether to use TTA wrapper
        device: Device to run inference on

    Returns:
        TTAEmbedder or raw model
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from training.embedding.train import CardEmbeddingModel

    # Load model
    model = CardEmbeddingModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        pretrained=False,
    )

    # Load weights
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if use_tta:
        return TTAEmbedder(model, device=device)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Example usage
    print("TTA Embedder Example")
    print("=" * 50)

    # This is just for demonstration - in practice you'd load a trained model
    print("\nAvailable TTA augmentations:")
    for name in TTAEmbedder.DEFAULT_AUGMENTATIONS:
        print(f"  - {name}")

    print("\nTo use the TTA embedder:")
    print("""
    from ml.inference.tta_embedder import load_embedder, TTAEmbedder

    # Load with TTA (recommended for highest accuracy)
    embedder = load_embedder(
        model_path="models/embedding/final_model.pt",
        backbone="fastvit_t12",
        embedding_dim=384,
        use_tta=True,
    )

    # Get embedding for an image
    from PIL import Image
    image = Image.open("path/to/card.jpg")
    embedding = embedder.get_embedding(image)

    # Search in vector index
    from annoy import AnnoyIndex
    index = AnnoyIndex(384, 'angular')
    index.load("models/indices/riftbound.ann")

    indices, distances = index.get_nns_by_vector(
        embedding, 5, include_distances=True
    )

    # Get confidence
    confidence = embedder.get_confidence(distances)
    print(f"Top match confidence: {confidence:.2%}")
    """)
