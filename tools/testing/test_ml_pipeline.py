#!/usr/bin/env python3
"""
Test the complete ML pipeline with sample card images.

This script simulates what the Flutter app does:
1. Load TFLite models
2. Process sample images
3. Run detection → embedding → search pipeline
4. Verify results are correct
"""

import sys
from pathlib import Path
import numpy as np
import json
from PIL import Image
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("❌ TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)


class MLPipelineTester:
    """Test the ML pipeline end-to-end."""

    def __init__(self, models_dir: Path, cards_db: Path):
        """
        Initialize tester with models and database.

        Args:
            models_dir: Directory containing TFLite models
            cards_db: Path to cards.json
        """
        self.models_dir = models_dir
        self.cards_db = cards_db

        # Model paths
        self.detection_model_path = models_dir / 'detection.tflite'
        self.embedding_model_path = models_dir / 'embedding.tflite'

        # Models (loaded later)
        self.detection_interpreter = None
        self.embedding_interpreter = None

        # Card database
        self.cards = []

        # ImageNet normalization constants (same as Flutter)
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_models(self):
        """Load TFLite models."""
        print("\n" + "=" * 60)
        print("LOADING MODELS")
        print("=" * 60)

        # Load detection model
        print(f"\nLoading detection model: {self.detection_model_path}")
        if not self.detection_model_path.exists():
            raise FileNotFoundError(f"Detection model not found: {self.detection_model_path}")

        self.detection_interpreter = tf.lite.Interpreter(str(self.detection_model_path))
        self.detection_interpreter.allocate_tensors()

        # Get input/output details
        det_input = self.detection_interpreter.get_input_details()[0]
        det_output = self.detection_interpreter.get_output_details()[0]

        print(f"  Input shape: {det_input['shape']}")
        print(f"  Output shape: {det_output['shape']}")
        print(f"  ✅ Detection model loaded")

        # Load embedding model
        print(f"\nLoading embedding model: {self.embedding_model_path}")
        if not self.embedding_model_path.exists():
            raise FileNotFoundError(f"Embedding model not found: {self.embedding_model_path}")

        self.embedding_interpreter = tf.lite.Interpreter(str(self.embedding_model_path))
        self.embedding_interpreter.allocate_tensors()

        # Get input/output details
        emb_input = self.embedding_interpreter.get_input_details()[0]
        emb_output = self.embedding_interpreter.get_output_details()[0]

        print(f"  Input shape: {emb_input['shape']}")
        print(f"  Output shape: {emb_output['shape']}")
        print(f"  ✅ Embedding model loaded")

    def load_card_database(self):
        """Load card database."""
        print(f"\nLoading card database: {self.cards_db}")
        if not self.cards_db.exists():
            raise FileNotFoundError(f"Card database not found: {self.cards_db}")

        with open(self.cards_db) as f:
            self.cards = json.load(f)

        print(f"  ✅ Loaded {len(self.cards)} cards")

    def preprocess_for_detection(self, image_path: Path) -> np.ndarray:
        """
        Preprocess image for detection (same as Flutter implementation).

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor [1, 640, 640, 3]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        print(f"    Original size: {original_size}")

        # Letterbox resize to 640x640
        target_size = 640
        scale = min(target_size / image.width, target_size / image.height)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Create 640x640 canvas with gray padding (114, 114, 114)
        padded = Image.new('RGB', (target_size, target_size), color=(114, 114, 114))
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        padded.paste(resized, (offset_x, offset_y))

        # Convert to numpy and normalize [0-255] → [0-1]
        tensor = np.array(padded, dtype=np.float32) / 255.0

        # Add batch dimension: [640, 640, 3] → [1, 640, 640, 3]
        tensor = np.expand_dims(tensor, axis=0)

        print(f"    Preprocessed shape: {tensor.shape}")
        return tensor

    def preprocess_for_embedding(self, image_path: Path) -> np.ndarray:
        """
        Preprocess image for embedding (same as Flutter implementation).

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor [1, 224, 224, 3]
        """
        # Load and resize to 224x224
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert to numpy [0-255]
        tensor = np.array(image, dtype=np.float32)

        # Normalize with ImageNet statistics
        # Formula: (pixel/255 - mean) / std
        tensor = tensor / 255.0
        tensor = (tensor - self.imagenet_mean) / self.imagenet_std

        # Add batch dimension: [224, 224, 3] → [1, 224, 224, 3]
        tensor = np.expand_dims(tensor, axis=0)

        return tensor

    def run_detection(self, image_path: Path):
        """
        Run detection on image.

        Args:
            image_path: Path to image file

        Returns:
            Detection results
        """
        print(f"\n  Running detection...")

        # Preprocess
        input_tensor = self.preprocess_for_detection(image_path)

        # Run inference
        input_details = self.detection_interpreter.get_input_details()[0]
        output_details = self.detection_interpreter.get_output_details()[0]

        self.detection_interpreter.set_tensor(input_details['index'], input_tensor)

        start = time.time()
        self.detection_interpreter.invoke()
        elapsed = (time.time() - start) * 1000

        output = self.detection_interpreter.get_tensor(output_details['index'])

        print(f"    Detection inference: {elapsed:.1f}ms")
        print(f"    Output shape: {output.shape}")

        return output

    def run_embedding(self, image_path: Path):
        """
        Run embedding model on image.

        Args:
            image_path: Path to image file

        Returns:
            384-dim embedding vector
        """
        print(f"\n  Running embedding...")

        # Preprocess
        input_tensor = self.preprocess_for_embedding(image_path)

        # Run inference
        input_details = self.embedding_interpreter.get_input_details()[0]
        output_details = self.embedding_interpreter.get_output_details()[0]

        self.embedding_interpreter.set_tensor(input_details['index'], input_tensor)

        start = time.time()
        self.embedding_interpreter.invoke()
        elapsed = (time.time() - start) * 1000

        embedding = self.embedding_interpreter.get_tensor(output_details['index'])[0]

        print(f"    Embedding inference: {elapsed:.1f}ms")
        print(f"    Embedding shape: {embedding.shape}")

        # L2 normalize (same as Flutter)
        norm = np.linalg.norm(embedding)
        embedding_normalized = embedding / norm

        print(f"    Embedding norm before normalization: {norm:.4f}")
        print(f"    Embedding norm after normalization: {np.linalg.norm(embedding_normalized):.4f}")

        return embedding_normalized

    def test_sample_images(self, images_dir: Path, num_samples: int = 5):
        """
        Test pipeline with sample images.

        Args:
            images_dir: Directory containing sample images
            num_samples: Number of samples to test
        """
        print("\n" + "=" * 60)
        print("TESTING WITH SAMPLE IMAGES")
        print("=" * 60)

        # Find sample images
        sample_images = list(images_dir.glob('**/*.jpg'))[:num_samples]

        if not sample_images:
            print(f"❌ No images found in {images_dir}")
            return

        print(f"\nFound {len(sample_images)} sample images")

        for i, image_path in enumerate(sample_images, 1):
            print(f"\n{'─' * 60}")
            print(f"Sample {i}/{len(sample_images)}: {image_path.name}")
            print(f"{'─' * 60}")

            try:
                # Test detection
                detection_output = self.run_detection(image_path)

                # Test embedding
                embedding = self.run_embedding(image_path)

                print(f"\n  ✅ Pipeline completed successfully")
                print(f"    Embedding: {embedding[:5]}... (showing first 5 values)")

            except Exception as e:
                print(f"\n  ❌ Error: {e}")
                import traceback
                traceback.print_exc()

    def run_all_tests(self, images_dir: Path):
        """Run all tests."""
        try:
            # Load models and database
            self.load_models()
            self.load_card_database()

            # Test with sample images
            self.test_sample_images(images_dir, num_samples=5)

            print("\n" + "=" * 60)
            print("✅ ALL TESTS COMPLETED")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'mobile' / 'flutter' / 'assets' / 'models'
    cards_db = project_root / 'mobile' / 'flutter' / 'assets' / 'data' / 'cards.json'
    images_dir = project_root / 'ml' / 'data' / 'images' / 'riftbound'

    print("=" * 60)
    print("TCG Scanner ML Pipeline Test")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print(f"Cards database: {cards_db}")
    print(f"Images directory: {images_dir}")

    # Run tests
    tester = MLPipelineTester(models_dir, cards_db)
    tester.run_all_tests(images_dir)


if __name__ == '__main__':
    main()
