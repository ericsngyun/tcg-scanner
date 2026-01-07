#!/usr/bin/env python3
"""
Card Identification Accuracy Testing Script

Tests the pure identification accuracy of the embedding model using pre-cropped card images.
Generates an HTML report showing input images, top-3 matches, and confidence scores.

Usage:
    python test_identification.py --test-folder path/to/test/images [--output report.html]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MatchResult:
    """Result from searching the index."""
    product_id: str
    name: str
    clean_name: str
    similarity: float
    rank: int
    image_path: Optional[str] = None


@dataclass
class TestResult:
    """Result from testing a single image."""
    input_path: str
    input_filename: str
    ground_truth_id: Optional[str]
    ground_truth_name: Optional[str]
    top_matches: list[MatchResult]
    inference_time_ms: float
    is_correct: Optional[bool] = None
    correct_rank: Optional[int] = None


class IdentificationTester:
    """Tests identification accuracy using the embedding model and index."""

    def __init__(
        self,
        model_path: str,
        index_path: str,
        product_ids_path: str,
        products_path: str,
        images_base_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.index_path = Path(index_path)
        self.product_ids_path = Path(product_ids_path)
        self.products_path = Path(products_path)
        self.images_base_path = Path(images_base_path) if images_base_path else None

        self.model = None
        self.index = None
        self.product_ids = None
        self.products_map = None
        self.device = None

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = 224

    def load(self):
        """Load model, index, and metadata."""
        print("Loading model and index...")

        # Try to load PyTorch model
        try:
            import torch
            import timm

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            # Load the trained model
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

                # Create model architecture
                self.model = timm.create_model(
                    "fastvit_t12",
                    pretrained=False,
                    num_classes=0,  # Remove classification head
                )

                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                # Remove prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        new_state_dict[k.replace("backbone.", "")] = v
                    elif k.startswith("model."):
                        new_state_dict[k.replace("model.", "")] = v
                    else:
                        new_state_dict[k] = v

                # Load only matching keys
                model_dict = self.model.state_dict()
                filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict, strict=False)

                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"Loaded model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}")

        except ImportError:
            raise RuntimeError("PyTorch and timm required. Install with: pip install torch timm")

        # Load Annoy index
        try:
            from annoy import AnnoyIndex

            if self.index_path.exists():
                # Get embedding dimension from config (384 for FastViT-T12)
                self.index = AnnoyIndex(384, "angular")
                self.index.load(str(self.index_path))
                print(f"Loaded index from {self.index_path} with {self.index.get_n_items()} items")
            else:
                raise FileNotFoundError(f"Index not found at {self.index_path}")

        except ImportError:
            raise RuntimeError("Annoy required. Install with: pip install annoy")

        # Load product IDs mapping
        if self.product_ids_path.exists():
            with open(self.product_ids_path) as f:
                self.product_ids = json.load(f)
            print(f"Loaded {len(self.product_ids)} product IDs")
        else:
            raise FileNotFoundError(f"Product IDs not found at {self.product_ids_path}")

        # Load products metadata
        if self.products_path.exists():
            with open(self.products_path) as f:
                products = json.load(f)
            self.products_map = {str(p["product_id"]): p for p in products}
            print(f"Loaded metadata for {len(self.products_map)} products")
        else:
            raise FileNotFoundError(f"Products not found at {self.products_path}")

    def preprocess_image(self, image_path: str) -> "torch.Tensor":
        """Preprocess image for model input."""
        import torch
        from torchvision import transforms

        # Load and convert to RGB
        img = Image.open(image_path).convert("RGB")

        # Resize and normalize
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
        ])

        tensor = transform(img)
        return tensor.unsqueeze(0).to(self.device)  # Add batch dimension

    def generate_embedding(self, image_path: str) -> tuple[np.ndarray, float]:
        """Generate embedding for an image. Returns embedding and inference time in ms."""
        import torch

        # Preprocess
        input_tensor = self.preprocess_image(image_path)

        # Inference
        start_time = time.perf_counter()
        with torch.no_grad():
            embedding = self.model(input_tensor)
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        # Normalize embedding
        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding, inference_time

    def search_similar(self, embedding: np.ndarray, top_k: int = 3) -> list[MatchResult]:
        """Search for similar cards in the index."""
        # Query the Annoy index
        indices, distances = self.index.get_nns_by_vector(
            embedding.tolist(), top_k, include_distances=True
        )

        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
            product_id = self.product_ids[idx]
            product = self.products_map.get(product_id, {})

            # Convert angular distance to similarity (cosine)
            # Annoy angular distance: sqrt(2 * (1 - cos(u, v)))
            # So: cos = 1 - (dist^2 / 2)
            similarity = 1 - (dist ** 2 / 2)

            # Find reference image path
            image_path = None
            if self.images_base_path:
                # Search for image in images folder
                for img_file in self.images_base_path.rglob(f"{product_id}_*.jpg"):
                    image_path = str(img_file)
                    break

            results.append(MatchResult(
                product_id=product_id,
                name=product.get("name", "Unknown"),
                clean_name=product.get("clean_name", "Unknown"),
                similarity=similarity,
                rank=rank,
                image_path=image_path,
            ))

        return results

    def extract_ground_truth(self, filename: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract ground truth from filename if present.
        Expected formats:
        - {product_id}_{name}.jpg
        - {product_id}.jpg
        """
        stem = Path(filename).stem

        # Try to extract product ID from filename
        parts = stem.split("_")
        if parts and parts[0].isdigit():
            product_id = parts[0]
            product = self.products_map.get(product_id, {})
            return product_id, product.get("name")

        return None, None

    def test_image(self, image_path: str, top_k: int = 3) -> TestResult:
        """Test identification for a single image."""
        path = Path(image_path)

        # Extract ground truth if available
        gt_id, gt_name = self.extract_ground_truth(path.name)

        # Generate embedding and search
        embedding, inference_time = self.generate_embedding(str(path))
        matches = self.search_similar(embedding, top_k=top_k)

        # Check if correct (if ground truth available)
        is_correct = None
        correct_rank = None
        if gt_id:
            for match in matches:
                if match.product_id == gt_id:
                    is_correct = True
                    correct_rank = match.rank
                    break
            if is_correct is None:
                is_correct = False

        return TestResult(
            input_path=str(path),
            input_filename=path.name,
            ground_truth_id=gt_id,
            ground_truth_name=gt_name,
            top_matches=matches,
            inference_time_ms=inference_time,
            is_correct=is_correct,
            correct_rank=correct_rank,
        )

    def test_folder(self, folder_path: str, top_k: int = 3) -> list[TestResult]:
        """Test all images in a folder."""
        folder = Path(folder_path)

        # Find all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = [
            f for f in folder.rglob("*")
            if f.suffix.lower() in image_extensions
        ]

        print(f"Found {len(image_files)} images to test")

        results = []
        for i, img_path in enumerate(image_files, start=1):
            print(f"Testing {i}/{len(image_files)}: {img_path.name}")
            result = self.test_image(str(img_path), top_k=top_k)
            results.append(result)

        return results


def generate_html_report(
    results: list[TestResult],
    output_path: str,
    title: str = "Card Identification Accuracy Report",
) -> None:
    """Generate an HTML report from test results."""

    # Calculate statistics
    total = len(results)
    with_ground_truth = [r for r in results if r.ground_truth_id]
    correct = [r for r in with_ground_truth if r.is_correct]

    accuracy = (len(correct) / len(with_ground_truth) * 100) if with_ground_truth else 0
    avg_inference_time = sum(r.inference_time_ms for r in results) / total if total else 0

    # Count by rank
    rank_counts = {1: 0, 2: 0, 3: 0}
    for r in correct:
        if r.correct_rank in rank_counts:
            rank_counts[r.correct_rank] += 1

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            color: #fff;
            font-size: 2rem;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: #1a1a1a;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #333;
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #4ade80;
        }}
        .stat-value.warning {{
            color: #fbbf24;
        }}
        .stat-value.danger {{
            color: #f87171;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        .results-grid {{
            display: grid;
            gap: 20px;
        }}
        .result-card {{
            background: #1a1a1a;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #333;
        }}
        .result-card.correct {{
            border-color: #4ade80;
        }}
        .result-card.incorrect {{
            border-color: #f87171;
        }}
        .result-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }}
        .result-filename {{
            font-weight: 600;
            color: #fff;
            font-size: 1.1rem;
        }}
        .result-status {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .result-status.correct {{
            background: #166534;
            color: #4ade80;
        }}
        .result-status.incorrect {{
            background: #7f1d1d;
            color: #f87171;
        }}
        .result-status.unknown {{
            background: #374151;
            color: #9ca3af;
        }}
        .result-content {{
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 20px;
        }}
        .input-section {{
            text-align: center;
        }}
        .input-section h4 {{
            color: #888;
            font-size: 0.8rem;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .input-image {{
            width: 180px;
            height: 252px;
            object-fit: contain;
            border-radius: 8px;
            background: #000;
            border: 1px solid #333;
        }}
        .ground-truth {{
            margin-top: 10px;
            font-size: 0.85rem;
            color: #888;
        }}
        .matches-section h4 {{
            color: #888;
            font-size: 0.8rem;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .matches-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        .match-card {{
            background: #252525;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .match-card.rank-1 {{
            background: linear-gradient(135deg, #1e3a1e 0%, #252525 100%);
            border: 1px solid #4ade80;
        }}
        .match-rank {{
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 8px;
        }}
        .match-image {{
            width: 120px;
            height: 168px;
            object-fit: contain;
            border-radius: 6px;
            background: #000;
            margin-bottom: 10px;
        }}
        .match-name {{
            font-size: 0.85rem;
            font-weight: 500;
            color: #fff;
            margin-bottom: 5px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .match-id {{
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 8px;
        }}
        .similarity-bar {{
            background: #333;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
            margin-bottom: 5px;
        }}
        .similarity-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .similarity-fill.high {{
            background: linear-gradient(90deg, #4ade80, #22c55e);
        }}
        .similarity-fill.medium {{
            background: linear-gradient(90deg, #fbbf24, #f59e0b);
        }}
        .similarity-fill.low {{
            background: linear-gradient(90deg, #f87171, #ef4444);
        }}
        .similarity-value {{
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .similarity-value.high {{
            color: #4ade80;
        }}
        .similarity-value.medium {{
            color: #fbbf24;
        }}
        .similarity-value.low {{
            color: #f87171;
        }}
        .inference-time {{
            text-align: right;
            font-size: 0.75rem;
            color: #666;
            margin-top: 10px;
        }}
        .filter-bar {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #888;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-btn:hover {{
            border-color: #4ade80;
            color: #4ade80;
        }}
        .filter-btn.active {{
            background: #166534;
            border-color: #4ade80;
            color: #4ade80;
        }}
        .no-image {{
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #1a1a1a;
            color: #666;
            font-size: 0.8rem;
        }}
        @media (max-width: 900px) {{
            .result-content {{
                grid-template-columns: 1fr;
            }}
            .matches-grid {{
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Images Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'danger' if accuracy < 80 else 'warning' if accuracy < 95 else ''}">{accuracy:.1f}%</div>
                <div class="stat-label">Top-3 Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{rank_counts.get(1, 0)}/{len(with_ground_truth)}</div>
                <div class="stat-label">Top-1 Correct</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_inference_time:.1f}ms</div>
                <div class="stat-label">Avg Inference Time</div>
            </div>
        </div>

        <div class="filter-bar">
            <button class="filter-btn active" onclick="filterResults('all')">All ({total})</button>
            <button class="filter-btn" onclick="filterResults('correct')">Correct ({len(correct)})</button>
            <button class="filter-btn" onclick="filterResults('incorrect')">Incorrect ({len(with_ground_truth) - len(correct)})</button>
            <button class="filter-btn" onclick="filterResults('unknown')">Unknown GT ({total - len(with_ground_truth)})</button>
        </div>

        <div class="results-grid">
"""

    for i, result in enumerate(results):
        # Determine status
        if result.is_correct is True:
            status_class = "correct"
            status_text = f"Correct (Rank {result.correct_rank})"
        elif result.is_correct is False:
            status_class = "incorrect"
            status_text = "Incorrect"
        else:
            status_class = "unknown"
            status_text = "No Ground Truth"

        # Build matches HTML
        matches_html = ""
        for match in result.top_matches:
            # Determine similarity class
            if match.similarity >= 0.85:
                sim_class = "high"
            elif match.similarity >= 0.7:
                sim_class = "medium"
            else:
                sim_class = "low"

            # Image handling
            if match.image_path:
                img_tag = f'<img class="match-image" src="file:///{match.image_path.replace(chr(92), "/")}" alt="{match.clean_name}" loading="lazy">'
            else:
                img_tag = '<div class="match-image"><div class="no-image">No Image</div></div>'

            rank_class = "rank-1" if match.rank == 1 else ""
            matches_html += f"""
            <div class="match-card {rank_class}">
                <div class="match-rank">#{match.rank}</div>
                {img_tag}
                <div class="match-name" title="{match.name}">{match.clean_name[:30]}{'...' if len(match.clean_name) > 30 else ''}</div>
                <div class="match-id">ID: {match.product_id}</div>
                <div class="similarity-bar">
                    <div class="similarity-fill {sim_class}" style="width: {match.similarity * 100}%"></div>
                </div>
                <div class="similarity-value {sim_class}">{match.similarity * 100:.1f}%</div>
            </div>
"""

        # Input image
        input_img_tag = f'<img class="input-image" src="file:///{result.input_path.replace(chr(92), "/")}" alt="Input">'

        # Ground truth info
        gt_html = ""
        if result.ground_truth_name:
            gt_html = f'<div class="ground-truth">Expected: {result.ground_truth_name}</div>'

        html += f"""
            <div class="result-card {status_class}" data-status="{status_class}">
                <div class="result-header">
                    <span class="result-filename">{result.input_filename}</span>
                    <span class="result-status {status_class}">{status_text}</span>
                </div>
                <div class="result-content">
                    <div class="input-section">
                        <h4>Input Image</h4>
                        {input_img_tag}
                        {gt_html}
                    </div>
                    <div class="matches-section">
                        <h4>Top 3 Matches</h4>
                        <div class="matches-grid">
                            {matches_html}
                        </div>
                    </div>
                </div>
                <div class="inference-time">Inference: {result.inference_time_ms:.1f}ms</div>
            </div>
"""

    html += """
        </div>
    </div>

    <script>
        function filterResults(filter) {
            // Update button states
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Filter cards
            document.querySelectorAll('.result-card').forEach(card => {
                const status = card.dataset.status;
                if (filter === 'all') {
                    card.style.display = 'block';
                } else if (filter === status) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

    # Write to file
    output = Path(output_path)
    output.write_text(html, encoding="utf-8")
    print(f"\nReport saved to: {output.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Test card identification accuracy using pre-cropped images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test images in a folder
    python test_identification.py --test-folder ./test_images

    # Specify custom output path
    python test_identification.py --test-folder ./test_images --output results.html

    # Use custom model and index paths
    python test_identification.py --test-folder ./test_images \\
        --model ../models/embedding/embedding_model.pt \\
        --index ../models/indices/riftbound.ann

Notes:
    - Test images should be pre-cropped to show just the card
    - If filenames start with product_id (e.g., 652771_CardName.jpg),
      accuracy metrics will be calculated
    - The report shows input image, top-3 matches with confidence scores
        """
    )

    parser.add_argument(
        "--test-folder",
        type=str,
        required=True,
        help="Path to folder containing test images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="identification_report.html",
        help="Output HTML report path (default: identification_report.html)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to embedding model (default: auto-detect)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to Annoy index (default: auto-detect)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top matches to show (default: 3)",
    )

    args = parser.parse_args()

    # Find paths relative to script location
    script_dir = Path(__file__).parent.parent

    # Default paths
    model_path = args.model or script_dir / "models" / "embedding" / "embedding_model.pt"
    index_path = args.index or script_dir / "models" / "indices" / "riftbound.ann"
    product_ids_path = script_dir / "data" / "embeddings" / "riftbound_product_ids.json"
    products_path = script_dir / "data" / "raw" / "riftbound" / "all_products.json"
    images_base_path = script_dir / "data" / "images" / "riftbound"

    # Validate test folder
    test_folder = Path(args.test_folder)
    if not test_folder.exists():
        print(f"Error: Test folder not found: {test_folder}")
        sys.exit(1)

    # Initialize tester
    tester = IdentificationTester(
        model_path=str(model_path),
        index_path=str(index_path),
        product_ids_path=str(product_ids_path),
        products_path=str(products_path),
        images_base_path=str(images_base_path),
    )

    # Load models and data
    try:
        tester.load()
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # Run tests
    print(f"\nTesting images in: {test_folder}")
    results = tester.test_folder(str(test_folder), top_k=args.top_k)

    # Generate report
    generate_html_report(
        results=results,
        output_path=args.output,
        title=f"Card Identification Report - {test_folder.name}",
    )

    # Print summary
    with_gt = [r for r in results if r.ground_truth_id]
    correct = [r for r in with_gt if r.is_correct]

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total images tested: {len(results)}")
    print(f"Images with ground truth: {len(with_gt)}")
    if with_gt:
        print(f"Top-3 Accuracy: {len(correct) / len(with_gt) * 100:.1f}%")
        rank_1 = sum(1 for r in correct if r.correct_rank == 1)
        print(f"Top-1 Accuracy: {rank_1 / len(with_gt) * 100:.1f}%")
    print(f"Avg inference time: {sum(r.inference_time_ms for r in results) / len(results):.1f}ms")
    print(f"\nReport saved to: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()
