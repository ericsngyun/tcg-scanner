#!/usr/bin/env python3
"""
Convert PyTorch models to TFLite format for Flutter mobile app.

This script converts:
1. YOLOv8 detection model (best.pt) -> detection.tflite
2. FastViT embedding model (embedding_model.pt) -> embedding.tflite

Usage:
    python ml/scripts/convert_to_tflite.py

Output:
    mobile/flutter/assets/models/detection.tflite
    mobile/flutter/assets/models/embedding.tflite
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    required = ['torch', 'ultralytics', 'onnx', 'tensorflow']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install -r ml/requirements-convert.txt")
        return False

    return True


def convert_detection_model(input_path: Path, output_path: Path):
    """Convert YOLOv8 detection model to TFLite."""
    print("\n" + "="*50)
    print("Converting Detection Model (YOLOv8)")
    print("="*50)

    from ultralytics import YOLO

    # Load the trained model
    print(f"Loading model from: {input_path}")
    model = YOLO(str(input_path))

    # Export to TFLite
    print("Exporting to TFLite (this may take a few minutes)...")
    export_path = model.export(
        format='tflite',
        imgsz=640,
        int8=False,  # Use float16 for better compatibility
        half=True,
    )

    # Move to output location
    if export_path:
        export_file = Path(export_path)
        if export_file.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(export_file, output_path)
            print(f"Detection model saved to: {output_path}")
            print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True

    print("Warning: TFLite export may have failed. Check for .tflite files.")
    return False


def convert_embedding_model(input_path: Path, output_path: Path):
    """Convert FastViT embedding model to TFLite via ONNX."""
    print("\n" + "="*50)
    print("Converting Embedding Model (FastViT)")
    print("="*50)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    # Define the model architecture (must match training)
    class EmbeddingModel(nn.Module):
        def __init__(self, backbone='fastvit_t12', embedding_dim=384):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)

            # Get feature dimension
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy)
                feature_dim = features.shape[-1]

            self.head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, embedding_dim),
            )

        def forward(self, x):
            features = self.backbone(x)
            embeddings = self.head(features)
            return F.normalize(embeddings, p=2, dim=1)

    # Load trained weights
    print(f"Loading model from: {input_path}")
    model = EmbeddingModel()
    state_dict = torch.load(input_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = input_path.parent / 'embedding.onnx'
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved: {onnx_path}")

    # Convert ONNX to TFLite
    print("Converting ONNX to TFLite...")
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))

        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_path = input_path.parent / 'embedding_tf'
        tf_rep.export_graph(str(tf_path))

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Embedding model saved to: {output_path}")
        print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Cleanup
        shutil.rmtree(tf_path, ignore_errors=True)

        return True

    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        print("\nAlternative: Use the ONNX model directly with onnxruntime")
        print(f"ONNX model available at: {onnx_path}")
        return False


def copy_assets(project_root: Path):
    """Copy additional assets needed for the Flutter app."""
    print("\n" + "="*50)
    print("Copying Additional Assets")
    print("="*50)

    flutter_assets = project_root / 'mobile/flutter/assets'

    # Copy Annoy index
    ann_src = project_root / 'ml/models/indices/riftbound.ann'
    ann_dst = flutter_assets / 'indices/riftbound.ann'
    if ann_src.exists():
        ann_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ann_src, ann_dst)
        print(f"Copied: {ann_dst}")

    # Verify cards.json exists
    cards_path = flutter_assets / 'data/cards.json'
    if cards_path.exists():
        print(f"Verified: {cards_path}")
    else:
        print(f"Warning: cards.json not found at {cards_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert models to TFLite')
    parser.add_argument('--skip-detection', action='store_true',
                        help='Skip detection model conversion')
    parser.add_argument('--skip-embedding', action='store_true',
                        help='Skip embedding model conversion')
    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    print("TCG Scanner - Model Conversion to TFLite")
    print(f"Project root: {project_root}")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Paths
    detection_input = project_root / 'ml/models/detection/best.pt'
    detection_output = project_root / 'mobile/flutter/assets/models/detection.tflite'

    embedding_input = project_root / 'ml/models/embedding/embedding_model.pt'
    embedding_output = project_root / 'mobile/flutter/assets/models/embedding.tflite'

    success = True

    # Convert detection model
    if not args.skip_detection:
        if detection_input.exists():
            if not convert_detection_model(detection_input, detection_output):
                success = False
        else:
            print(f"Detection model not found: {detection_input}")
            success = False

    # Convert embedding model
    if not args.skip_embedding:
        if embedding_input.exists():
            if not convert_embedding_model(embedding_input, embedding_output):
                success = False
        else:
            print(f"Embedding model not found: {embedding_input}")
            success = False

    # Copy additional assets
    copy_assets(project_root)

    # Summary
    print("\n" + "="*50)
    print("Conversion Summary")
    print("="*50)

    models_dir = project_root / 'mobile/flutter/assets/models'
    if models_dir.exists():
        for f in models_dir.glob('*.tflite'):
            print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.2f} MB")

    if success:
        print("\nConversion complete! Models ready for Flutter app.")
    else:
        print("\nSome conversions failed. Check errors above.")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
