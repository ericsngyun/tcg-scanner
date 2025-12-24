#!/usr/bin/env python3
"""
Export PyTorch Model to TensorFlow Lite

Converts trained FastViT embedding model to TFLite format for Android deployment.
Converts via ONNX intermediate format.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    input_shape: tuple,
    output_path: str,
) -> None:
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, *input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to {output_path}")


def convert_onnx_to_tflite(
    onnx_path: str,
    output_path: str,
    quantize: bool = True,
) -> None:
    """Convert ONNX model to TFLite using onnx-tf and TFLite converter."""
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError:
        raise ImportError(
            "Please install tensorflow and onnx-tf: "
            "pip install tensorflow onnx-tf"
        )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert to TensorFlow SavedModel
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_path = Path(tmpdir) / "saved_model"
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_path))

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)

    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export model to TFLite")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/exports/card_embedding.tflite",
        help="Output TFLite model path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization",
    )
    args = parser.parse_args()

    # Import model class
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "embedding"))
    from train import CardEmbeddingModel

    # Load model
    print(f"Loading model from {args.model}...")
    model = CardEmbeddingModel(
        backbone="fastvit_t8",
        embedding_dim=256,
        pretrained=False,
    )
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # Export to ONNX first
    output_path = Path(args.output)
    onnx_path = output_path.with_suffix(".onnx")

    print("Exporting to ONNX...")
    export_to_onnx(
        model=model,
        input_shape=(3, args.input_size, args.input_size),
        output_path=str(onnx_path),
    )

    # Convert to TFLite
    print("Converting to TFLite...")
    convert_onnx_to_tflite(
        onnx_path=str(onnx_path),
        output_path=str(output_path),
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
