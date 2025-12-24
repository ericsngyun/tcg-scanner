#!/usr/bin/env python3
"""
Export PyTorch Model to CoreML

Converts trained FastViT embedding model to CoreML format for iOS deployment.
"""

import argparse
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn


def load_pytorch_model(model_path: str, model_class, **kwargs):
    """Load a PyTorch model from checkpoint."""
    model = model_class(**kwargs)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_coreml(
    model: nn.Module,
    input_shape: tuple,
    output_path: str,
    quantize: bool = True,
) -> None:
    """Export PyTorch model to CoreML format."""
    # Trace model
    example_input = torch.randn(1, *input_shape)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, input_shape[1], input_shape[2]),
                scale=1 / 255.0,
                bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS17,
    )

    # Quantize if requested
    if quantize:
        print("Applying quantization...")
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=8
        )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".mlpackage":
        mlmodel.save(str(output_path))
    else:
        mlmodel.save(str(output_path.with_suffix(".mlmodel")))

    print(f"CoreML model saved to {output_path}")

    # Print model info
    spec = mlmodel.get_spec()
    print(f"\nModel info:")
    print(f"  Input: {spec.description.input[0].name}")
    print(f"  Output: {spec.description.output[0].name}")


def main():
    parser = argparse.ArgumentParser(description="Export model to CoreML")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/exports/card_embedding.mlmodel",
        help="Output CoreML model path",
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

    # Import model class (adjust import based on your model location)
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

    # Export
    print("Converting to CoreML...")
    export_to_coreml(
        model=model,
        input_shape=(3, args.input_size, args.input_size),
        output_path=args.output,
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
