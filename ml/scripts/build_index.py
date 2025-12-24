#!/usr/bin/env python3
"""
Build Vector Index Script

Creates an Annoy index from card embeddings for fast similarity search.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from annoy import AnnoyIndex
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_annoy_index(
    embeddings: np.ndarray,
    embedding_dim: int,
    num_trees: int,
    metric: str = "angular",
) -> AnnoyIndex:
    """Build Annoy index from embeddings."""
    index = AnnoyIndex(embedding_dim, metric)

    for i, embedding in enumerate(tqdm(embeddings, desc="Adding vectors")):
        index.add_item(i, embedding)

    print(f"Building index with {num_trees} trees...")
    index.build(num_trees)

    return index


def load_card_metadata(cards_file: str) -> List[Dict]:
    """Load card metadata from JSON file."""
    with open(cards_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Build vector index for card search")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vector_index.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=["riftbound", "optcg", "pokemon"],
        help="Game to build index for",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    game_config = config["games"][args.game]
    index_config = config["index"]
    annoy_config = config["annoy"]

    # Load embeddings
    print(f"Loading embeddings from {game_config['embeddings_file']}...")
    embeddings = np.load(game_config["embeddings_file"])
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Verify embedding dimension
    assert embeddings.shape[1] == index_config["embedding_dim"], (
        f"Embedding dimension mismatch: "
        f"expected {index_config['embedding_dim']}, got {embeddings.shape[1]}"
    )

    # Build index
    index = build_annoy_index(
        embeddings=embeddings,
        embedding_dim=index_config["embedding_dim"],
        num_trees=annoy_config["num_trees"],
        metric=index_config["metric"],
    )

    # Save index
    output_path = Path(game_config["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output_path))
    print(f"Index saved to {output_path}")

    # Verify index
    print("\nVerifying index...")
    test_idx = 0
    test_embedding = embeddings[test_idx]
    neighbors, distances = index.get_nns_by_vector(
        test_embedding, n=5, include_distances=True
    )
    print(f"Test query (card 0) nearest neighbors: {neighbors}")
    print(f"Distances: {distances}")

    # Print stats
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\nIndex stats:")
    print(f"  Cards indexed: {len(embeddings)}")
    print(f"  Embedding dim: {index_config['embedding_dim']}")
    print(f"  Number of trees: {annoy_config['num_trees']}")
    print(f"  File size: {file_size:.2f} MB")


if __name__ == "__main__":
    main()
