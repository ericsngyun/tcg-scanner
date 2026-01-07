# TCG Scanner - Claude Code Context

This file provides context for Claude Code sessions working on this project.

## Project Overview

TCG Scanner is a mobile app for scanning and identifying trading card game (TCG) cards in real-time. Currently focused on **Riftbound** (League of Legends TCG) with plans to expand to other games.

## Architecture

**Two-stage ML pipeline:**
1. **Detection:** YOLOv8-nano (640x640) - locates cards in camera frame
2. **Recognition:** FastViT-T12 (224x224, 384-dim embeddings) + Annoy index search

**Stack:**
- ML Training: PyTorch, timm, ultralytics
- Mobile: Flutter with TFLite inference
- Vector Search: Annoy (10 trees, angular/cosine metric)

## Key Directories

```
ml/
├── configs/           # YAML configs for models and training
├── models/
│   ├── detection/     # YOLOv8 model (best.pt)
│   ├── embedding/     # FastViT model + embeddings (.pt, .npy)
│   └── indices/       # Annoy index files (.ann)
├── scripts/           # Python scripts (build_index.py, test_identification.py)
├── notebooks/         # Colab training notebooks
└── data/
    ├── images/        # Card images organized by game/group
    ├── raw/           # Raw API data (all_products.json, etc.)
    ├── processed/     # Training manifests, prices
    └── embeddings/    # Product ID mappings

mobile/flutter/        # Flutter mobile app
├── lib/core/services/ # ML inference service
└── lib/features/      # Scanner, collection features

docs/                  # Documentation
└── SCANNING_SYSTEM_AUDIT.md  # Detailed audit with mobile optimization tasks
```

## Current Status (Jan 2026)

**Working:**
- Trained embedding model (FastViT-T12) with 674 Riftbound cards
- Trained detection model (YOLOv8-nano)
- Annoy index built and ready
- Identification testing script created

**Needs Work (see `docs/SCANNING_SYSTEM_AUDIT.md`):**
- Mobile: Image preprocessing not implemented (`ml_service.dart:201-237`)
- Mobile: Embedding normalization bug (`ml_service.dart:252-259`)
- Mobile: Hardware acceleration not configured
- Mobile: YOLO output parsing not implemented
- TFLite model export needs execution

## Common Tasks

### Test Identification Accuracy
```bash
cd ml
python scripts/test_identification.py --test-folder path/to/cropped/images
```

### Build Vector Index
```bash
cd ml
python scripts/build_index.py --game riftbound --config configs/vector_index.yaml
```

### Fetch Card Data
```bash
cd tools/data_collection
python fetch_tcgcsv.py --game riftbound
```

## Key Configuration Values

| Parameter | Value | File |
|-----------|-------|------|
| Embedding dim | 384 | `configs/embedding.yaml` |
| Detection input | 640x640 | `configs/detection.yaml` |
| Embedding input | 224x224 | `configs/embedding.yaml` |
| Annoy trees | 10 | `configs/vector_index.yaml` |
| ImageNet mean | [0.485, 0.456, 0.406] | preprocessing |
| ImageNet std | [0.229, 0.224, 0.225] | preprocessing |

## Data Sources

- **Card data:** tcgcsv.com API (fetched via `fetch_tcgcsv.py`)
- **Product IDs:** `ml/data/embeddings/riftbound_product_ids.json` (674 cards)
- **Metadata:** `ml/data/raw/riftbound/all_products.json`

## Important Notes

1. **Single sample per class:** We only have one official image per card, so training uses heavy augmentation and multi-view learning (4 augmented views per card per batch)

2. **Mobile vector search:** Flutter doesn't have native Annoy bindings. Current implementation uses linear search which is acceptable for <1000 cards but needs optimization for scale.

3. **Model conversion:** Use `notebooks/convert_to_tflite.ipynb` for PyTorch to TFLite conversion. FastViT requires ONNX intermediate step.
