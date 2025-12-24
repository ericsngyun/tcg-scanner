# TCG Card Scanner

A high-performance, on-device mobile card scanner for Trading Card Games (Riftbound, OPTCG, Pokemon, and more).

## Features

- **Real-time recognition** - Sub-200ms card identification
- **On-device ML** - Privacy-first, works offline
- **Multi-TCG support** - Modular architecture for multiple games
- **99%+ accuracy** - Robust to lighting, angles, and occlusions

## Architecture

```
Camera → Detection (YOLOv8) → Feature Extraction (FastViT) → Vector Search (Annoy) → Results
```

## Project Structure

```
tcg-scanner/
├── mobile/          # iOS and Android apps
├── ml/              # Model training and optimization
├── backend/         # Metadata and pricing APIs
├── tools/           # Data collection and benchmarking
└── docs/            # Technical documentation
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- CUDA 12.x (for training)
- Xcode 15+ (iOS development)
- Android Studio (Android development)

### Setup ML Environment

```bash
cd ml
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Train Models

```bash
# Train detection model
python training/detection/train.py --config configs/yolov8_card.yaml

# Train embedding model
python training/embedding/train.py --config configs/fastvit_arcface.yaml

# Build vector index
python scripts/build_index.py --embeddings data/embeddings.npy --output models/cards.ann
```

### Export for Mobile

```bash
# Export to CoreML (iOS)
python scripts/export_coreml.py --model models/fastvit.pt --output models/exports/

# Export to TFLite (Android)
python scripts/export_tflite.py --model models/fastvit.pt --output models/exports/
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection | <30ms | YOLOv8-nano |
| Embedding | <50ms | FastViT-T8 quantized |
| Search | <10ms | Annoy with 10K vectors |
| Total | <200ms | End-to-end pipeline |
| Accuracy | >99% | Clean conditions |
| Model Size | <10MB | Per-game models |

## Tech Stack

- **Training**: PyTorch 2.x, Ultralytics (YOLOv8)
- **iOS**: Swift, CoreML, Vision
- **Android**: Kotlin, LiteRT (TensorFlow Lite)
- **Vector Search**: Annoy / USearch
- **Backend**: Node.js / Python FastAPI

## Documentation

- [Technical Outline](docs/TECHNICAL_OUTLINE.md) - Detailed architecture and design decisions
- [API Documentation](docs/API.md) - Backend API reference
- [Contributing](docs/CONTRIBUTING.md) - How to contribute

## License

TBD
