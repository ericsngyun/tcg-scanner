# TCG Scanner - ML Pipeline Documentation

Complete guide for training, converting, and deploying the card recognition ML models.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Environment Setup](#environment-setup)
4. [Data Collection](#data-collection)
5. [Model Training](#model-training)
6. [Model Conversion](#model-conversion)
7. [Flutter Integration](#flutter-integration)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The TCG Scanner uses a two-stage ML pipeline for real-time card recognition:

```
Camera Frame
    ↓
┌─────────────────────────────┐
│  Stage 1: Card Detection    │  YOLOv8-nano (~3MB)
│  - Locates cards in frame   │  Input: 640x640 RGB
│  - Returns bounding boxes   │  Output: Bounding boxes + confidence
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Stage 2: Card Embedding    │  FastViT-T12 (~4MB)
│  - Extracts card features   │  Input: 224x224 RGB (cropped card)
│  - Generates 384-dim vector │  Output: Normalized embedding
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Stage 3: Vector Search     │  Annoy Index (~1MB)
│  - Finds similar cards      │  Input: 384-dim embedding
│  - Returns top-K matches    │  Output: Product IDs + distances
└─────────────────────────────┘
    ↓
Card Information + Pricing
```

**Target Performance:** <200ms total latency on mobile devices

---

## Architecture

### Directory Structure

```
tcg-scanner/
├── ml/
│   ├── configs/                    # Training configurations
│   │   ├── detection.yaml
│   │   ├── embedding.yaml
│   │   └── vector_index.yaml
│   │
│   ├── data/
│   │   ├── images/                 # Card images (gitignored)
│   │   │   └── riftbound/
│   │   ├── processed/              # Processed data
│   │   │   └── riftbound/
│   │   │       ├── training_manifest.json
│   │   │       ├── latest_prices.json
│   │   │       └── price_history/
│   │   └── embeddings/
│   │       └── riftbound_product_ids.json
│   │
│   ├── models/                     # Trained models (gitignored)
│   │   ├── detection/
│   │   │   └── best.pt
│   │   ├── embedding/
│   │   │   ├── embedding_model.pt
│   │   │   └── riftbound_embeddings.npy
│   │   └── indices/
│   │       └── riftbound.ann
│   │
│   ├── notebooks/                  # Colab training notebooks
│   │   ├── train_detection.ipynb
│   │   └── train_embedding_v2.ipynb
│   │
│   ├── scripts/
│   │   └── convert_to_tflite.py
│   │
│   ├── training/                   # Training scripts
│   │   ├── detection/
│   │   └── embedding/
│   │
│   ├── requirements.txt            # Training requirements
│   └── requirements-convert.txt    # Conversion requirements
│
├── mobile/flutter/                 # Flutter mobile app
│   ├── lib/
│   │   ├── core/
│   │   │   ├── models/
│   │   │   └── services/
│   │   └── features/
│   │       ├── scanner/
│   │       └── collection/
│   │
│   └── assets/
│       ├── models/                 # TFLite models
│       │   ├── detection.tflite
│       │   └── embedding.tflite
│       ├── indices/
│       │   └── riftbound.ann
│       └── data/
│           └── cards.json
│
└── tools/data_collection/          # Data fetching scripts
    ├── fetch_tcgcsv.py
    └── update_prices.py
```

### Model Specifications

| Model | Architecture | Input | Output | Size |
|-------|-------------|-------|--------|------|
| Detection | YOLOv8-nano | 640×640×3 | Bounding boxes | ~3MB |
| Embedding | FastViT-T12 | 224×224×3 | 384-dim vector | ~4MB |
| Index | Annoy (10 trees) | 384-dim vector | Top-K IDs | ~1MB |

---

## Environment Setup

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Git
- Google account (for Colab training)
- ~50GB free disk space

### Windows Setup

```powershell
# 1. Clone repository
git clone https://github.com/yourusername/tcg-scanner.git
cd tcg-scanner

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r ml/requirements.txt

# 4. For model conversion (optional, can use Colab)
pip install -r ml/requirements-convert.txt
```

### macOS Setup

```bash
# 1. Install Python (if needed)
brew install python@3.11

# 2. Clone repository
git clone https://github.com/yourusername/tcg-scanner.git
cd tcg-scanner

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r ml/requirements.txt

# 5. For model conversion
pip install -r ml/requirements-convert.txt
```

### Linux Setup

```bash
# 1. Install Python and dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# 2. Clone repository
git clone https://github.com/yourusername/tcg-scanner.git
cd tcg-scanner

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r ml/requirements.txt
pip install -r ml/requirements-convert.txt
```

---

## Data Collection

### Step 1: Fetch Card Data

```bash
# Activate virtual environment first
cd tcg-scanner

# Fetch all data (images + metadata + prices)
python tools/data_collection/fetch_tcgcsv.py

# Or fetch specific components
python tools/data_collection/fetch_tcgcsv.py --images-only
python tools/data_collection/fetch_tcgcsv.py --prices-only
python tools/data_collection/fetch_tcgcsv.py --metadata-only
```

**Output:**
- `ml/data/images/riftbound/` - 673 card images (~29MB)
- `ml/data/processed/riftbound/training_manifest.json` - Card metadata
- `ml/data/processed/riftbound/latest_prices.json` - Current prices

### Step 2: Verify Data

```bash
# Check image count
find ml/data/images/riftbound -name "*.jpg" | wc -l
# Expected: 673

# Check manifest
python -c "import json; print(len(json.load(open('ml/data/processed/riftbound/training_manifest.json'))))"
# Expected: 673
```

---

## Model Training

Training is done on Google Colab for free GPU access.

### Step 1: Upload to Google Drive

Create this structure in Google Drive:

```
My Drive/
└── tcg-scanner/
    └── ml/
        ├── data/
        │   ├── images/riftbound/     ← Upload all images
        │   └── processed/riftbound/
        │       └── training_manifest.json
        └── notebooks/
            ├── train_detection.ipynb
            └── train_embedding_v2.ipynb
```

### Step 2: Train Detection Model

1. Open `train_detection.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells

**Training time:** ~2-4 hours
**Output:** `models/detection/best.pt`

**Expected metrics:**
- mAP50: >0.94
- Precision: >0.95
- Recall: >0.85

### Step 3: Train Embedding Model

1. Open `train_embedding_v2.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells

**Training time:** ~6-12 hours
**Output:**
- `models/embedding/embedding_model.pt`
- `models/indices/riftbound.ann`
- `ml/data/embeddings/riftbound_product_ids.json`

### Step 4: Download Trained Models

Download from Google Drive to local:

```
Drive: tcg-scanner/models/detection/best.pt
  → Local: ml/models/detection/best.pt

Drive: tcg-scanner/models/embedding/embedding_model.pt
  → Local: ml/models/embedding/embedding_model.pt

Drive: tcg-scanner/models/indices/riftbound.ann
  → Local: ml/models/indices/riftbound.ann

Drive: tcg-scanner/ml/data/embeddings/riftbound_product_ids.json
  → Local: ml/data/embeddings/riftbound_product_ids.json
```

---

## Model Conversion

Convert PyTorch models to TFLite for mobile deployment.

### Option A: Colab Conversion (Recommended)

Use Google Colab for conversion - this avoids TensorFlow installation issues on Windows.

1. Upload `convert_to_tflite.ipynb` to Google Drive:
   ```
   My Drive/tcg-scanner/ml/notebooks/convert_to_tflite.ipynb
   ```

2. Ensure your trained models are in Google Drive:
   ```
   My Drive/tcg-scanner/models/detection/best.pt
   My Drive/tcg-scanner/models/embedding/embedding_model.pt
   ```

3. Open `convert_to_tflite.ipynb` in Google Colab

4. Runtime -> Change runtime type -> **T4 GPU** (optional but faster)

5. Run all cells

6. Download the converted models from:
   ```
   My Drive/tcg-scanner/models/tflite/detection.tflite
   My Drive/tcg-scanner/models/tflite/embedding.tflite
   ```

7. Copy to your Flutter project:
   ```
   mobile/flutter/assets/models/detection.tflite
   mobile/flutter/assets/models/embedding.tflite
   ```

### Option B: Local Conversion

If you have TensorFlow installed locally (works best on Linux/macOS):

```bash
# Activate virtual environment
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Install conversion dependencies
pip install -r ml/requirements-convert.txt

# Run conversion script
python ml/scripts/convert_to_tflite.py
```

**Note:** Windows users may encounter path length issues with TensorFlow/ONNX packages. Use Option A (Colab) instead.

### Verify Conversion

```bash
# Check TFLite files (Windows)
dir mobile\flutter\assets\models\

# Check TFLite files (macOS/Linux)
ls -lh mobile/flutter/assets/models/

# Expected:
# detection.tflite  ~3-6MB
# embedding.tflite  ~4-8MB
```

---

## Flutter Integration

### Step 1: Verify Assets

Ensure all required files are in place:

```
mobile/flutter/assets/
├── models/
│   ├── detection.tflite    ✓
│   └── embedding.tflite    ✓
├── indices/
│   └── riftbound.ann       ✓
└── data/
    └── cards.json          ✓
```

### Step 2: Generate cards.json

If not already created:

```bash
python -c "
import json
from pathlib import Path

# Load data
manifest = json.load(open('ml/data/processed/riftbound/training_manifest.json'))
prices = json.load(open('ml/data/processed/riftbound/latest_prices.json'))
product_ids = json.load(open('ml/data/embeddings/riftbound_product_ids.json'))

# Build cards list
cards = []
for pid in product_ids:
    card = next((c for c in manifest if str(c['product_id']) == pid), None)
    if card:
        price_info = prices.get(pid, {})
        cards.append({
            'product_id': card['product_id'],
            'name': card['name'],
            'clean_name': card['clean_name'],
            'group_name': card['group_name'],
            'rarity': card.get('rarity', 'Unknown'),
            'card_type': card.get('card_type', 'Unknown'),
            'pricing': {
                'market': price_info.get('market_price'),
                'low': price_info.get('low_price'),
            } if price_info else None
        })

# Save
Path('mobile/flutter/assets/data').mkdir(parents=True, exist_ok=True)
json.dump(cards, open('mobile/flutter/assets/data/cards.json', 'w'), indent=2)
print(f'Created cards.json with {len(cards)} cards')
"
```

### Step 3: Run Flutter App

```bash
cd mobile/flutter

# Install dependencies
flutter pub get

# Run on device/simulator
flutter run
```

---

## Testing

### Test Detection Model

```python
from ultralytics import YOLO
from PIL import Image

model = YOLO('ml/models/detection/best.pt')
results = model('path/to/test/image.jpg')
results[0].show()
```

### Test Embedding Model

```python
import torch
import numpy as np
from annoy import AnnoyIndex
from PIL import Image
import json

# Load model and index
# ... (see ml/scripts/test_embedding.py)
```

### Test Full Pipeline

```python
# End-to-end test
python ml/scripts/test_pipeline.py --image path/to/card.jpg
```

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory during training**
- Reduce batch size in notebook CONFIG
- Use smaller image size (imgsz=480 instead of 640)

**2. Windows path errors in Colab**
- Fixed in train_embedding_v2.ipynb
- Paths are converted: `\\` → `/`

**3. TFLite conversion fails**
- Try using Colab for conversion
- Check TensorFlow version compatibility
- Use ONNX as intermediate format

**4. val_recall_at_1 = 0 during training**
- Expected with 1 sample per class
- Monitor `val_loss` instead
- Final metrics calculated after training

**5. Flutter app crashes on model load**
- Verify TFLite files are valid
- Check file sizes match expected
- Ensure assets are listed in pubspec.yaml

### Getting Help

- Check GitHub Issues
- Review training notebook outputs
- Verify all file paths match documentation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-26 | Initial pipeline with Riftbound support |

---

## Next Steps

1. **Add more TCGs** - OPTCG, Pokemon, etc.
2. **Improve detection** - Train on real-world photos
3. **Optimize models** - Quantization, pruning
4. **Add features** - Collection tracking, price alerts
