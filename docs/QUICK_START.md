# TCG Scanner - Quick Start Guide

Get the TCG Scanner running from scratch in under 2 hours (plus training time).

## Prerequisites

- Python 3.10+
- Google account (for Colab)
- Git

---

## 1. Clone & Setup (5 minutes)

```bash
# Clone
git clone https://github.com/yourusername/tcg-scanner.git
cd tcg-scanner

# Create virtual environment
python -m venv venv

# Activate (choose your OS)
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\Activate.ps1     # Windows PowerShell
.\venv\Scripts\activate.bat     # Windows CMD

# Install dependencies
pip install -r ml/requirements.txt
```

---

## 2. Collect Card Data (10 minutes)

```bash
python tools/data_collection/fetch_tcgcsv.py
```

This downloads:
- 673 card images (~29MB)
- Card metadata (names, rarities, etc.)
- Current prices

---

## 3. Upload to Google Drive (10 minutes)

Create in Google Drive:
```
My Drive/tcg-scanner/ml/data/images/riftbound/  ← Upload image folders
My Drive/tcg-scanner/ml/data/processed/riftbound/training_manifest.json
My Drive/tcg-scanner/ml/notebooks/  ← Upload notebooks
```

---

## 4. Train Models (4-12 hours)

### Detection Model (~2-4 hours)
1. Open `train_detection.ipynb` in Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Model saves automatically to Drive

### Embedding Model (~6-12 hours)
1. Open `train_embedding_v2.ipynb` in Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Model + index save automatically to Drive

---

## 5. Download Models (5 minutes)

From Google Drive, download to local:

| From Drive | To Local |
|------------|----------|
| `models/detection/best.pt` | `ml/models/detection/` |
| `models/embedding/embedding_model.pt` | `ml/models/embedding/` |
| `models/indices/riftbound.ann` | `ml/models/indices/` |
| `ml/data/embeddings/riftbound_product_ids.json` | `ml/data/embeddings/` |

---

## 6. Convert to TFLite (10 minutes)

### Option A: Using Colab (Recommended)
1. Upload `ml/notebooks/convert_to_tflite.ipynb` to Google Drive
2. Open in Colab and run all cells
3. Download TFLite files from `models/tflite/`

### Option B: Local (Linux/macOS only)
```bash
pip install -r ml/requirements-convert.txt
python ml/scripts/convert_to_tflite.py
```

Copy the output files to:
- `mobile/flutter/assets/models/detection.tflite`
- `mobile/flutter/assets/models/embedding.tflite`

---

## 7. Run Flutter App

```bash
cd mobile/flutter
flutter pub get
flutter run
```

---

## Quick Commands Reference

```bash
# Fetch latest prices
python tools/data_collection/update_prices.py

# Convert models
python ml/scripts/convert_to_tflite.py

# Run Flutter app
cd mobile/flutter && flutter run
```

---

## Need Help?

- Full documentation: [docs/ML_PIPELINE.md](./ML_PIPELINE.md)
- Issues: GitHub Issues
