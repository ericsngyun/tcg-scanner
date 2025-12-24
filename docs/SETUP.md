# TCG Scanner - Complete Setup Guide

This guide covers everything you need to know to set up, run, and extend the TCG Scanner project.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Data Collection Pipeline](#data-collection-pipeline)
6. [Automated Price Updates](#automated-price-updates)
7. [ML Training Pipeline](#ml-training-pipeline)
8. [Development Workflow](#development-workflow)
9. [Troubleshooting](#troubleshooting)

---

## Overview

TCG Scanner is a mobile card recognition app that identifies Trading Card Game cards in real-time using on-device machine learning.

### Current State

| Component | Status | Description |
|-----------|--------|-------------|
| Data Collection | **Active** | Fetches card images and prices from tcgcsv.com |
| Price Updates | **Automated** | GitHub Actions runs daily at 2 PM EST |
| ML Training | Ready | Scripts ready, awaiting training data |
| iOS App | Planned | Not started |
| Android App | Planned | Not started |

### Supported TCGs

- **Riftbound** - Primary focus, fully configured
- OPTCG - Planned
- Pokemon - Planned

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Data collection, ML training |
| Git | Latest | Version control |
| pip | Latest | Python package management |

### Optional (for ML training)

| Software | Version | Purpose |
|----------|---------|---------|
| CUDA | 12.x | GPU acceleration |
| PyTorch | 2.1+ | Deep learning framework |

### Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Data Collection | Any modern PC | Any modern PC |
| ML Training | 8GB RAM, CPU | 16GB RAM, NVIDIA GPU (8GB+ VRAM) |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ericsngyun/tcg-scanner.git
cd tcg-scanner
```

### 2. Install Dependencies

```bash
# Core dependencies for data collection
pip install requests pyyaml tqdm

# Full ML dependencies (optional, for training)
pip install -r ml/requirements.txt
```

### 3. Run Data Collection

```bash
# Fetch prices only (fast, ~15 seconds)
python tools/data_collection/fetch_tcgcsv.py --prices-only

# Full collection with images (slower, downloads all card images)
python tools/data_collection/fetch_tcgcsv.py
```

---

## Project Structure

```
tcg-scanner/
├── .github/
│   └── workflows/
│       └── daily-price-update.yml    # Automated daily price fetching
│
├── docs/
│   ├── SETUP.md                      # This file
│   ├── TECHNICAL_OUTLINE.md          # Architecture & design decisions
│   └── CONTRIBUTING.md               # Contribution guidelines
│
├── ml/
│   ├── configs/                      # Training configurations
│   │   ├── detection.yaml            # YOLOv8 card detection config
│   │   ├── embedding.yaml            # FastViT embedding config
│   │   └── vector_index.yaml         # Vector search index config
│   │
│   ├── data/
│   │   ├── tcgcsv_config.yaml        # Data source configuration
│   │   ├── raw/                      # Raw API responses (gitignored)
│   │   ├── images/                   # Downloaded card images (gitignored)
│   │   └── processed/
│   │       └── riftbound/
│   │           ├── latest_prices.json      # Current prices (tracked)
│   │           └── price_history/          # Historical prices (tracked)
│   │
│   ├── models/                       # Trained models (gitignored)
│   ├── training/
│   │   ├── detection/train.py        # YOLOv8 training script
│   │   └── embedding/train.py        # FastViT training script
│   └── scripts/
│       ├── build_index.py            # Vector index builder
│       ├── export_coreml.py          # iOS model export
│       └── export_tflite.py          # Android model export
│
├── tools/
│   └── data_collection/
│       ├── fetch_tcgcsv.py           # Main data collection script
│       ├── update_prices.py          # Standalone price updater
│       ├── prepare_dataset.py        # Dataset preparation for training
│       ├── run_price_update.bat      # Windows scheduler wrapper
│       └── setup_scheduled_task.ps1  # Windows Task Scheduler setup
│
├── mobile/
│   ├── ios/                          # iOS app (not started)
│   └── android/                      # Android app (not started)
│
├── backend/
│   └── api/                          # Backend API (not started)
│
└── logs/                             # Log files (gitignored)
```

---

## Data Collection Pipeline

### Overview

The data collection pipeline fetches card data from [tcgcsv.com](https://tcgcsv.com), which provides TCGplayer data in accessible formats.

### Data Sources

| Endpoint | Data | Format |
|----------|------|--------|
| `/tcgplayer/{cat}/{group}/products` | Card metadata | JSON |
| `/tcgplayer/{cat}/{group}/ProductsAndPrices.csv` | Cards + Prices | CSV |
| TCGplayer CDN | Card images | JPG (200w, 400w) |

### Configuration

The data source is configured in `ml/data/tcgcsv_config.yaml`:

```yaml
api:
  base_url: "https://tcgcsv.com/tcgplayer"
  cdn_url: "https://tcgplayer-cdn.tcgplayer.com/product"
  request_delay_seconds: 1.0  # Rate limiting
  image_delay_seconds: 0.5

riftbound:
  category_id: 89
  groups:
    - id: 24344
      name: "Origins"
    - id: 24519
      name: "Spiritforged"
    # ... more groups
```

### Running Data Collection

#### Full Collection (Metadata + Images + Prices)

```bash
python tools/data_collection/fetch_tcgcsv.py
```

This will:
1. Fetch all product metadata from tcgcsv API
2. Download card images (400x400 resolution)
3. Fetch current prices
4. Create training manifest

**Output:**
- `ml/data/raw/riftbound/` - Raw API responses
- `ml/data/images/riftbound/` - Card images
- `ml/data/processed/riftbound/latest_prices.json` - Current prices
- `ml/data/processed/riftbound/training_manifest.json` - ML training data

#### Prices Only (Fast)

```bash
python tools/data_collection/fetch_tcgcsv.py --prices-only
```

This fetches only price data (~15 seconds). Used for daily updates.

#### Command Line Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to config file (default: `ml/data/tcgcsv_config.yaml`) |
| `--output PATH` | Output directory (default: `ml/data`) |
| `--prices-only` | Only fetch prices, skip images |
| `--metadata-only` | Only fetch metadata, skip images |
| `--images-only` | Only download images (requires prior metadata fetch) |
| `--force-refresh` | Force refresh metadata from API (ignore cache) |
| `--with-prices` | Include price data (default: true) |

### Data Format

#### Product Data (`all_products.json`)

```json
{
  "product_id": 635369,
  "name": "Card Name",
  "clean_name": "card_name",
  "image_url": "https://...",
  "group_id": 24344,
  "group_name": "Origins",
  "rarity": "Rare",
  "number": "001",
  "card_type": "Champion",
  "low_price": 16.30,
  "mid_price": 18.02,
  "high_price": 18.80,
  "market_price": 17.67
}
```

#### Price Data (`latest_prices.json`)

```json
{
  "635369": {
    "product_id": 635369,
    "low_price": 16.30,
    "mid_price": 18.02,
    "high_price": 18.80,
    "market_price": 17.67,
    "direct_low_price": null,
    "fetched_at": "2025-12-24T14:04:26.913657"
  }
}
```

---

## Automated Price Updates

### GitHub Actions (Cloud)

Price updates run automatically via GitHub Actions:

- **Schedule:** Daily at 2:00 PM EST (7:00 PM UTC)
- **Workflow:** `.github/workflows/daily-price-update.yml`
- **What it does:**
  1. Fetches latest prices from tcgcsv.com
  2. Commits changes to `ml/data/processed/riftbound/`
  3. Maintains price history

#### Manual Trigger

1. Go to repository on GitHub
2. Click **Actions** tab
3. Select **Daily Price Update**
4. Click **Run workflow**

### Windows Task Scheduler (Local)

For local scheduled updates:

```powershell
# Run as Administrator
cd tools/data_collection
powershell -ExecutionPolicy Bypass -File setup_scheduled_task.ps1
```

This creates a task that runs `run_price_update.bat` daily at 2 PM.

---

## ML Training Pipeline

### Overview

The ML pipeline consists of:

1. **Card Detection** - YOLOv8-nano detects cards in camera frame
2. **Feature Extraction** - FastViT-T8 generates embeddings
3. **Vector Search** - Annoy index for fast similarity search

### Step 1: Prepare Dataset

After data collection:

```bash
python tools/data_collection/prepare_dataset.py
```

This creates:
- Train/validation split (80/20)
- Class directories for training
- Dataset statistics

### Step 2: Train Detection Model

```bash
python ml/training/detection/train.py --config ml/configs/detection.yaml
```

### Step 3: Train Embedding Model

```bash
python ml/training/embedding/train.py --config ml/configs/embedding.yaml
```

### Step 4: Build Vector Index

```bash
python ml/scripts/build_index.py --game riftbound
```

### Step 5: Export for Mobile

```bash
# iOS (CoreML)
python ml/scripts/export_coreml.py

# Android (TFLite)
python ml/scripts/export_tflite.py
```

---

## Development Workflow

### Adding a New TCG

1. **Add configuration** in `ml/data/tcgcsv_config.yaml`:

```yaml
new_game:
  category_id: XX
  groups:
    - id: XXXXX
      name: "Set Name"
```

2. **Update vector index config** in `ml/configs/vector_index.yaml`

3. **Run data collection** for the new game

4. **Train game-specific models**

### Updating Price Data Manually

```bash
# Quick price update
python tools/data_collection/fetch_tcgcsv.py --prices-only

# Or use the standalone script
python tools/data_collection/update_prices.py
```

### Checking Price History

Price history is stored in:
```
ml/data/processed/riftbound/price_history/prices_YYYYMMDD_HHMMSS.json
```

Each file is a complete snapshot of all prices at that time.

---

## Troubleshooting

### Common Issues

#### "No module named 'requests'"

```bash
pip install requests pyyaml tqdm
```

#### "Config file not found"

Make sure you're running from the project root:

```bash
cd /path/to/tcg-scanner
python tools/data_collection/fetch_tcgcsv.py
```

#### Rate limiting / Connection errors

The script has built-in retry logic. If issues persist:
- Check your internet connection
- Try again later (tcgcsv.com may be down)
- Increase `retry_delay_seconds` in config

#### Images not downloading

- Check if `ml/data/images/` directory exists
- Verify the CDN URLs are accessible
- Some products may not have images

### Logs

- Data collection logs: Console output
- Scheduled task logs: `logs/price_updates.log`
- GitHub Actions logs: Repository Actions tab

---

## API Reference

### fetch_tcgcsv.py Classes

#### `Product`
Represents a TCG card with metadata and prices.

#### `PriceData`
Price information for a single product.

#### `TCGCSVFetcher`
Main class for fetching data from tcgcsv.com.

Key methods:
- `fetch_all_riftbound_products()` - Fetch all products
- `fetch_all_prices()` - Fetch current prices
- `download_all_images()` - Download card images
- `save_price_history()` - Save prices with timestamp

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Questions?

Open an issue on GitHub or check the [Technical Outline](TECHNICAL_OUTLINE.md) for architecture details.
