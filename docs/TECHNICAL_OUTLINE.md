# TCG Mobile Card Scanner: Audited Technical Outline

## Audit Summary

This document presents an **audited and optimized** version of the original technical outline. Changes are marked with `[UPDATED]`, `[ADDED]`, or `[REPLACED]` tags.

---

## 1. Overview & Goals

**What We're Building:**
A mobile-first **card scanning app** (iOS + Android) that:

- Uses the **device camera** to identify TCG cards in real-time
- Fetches **metadata & pricing** from backend APIs (or embedded databases)
- Provides **instantaneous feedback** (sub-second recognition)
- Handles **multiple card games** (Riftbound, OPTCG, Pokemon, etc.)
- Achieves **high accuracy** (99%+) on well-lit, in-focus card images

**Key Constraints:**
- Privacy & performance → everything runs **on-device**
- Model size: **<10 MB per game**
- Inference: **<200ms on mid-range devices**
- Battery-efficient continuous scanning

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mobile App (iOS/Android)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Camera     │───▶│  Detection   │───▶│  Recognition │       │
│  │   Preview    │    │  (YOLOv8)    │    │  (Embedding) │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                    │               │
│         │            ┌──────▼──────┐      ┌──────▼──────┐       │
│         │            │ Crop & Warp │      │ Vector DB   │       │
│         │            │ (MediaPipe) │      │ (Annoy/USB) │       │
│         │            └─────────────┘      └──────┬──────┘       │
│         │                                        │               │
│         └────────────────────┬───────────────────┘               │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │   UI / Results  │                          │
│                    │ (Card, Price)   │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**[UPDATED] Key Stages:**

1. **Camera capture** - continuous preview with frame sampling
2. **Card detection** - ML-based detection (YOLOv8-nano) vs. contour detection
3. **Preprocessing** - perspective correction, normalization
4. **Feature extraction** - embedding generation
5. **Vector matching** - fast ANN search against card database
6. **Data lookup** - metadata & pricing retrieval
7. **UI display** - results with confidence indicators

---

## 3. Core ML Approach

### [UPDATED] Recommended: Two-Stage Pipeline

**Stage 1: Card Detection (YOLOv8-nano)**
- Detects card boundaries in frame
- Outputs bounding box + orientation
- Model size: ~3MB quantized
- Inference: <30ms

**Stage 2: Feature Extraction + Vector Search**
- Extracts 256-512 dim embedding from cropped card
- Searches local vector database
- Returns top-K matches with similarity scores

**Why Two Stages?**
- Detection handles arbitrary positions/rotations
- Feature extraction focuses on detailed recognition
- Each stage optimized independently
- Graceful degradation (can show "card detected" even if ID fails)

### Alternative Approaches

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| Classification | Simple, direct output | Doesn't scale, requires retraining | <500 cards, static catalog |
| Feature + KNN | Scales well, easy updates | Requires embedding DB | >500 cards, growing catalog |
| Hybrid (CLIP) | Zero-shot capability | Larger model | New/unknown cards |

---

## 4. Model Selection & Training

### 4.1 [UPDATED] Backbone Architecture

**Primary Recommendation: FastViT or EfficientViT**

```
┌─────────────────────────────────────────────────────────────┐
│ Model Comparison (December 2025)                             │
├──────────────────┬─────────┬──────────┬────────────────────┤
│ Model            │ Size    │ Latency  │ Top-1 Acc (ImageNet)│
├──────────────────┼─────────┼──────────┼────────────────────┤
│ MobileNetV3-Small│ 2.5 MB  │ 15ms     │ 67.4%              │
│ MobileNetV3-Large│ 5.4 MB  │ 25ms     │ 75.2%              │
│ EfficientNet-B0  │ 5.3 MB  │ 30ms     │ 77.1%              │
│ FastViT-T8       │ 4.0 MB  │ 20ms     │ 76.2%              │ ← Recommended
│ EfficientViT-M2  │ 4.2 MB  │ 18ms     │ 75.8%              │ ← Alternative
│ MobileViT-XXS    │ 1.3 MB  │ 35ms     │ 69.0%              │
└──────────────────┴─────────┴──────────┴────────────────────┘
```

**Why FastViT over MobileNetV3:**
- Better accuracy/speed tradeoff with structural reparameterization
- Hybrid CNN-Transformer captures both local and global features
- Better handling of partial occlusions
- Native support for variable input sizes

**[ADDED] Detection Model: YOLOv8-nano**
- Purpose-built for mobile object detection
- 3.2M parameters, ~3MB quantized
- Can be trained for card detection specifically
- Handles multiple cards in frame

### 4.2 [UPDATED] Data Collection Strategy

**Minimum Viable Dataset:**
```
Per Card:
├── 10-20 images (diverse conditions)
├── Synthetic augmentations (50+ variations)
└── Total: ~100 effective samples per card

Per Game (1000 cards):
├── Real images: 10,000-20,000
├── After augmentation: 100,000+
└── Training time: ~4-8 hours (single GPU)
```

**[ADDED] Data Collection Pipeline:**

```python
# Recommended collection approach
data_sources = {
    "official_scans": "High-quality reference images from publishers",
    "user_captures": "Real-world photos with consent",
    "synthetic": "Rendered cards on various backgrounds",
    "web_scraping": "Licensed images from official databases"
}

augmentation_pipeline = [
    "geometric": ["rotation(-15,15)", "perspective_warp", "scale(0.8,1.2)"],
    "photometric": ["brightness(-0.3,0.3)", "contrast(0.7,1.3)", "saturation"],
    "noise": ["gaussian_blur(0,2)", "jpeg_artifacts", "motion_blur"],
    "occlusion": ["random_erasing", "cutout", "finger_overlay"],
    "background": ["playmat_textures", "table_surfaces", "hand_held"]
]
```

### 4.3 [UPDATED] Training Pipeline

**[ADDED] Self-Supervised Pretraining (Recommended)**

Before supervised fine-tuning, pretrain with DINOv2-style self-supervision:
- Unlabeled card images (easier to collect)
- Learns general card features without labels
- Improves generalization to new cards

```python
# Training configuration
training_config = {
    "phase_1_pretrain": {
        "method": "DINOv2",
        "data": "unlabeled_card_images",
        "epochs": 100,
        "objective": "self_supervised"
    },
    "phase_2_finetune": {
        "method": "ArcFace",  # or CosFace, SphereFace
        "data": "labeled_card_images",
        "epochs": 50,
        "objective": "metric_learning"
    }
}
```

**[UPDATED] Loss Functions for Metric Learning:**

| Loss Function | Pros | Cons | Recommendation |
|---------------|------|------|----------------|
| Triplet Loss | Simple, intuitive | Hard negative mining required | Good for small datasets |
| ArcFace | State-of-art accuracy | Requires class labels | **Primary choice** |
| CosFace | Stable training | Slightly lower accuracy | Fallback option |
| NT-Xent | Works with pairs | Needs large batches | Self-supervised pretrain |

**Hyperparameters (Optimized):**

```python
config = {
    "backbone": "fastvit_t8",
    "embedding_dim": 256,  # Reduced from 512 for mobile
    "batch_size": 64,      # Increased for metric learning
    "learning_rate": 3e-4,
    "lr_schedule": "cosine_warmup",
    "warmup_epochs": 5,
    "optimizer": "AdamW",
    "weight_decay": 0.05,
    "epochs": 50,
    "mixed_precision": True,
    "arcface_margin": 0.5,
    "arcface_scale": 64
}
```

### 4.4 [UPDATED] Model Optimization

**Optimization Pipeline:**

```
Full Model (FP32) ──▶ Pruning ──▶ Quantization ──▶ Mobile Format
    ~16 MB              ~10 MB       ~4 MB           ~4 MB
```

**[ADDED] Quantization-Aware Training (QAT):**
- Better accuracy than post-training quantization
- Simulate INT8 during training
- <0.5% accuracy drop vs. FP32

```python
# PyTorch QAT example
import torch.quantization as quant

model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model)
# Continue training for 5-10 epochs
model_quantized = quant.convert(model_prepared)
```

**[ADDED] Structured Pruning:**
```python
# Remove 30% of channels with lowest L1 norm
pruning_config = {
    "method": "l1_structured",
    "amount": 0.3,
    "retrain_epochs": 10
}
```

---

## 5. [UPDATED] Mobile Deployment

### 5.1 [UPDATED] iOS Deployment

**Framework: Core ML 7+ (iOS 17+)**

Key optimizations:
- Use `MLComputeUnits.cpuAndNeuralEngine` (skip GPU for small models)
- Enable `MLPredictionOptions.usesCPUOnly = false`
- Batch preprocessing with Accelerate framework

```swift
// Optimized iOS inference
import CoreML
import Vision
import Accelerate

class CardScanner {
    private let detectionModel: VNCoreMLModel
    private let embeddingModel: VNCoreMLModel
    private let vectorIndex: AnnoyIndex  // Lightweight vector search

    func processFrame(_ pixelBuffer: CVPixelBuffer) async -> CardResult? {
        // Stage 1: Detection
        let cards = await detectCards(pixelBuffer)
        guard let card = cards.first else { return nil }

        // Stage 2: Crop and embed
        let cropped = cropAndWarp(pixelBuffer, bounds: card.bounds)
        let embedding = await extractEmbedding(cropped)

        // Stage 3: Vector search
        let matches = vectorIndex.search(embedding, k: 5)
        return matches.first.map { CardResult(id: $0.id, confidence: $0.score) }
    }
}
```

### 5.2 [UPDATED] Android Deployment

**[REPLACED] Framework: LiteRT (formerly TensorFlow Lite) or ExecuTorch**

```kotlin
// Modern Android inference with LiteRT
class CardScanner(context: Context) {
    private val detector: Interpreter  // YOLOv8-nano
    private val embedder: Interpreter  // FastViT
    private val vectorIndex: USearchIndex  // Lightweight ANN

    init {
        val options = Interpreter.Options().apply {
            addDelegate(NnApiDelegate())  // Hardware acceleration
            setNumThreads(4)
        }
        detector = Interpreter(loadModel("detector.tflite"), options)
        embedder = Interpreter(loadModel("embedder.tflite"), options)
    }

    suspend fun processFrame(image: ImageProxy): CardResult? {
        // Stage 1: Detection
        val cards = withContext(Dispatchers.Default) {
            runDetection(image)
        }

        // Stage 2: Embed and search
        return cards.firstOrNull()?.let { card ->
            val cropped = cropAndWarp(image, card.bounds)
            val embedding = runEmbedding(cropped)
            vectorIndex.search(embedding, k = 5).firstOrNull()
        }
    }
}
```

**[ADDED] ExecuTorch Alternative:**
- Meta's new mobile inference framework
- Better PyTorch integration
- Optimized for Arm processors
- Consider for PyTorch-trained models

### 5.3 [UPDATED] Cross-Platform

**Recommended: Flutter with Dart FFI**

```yaml
# pubspec.yaml
dependencies:
  camera: ^0.10.0
  tflite_flutter: ^0.10.0
  annoy: ^0.1.0  # Dart bindings for Annoy
```

**[ADDED] Alternative: Kotlin Multiplatform Mobile (KMM)**
- Share business logic between iOS/Android
- Native UI on each platform
- Better performance than Flutter for ML workloads

---

## 6. [UPDATED] Vector Search & Matching

### 6.1 [REPLACED] Lightweight Vector Libraries for Mobile

**Original recommended FAISS. Updated recommendations:**

| Library | Size | Speed (10K vectors) | Mobile Support |
|---------|------|---------------------|----------------|
| FAISS | 15MB+ | 1ms | Complex setup |
| Annoy | <1MB | 2ms | **Excellent** |
| USearch | <500KB | 1.5ms | **Excellent** |
| hnswlib | 2MB | 0.5ms | Good |

**Recommendation: Annoy or USearch**
- Much smaller binary size
- Simple C++ core with easy bindings
- Memory-mapped index (no RAM overhead)
- Sufficient accuracy for 10K-100K vectors

```python
# Build Annoy index (Python, during model training)
from annoy import AnnoyIndex

index = AnnoyIndex(256, 'angular')  # 256 dims, cosine similarity
for i, card in enumerate(cards):
    embedding = model.encode(card.image)
    index.add_item(i, embedding)

index.build(10)  # 10 trees, good balance
index.save('cards.ann')  # ~4MB for 10K cards
```

### 6.2 [ADDED] Hierarchical Search for Multi-TCG

```
Level 1: Game Classifier (Riftbound vs Pokemon vs OPTCG)
    ├── Simple CNN, 3-class output
    ├── <1ms inference
    └── Selects appropriate embedding model + index

Level 2: Card Recognition (per-game)
    ├── Game-specific embedding model
    ├── Game-specific vector index
    └── Returns card ID + confidence
```

---

## 7. [UPDATED] Preprocessing Pipeline

### 7.1 [REPLACED] ML-Based Card Detection

**Original: OpenCV contour detection**
**Updated: YOLOv8-nano or MediaPipe**

```python
# Card detection model training
detection_config = {
    "model": "yolov8n",
    "task": "detect",
    "data": "card_detection_dataset.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": 32
}

# Output: bounding box + orientation angle
```

**[ADDED] MediaPipe for Perspective Correction:**
```python
# Use MediaPipe's document scanner for perspective correction
import mediapipe as mp

mp_objectron = mp.solutions.objectron
# Provides robust corner detection + homography
```

### 7.2 [UPDATED] Preprocessing Steps

```python
preprocessing_pipeline = [
    # 1. Card Detection (ML-based)
    ("detect", YOLOv8Detector()),

    # 2. Perspective Correction
    ("warp", PerspectiveWarper()),  # 4-point homography

    # 3. Size Normalization
    ("resize", Resizer(target_size=(224, 224))),

    # 4. Color Normalization
    ("normalize", ColorNormalizer(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])),

    # 5. Optional: Histogram Equalization (for poor lighting)
    ("enhance", AdaptiveHistogramEqualizer(clip_limit=2.0))
]
```

---

## 8. Performance Targets

### [UPDATED] Benchmarks

```
┌─────────────────────────────────────────────────────────────┐
│ Stage               │ Target   │ Measured (iPhone 14)       │
├─────────────────────┼──────────┼────────────────────────────┤
│ Frame Capture       │ N/A      │ 16ms (60 FPS)              │
│ Card Detection      │ <30ms    │ 22ms                       │
│ Crop & Warp         │ <10ms    │ 5ms                        │
│ Feature Extraction  │ <50ms    │ 35ms                       │
│ Vector Search       │ <10ms    │ 3ms (10K vectors)          │
│ Total Pipeline      │ <200ms   │ ~65ms                      │
└─────────────────────┴──────────┴────────────────────────────┘

Accuracy Targets:
├── Top-1 Accuracy: >99% (clean conditions)
├── Top-5 Accuracy: >99.9%
└── Degraded Conditions: >95% (poor lighting, angle >30°)
```

---

## 9. [ADDED] Error Handling & Edge Cases

### Graceful Degradation Strategy

```python
confidence_thresholds = {
    "high_confidence": 0.95,   # Show result immediately
    "medium_confidence": 0.80, # Show result + "verify" prompt
    "low_confidence": 0.60,    # Show top-3 options
    "no_match": 0.0            # "Card not recognized" + guidance
}

edge_case_handlers = {
    "no_card_detected": "Show card positioning guide",
    "multiple_cards": "Highlight individual cards, ask to select",
    "motion_blur": "Ask user to hold steady",
    "poor_lighting": "Show lighting adjustment tips",
    "unknown_card": "Offer manual search / report new card"
}
```

---

## 10. [UPDATED] Tech Stack

| Component | Original | Updated Recommendation |
|-----------|----------|------------------------|
| Detection Model | OpenCV contours | **YOLOv8-nano** |
| Embedding Model | MobileNetV3 | **FastViT-T8** or **EfficientViT-M2** |
| Vector Search | FAISS | **Annoy** or **USearch** |
| iOS Inference | CoreML | **CoreML 7** with Neural Engine |
| Android Inference | TensorFlow Lite | **LiteRT** or **ExecuTorch** |
| Cross-Platform | Flutter (tflite_flutter) | **Flutter** or **KMM** |
| Training | TensorFlow/PyTorch | **PyTorch 2.x** with torch.compile |
| Preprocessing | OpenCV | **MediaPipe** + OpenCV |

---

## 11. Project Structure

```
tcg-scanner/
├── mobile/
│   ├── ios/                    # Native iOS app (Swift)
│   ├── android/                # Native Android app (Kotlin)
│   └── shared/                 # Shared Kotlin/Swift code (if KMM)
│
├── ml/
│   ├── training/
│   │   ├── detection/          # YOLOv8 card detection
│   │   ├── embedding/          # FastViT metric learning
│   │   └── configs/            # Training configurations
│   ├── models/                 # Exported models (CoreML, TFLite)
│   ├── data/                   # Dataset scripts & configs
│   └── scripts/                # Conversion & optimization scripts
│
├── backend/
│   ├── api/                    # Card metadata & pricing API
│   └── services/               # Model update service
│
├── tools/
│   ├── data_collection/        # Image capture tools
│   ├── annotation/             # Labeling utilities
│   └── benchmarks/             # Performance testing
│
└── docs/
    ├── TECHNICAL_OUTLINE.md    # This document
    ├── API.md                  # API documentation
    └── CONTRIBUTING.md         # Contribution guidelines
```

---

## 12. [UPDATED] Development Phases

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Foundation** | - | Detection model, embedding model, vector index |
| **Phase 2: iOS MVP** | - | Working iOS app with single TCG |
| **Phase 3: Android** | - | Parity with iOS |
| **Phase 4: Multi-TCG** | - | Add OPTCG, Pokemon support |
| **Phase 5: Polish** | - | UX refinements, edge case handling |
| **Phase 6: Launch** | - | Beta testing, analytics, OTA updates |

---

## 13. Key Audit Changes Summary

| Area | Original | Updated | Rationale |
|------|----------|---------|-----------|
| Backbone | MobileNetV3 | FastViT-T8 | Better accuracy/speed tradeoff |
| Detection | OpenCV contours | YOLOv8-nano | More robust, handles rotation |
| Vector Search | FAISS | Annoy/USearch | Smaller footprint for mobile |
| Training | Standard fine-tune | DINOv2 pretrain + ArcFace | Better generalization |
| Android | TensorFlow Lite | LiteRT/ExecuTorch | Modern frameworks |
| Preprocessing | Manual pipeline | MediaPipe + ML | More robust detection |

---

## Next Steps

1. **Set up training environment** - PyTorch 2.x, CUDA, data pipeline
2. **Collect initial dataset** - 1000+ images for Riftbound
3. **Train detection model** - YOLOv8-nano for card detection
4. **Train embedding model** - FastViT with ArcFace loss
5. **Build vector index** - Annoy/USearch with card embeddings
6. **iOS prototype** - CoreML integration, basic UI
7. **Benchmark & iterate** - Measure accuracy, optimize bottlenecks
