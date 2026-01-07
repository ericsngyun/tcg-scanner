# TCG Scanner System Audit Report

**Date:** January 2026
**Scope:** Full scanning system architecture review with mobile optimization focus

---

## Executive Summary

The TCG Scanner uses a **two-stage recognition pipeline**:
1. **YOLOv8-nano** for card detection (640x640 input)
2. **FastViT-T12** for embedding generation (224x224 input, 384-dim output)
3. **Annoy index** for vector similarity search (10 trees, angular metric)

**Current Status:** Core ML pipeline is well-designed. Mobile implementation has significant gaps that need completion before deployment.

---

## 1. Architecture Overview

```
Camera Frame (60 FPS)
        |
        v
   Frame Throttle (100ms / 10 FPS)
        |
        v
+------------------+     +------------------+     +------------------+
|   YOLOv8-nano    | --> |   FastViT-T12    | --> |   Annoy Search   |
|   Detection      |     |   Embedding      |     |   Top-K Match    |
|   <30ms target   |     |   <50ms target   |     |   <10ms target   |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
   Bounding Box            384-dim Vector           Card Results
   + Confidence            (Normalized L2)          + Similarity
```

**Target Latency:** <200ms end-to-end
**Model Sizes:** Detection ~3MB, Embedding ~4MB (quantized)

---

## 2. Component Analysis

### 2.1 Detection Model (YOLOv8-nano)

| Aspect | Status | Notes |
|--------|--------|-------|
| Architecture | Good | YOLOv8-nano is appropriate for mobile |
| Training Config | Good | 150 epochs, heavy augmentation, copy-paste |
| Input Size | Good | 640x640 balanced for accuracy/speed |
| Quantization | Configured | INT8 with QAT planned |
| Model File | Present | `models/detection/best.pt` exists |
| TFLite Export | Needed | Conversion script exists but needs execution |

**Strengths:**
- Nano variant optimized for edge devices
- Heavy augmentation for real-world robustness

**Gaps:**
- TFLite model not yet generated
- No CoreML export for iOS Neural Engine optimization

### 2.2 Embedding Model (FastViT-T12)

| Aspect | Status | Notes |
|--------|--------|-------|
| Architecture | Excellent | FastViT hybrid CNN-Transformer |
| Embedding Dim | Good | 384-dim provides good capacity |
| Loss Function | Excellent | ArcFace with hard mining |
| Multi-View Training | Excellent | 4 views per card handles single-sample scenario |
| Augmentation | Comprehensive | Covers real-world conditions well |
| Model File | Present | `models/embedding/embedding_model.pt` exists |
| Embeddings | Present | `riftbound_embeddings.npy` with 674 cards |

**Strengths:**
- FastViT designed for mobile efficiency
- Heavy augmentation pipeline simulates camera conditions
- Memory bank for cross-batch hard negative mining
- Test-time augmentation support

**Gaps:**
- TFLite conversion needs proper architecture handling
- Float16 quantization should be verified for accuracy retention

### 2.3 Vector Search (Annoy)

| Aspect | Status | Notes |
|--------|--------|-------|
| Library | Good | Annoy is lightweight and fast |
| Trees | Good | 10 trees provides good recall |
| Metric | Correct | Angular (cosine) matches normalized embeddings |
| Index File | Present | `models/indices/riftbound.ann` exists |
| Product Mapping | Present | 674 products indexed |

**Strengths:**
- Annoy is efficient for mobile
- Index size ~1MB for 674 cards

**Mobile Consideration:**
- Flutter doesn't have native Annoy bindings
- Current implementation uses linear search (see optimization section)

### 2.4 Mobile Implementation (Flutter)

| Component | Status | Issue |
|-----------|--------|-------|
| MLService | Partial | Image preprocessing is TODO |
| TFLite Loading | Present | Basic interpreter setup done |
| Detection Parsing | TODO | YOLO output parsing not implemented |
| Embedding Normalization | Bug | Uses incorrect L2 norm formula |
| Vector Search | Suboptimal | Linear search instead of Annoy |
| Image Cropping | TODO | Not implemented |
| Hardware Acceleration | Partial | NNAPI/CoreML delegates not configured |

---

## 3. Critical Mobile Optimization Opportunities

### 3.1 HIGH PRIORITY - Image Preprocessing Implementation

**Current Issue:** `ml_service.dart:201-237` - Preprocessing functions return placeholder data.

**Required Implementation:**
```dart
// Detection preprocessing (640x640, 0-1 normalized)
Future<List<List<List<List<double>>>>> _preprocessForDetection(
  Uint8List imageBytes,
) async {
  final img = decodeImage(imageBytes);
  final resized = copyResize(img, width: 640, height: 640);

  // Convert to float tensor [1, 640, 640, 3] normalized to 0-1
  final tensor = List.generate(1, (_) =>
    List.generate(640, (y) =>
      List.generate(640, (x) {
        final pixel = resized.getPixel(x, y);
        return [
          getRed(pixel) / 255.0,
          getGreen(pixel) / 255.0,
          getBlue(pixel) / 255.0,
        ];
      })
    )
  );
  return tensor;
}

// Embedding preprocessing (224x224, ImageNet normalized)
Future<List<List<List<List<double>>>>> _preprocessForEmbedding(
  Uint8List cardImage,
) async {
  final img = decodeImage(cardImage);
  final resized = copyResize(img, width: 224, height: 224);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  final tensor = List.generate(1, (_) =>
    List.generate(224, (y) =>
      List.generate(224, (x) {
        final pixel = resized.getPixel(x, y);
        return [
          (getRed(pixel) / 255.0 - mean[0]) / std[0],
          (getGreen(pixel) / 255.0 - mean[1]) / std[1],
          (getBlue(pixel) / 255.0 - mean[2]) / std[2],
        ];
      })
    )
  );
  return tensor;
}
```

### 3.2 HIGH PRIORITY - Fix Embedding Normalization Bug

**Current Code (WRONG):**
```dart
List<double> _normalizeEmbedding(List<double> embedding) {
  final norm = embedding.fold<double>(0.0, (sum, val) => sum + val * val);
  final magnitude = norm > 0 ? 1.0 / (norm * norm) : 1.0;  // BUG: Should be sqrt(norm)
  return embedding.map((v) => v * magnitude).toList();
}
```

**Fixed Code:**
```dart
List<double> _normalizeEmbedding(List<double> embedding) {
  final sumSquares = embedding.fold<double>(0.0, (sum, val) => sum + val * val);
  final magnitude = sumSquares > 0 ? sqrt(sumSquares) : 1.0;
  return embedding.map((v) => v / magnitude).toList();
}
```

### 3.3 HIGH PRIORITY - Hardware Acceleration

**Current:** Basic 4-thread CPU inference
**Opportunity:** 2-5x speedup with hardware delegates

```dart
// iOS - CoreML Delegate
final coremlOptions = CoreMlDelegateOptions()
  ..coremlVersion = 3
  ..enableOnSubgraphs = true;
final coremlDelegate = CoreMlDelegate(options: coremlOptions);

_embeddingModel = await Interpreter.fromAsset(
  'assets/models/embedding.tflite',
  options: InterpreterOptions()
    ..addDelegate(coremlDelegate),
);

// Android - NNAPI Delegate
final nnapiDelegate = NnApiDelegate(
  options: NnApiDelegateOptions()
    ..allowFp16 = true
    ..useNnapiCpu = false,
);

_embeddingModel = await Interpreter.fromAsset(
  'assets/models/embedding.tflite',
  options: InterpreterOptions()
    ..addDelegate(nnapiDelegate),
);
```

### 3.4 MEDIUM PRIORITY - Efficient Vector Search

**Current:** O(n) linear search over all embeddings
**Options:**

1. **Pre-compute on build** - Ship binary embedding file, use linear search (acceptable for <1000 cards)
2. **Port Annoy to Dart** - Complex but provides O(log n) search
3. **Use usearch-dart** - If available, provides HNSW search
4. **Native bridge** - Platform channels to C++ Annoy (most performant)

For 674 cards, linear search is acceptable (~3ms). For scaling to 10K+ cards, implement option 4.

### 3.5 MEDIUM PRIORITY - YOLO Output Parsing

**Required Implementation:**
```dart
List<DetectionResult> _parseDetections(
  List<List<List<double>>> output, {
  double confidenceThreshold = 0.5,
  double nmsThreshold = 0.5,
}) {
  // YOLOv8 output shape: [1, num_boxes, 5] for (x_center, y_center, width, height, confidence)
  final boxes = <DetectionResult>[];

  for (final detection in output[0]) {
    final confidence = detection[4];
    if (confidence < confidenceThreshold) continue;

    final xCenter = detection[0];
    final yCenter = detection[1];
    final width = detection[2];
    final height = detection[3];

    boxes.add(DetectionResult(
      boundingBox: Rect.fromCenter(
        center: Offset(xCenter, yCenter),
        width: width,
        height: height,
      ),
      confidence: confidence,
      angle: 0.0,  // Add rotation handling if needed
    ));
  }

  // Apply NMS
  return _nonMaxSuppression(boxes, nmsThreshold);
}
```

### 3.6 LOW PRIORITY - Batch Processing

For scanning multiple cards simultaneously:
```dart
Future<List<List<double>>> generateEmbeddingsBatch(
  List<Uint8List> cardImages,
) async {
  // Process in batches of 4 for efficiency
  final batchSize = 4;
  final results = <List<double>>[];

  for (var i = 0; i < cardImages.length; i += batchSize) {
    final batch = cardImages.skip(i).take(batchSize).toList();
    final inputs = await Future.wait(batch.map(_preprocessForEmbedding));

    // Concatenate to single batch tensor
    final batchInput = _stackBatch(inputs);
    final batchOutput = List.filled(batch.length * embeddingDim, 0.0)
        .reshape([batch.length, embeddingDim]);

    _embeddingModel!.run(batchInput, batchOutput);

    for (var j = 0; j < batch.length; j++) {
      results.add(_normalizeEmbedding(batchOutput[j]));
    }
  }

  return results;
}
```

---

## 4. Model Conversion Checklist

### 4.1 Detection Model (YOLOv8)
```bash
# Export to TFLite
yolo export model=models/detection/best.pt format=tflite int8=True

# Export to CoreML
yolo export model=models/detection/best.pt format=coreml nms=True
```

### 4.2 Embedding Model (FastViT)
The conversion requires careful handling of FastViT architecture:

```python
import torch
import timm

# Load trained model
model = timm.create_model('fastvit_t12', pretrained=False, num_classes=0)
checkpoint = torch.load('models/embedding/embedding_model.pt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'embedding_model.onnx',
    input_names=['input'],
    output_names=['embedding'],
    dynamic_axes={'input': {0: 'batch'}},
    opset_version=13,
)

# Convert ONNX to TFLite
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load('embedding_model.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('embedding_model_tf')

# Convert to TFLite with float16 quantization
converter = tf.lite.TFLiteConverter.from_saved_model('embedding_model_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('embedding.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 5. Performance Targets

| Metric | Target | Current Estimate |
|--------|--------|------------------|
| Detection Latency | <30ms | ~22ms (iPhone 14) |
| Embedding Latency | <50ms | ~35ms (iPhone 14) |
| Vector Search | <10ms | ~3ms (674 cards) |
| Total Pipeline | <200ms | ~65ms (with hardware accel) |
| Top-1 Accuracy | >99% | TBD (run test script) |
| Model Size (Total) | <10MB | ~7-8MB |

---

## 6. Testing Recommendations

### 6.1 Use the Identification Test Script
```bash
cd ml
python scripts/test_identification.py \
    --test-folder path/to/cropped/test/images \
    --output identification_report.html
```

This tests the **pure identification accuracy** without detection, isolating embedding model performance.

### 6.2 Create Test Image Dataset
1. Capture real photos of cards under various conditions:
   - Good lighting
   - Poor lighting
   - Angled shots (15-30 degrees)
   - Partial occlusion
   - Foil/holographic cards
   - Cards in sleeves

2. Name files with product ID for automatic accuracy calculation:
   - `652771_test1.jpg`
   - `652771_test2.jpg`
   - etc.

### 6.3 Benchmark Mobile Performance
```dart
// Add to MLService for benchmarking
Future<Map<String, double>> benchmark(Uint8List testImage) async {
  final results = <String, double>{};

  // Detection benchmark
  final detStart = DateTime.now();
  await detectCards(testImage);
  results['detection_ms'] = DateTime.now().difference(detStart).inMilliseconds.toDouble();

  // Embedding benchmark
  final embStart = DateTime.now();
  await generateEmbedding(testImage);
  results['embedding_ms'] = DateTime.now().difference(embStart).inMilliseconds.toDouble();

  // Search benchmark (with dummy embedding)
  final searchStart = DateTime.now();
  await searchCards(List.filled(384, 0.0));
  results['search_ms'] = DateTime.now().difference(searchStart).inMilliseconds.toDouble();

  return results;
}
```

---

## 7. Action Items Summary

### Immediate (Before Testing)
1. [ ] Fix embedding normalization bug in `ml_service.dart:252-259`
2. [ ] Implement image preprocessing functions
3. [ ] Run TFLite conversion for both models
4. [ ] Create test image dataset

### Short-term (Before Beta)
5. [ ] Add hardware acceleration delegates
6. [ ] Implement YOLO output parsing
7. [ ] Add image cropping implementation
8. [ ] Benchmark on target devices

### Medium-term (Production)
9. [ ] Implement efficient vector search (if scaling beyond 1K cards)
10. [ ] Add batch processing for multi-card scans
11. [ ] Implement Test-Time Augmentation for difficult images
12. [ ] Add confidence calibration

---

## 8. Appendix: File Reference

| File | Purpose |
|------|---------|
| `ml/configs/embedding.yaml` | Embedding model training config |
| `ml/configs/detection.yaml` | Detection model training config |
| `ml/configs/vector_index.yaml` | Index building config |
| `ml/models/embedding/embedding_model.pt` | Trained PyTorch embedding model |
| `ml/models/embedding/riftbound_embeddings.npy` | Pre-computed card embeddings |
| `ml/models/detection/best.pt` | Trained YOLOv8 detection model |
| `ml/models/indices/riftbound.ann` | Annoy search index |
| `ml/data/embeddings/riftbound_product_ids.json` | Product ID mapping for index |
| `ml/data/raw/riftbound/all_products.json` | Card metadata |
| `mobile/flutter/lib/core/services/ml_service.dart` | Flutter ML inference service |
| `ml/scripts/test_identification.py` | Identification accuracy testing script |
