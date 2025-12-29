import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

import '../models/card.dart';
import '../models/frame_data.dart';
import '../models/recognition_result.dart';

/// Singleton service for ML model inference.
class MLService {
  static final MLService instance = MLService._();
  MLService._();

  Interpreter? _detectionModel;
  Interpreter? _embeddingModel;
  List<Card>? _cardDatabase;
  List<List<double>>? _embeddingIndex;

  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  /// Model input sizes.
  static const int detectionInputSize = 640;
  static const int embeddingInputSize = 224;
  static const int embeddingDim = 384;

  /// Initialize ML models and card database.
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Load detection model (YOLOv8-nano)
      _detectionModel = await Interpreter.fromAsset(
        'assets/models/detection.tflite',
        options: InterpreterOptions()..threads = 4,
      );

      // Load embedding model (FastViT-T12)
      _embeddingModel = await Interpreter.fromAsset(
        'assets/models/embedding.tflite',
        options: InterpreterOptions()..threads = 4,
      );

      // Load card database and embeddings
      await _loadCardDatabase();
      await _loadEmbeddingIndex();

      _isInitialized = true;
    } catch (e) {
      // Models not yet available - will be added after training
      print('ML models not yet available: $e');
      _isInitialized = false;
    }
  }

  Future<void> _loadCardDatabase() async {
    try {
      final jsonStr = await rootBundle.loadString('assets/data/cards.json');

      // Parse JSON array
      final jsonList = json.decode(jsonStr) as List<dynamic>;

      // Convert each JSON object to Card model
      _cardDatabase = jsonList
          .map((json) => Card.fromJson(json as Map<String, dynamic>))
          .toList();

      print('✅ Loaded ${_cardDatabase!.length} cards from database');
    } catch (e) {
      print('Failed to load card database: $e');
      _cardDatabase = null;
    }
  }

  Future<void> _loadEmbeddingIndex() async {
    try {
      final bytes = await rootBundle.load('assets/indices/riftbound.bin');

      // Parse binary format: [num_cards × 384] float32 array
      final floatList = bytes.buffer.asFloat32List();

      _embeddingIndex = [];
      for (var i = 0; i < floatList.length; i += embeddingDim) {
        if (i + embeddingDim <= floatList.length) {
          final embedding = floatList.sublist(i, i + embeddingDim);
          _embeddingIndex!.add(List<double>.from(embedding));
        }
      }

      print('✅ Loaded ${_embeddingIndex!.length} pre-computed embeddings');

      // Verify count matches database
      if (_cardDatabase != null &&
          _embeddingIndex!.length != _cardDatabase!.length) {
        print('⚠️  Embedding count mismatch!');
        print('   Cards: ${_cardDatabase!.length}');
        print('   Embeddings: ${_embeddingIndex!.length}');
      }
    } catch (e) {
      print('Embedding index not available: $e');
      print('   Will compute embeddings on-demand (slower startup)');
      _embeddingIndex = null;
    }
  }

  /// Detect cards in an image frame.
  /// Returns list of detected card regions.
  Future<List<DetectionResult>> detectCards(FrameData frameData) async {
    if (_detectionModel == null) {
      return [];
    }

    final stopwatch = Stopwatch()..start();

    // Preprocess image to model input format
    final input = await _preprocessForDetection(frameData);

    // Run inference
    final outputShape = _detectionModel!.getOutputTensor(0).shape;
    final output = List.generate(
      outputShape[0],
      (_) => List.generate(
        outputShape[1],
        (_) => List.filled(outputShape[2], 0.0),
      ),
    );

    _detectionModel!.run(input, output);

    // Parse YOLO output to detection results
    final detections = _parseDetections(output, confidenceThreshold: 0.5);

    stopwatch.stop();
    print('Detection took ${stopwatch.elapsedMilliseconds}ms');

    return detections;
  }

  /// Generate embedding for a cropped card image.
  Future<List<double>> generateEmbedding(Uint8List cardImage) async {
    if (_embeddingModel == null) {
      return List.filled(embeddingDim, 0.0);
    }

    final stopwatch = Stopwatch()..start();

    // Preprocess image
    final input = await _preprocessForEmbedding(cardImage);

    // Run inference
    final output = List.filled(embeddingDim, 0.0).reshape([1, embeddingDim]);
    _embeddingModel!.run(input, output);

    // Normalize embedding
    final embedding = _normalizeEmbedding(output[0] as List<double>);

    stopwatch.stop();
    print('Embedding took ${stopwatch.elapsedMilliseconds}ms');

    return embedding;
  }

  /// Search for matching cards given an embedding.
  Future<List<RecognitionResult>> searchCards(
    List<double> embedding, {
    int topK = 5,
  }) async {
    if (_embeddingIndex == null || _cardDatabase == null) {
      return [];
    }

    final stopwatch = Stopwatch()..start();

    // Compute cosine similarity with all cards
    final similarities = <MapEntry<int, double>>[];
    for (var i = 0; i < _embeddingIndex!.length; i++) {
      final sim = _cosineSimilarity(embedding, _embeddingIndex![i]);
      similarities.add(MapEntry(i, sim));
    }

    // Sort by similarity and take top K
    similarities.sort((a, b) => b.value.compareTo(a.value));
    final topMatches = similarities.take(topK);

    stopwatch.stop();

    return topMatches.map((entry) {
      return RecognitionResult(
        card: _cardDatabase![entry.key],
        similarity: entry.value,
        latencyMs: stopwatch.elapsedMilliseconds,
      );
    }).toList();
  }

  /// Full pipeline: detect cards and recognize each one.
  Future<List<ScanResult>> scanFrame(FrameData frameData) async {
    final totalStopwatch = Stopwatch()..start();

    // Step 1: Detect cards
    final detections = await detectCards(frameData);

    final results = <ScanResult>[];

    // Convert FrameData to Image for cropping (temporary until we optimize cropping)
    final image = img.Image.fromBytes(
      width: frameData.width,
      height: frameData.height,
      bytes: frameData.bytes.buffer,
      numChannels: 3,
    );

    for (final detection in detections) {
      // Step 2: Crop detected region
      final cardImage = await _cropRegion(image, detection.boundingBox);

      // Step 3: Generate embedding
      final embedding = await generateEmbedding(cardImage);

      // Step 4: Search for matches
      final matches = await searchCards(embedding);

      results.add(ScanResult(
        detection: detection,
        matches: matches,
        totalLatencyMs: totalStopwatch.elapsedMilliseconds,
      ));
    }

    totalStopwatch.stop();
    print('Total scan took ${totalStopwatch.elapsedMilliseconds}ms');

    return results;
  }

  // --- Private Helper Methods ---

  Future<List<List<List<List<double>>>>> _preprocessForDetection(
    FrameData frameData,
  ) async {
    // Step 1: Create Image from raw RGB bytes (no JPEG decoding needed)
    img.Image image = img.Image.fromBytes(
      width: frameData.width,
      height: frameData.height,
      bytes: frameData.bytes.buffer,
      numChannels: 3, // RGB format
    );


    // Step 2: Resize with letterboxing to 640x640
    // Letterboxing maintains aspect ratio and adds gray padding
    // This prevents card distortion which would hurt detection accuracy
    final int targetSize = detectionInputSize;

    // Calculate scale to fit image within 640x640
    final double scale = min(
      targetSize / image.width,
      targetSize / image.height,
    );

    final int newWidth = (image.width * scale).round();
    final int newHeight = (image.height * scale).round();

    // Resize image to fit within target size
    img.Image resized = img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.linear,
    );

    // Create 640x640 canvas with gray padding (value: 114)
    // Gray padding is standard for YOLO preprocessing
    // Explicitly specify 3 channels for RGB
    img.Image padded = img.Image(
      width: targetSize,
      height: targetSize,
      numChannels: 3, // RGB: 3 channels
    );
    img.fill(padded, color: img.ColorRgb8(114, 114, 114));

    // Center the resized image on the padded canvas
    final int offsetX = (targetSize - newWidth) ~/ 2;
    final int offsetY = (targetSize - newHeight) ~/ 2;
    img.compositeImage(padded, resized, dstX: offsetX, dstY: offsetY);

    // Step 3: Convert to normalized tensor in NHWC format [1, 640, 640, 3]
    // TFLite model expects channels-last format (Height, Width, Channels)
    final tensor = List.generate(
      1, // batch size
      (_) => List.generate(
        targetSize, // height
        (y) => List.generate(
          targetSize, // width
          (x) {
            final pixel = padded.getPixel(x, y);

            // Return RGB values as a list [R, G, B], normalized [0-255] → [0-1]
            return [
              pixel.r / 255.0, // Red channel
              pixel.g / 255.0, // Green channel
              pixel.b / 255.0, // Blue channel
            ];
          },
        ),
      ),
    );

    return tensor;
  }

  Future<List<List<List<List<double>>>>> _preprocessForEmbedding(
    Uint8List cardImage,
  ) async {
    // Step 1: Decode image from bytes
    img.Image? image = img.decodeImage(cardImage);
    if (image == null) {
      throw Exception('Failed to decode card image');
    }

    // Step 2: Resize to 224x224 (FastViT input size)
    final int targetSize = embeddingInputSize;

    img.Image resized = img.copyResize(
      image,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.linear,
    );

    // Step 3: Extract pixel data and normalize with ImageNet statistics
    const imagenetMean = [0.485, 0.456, 0.406]; // RGB means
    const imagenetStd = [0.229, 0.224, 0.225];  // RGB standard deviations

    // Create tensor in NCHW format: [1, 3, 224, 224]
    // TFLite model expects channels-first format!
    final tensor = List.generate(
      1, // batch size
      (_) => List.generate(
        3, // channels (R, G, B)
        (c) => List.generate(
          targetSize, // height
          (y) => List.generate(
            targetSize, // width
            (x) {
              final pixel = resized.getPixel(x, y);
              double value;

              // Get the appropriate channel value
              if (c == 0) {
                value = pixel.r / 255.0; // Red channel
              } else if (c == 1) {
                value = pixel.g / 255.0; // Green channel
              } else {
                value = pixel.b / 255.0; // Blue channel
              }

              // Apply ImageNet normalization: (value - mean) / std
              return (value - imagenetMean[c]) / imagenetStd[c];
            },
          ),
        ),
      ),
    );

    return tensor;
  }

  List<DetectionResult> _parseDetections(
    List<List<List<double>>> output, {
    double confidenceThreshold = 0.5,
  }) {
    final detections = <DetectionResult>[];

    // YOLOv8 output format: [1, num_detections, 5]
    // Each detection: [x_center, y_center, width, height, confidence]
    // Coordinates are in pixels relative to 640x640 input image

    if (output.isEmpty || output[0].isEmpty) {
      return detections;
    }

    final numDetections = output[0].length;

    // Step 1: Parse and filter by confidence
    for (var i = 0; i < numDetections; i++) {
      final detection = output[0][i];

      if (detection.length < 5) continue;

      final confidence = detection[4];

      // Filter low-confidence detections
      if (confidence < confidenceThreshold) continue;

      // Extract box coordinates (center format, in pixels)
      final xCenter = detection[0];
      final yCenter = detection[1];
      final width = detection[2];
      final height = detection[3];

      // Convert center format to corner format (x1, y1, x2, y2)
      final x1 = xCenter - width / 2;
      final y1 = yCenter - height / 2;
      final x2 = xCenter + width / 2;
      final y2 = yCenter + height / 2;

      // Normalize coordinates to [0, 1] range
      // YOLO output is in pixels relative to 640x640 input
      final normalizedBox = Rect.fromLTRB(
        (x1 / detectionInputSize).clamp(0.0, 1.0),
        (y1 / detectionInputSize).clamp(0.0, 1.0),
        (x2 / detectionInputSize).clamp(0.0, 1.0),
        (y2 / detectionInputSize).clamp(0.0, 1.0),
      );

      detections.add(DetectionResult(
        boundingBox: normalizedBox,
        confidence: confidence,
        angle: 0.0, // YOLOv8 doesn't predict rotation by default
      ));
    }

    // Step 2: Apply Non-Maximum Suppression (NMS)
    // Remove overlapping detections, keeping only the highest confidence ones
    final filteredDetections = _applyNMS(detections, iouThreshold: 0.45);

    return filteredDetections;
  }

  /// Apply Non-Maximum Suppression to remove duplicate detections.
  List<DetectionResult> _applyNMS(
    List<DetectionResult> detections, {
    double iouThreshold = 0.45,
  }) {
    if (detections.length <= 1) return detections;

    // Sort by confidence (highest first)
    final sorted = List<DetectionResult>.from(detections)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final kept = <DetectionResult>[];

    for (final detection in sorted) {
      var shouldKeep = true;

      // Check if this detection overlaps significantly with any kept detection
      for (final keptDetection in kept) {
        final iou = _calculateIoU(
          detection.boundingBox,
          keptDetection.boundingBox,
        );

        // If overlap is too high, discard this detection
        if (iou > iouThreshold) {
          shouldKeep = false;
          break;
        }
      }

      if (shouldKeep) {
        kept.add(detection);
      }
    }

    return kept;
  }

  /// Calculate Intersection over Union (IoU) between two bounding boxes.
  double _calculateIoU(Rect box1, Rect box2) {
    // Calculate intersection area
    final intersectionLeft = max(box1.left, box2.left);
    final intersectionTop = max(box1.top, box2.top);
    final intersectionRight = min(box1.right, box2.right);
    final intersectionBottom = min(box1.bottom, box2.bottom);

    if (intersectionRight < intersectionLeft ||
        intersectionBottom < intersectionTop) {
      return 0.0; // No intersection
    }

    final intersectionWidth = intersectionRight - intersectionLeft;
    final intersectionHeight = intersectionBottom - intersectionTop;
    final intersectionArea = intersectionWidth * intersectionHeight;

    // Calculate union area
    final box1Area = box1.width * box1.height;
    final box2Area = box2.width * box2.height;
    final unionArea = box1Area + box2Area - intersectionArea;

    if (unionArea == 0.0) return 0.0;

    return intersectionArea / unionArea;
  }

  List<double> _normalizeEmbedding(List<double> embedding) {
    // Calculate L2 norm: sqrt(sum of squares)
    final sumOfSquares = embedding.fold<double>(
      0.0,
      (sum, val) => sum + val * val,
    );
    final magnitude = sumOfSquares > 0 ? sqrt(sumOfSquares) : 1.0;

    // Normalize: divide each element by magnitude
    return embedding.map((v) => v / magnitude).toList();
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    var dotProduct = 0.0;
    for (var i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
    }
    return dotProduct; // Embeddings are already normalized
  }

  Future<Uint8List> _cropRegion(img.Image image, ui.Rect rect) async {
    // Step 1: Convert normalized coordinates to pixel coordinates
    // rect is in normalized [0, 1] range, convert to actual pixels
    final int x = (rect.left * image.width).round().clamp(0, image.width - 1);
    final int y = (rect.top * image.height).round().clamp(0, image.height - 1);
    final int width = (rect.width * image.width).round();
    final int height = (rect.height * image.height).round();

    // Ensure crop dimensions are valid
    final int cropWidth = min(width, image.width - x);
    final int cropHeight = min(height, image.height - y);

    if (cropWidth <= 0 || cropHeight <= 0) {
      throw Exception('Invalid crop dimensions');
    }

    // Step 3: Crop the region
    img.Image cropped = img.copyCrop(
      image,
      x: x,
      y: y,
      width: cropWidth,
      height: cropHeight,
    );

    // Step 4: Encode back to bytes (JPEG for efficiency)
    final croppedBytes = img.encodeJpg(cropped, quality: 95);

    return Uint8List.fromList(croppedBytes);
  }

  /// Release model resources.
  void dispose() {
    _detectionModel?.close();
    _embeddingModel?.close();
    _isInitialized = false;
  }
}
