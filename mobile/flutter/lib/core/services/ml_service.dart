import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/card.dart';
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
      // Parse JSON and populate _cardDatabase
      // TODO: Implement after training
    } catch (e) {
      print('Card database not yet available: $e');
    }
  }

  Future<void> _loadEmbeddingIndex() async {
    try {
      final bytes = await rootBundle.load('assets/indices/riftbound.bin');
      // Parse binary embedding index
      // TODO: Implement after training
    } catch (e) {
      print('Embedding index not yet available: $e');
    }
  }

  /// Detect cards in an image frame.
  /// Returns list of detected card regions.
  Future<List<DetectionResult>> detectCards(Uint8List imageBytes) async {
    if (_detectionModel == null) {
      return [];
    }

    final stopwatch = Stopwatch()..start();

    // Preprocess image to model input format
    final input = await _preprocessForDetection(imageBytes);

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
  Future<List<ScanResult>> scanFrame(Uint8List imageBytes) async {
    final totalStopwatch = Stopwatch()..start();

    // Step 1: Detect cards
    final detections = await detectCards(imageBytes);

    final results = <ScanResult>[];

    for (final detection in detections) {
      // Step 2: Crop detected region
      final cardImage = await _cropRegion(imageBytes, detection.boundingBox);

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
    Uint8List imageBytes,
  ) async {
    // Decode image and resize to 640x640
    // Normalize to 0-1 range
    // Return as [1, 640, 640, 3] tensor
    // TODO: Implement with image package
    return List.generate(
      1,
      (_) => List.generate(
        detectionInputSize,
        (_) => List.generate(
          detectionInputSize,
          (_) => List.filled(3, 0.0),
        ),
      ),
    );
  }

  Future<List<List<List<List<double>>>>> _preprocessForEmbedding(
    Uint8List cardImage,
  ) async {
    // Resize to 224x224
    // Normalize with ImageNet mean/std
    // Return as [1, 224, 224, 3] tensor
    // TODO: Implement with image package
    return List.generate(
      1,
      (_) => List.generate(
        embeddingInputSize,
        (_) => List.generate(
          embeddingInputSize,
          (_) => List.filled(3, 0.0),
        ),
      ),
    );
  }

  List<DetectionResult> _parseDetections(
    List<List<List<double>>> output, {
    double confidenceThreshold = 0.5,
  }) {
    final detections = <DetectionResult>[];

    // Parse YOLOv8 output format
    // Output shape is typically [1, num_detections, 6] for (x, y, w, h, conf, class)
    // TODO: Implement proper YOLO output parsing

    return detections;
  }

  List<double> _normalizeEmbedding(List<double> embedding) {
    final norm = embedding.fold<double>(
      0.0,
      (sum, val) => sum + val * val,
    );
    final magnitude = norm > 0 ? 1.0 / (norm * norm) : 1.0;
    return embedding.map((v) => v * magnitude).toList();
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    var dotProduct = 0.0;
    for (var i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
    }
    return dotProduct; // Embeddings are already normalized
  }

  Future<Uint8List> _cropRegion(Uint8List imageBytes, ui.Rect rect) async {
    // Crop the detected region from the image
    // TODO: Implement with image package
    return imageBytes;
  }

  /// Release model resources.
  void dispose() {
    _detectionModel?.close();
    _embeddingModel?.close();
    _isInitialized = false;
  }
}
