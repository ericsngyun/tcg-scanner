import 'dart:ui';

import 'card.dart';

/// Result of card detection in an image frame.
class DetectionResult {
  /// Bounding box in normalized coordinates (0-1).
  final Rect boundingBox;

  /// Detection confidence (0-1).
  final double confidence;

  /// Detected orientation angle in degrees.
  final double angle;

  const DetectionResult({
    required this.boundingBox,
    required this.confidence,
    this.angle = 0.0,
  });

  /// Converts normalized coordinates to pixel coordinates.
  Rect toPixelRect(Size imageSize) {
    return Rect.fromLTWH(
      boundingBox.left * imageSize.width,
      boundingBox.top * imageSize.height,
      boundingBox.width * imageSize.width,
      boundingBox.height * imageSize.height,
    );
  }
}

/// Result of card identification via embedding similarity search.
class RecognitionResult {
  /// The identified card.
  final Card card;

  /// Similarity score (0-1, higher is better).
  final double similarity;

  /// Time taken for recognition in milliseconds.
  final int latencyMs;

  const RecognitionResult({
    required this.card,
    required this.similarity,
    required this.latencyMs,
  });

  /// Returns true if this is a confident match.
  bool get isConfident => similarity >= 0.8;
}

/// Complete result of scanning a card (detection + recognition).
class ScanResult {
  /// Detection result with bounding box.
  final DetectionResult detection;

  /// Top recognition matches (sorted by similarity).
  final List<RecognitionResult> matches;

  /// Total processing time in milliseconds.
  final int totalLatencyMs;

  const ScanResult({
    required this.detection,
    required this.matches,
    required this.totalLatencyMs,
  });

  /// Returns the best match if confidence threshold is met.
  RecognitionResult? get bestMatch =>
      matches.isNotEmpty && matches.first.isConfident ? matches.first : null;
}
