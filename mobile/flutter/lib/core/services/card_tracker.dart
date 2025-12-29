import 'dart:math';
import 'dart:ui';

import '../models/recognition_result.dart';

/// Tracked card with smoothed position and stable identity.
class TrackedCard {
  /// Unique ID for this tracked card.
  final int id;
  
  /// Current smoothed bounding box (normalized 0-1).
  Rect smoothedBox;
  
  /// Raw detection from last update.
  DetectionResult? lastDetection;
  
  /// Best recognition result for this card.
  RecognitionResult? recognition;
  
  /// Velocity for predictive tracking (normalized units per frame).
  Offset velocity;
  
  /// Frames since this card was last detected.
  int framesSinceLastSeen;
  
  /// Total frames this card has been tracked.
  int framesTracked;
  
  /// Timestamp of last update.
  DateTime lastUpdateTime;

  TrackedCard({
    required this.id,
    required this.smoothedBox,
    this.lastDetection,
    this.recognition,
    this.velocity = Offset.zero,
    this.framesSinceLastSeen = 0,
    this.framesTracked = 0,
  }) : lastUpdateTime = DateTime.now();
  
  /// Returns true if this card is considered stable (tracked for multiple frames).
  bool get isStable => framesTracked >= 3;
  
  /// Returns true if this card should be removed (not seen for too long).
  bool get isExpired => framesSinceLastSeen > 10;
  
  /// Center point of the bounding box.
  Offset get center => Offset(
    smoothedBox.left + smoothedBox.width / 2,
    smoothedBox.top + smoothedBox.height / 2,
  );
}

/// Manages card tracking with temporal smoothing and identity persistence.
/// 
/// Provides:
/// - Smooth bounding box transitions using EMA
/// - Object tracking across frames using IoU matching
/// - Predictive positioning based on velocity
/// - Recognition stabilization
class CardTracker {
  /// Smoothing factor for exponential moving average (0-1).
  /// Lower = smoother but more lag. Higher = more responsive but jittery.
  final double smoothingFactor;
  
  /// IoU threshold for matching cards between frames.
  final double iouThreshold;
  
  /// Maximum frames a card can be missing before removal.
  final int maxMissingFrames;
  
  /// Currently tracked cards.
  final Map<int, TrackedCard> _trackedCards = {};
  
  /// Next ID to assign.
  int _nextId = 0;

  CardTracker({
    this.smoothingFactor = 0.35,
    this.iouThreshold = 0.3,
    this.maxMissingFrames = 10,
  });

  /// Update tracked cards with new detections.
  /// 
  /// Returns list of tracked cards with smoothed positions.
  List<TrackedCard> update(List<DetectionResult> detections) {
    // Mark all cards as potentially missing
    for (final card in _trackedCards.values) {
      card.framesSinceLastSeen++;
    }

    // Match new detections to existing tracked cards
    final matchedDetections = <int>{};
    
    for (final detection in detections) {
      final matchedId = _findBestMatch(detection);
      
      if (matchedId != null) {
        // Update existing tracked card
        _updateTrackedCard(matchedId, detection);
        matchedDetections.add(matchedId);
      } else {
        // Create new tracked card
        _createTrackedCard(detection);
      }
    }

    // Apply predictive motion to unmatched cards
    for (final card in _trackedCards.values) {
      if (!matchedDetections.contains(card.id) && card.velocity != Offset.zero) {
        _applyPredictiveMotion(card);
      }
    }

    // Remove expired cards
    _trackedCards.removeWhere((id, card) => card.isExpired);

    // Return sorted list (most confident first)
    final cards = _trackedCards.values.toList();
    cards.sort((a, b) {
      // Prioritize stable cards
      if (a.isStable != b.isStable) return a.isStable ? -1 : 1;
      // Then by confidence
      final aConf = a.lastDetection?.confidence ?? 0;
      final bConf = b.lastDetection?.confidence ?? 0;
      return bConf.compareTo(aConf);
    });
    
    return cards;
  }

  /// Find the best matching tracked card for a detection.
  int? _findBestMatch(DetectionResult detection) {
    int? bestId;
    double bestIou = iouThreshold;

    for (final entry in _trackedCards.entries) {
      final card = entry.value;
      final iou = _calculateIoU(detection.boundingBox, card.smoothedBox);
      
      if (iou > bestIou) {
        bestIou = iou;
        bestId = entry.key;
      }
    }

    return bestId;
  }

  /// Update an existing tracked card with a new detection.
  void _updateTrackedCard(int id, DetectionResult detection) {
    final card = _trackedCards[id];
    if (card == null) return;

    final newBox = detection.boundingBox;
    final oldCenter = card.center;
    
    // Apply exponential moving average smoothing
    final smoothedBox = Rect.fromLTRB(
      _lerp(card.smoothedBox.left, newBox.left, smoothingFactor),
      _lerp(card.smoothedBox.top, newBox.top, smoothingFactor),
      _lerp(card.smoothedBox.right, newBox.right, smoothingFactor),
      _lerp(card.smoothedBox.bottom, newBox.bottom, smoothingFactor),
    );
    
    // Calculate velocity (for predictive tracking)
    final newCenter = Offset(
      smoothedBox.left + smoothedBox.width / 2,
      smoothedBox.top + smoothedBox.height / 2,
    );
    final velocityAlpha = 0.3;
    card.velocity = Offset(
      _lerp(card.velocity.dx, newCenter.dx - oldCenter.dx, velocityAlpha),
      _lerp(card.velocity.dy, newCenter.dy - oldCenter.dy, velocityAlpha),
    );

    card.smoothedBox = smoothedBox;
    card.lastDetection = detection;
    card.framesSinceLastSeen = 0;
    card.framesTracked++;
    card.lastUpdateTime = DateTime.now();
  }

  /// Create a new tracked card from a detection.
  void _createTrackedCard(DetectionResult detection) {
    final id = _nextId++;
    _trackedCards[id] = TrackedCard(
      id: id,
      smoothedBox: detection.boundingBox,
      lastDetection: detection,
      framesTracked: 1,
    );
  }

  /// Apply predictive motion to a card that wasn't detected this frame.
  void _applyPredictiveMotion(TrackedCard card) {
    // Decay velocity over time
    card.velocity = card.velocity * 0.8;
    
    // Apply velocity to position
    card.smoothedBox = card.smoothedBox.translate(
      card.velocity.dx,
      card.velocity.dy,
    );
    
    // Clamp to valid range
    card.smoothedBox = Rect.fromLTRB(
      card.smoothedBox.left.clamp(0.0, 1.0),
      card.smoothedBox.top.clamp(0.0, 1.0),
      card.smoothedBox.right.clamp(0.0, 1.0),
      card.smoothedBox.bottom.clamp(0.0, 1.0),
    );
  }

  /// Update recognition result for a tracked card.
  void updateRecognition(int cardId, RecognitionResult? recognition) {
    _trackedCards[cardId]?.recognition = recognition;
  }

  /// Clear all tracked cards.
  void reset() {
    _trackedCards.clear();
    _nextId = 0;
  }

  /// Linear interpolation.
  double _lerp(double a, double b, double t) => a + (b - a) * t;

  /// Calculate Intersection over Union between two rectangles.
  double _calculateIoU(Rect a, Rect b) {
    final intersectionLeft = max(a.left, b.left);
    final intersectionTop = max(a.top, b.top);
    final intersectionRight = min(a.right, b.right);
    final intersectionBottom = min(a.bottom, b.bottom);

    if (intersectionRight <= intersectionLeft || intersectionBottom <= intersectionTop) {
      return 0.0;
    }

    final intersectionArea = 
        (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
    final aArea = a.width * a.height;
    final bArea = b.width * b.height;
    final unionArea = aArea + bArea - intersectionArea;

    if (unionArea <= 0) return 0.0;
    return intersectionArea / unionArea;
  }

  /// Get current tracked cards.
  List<TrackedCard> get trackedCards => _trackedCards.values.toList();
  
  /// Get number of tracked cards.
  int get count => _trackedCards.length;
}
