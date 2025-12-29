import '../models/recognition_result.dart';

/// Stabilizes recognition results to prevent flickering between similar cards.
/// 
/// Uses a voting system to lock onto a card after consistent detections,
/// preventing the UI from jumping between similar matches.
class RecognitionStabilizer {
  /// Number of consistent frames required to lock onto a card.
  final int lockThreshold;
  
  /// Number of frames to maintain lock after losing detection.
  final int holdFrames;
  
  /// Minimum similarity difference to consider switching cards.
  final double switchThreshold;
  
  /// Vote counts per card ID.
  final Map<String, int> _votes = {};
  
  /// Currently locked card ID.
  String? _lockedCardId;
  
  /// Frames remaining in hold mode.
  int _holdRemaining = 0;
  
  /// Last stable recognition result.
  RecognitionResult? _lastStableResult;

  RecognitionStabilizer({
    this.lockThreshold = 4,
    this.holdFrames = 8,
    this.switchThreshold = 0.15,
  });

  /// Process recognition candidates and return stabilized result.
  /// 
  /// Returns the most stable recognition, which may be:
  /// - The locked card if consistently detected
  /// - null if not yet confident
  /// - A held result if card temporarily lost
  RecognitionResult? stabilize(List<RecognitionResult> candidates) {
    if (candidates.isEmpty) {
      return _handleNoDetection();
    }

    final topCandidate = candidates.first;
    final topCardId = topCandidate.card.productId.toString();
    
    // Update votes
    _votes[topCardId] = (_votes[topCardId] ?? 0) + 1;
    
    // Decay other votes
    _votes.forEach((id, count) {
      if (id != topCardId && count > 0) {
        _votes[id] = count - 1;
      }
    });
    _votes.removeWhere((id, count) => count <= 0);

    // Check if we should lock onto this card
    final topVotes = _votes[topCardId] ?? 0;
    
    if (_lockedCardId == null) {
      // Not locked yet - check if we have enough votes
      if (topVotes >= lockThreshold) {
        _lockOnto(topCandidate);
        return _lastStableResult;
      }
      // Not confident enough yet
      return null;
    }
    
    // Already locked onto a card
    if (topCardId == _lockedCardId) {
      // Same card - refresh hold and return
      _holdRemaining = holdFrames;
      _lastStableResult = topCandidate;
      return _lastStableResult;
    }
    
    // Different card detected - should we switch?
    final lockedVotes = _votes[_lockedCardId] ?? 0;
    
    // Only switch if new card has significantly more votes
    if (topVotes > lockedVotes + lockThreshold) {
      // Check similarity difference
      final lockedResult = candidates.where((r) => r.card.productId.toString() == _lockedCardId).firstOrNull;
      if (lockedResult == null || 
          topCandidate.similarity - lockedResult.similarity > switchThreshold) {
        _lockOnto(topCandidate);
        return _lastStableResult;
      }
    }
    
    // Keep current lock
    return _lastStableResult;
  }

  /// Handle case when no cards are detected.
  RecognitionResult? _handleNoDetection() {
    if (_holdRemaining > 0) {
      _holdRemaining--;
      return _lastStableResult;
    }
    
    // Hold expired - reset
    reset();
    return null;
  }

  /// Lock onto a new card.
  void _lockOnto(RecognitionResult result) {
    _lockedCardId = result.card.productId.toString();
    _lastStableResult = result;
    _holdRemaining = holdFrames;
    
    // Clear other votes to solidify the lock
    _votes.clear();
    _votes[result.card.productId.toString()] = lockThreshold;
  }

  /// Reset the stabilizer.
  void reset() {
    _votes.clear();
    _lockedCardId = null;
    _holdRemaining = 0;
    _lastStableResult = null;
  }

  /// Check if currently locked onto a card.
  bool get isLocked => _lockedCardId != null;
  
  /// Get the currently locked card ID.
  String? get lockedCardId => _lockedCardId;
  
  /// Get the current stable result.
  RecognitionResult? get currentResult => _lastStableResult;
  
  /// Get confidence level (0-1) based on vote strength.
  double get confidence {
    if (_lockedCardId == null) return 0.0;
    final votes = _votes[_lockedCardId] ?? 0;
    return (votes / (lockThreshold * 2)).clamp(0.0, 1.0);
  }
}

/// Manages multiple recognition stabilizers for multiple tracked cards.
class MultiCardRecognitionStabilizer {
  final Map<int, RecognitionStabilizer> _stabilizers = {};
  
  /// Get or create a stabilizer for a tracked card.
  RecognitionStabilizer getStabilizer(int trackedCardId) {
    return _stabilizers.putIfAbsent(
      trackedCardId,
      () => RecognitionStabilizer(),
    );
  }
  
  /// Remove stabilizer for a tracked card.
  void removeStabilizer(int trackedCardId) {
    _stabilizers.remove(trackedCardId);
  }
  
  /// Reset all stabilizers.
  void reset() {
    _stabilizers.clear();
  }
  
  /// Clean up stabilizers for cards that are no longer tracked.
  void cleanup(Set<int> activeCardIds) {
    _stabilizers.removeWhere((id, _) => !activeCardIds.contains(id));
  }
}
