import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../../core/models/recognition_result.dart';
import '../../../../core/services/ml_service.dart';

/// State for the scanner feature.
class ScannerState {
  final bool isProcessing;
  final bool isModelLoaded;
  final List<DetectionResult> detections;
  final List<ScanResult> results;
  final String? error;

  const ScannerState({
    this.isProcessing = false,
    this.isModelLoaded = false,
    this.detections = const [],
    this.results = const [],
    this.error,
  });

  ScannerState copyWith({
    bool? isProcessing,
    bool? isModelLoaded,
    List<DetectionResult>? detections,
    List<ScanResult>? results,
    String? error,
  }) {
    return ScannerState(
      isProcessing: isProcessing ?? this.isProcessing,
      isModelLoaded: isModelLoaded ?? this.isModelLoaded,
      detections: detections ?? this.detections,
      results: results ?? this.results,
      error: error,
    );
  }
}

/// Provider for scanner state management.
class ScannerNotifier extends StateNotifier<ScannerState> {
  ScannerNotifier() : super(const ScannerState()) {
    _initialize();
  }

  Timer? _throttleTimer;
  bool _isProcessingFrame = false;

  /// Minimum interval between frame processing (ms).
  static const _frameInterval = Duration(milliseconds: 100);

  Future<void> _initialize() async {
    final mlService = MLService.instance;
    state = state.copyWith(isModelLoaded: mlService.isInitialized);
  }

  /// Process a camera frame for card detection.
  Future<void> processFrame(CameraImage image) async {
    // Skip if already processing or throttled
    if (_isProcessingFrame || _throttleTimer?.isActive == true) {
      return;
    }

    // Skip if models not loaded
    if (!MLService.instance.isInitialized) {
      return;
    }

    _isProcessingFrame = true;
    _throttleTimer = Timer(_frameInterval, () {});

    state = state.copyWith(isProcessing: true);

    try {
      // Convert camera image to bytes
      final bytes = _convertCameraImage(image);

      // Run ML pipeline
      final results = await MLService.instance.scanFrame(bytes);

      // Extract detections for overlay
      final detections = results.map((r) => r.detection).toList();

      state = state.copyWith(
        isProcessing: false,
        detections: detections,
        results: results,
        error: null,
      );
    } catch (e) {
      state = state.copyWith(
        isProcessing: false,
        error: e.toString(),
      );
    } finally {
      _isProcessingFrame = false;
    }
  }

  /// Convert CameraImage to Uint8List for processing.
  Uint8List _convertCameraImage(CameraImage image) {
    // For YUV420 format, extract Y plane and convert to RGB
    // This is a simplified version - full implementation would handle
    // different formats and do proper color space conversion

    final yPlane = image.planes[0];
    final bytes = yPlane.bytes;

    // For now, return raw bytes - proper implementation would
    // convert YUV to RGB using image processing
    return bytes;
  }

  /// Clear current detection results.
  void clearResults() {
    state = state.copyWith(
      detections: [],
      results: [],
    );
  }

  @override
  void dispose() {
    _throttleTimer?.cancel();
    super.dispose();
  }
}

/// Global provider for scanner state.
final scannerProvider =
    StateNotifierProvider<ScannerNotifier, ScannerState>((ref) {
  return ScannerNotifier();
});
