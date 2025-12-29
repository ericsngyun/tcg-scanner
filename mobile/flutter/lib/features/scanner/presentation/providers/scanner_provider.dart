import 'dart:async';
import 'dart:typed_data';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../../core/models/frame_data.dart';
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

/// Notifier for scanner state management (Riverpod 3.x compatible).
class ScannerNotifier extends Notifier<ScannerState> {
  /// Sensor orientation for camera (needed for rotation).
  int _sensorOrientation = 90; // Default for iOS back camera

  Timer? _throttleTimer;
  bool _isProcessingFrame = false;

  /// Minimum interval between frame processing (ms).
  static const _frameInterval = Duration(milliseconds: 100);

  @override
  ScannerState build() {
    // Initialize with current ML service state
    final mlService = MLService.instance;
    return ScannerState(isModelLoaded: mlService.isInitialized);
  }
  
  /// Set the sensor orientation (call this after camera initialization).
  void setSensorOrientation(int orientation) {
    _sensorOrientation = orientation;
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
      // Convert camera image to FrameData with proper rotation
      final frameData = _convertCameraImage(image);

      // Run ML pipeline
      final results = await MLService.instance.scanFrame(frameData);

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

  /// Convert CameraImage (YUV420) to FrameData (RGB) with rotation.
  FrameData _convertCameraImage(CameraImage image) {
    final int rawWidth = image.width;
    final int rawHeight = image.height;
    
    // Determine if we need to rotate (iOS back camera is typically 90 degrees)
    final bool needsRotation = Platform.isIOS && 
        (_sensorOrientation == 90 || _sensorOrientation == 270);
    
    // Output dimensions (swapped if rotating 90/270 degrees)
    final int outWidth = needsRotation ? rawHeight : rawWidth;
    final int outHeight = needsRotation ? rawWidth : rawHeight;

    // Create RGB byte buffer (RGB888 format: 3 bytes per pixel)
    final rgbBytes = Uint8List(outWidth * outHeight * 3);

    // Get Y plane (always first plane)
    final yPlane = image.planes[0];

    // Check if we have NV12 (2 planes) or I420/YV12 (3 planes)
    final bool isNV12 = image.planes.length == 2;

    if (isNV12) {
      // iOS NV12 format: Plane 0 = Y, Plane 1 = interleaved UV
      final uvPlane = image.planes[1];

      for (int row = 0; row < rawHeight; row++) {
        for (int col = 0; col < rawWidth; col++) {
          // Get Y value
          final int yIndex = row * yPlane.bytesPerRow + col;
          final int y = yPlane.bytes[yIndex];

          // Get UV values
          final int uvRow = row ~/ 2;
          final int uvCol = col ~/ 2;
          final int uvIndex = uvRow * uvPlane.bytesPerRow + (uvCol * 2);

          final int u = uvPlane.bytes[uvIndex] - 128;
          final int v = uvPlane.bytes[uvIndex + 1] - 128;

          // Convert to RGB using BT.601
          int r = (y + 1.402 * v).round().clamp(0, 255);
          int g = (y - 0.344136 * u - 0.714136 * v).round().clamp(0, 255);
          int b = (y + 1.772 * u).round().clamp(0, 255);

          // Calculate output position with rotation
          int outRow, outCol;
          if (needsRotation) {
            if (_sensorOrientation == 90) {
              outRow = col;
              outCol = rawHeight - 1 - row;
            } else {
              outRow = rawWidth - 1 - col;
              outCol = row;
            }
          } else {
            outRow = row;
            outCol = col;
          }

          // Store RGB bytes
          final int pixelIndex = (outRow * outWidth + outCol) * 3;
          rgbBytes[pixelIndex] = r;
          rgbBytes[pixelIndex + 1] = g;
          rgbBytes[pixelIndex + 2] = b;
        }
      }
    } else {
      // I420 or YV12 format: 3 separate planes
      final uPlane = image.planes[1];
      final vPlane = image.planes[2];

      for (int row = 0; row < rawHeight; row++) {
        for (int col = 0; col < rawWidth; col++) {
          final int yIndex = row * yPlane.bytesPerRow + col;
          final int uvRow = row ~/ 2;
          final int uvCol = col ~/ 2;
          final int uvIndex = uvRow * uPlane.bytesPerRow + uvCol;

          final int y = yPlane.bytes[yIndex];
          final int u = uPlane.bytes[uvIndex] - 128;
          final int v = vPlane.bytes[uvIndex] - 128;

          int r = (y + 1.402 * v).round().clamp(0, 255);
          int g = (y - 0.344136 * u - 0.714136 * v).round().clamp(0, 255);
          int b = (y + 1.772 * u).round().clamp(0, 255);

          int outRow, outCol;
          if (needsRotation) {
            if (_sensorOrientation == 90) {
              outRow = col;
              outCol = rawHeight - 1 - row;
            } else {
              outRow = rawWidth - 1 - col;
              outCol = row;
            }
          } else {
            outRow = row;
            outCol = col;
          }

          final int pixelIndex = (outRow * outWidth + outCol) * 3;
          rgbBytes[pixelIndex] = r;
          rgbBytes[pixelIndex + 1] = g;
          rgbBytes[pixelIndex + 2] = b;
        }
      }
    }

    return FrameData(
      bytes: rgbBytes,
      width: outWidth,
      height: outHeight,
    );
  }

  /// Clear current detection results.
  void clearResults() {
    state = state.copyWith(
      detections: [],
      results: [],
    );
  }
}

/// Global provider for scanner state.
final scannerProvider = NotifierProvider<ScannerNotifier, ScannerState>(() {
  return ScannerNotifier();
});
