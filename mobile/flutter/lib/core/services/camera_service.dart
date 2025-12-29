import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

import '../models/frame_data.dart';

/// High-performance camera service optimized for iOS.
class CameraService {
  CameraController? _controller;
  bool _isProcessing = false;
  int _frameCount = 0;

  /// Frame processing callback.
  Function(FrameData frameData)? onFrameAvailable;

  /// Initialize camera with optimal settings for card scanning.
  Future<void> initialize() async {
    final cameras = await availableCameras();

    if (cameras.isEmpty) {
      throw Exception('No cameras available');
    }

    // Use back camera for scanning
    final camera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      camera,
      ResolutionPreset.medium, // 720p - balance between quality and performance
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420, // Efficient format
    );

    await _controller!.initialize();

    // Configure for best performance
    await _controller!.setFocusMode(FocusMode.auto);
    await _controller!.setExposureMode(ExposureMode.auto);

    print('✅ Camera initialized: ${_controller!.value.previewSize}');
  }

  /// Start processing camera frames.
  void startProcessing({int frameSkip = 15}) {
    if (_controller == null || !_controller!.value.isInitialized) {
      throw Exception('Camera not initialized');
    }

    _controller!.startImageStream((CameraImage image) {
      _frameCount++;

      // Throttle: Process every Nth frame (~5 FPS at 30 FPS camera)
      if (_frameCount % frameSkip != 0) return;

      // Skip if still processing previous frame
      if (_isProcessing) return;

      // Process frame without blocking camera stream
      _processFrame(image).then((_) {
        // Frame processed successfully
      }).catchError((e) {
        print('⚠️  Frame processing error: $e');
      });
    });

    print('✅ Camera processing started (every ${frameSkip}th frame)');
  }

  /// Stop processing camera frames.
  Future<void> stopProcessing() async {
    if (_controller == null) return;

    try {
      await _controller!.stopImageStream();
      print('✅ Camera processing stopped');
    } catch (e) {
      // Already stopped
    }
  }

  /// Process a single camera frame.
  Future<void> _processFrame(CameraImage image) async {
    _isProcessing = true;

    try {
      // Convert CameraImage to RGB bytes
      final imageBytes = await _convertCameraImage(image);

      // Send to ML pipeline
      onFrameAvailable?.call(imageBytes);
    } catch (e) {
      print('⚠️  Frame processing error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  /// Convert CameraImage (YUV420) to RGB bytes.
  ///
  /// Optimized for iOS camera format.
  Future<FrameData> _convertCameraImage(CameraImage image) async {
    try {
      // iOS uses YUV420 format (efficient)
      if (image.format.group == ImageFormatGroup.yuv420) {
        return _convertYUV420toRGB(image);
      }

      // Fallback for other formats
      throw UnsupportedError('Unsupported image format: ${image.format.group}');
    } catch (e) {
      print('⚠️  Image conversion error: $e');
      rethrow;
    }
  }

  /// Convert YUV420 to RGB (optimized implementation).
  ///
  /// iOS uses NV12 format (biplanar): Y plane + interleaved UV plane
  /// Android may use YV12 or I420 (triplanar): Y + U + V separate planes
  FrameData _convertYUV420toRGB(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    // Create RGB byte buffer directly (RGB888 format: 3 bytes per pixel)
    final rgbBytes = Uint8List(width * height * 3);

    // Get Y plane (always first plane)
    final yPlane = image.planes[0];

    // Check if we have NV12 (2 planes) or I420/YV12 (3 planes)
    final bool isNV12 = image.planes.length == 2;

    if (isNV12) {
      // iOS NV12 format: Plane 0 = Y, Plane 1 = interleaved UV
      final uvPlane = image.planes[1];

      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          // Get Y value
          final int yIndex = row * yPlane.bytesPerRow + col;
          final int y = yPlane.bytes[yIndex];

          // Get UV values (interleaved: UVUVUV...)
          // UV plane is half resolution (4:2:0 subsampling)
          final int uvRow = row ~/ 2;
          final int uvCol = col ~/ 2;
          final int uvIndex = uvRow * uvPlane.bytesPerRow + (uvCol * 2);

          // In NV12, U and V are interleaved: U at even indices, V at odd
          final int u = uvPlane.bytes[uvIndex] - 128;
          final int v = uvPlane.bytes[uvIndex + 1] - 128;

          // Convert to RGB using BT.601 color space conversion
          int r = (y + 1.402 * v).round().clamp(0, 255);
          int g = (y - 0.344136 * u - 0.714136 * v).round().clamp(0, 255);
          int b = (y + 1.772 * u).round().clamp(0, 255);

          // Store RGB bytes (RGB888 format)
          final int pixelIndex = (row * width + col) * 3;
          rgbBytes[pixelIndex] = r;
          rgbBytes[pixelIndex + 1] = g;
          rgbBytes[pixelIndex + 2] = b;
        }
      }
    } else {
      // I420 or YV12 format: 3 separate planes
      final uPlane = image.planes[1];
      final vPlane = image.planes[2];

      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          final int yIndex = row * yPlane.bytesPerRow + col;
          final int uvRow = row ~/ 2;
          final int uvCol = col ~/ 2;
          final int uvIndex = uvRow * uPlane.bytesPerRow + uvCol;

          final int y = yPlane.bytes[yIndex];
          final int u = uPlane.bytes[uvIndex] - 128;
          final int v = vPlane.bytes[uvIndex] - 128;

          // Convert to RGB
          int r = (y + 1.402 * v).round().clamp(0, 255);
          int g = (y - 0.344136 * u - 0.714136 * v).round().clamp(0, 255);
          int b = (y + 1.772 * u).round().clamp(0, 255);

          // Store RGB bytes (RGB888 format)
          final int pixelIndex = (row * width + col) * 3;
          rgbBytes[pixelIndex] = r;
          rgbBytes[pixelIndex + 1] = g;
          rgbBytes[pixelIndex + 2] = b;
        }
      }
    }

    // Return raw RGB bytes with dimensions (no JPEG encoding)
    return FrameData(
      bytes: rgbBytes,
      width: width,
      height: height,
    );
  }

  /// Get camera controller for preview.
  CameraController? get controller => _controller;

  /// Check if camera is initialized.
  bool get isInitialized => _controller?.value.isInitialized ?? false;

  /// Dispose camera resources.
  Future<void> dispose() async {
    await stopProcessing();
    await _controller?.dispose();
    _controller = null;
  }
}
