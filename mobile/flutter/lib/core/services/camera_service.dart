import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// High-performance camera service optimized for iOS.
class CameraService {
  CameraController? _controller;
  bool _isProcessing = false;
  int _frameCount = 0;

  /// Frame processing callback.
  Function(Uint8List imageBytes)? onFrameAvailable;

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
      ResolutionPreset.high, // 1080p for good quality
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
  void startProcessing({int frameSkip = 6}) {
    if (_controller == null || !_controller!.value.isInitialized) {
      throw Exception('Camera not initialized');
    }

    _controller!.startImageStream((CameraImage image) {
      _frameCount++;

      // Throttle: Process every Nth frame (~5 FPS at 30 FPS camera)
      if (_frameCount % frameSkip != 0) return;

      // Skip if still processing previous frame
      if (_isProcessing) return;

      _processFrame(image);
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
  Future<Uint8List> _convertCameraImage(CameraImage image) async {
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
  Uint8List _convertYUV420toRGB(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    // Get YUV planes
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    // Create RGB image
    final rgbImage = img.Image(width: width, height: height);

    // YUV to RGB conversion
    // Formula: Standard BT.601 color space conversion
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

        rgbImage.setPixelRgba(col, row, r, g, b, 255);
      }
    }

    // Encode to JPEG for ML pipeline
    final jpegBytes = img.encodeJpg(rgbImage, quality: 90);
    return Uint8List.fromList(jpegBytes);
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
