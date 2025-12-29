import 'dart:typed_data';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

import '../models/frame_data.dart';

/// High-performance camera service optimized for iOS.
/// 
/// Handles camera orientation correctly for portrait mode scanning,
/// rotating frames from landscape sensor to portrait output.
class CameraService {
  CameraController? _controller;
  bool _isProcessing = false;
  int _frameCount = 0;
  
  /// Sensor orientation in degrees (90 for most iOS back cameras).
  int _sensorOrientation = 0;

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
    
    // Store sensor orientation for rotation handling
    _sensorOrientation = camera.sensorOrientation;
    print('📷 Camera sensor orientation: $_sensorOrientation degrees');

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
  /// 
  /// [frameSkip] controls processing frequency. Lower = more responsive but higher CPU.
  /// - 5: ~6 fps processing (recommended for fluid tracking)
  /// - 10: ~3 fps processing (balanced)
  /// - 15: ~2 fps processing (power saving)
  void startProcessing({int frameSkip = 5}) {
    if (_controller == null || !_controller!.value.isInitialized) {
      throw Exception('Camera not initialized');
    }

    _controller!.startImageStream((CameraImage image) {
      _frameCount++;

      // Throttle: Process every Nth frame
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
      // Convert CameraImage to RGB bytes with rotation handling
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
  /// Handles iOS camera rotation so output matches portrait screen orientation.
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

  /// Convert YUV420 to RGB with proper rotation for iOS.
  ///
  /// iOS cameras output in landscape orientation with sensorOrientation = 90.
  /// This method converts and rotates to portrait orientation.
  FrameData _convertYUV420toRGB(CameraImage image) {
    final int rawWidth = image.width;
    final int rawHeight = image.height;
    
    // Determine if we need to rotate (iOS back camera is typically 90 degrees)
    final bool needsRotation = Platform.isIOS && (_sensorOrientation == 90 || _sensorOrientation == 270);
    
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

          // Calculate output position with rotation
          int outRow, outCol;
          if (needsRotation) {
            if (_sensorOrientation == 90) {
              // Rotate 90 degrees clockwise: (row, col) -> (col, height - 1 - row)
              outRow = col;
              outCol = rawHeight - 1 - row;
            } else {
              // Rotate 270 degrees clockwise (or 90 CCW): (row, col) -> (width - 1 - col, row)
              outRow = rawWidth - 1 - col;
              outCol = row;
            }
          } else {
            outRow = row;
            outCol = col;
          }

          // Store RGB bytes (RGB888 format)
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

          // Convert to RGB
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

          // Store RGB bytes (RGB888 format)
          final int pixelIndex = (outRow * outWidth + outCol) * 3;
          rgbBytes[pixelIndex] = r;
          rgbBytes[pixelIndex + 1] = g;
          rgbBytes[pixelIndex + 2] = b;
        }
      }
    }

    // Return rotated RGB bytes with correct portrait dimensions
    return FrameData(
      bytes: rgbBytes,
      width: outWidth,
      height: outHeight,
    );
  }

  /// Get camera controller for preview.
  CameraController? get controller => _controller;

  /// Check if camera is initialized.
  bool get isInitialized => _controller?.value.isInitialized ?? false;
  
  /// Get the sensor orientation.
  int get sensorOrientation => _sensorOrientation;

  /// Dispose camera resources.
  Future<void> dispose() async {
    await stopProcessing();
    await _controller?.dispose();
    _controller = null;
  }
}
