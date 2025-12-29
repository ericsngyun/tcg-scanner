import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

import '../models/frame_data.dart';
import 'ml_service.dart';

/// Static image testing for ML pipeline debugging.
/// 
/// Tests detection/identification on known images to isolate
/// whether issues are in the camera pipeline or ML models.
class MLTestService {
  static final MLTestService instance = MLTestService._();
  MLTestService._();
  
  /// Test detection on a static image from assets.
  /// 
  /// Returns detection results and debug info.
  Future<Map<String, dynamic>> testDetectionOnAsset(String assetPath) async {
    final stopwatch = Stopwatch()..start();
    
    try {
      // Load image from assets
      final ByteData data = await rootBundle.load(assetPath);
      final Uint8List bytes = data.buffer.asUint8List();
      
      // Decode image
      final image = img.decodeImage(bytes);
      if (image == null) {
        return {
          'success': false,
          'error': 'Failed to decode image from $assetPath',
        };
      }
      
      print('🧪 Testing static image: ${image.width}x${image.height}');
      
      // Convert to RGB bytes (FrameData format)
      final rgbBytes = Uint8List(image.width * image.height * 3);
      int idx = 0;
      for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
          final pixel = image.getPixel(x, y);
          rgbBytes[idx++] = pixel.r.toInt();
          rgbBytes[idx++] = pixel.g.toInt();
          rgbBytes[idx++] = pixel.b.toInt();
        }
      }
      
      final frameData = FrameData(
        bytes: rgbBytes,
        width: image.width,
        height: image.height,
      );
      
      // Run ML pipeline
      final results = await MLService.instance.scanFrame(frameData);
      
      stopwatch.stop();
      
      return {
        'success': true,
        'imageWidth': image.width,
        'imageHeight': image.height,
        'detectionCount': results.length,
        'results': results.map((r) => {
          'detection': {
            'confidence': r.detection.confidence,
            'box': {
              'left': r.detection.boundingBox.left,
              'top': r.detection.boundingBox.top,
              'width': r.detection.boundingBox.width,
              'height': r.detection.boundingBox.height,
            },
          },
          'matchCount': r.matches.length,
          'bestMatch': r.bestMatch?.card.name,
          'bestSimilarity': r.bestMatch?.similarity,
        }).toList(),
        'processingTimeMs': stopwatch.elapsedMilliseconds,
      };
    } catch (e) {
      stopwatch.stop();
      return {
        'success': false,
        'error': e.toString(),
        'processingTimeMs': stopwatch.elapsedMilliseconds,
      };
    }
  }
  
  /// Test detection with a solid color image (should have NO detections).
  Future<Map<String, dynamic>> testWithBlankImage({
    int width = 640,
    int height = 480,
  }) async {
    // Create solid gray image
    final rgbBytes = Uint8List(width * height * 3);
    for (int i = 0; i < rgbBytes.length; i += 3) {
      rgbBytes[i] = 128;     // R
      rgbBytes[i + 1] = 128; // G
      rgbBytes[i + 2] = 128; // B
    }
    
    final frameData = FrameData(
      bytes: rgbBytes,
      width: width,
      height: height,
    );
    
    print('🧪 Testing blank image: ${width}x$height');
    
    final stopwatch = Stopwatch()..start();
    final results = await MLService.instance.scanFrame(frameData);
    stopwatch.stop();
    
    return {
      'success': true,
      'imageWidth': width,
      'imageHeight': height,
      'detectionCount': results.length,
      'expectedDetections': 0,
      'passed': results.isEmpty,
      'processingTimeMs': stopwatch.elapsedMilliseconds,
    };
  }
  
  /// Test detection with a known card image from URL.
  Future<Map<String, dynamic>> testWithCardImageUrl(String imageUrl) async {
    // This would download and test - but for now, just return placeholder
    return {
      'success': false,
      'error': 'URL testing not yet implemented',
    };
  }
}
