import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;

import '../models/frame_data.dart';
import '../models/recognition_result.dart';

/// Simple border-based card detection using edge detection and contour finding.
/// 
/// This provides a more robust alternative to YOLO-based detection,
/// especially since YOLO was trained on synthetic data.
/// 
/// Cards have distinct rectangular borders which can be detected via:
/// 1. Edge detection (Sobel/Canny-like)
/// 2. Contour finding
/// 3. Rectangle fitting
class BorderDetector {
  /// Minimum card size as fraction of frame (cards should be at least 10% of frame).
  final double minCardSize;
  
  /// Maximum card size as fraction of frame (cards shouldn't fill entire frame).
  final double maxCardSize;
  
  /// Expected aspect ratio of TCG cards (height/width ≈ 1.4).
  final double expectedAspectRatio;
  
  /// Tolerance for aspect ratio matching.
  final double aspectRatioTolerance;
  
  /// Edge detection threshold.
  final int edgeThreshold;

  BorderDetector({
    this.minCardSize = 0.05,
    this.maxCardSize = 0.7,
    this.expectedAspectRatio = 1.4, // Standard TCG card is 2.5 x 3.5 inches
    this.aspectRatioTolerance = 0.4,
    this.edgeThreshold = 30,
  });

  /// Detect cards using border/edge detection.
  /// 
  /// Returns normalized bounding boxes (0-1 range).
  List<DetectionResult> detectCards(FrameData frame) {
    final stopwatch = Stopwatch()..start();
    
    // Convert to grayscale image
    final grayscale = _toGrayscale(frame);
    
    // Apply edge detection
    final edges = _detectEdges(grayscale, frame.width, frame.height);
    
    // Find rectangular contours
    final rectangles = _findRectangles(edges, frame.width, frame.height);
    
    // Filter and convert to DetectionResult
    final detections = <DetectionResult>[];
    
    for (final rect in rectangles) {
      // Normalize to 0-1 range
      final normalized = Rect.fromLTRB(
        rect.left / frame.width,
        rect.top / frame.height,
        rect.right / frame.width,
        rect.bottom / frame.height,
      );
      
      // Filter by size
      final area = normalized.width * normalized.height;
      if (area < minCardSize || area > maxCardSize) continue;
      
      // Filter by aspect ratio
      final aspectRatio = normalized.height / normalized.width;
      if ((aspectRatio - expectedAspectRatio).abs() > aspectRatioTolerance) continue;
      
      // Calculate confidence based on how well it matches expected aspect ratio
      final aspectMatch = 1.0 - (aspectRatio - expectedAspectRatio).abs() / aspectRatioTolerance;
      final sizeScore = _calculateSizeScore(area);
      final confidence = (aspectMatch * 0.6 + sizeScore * 0.4).clamp(0.0, 1.0);
      
      detections.add(DetectionResult(
        boundingBox: normalized,
        confidence: confidence,
        angle: 0.0, // Could calculate rotation if needed
      ));
    }
    
    // Sort by confidence
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    
    // Apply non-maximum suppression
    final filtered = _applyNMS(detections, iouThreshold: 0.3);
    
    stopwatch.stop();
    print('🔲 Border detection: ${filtered.length} cards found in ${stopwatch.elapsedMilliseconds}ms');
    
    return filtered;
  }
  
  /// Convert RGB frame to grayscale.
  Uint8List _toGrayscale(FrameData frame) {
    final gray = Uint8List(frame.width * frame.height);
    for (int i = 0; i < gray.length; i++) {
      final r = frame.bytes[i * 3];
      final g = frame.bytes[i * 3 + 1];
      final b = frame.bytes[i * 3 + 2];
      // Luminance formula
      gray[i] = ((0.299 * r) + (0.587 * g) + (0.114 * b)).round();
    }
    return gray;
  }
  
  /// Simple Sobel edge detection.
  Uint8List _detectEdges(Uint8List gray, int width, int height) {
    final edges = Uint8List(width * height);
    
    // Sobel kernels
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
    
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        int gx = 0, gy = 0;
        
        // Apply 3x3 convolution
        for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
            final idx = (y + ky) * width + (x + kx);
            final kidx = (ky + 1) * 3 + (kx + 1);
            gx += gray[idx] * sobelX[kidx];
            gy += gray[idx] * sobelY[kidx];
          }
        }
        
        // Magnitude
        final magnitude = sqrt(gx * gx + gy * gy).round();
        edges[y * width + x] = magnitude > edgeThreshold ? 255 : 0;
      }
    }
    
    return edges;
  }
  
  /// Find rectangular contours in edge image.
  List<Rect> _findRectangles(Uint8List edges, int width, int height) {
    final rectangles = <Rect>[];
    
    // Simple connected component analysis
    // Find bounding boxes of connected white regions
    final visited = List.filled(width * height, false);
    
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final idx = y * width + x;
        if (edges[idx] > 0 && !visited[idx]) {
          // Flood fill to find connected component
          final bounds = _floodFill(edges, visited, x, y, width, height);
          if (bounds != null) {
            // Check if bounds form a reasonable rectangle
            final bw = bounds.width;
            final bh = bounds.height;
            final area = bw * bh;
            
            // Filter tiny/huge regions
            final minPixelArea = width * height * 0.01;
            final maxPixelArea = width * height * 0.8;
            
            if (area > minPixelArea && area < maxPixelArea) {
              rectangles.add(bounds);
            }
          }
        }
      }
    }
    
    return rectangles;
  }
  
  /// Flood fill to find connected component bounds.
  Rect? _floodFill(Uint8List edges, List<bool> visited, int startX, int startY, int width, int height) {
    final stack = <int>[];
    stack.add(startY * width + startX);
    
    int minX = startX, maxX = startX;
    int minY = startY, maxY = startY;
    int pixelCount = 0;
    
    while (stack.isNotEmpty) {
      final idx = stack.removeLast();
      if (idx < 0 || idx >= width * height || visited[idx]) continue;
      if (edges[idx] == 0) continue;
      
      visited[idx] = true;
      pixelCount++;
      
      final x = idx % width;
      final y = idx ~/ width;
      
      minX = min(minX, x);
      maxX = max(maxX, x);
      minY = min(minY, y);
      maxY = max(maxY, y);
      
      // Add neighbors (4-connectivity for speed)
      if (x > 0) stack.add(idx - 1);
      if (x < width - 1) stack.add(idx + 1);
      if (y > 0) stack.add(idx - width);
      if (y < height - 1) stack.add(idx + width);
      
      // Limit to prevent stack overflow on large regions
      if (pixelCount > 50000) break;
    }
    
    if (pixelCount < 50) return null; // Too small
    
    return Rect.fromLTRB(
      minX.toDouble(),
      minY.toDouble(),
      maxX.toDouble(),
      maxY.toDouble(),
    );
  }
  
  /// Calculate size score (cards should be ~10-40% of frame).
  double _calculateSizeScore(double area) {
    const idealMin = 0.1;
    const idealMax = 0.4;
    
    if (area >= idealMin && area <= idealMax) {
      return 1.0;
    } else if (area < idealMin) {
      return area / idealMin;
    } else {
      return idealMax / area;
    }
  }
  
  /// Non-maximum suppression to remove overlapping detections.
  List<DetectionResult> _applyNMS(List<DetectionResult> detections, {double iouThreshold = 0.3}) {
    if (detections.isEmpty) return [];
    
    final filtered = <DetectionResult>[];
    final suppressed = List.filled(detections.length, false);
    
    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;
      
      filtered.add(detections[i]);
      
      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;
        
        final iou = _calculateIoU(detections[i].boundingBox, detections[j].boundingBox);
        if (iou > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
    
    return filtered;
  }
  
  /// Calculate Intersection over Union.
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
}
