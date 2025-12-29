import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;

import '../models/frame_data.dart';
import '../models/recognition_result.dart';

/// Debug service for diagnosing ML pipeline issues.
/// 
/// Provides visibility into:
/// - Camera frames before/after processing
/// - Detection model input and raw outputs
/// - Identification crops and embeddings
class MLDebugService {
  static final MLDebugService instance = MLDebugService._();
  MLDebugService._();
  
  /// Whether debug mode is enabled.
  bool isEnabled = false;
  
  /// Directory for saving debug artifacts.
  Directory? _debugDir;
  
  /// Counter for saved frames.
  int _frameCounter = 0;
  
  /// Last detection stats for display.
  DebugStats? lastStats;
  
  /// Initialize debug directory.
  Future<void> initialize() async {
    final appDir = await getApplicationDocumentsDirectory();
    _debugDir = Directory('${appDir.path}/ml_debug');
    if (!await _debugDir!.exists()) {
      await _debugDir!.create(recursive: true);
    }
    print('📊 Debug directory: ${_debugDir!.path}');
  }
  
  /// Clear all debug artifacts.
  Future<void> clearDebugData() async {
    if (_debugDir != null && await _debugDir!.exists()) {
      await _debugDir!.delete(recursive: true);
      await _debugDir!.create(recursive: true);
    }
    _frameCounter = 0;
  }
  
  /// Save a camera frame for inspection.
  Future<String?> saveFrame(FrameData frame, {String prefix = 'frame'}) async {
    if (!isEnabled || _debugDir == null) return null;
    
    try {
      // Convert RGB bytes to image
      final image = img.Image.fromBytes(
        width: frame.width,
        height: frame.height,
        bytes: frame.bytes.buffer,
        numChannels: 3,
        order: img.ChannelOrder.rgb,
      );
      
      // Save as PNG
      final filename = '${prefix}_${_frameCounter.toString().padLeft(4, '0')}.png';
      final file = File('${_debugDir!.path}/$filename');
      await file.writeAsBytes(img.encodePng(image));
      
      _frameCounter++;
      print('📊 Saved debug frame: $filename (${frame.width}x${frame.height})');
      return file.path;
    } catch (e) {
      print('📊 Failed to save debug frame: $e');
      return null;
    }
  }
  
  /// Log detection model stats.
  void logDetectionStats({
    required int inputWidth,
    required int inputHeight,
    required List<int> inputShape,
    required List<int> outputShape,
    required int rawDetectionCount,
    required int afterNmsCount,
    required int afterFilterCount,
    required double maxConfidence,
    required Duration processingTime,
  }) {
    lastStats = DebugStats(
      inputWidth: inputWidth,
      inputHeight: inputHeight,
      inputShape: inputShape,
      outputShape: outputShape,
      rawDetectionCount: rawDetectionCount,
      afterNmsCount: afterNmsCount,
      afterFilterCount: afterFilterCount,
      maxConfidence: maxConfidence,
      processingTime: processingTime,
    );
    
    if (isEnabled) {
      print('📊 Detection Stats:');
      print('   Input: ${inputWidth}x${inputHeight} → tensor ${inputShape}');
      print('   Output: ${outputShape}');
      print('   Detections: $rawDetectionCount raw → $afterNmsCount NMS → $afterFilterCount final');
      print('   Max confidence: ${(maxConfidence * 100).toStringAsFixed(1)}%');
      print('   Processing: ${processingTime.inMilliseconds}ms');
    }
  }
  
  /// Log all raw detections before filtering.
  void logRawDetections(List<Map<String, dynamic>> detections) {
    if (!isEnabled) return;
    
    print('📊 Raw detections (${detections.length}):');
    for (int i = 0; i < detections.length && i < 10; i++) {
      final d = detections[i];
      final box = d['boundingBox'] as Map<String, dynamic>;
      final conf = d['confidence'] as double;
      print('   [$i] conf=${(conf * 100).toStringAsFixed(1)}% '
          'box=(${box['left']?.toStringAsFixed(2)}, ${box['top']?.toStringAsFixed(2)}) - '
          '(${box['right']?.toStringAsFixed(2)}, ${box['bottom']?.toStringAsFixed(2)})');
    }
    if (detections.length > 10) {
      print('   ... and ${detections.length - 10} more');
    }
  }
  
  /// Get debug directory path for display.
  String? get debugDirectoryPath => _debugDir?.path;
}

/// Debug statistics for display.
class DebugStats {
  final int inputWidth;
  final int inputHeight;
  final List<int> inputShape;
  final List<int> outputShape;
  final int rawDetectionCount;
  final int afterNmsCount;
  final int afterFilterCount;
  final double maxConfidence;
  final Duration processingTime;
  
  DebugStats({
    required this.inputWidth,
    required this.inputHeight,
    required this.inputShape,
    required this.outputShape,
    required this.rawDetectionCount,
    required this.afterNmsCount,
    required this.afterFilterCount,
    required this.maxConfidence,
    required this.processingTime,
  });
  
  @override
  String toString() {
    return 'Input: ${inputWidth}x$inputHeight | '
        'Raw: $rawDetectionCount → Final: $afterFilterCount | '
        'Max: ${(maxConfidence * 100).toStringAsFixed(0)}%';
  }
}
