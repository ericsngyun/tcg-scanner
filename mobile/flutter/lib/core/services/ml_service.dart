import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';

import '../models/card.dart';
import '../models/frame_data.dart';
import '../models/recognition_result.dart';
import 'ml_isolate_worker.dart';

/// Singleton service for ML model inference.
///
/// Uses a dedicated isolate for all ML processing to keep the main thread
/// responsive and achieve smooth 60fps camera preview.
class MLService {
  static final MLService instance = MLService._();
  MLService._();

  Isolate? _mlIsolate;
  SendPort? _mlSendPort;
  final _responseController = StreamController<Map<String, dynamic>>.broadcast();

  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  /// Initialize ML isolate and models.
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Load assets in main thread (Flutter bindings are available here)
      print('📦 Loading assets in main thread...');

      final jsonStr = await rootBundle.loadString('assets/data/cards.json');
      final jsonList = json.decode(jsonStr) as List<dynamic>;
      final cardJsonList = jsonList.map((e) => e as Map<String, dynamic>).toList();

      final embeddingBytes = await rootBundle.load('assets/indices/riftbound.bin');

      // Load TFLite model files as bytes (isolates can't use fromAsset)
      final detectionModelBytes = await rootBundle.load('assets/models/detection.tflite');
      final embeddingModelBytes = await rootBundle.load('assets/models/embedding.tflite');

      print('📦 Loaded ${cardJsonList.length} cards, ${embeddingBytes.lengthInBytes} bytes of embeddings, and ${(detectionModelBytes.lengthInBytes + embeddingModelBytes.lengthInBytes)} bytes of models');

      // Create a ReceivePort to get messages from the isolate
      final receivePort = ReceivePort();

      // Spawn the ML processing isolate
      _mlIsolate = await Isolate.spawn(
        mlIsolateEntryPoint,
        receivePort.sendPort,
      );

      print('✅ ML isolate spawned');

      // Set up response stream from isolate
      final completer = Completer<SendPort>();

      receivePort.listen((message) {
        if (message is SendPort) {
          // First message is the isolate's SendPort
          completer.complete(message);
        } else if (message is Map<String, dynamic>) {
          // Subsequent messages are responses
          _responseController.add(message);
        }
      });

      // Get the SendPort to send messages to the isolate
      _mlSendPort = await completer.future;

      // Initialize models in the isolate with loaded assets
      _mlSendPort!.send({
        'type': 'initialize',
        'data': {
          'cardJsonList': cardJsonList,
          'embeddingBytes': embeddingBytes.buffer.asUint8List(),
          'detectionModelBytes': detectionModelBytes.buffer.asUint8List(),
          'embeddingModelBytes': embeddingModelBytes.buffer.asUint8List(),
        },
      });

      // Wait for initialization to complete
      final initResult = await _responseController.stream
          .firstWhere((msg) => msg['type'] == 'initialized');

      if (initResult['success'] == true) {
        _isInitialized = true;
        print('✅ ML isolate initialized successfully');
      } else {
        throw Exception('ML isolate initialization failed: ${initResult['error']}');
      }
    } catch (e) {
      print('ML models not yet available: $e');
      _isInitialized = false;
    }
  }

  /// Full pipeline: detect cards and recognize each one.
  ///
  /// This method sends the frame to the ML isolate for processing,
  /// keeping the main thread responsive for smooth 60fps camera preview.
  Future<List<ScanResult>> scanFrame(FrameData frameData) async {
    if (!_isInitialized || _mlSendPort == null) {
      return [];
    }

    // Send frame to isolate for processing
    _mlSendPort!.send({
      'type': 'scanFrame',
      'data': {
        'bytes': frameData.bytes,
        'width': frameData.width,
        'height': frameData.height,
      },
    });

    // Wait for results from isolate
    final response = await _responseController.stream
        .firstWhere((msg) => msg['type'] == 'scanResult');

    // Convert serialized results back to Dart objects
    final resultsList = response['results'] as List<dynamic>;

    return resultsList.map((resultMap) {
      final detectionMap = resultMap['detection'] as Map<String, dynamic>;
      final boundingBoxMap = detectionMap['boundingBox'] as Map<String, dynamic>;

      final detection = DetectionResult(
        boundingBox: ui.Rect.fromLTRB(
          boundingBoxMap['left'] as double,
          boundingBoxMap['top'] as double,
          boundingBoxMap['right'] as double,
          boundingBoxMap['bottom'] as double,
        ),
        confidence: detectionMap['confidence'] as double,
        angle: detectionMap['angle'] as double,
      );

      final matchesList = resultMap['matches'] as List<dynamic>;
      final matches = matchesList.map((matchMap) {
        return RecognitionResult(
          card: Card.fromJson(matchMap['card'] as Map<String, dynamic>),
          similarity: matchMap['similarity'] as double,
          latencyMs: matchMap['latencyMs'] as int,
        );
      }).toList();

      return ScanResult(
        detection: detection,
        matches: matches,
        totalLatencyMs: resultMap['totalLatencyMs'] as int,
      );
    }).toList();
  }

  /// Release isolate resources.
  void dispose() {
    _mlSendPort?.send({
      'type': 'dispose',
      'data': null,
    });
    _mlIsolate?.kill(priority: Isolate.immediate);
    _mlIsolate = null;
    _mlSendPort = null;
    _responseController.close();
    _isInitialized = false;
  }
}
