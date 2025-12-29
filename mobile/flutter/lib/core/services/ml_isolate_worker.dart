import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math';

import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

import '../models/card.dart';
import '../models/frame_data.dart';
import '../models/recognition_result.dart';

/// Message types for isolate communication.
class IsolateMessage {
  final String type;
  final dynamic data;

  IsolateMessage(this.type, this.data);

  Map<String, dynamic> toJson() => {'type': type, 'data': data};
  factory IsolateMessage.fromJson(Map<String, dynamic> json) =>
      IsolateMessage(json['type'] as String, json['data']);
}

/// Isolate worker entry point for ML processing.
///
/// This runs in a separate isolate to keep the main thread (UI) responsive.
/// All TensorFlow Lite inference happens here, avoiding UI jank.
void mlIsolateEntryPoint(SendPort mainSendPort) async {
  // Create port for receiving messages from main isolate
  final receivePort = ReceivePort();

  // Send our SendPort to main isolate so it can send us messages
  mainSendPort.send(receivePort.sendPort);

  // ML models and data (initialized in this isolate)
  Interpreter? detectionModel;
  Interpreter? embeddingModel;
  List<Card>? cardDatabase;
  List<List<double>>? embeddingIndex;

  // Model constants
  const int detectionInputSize = 640;
  const int embeddingInputSize = 224;
  const int embeddingDim = 384;

  print('🔧 ML isolate started');

  // Listen for messages from main isolate
  await for (final message in receivePort) {
    if (message is Map<String, dynamic>) {
      final msg = IsolateMessage.fromJson(message);

      try {
        switch (msg.type) {
          case 'initialize':
            // Initialize ML models in this isolate
            try {
              // Get data passed from main thread
              final initData = msg.data as Map<String, dynamic>;
              final cardJsonList = initData['cardJsonList'] as List<dynamic>;
              final embeddingBytes = initData['embeddingBytes'] as Uint8List;
              final detectionModelBytes = initData['detectionModelBytes'] as Uint8List;
              final embeddingModelBytes = initData['embeddingModelBytes'] as Uint8List;

              // Use CPU with multi-threading
              final options = InterpreterOptions()..threads = 4;

              // Load detection model from bytes (no Flutter bindings needed!)
              detectionModel = Interpreter.fromBuffer(
                detectionModelBytes,
                options: options,
              );

              // Load embedding model from bytes (no Flutter bindings needed!)
              embeddingModel = Interpreter.fromBuffer(
                embeddingModelBytes,
                options: options,
              );

              // Parse card database from JSON (passed from main thread)
              cardDatabase = cardJsonList
                  .map((json) => Card.fromJson(json as Map<String, dynamic>))
                  .toList();

              // Parse embeddings from bytes (passed from main thread)
              final floatList = embeddingBytes.buffer.asFloat32List();

              embeddingIndex = [];
              for (var i = 0; i < floatList.length; i += embeddingDim) {
                if (i + embeddingDim <= floatList.length) {
                  final embedding = floatList.sublist(i, i + embeddingDim);
                  embeddingIndex!.add(List<double>.from(embedding));
                }
              }

              print('🔧 ML isolate initialized: ${cardDatabase!.length} cards, ${embeddingIndex!.length} embeddings');
              
              // Log model shapes for debugging
              final detInputTensor = detectionModel!.getInputTensor(0);
              final detOutputTensor = detectionModel!.getOutputTensor(0);
              print('📊 Detection model: input=${detInputTensor.shape}, output=${detOutputTensor.shape}');
              
              final embInputTensor = embeddingModel!.getInputTensor(0);
              final embOutputTensor = embeddingModel!.getOutputTensor(0);
              print('📊 Embedding model: input=${embInputTensor.shape}, output=${embOutputTensor.shape}');
              
              mainSendPort.send({'type': 'initialized', 'success': true});
            } catch (e) {
              print('🔧 ML isolate initialization failed: $e');
              mainSendPort.send({'type': 'initialized', 'success': false, 'error': e.toString()});
            }
            break;

          case 'scanFrame':
            // Process frame (this is the heavy work)
            if (detectionModel == null || embeddingModel == null) {
              mainSendPort.send({
                'type': 'scanResult',
                'results': <Map<String, dynamic>>[],
              });
              break;
            }

            final totalStopwatch = Stopwatch()..start();

            // Decode FrameData from message
            final frameDataMap = msg.data as Map<String, dynamic>;
            final frameData = FrameData(
              bytes: frameDataMap['bytes'] as Uint8List,
              width: frameDataMap['width'] as int,
              height: frameDataMap['height'] as int,
            );

            // Step 1: Detect cards
            final detections = await _detectCards(
              frameData,
              detectionModel,
              detectionInputSize,
            );

            final results = <Map<String, dynamic>>[];

            // Convert FrameData to Image for cropping
            final image = img.Image.fromBytes(
              width: frameData.width,
              height: frameData.height,
              bytes: frameData.bytes.buffer,
              numChannels: 3,
            );

            for (final detection in detections) {
              // Step 2: Crop detected region
              final cardImage = await _cropRegion(image, detection['boundingBox'] as Map<String, dynamic>);

              // Step 3: Generate embedding
              final embedding = await _generateEmbedding(
                cardImage,
                embeddingModel,
                embeddingInputSize,
                embeddingDim,
              );

              // Step 4: Search for matches
              final matches = await _searchCards(
                embedding,
                embeddingIndex!,
                cardDatabase!,
                embeddingDim,
              );

              // Log top match for debugging
              if (matches.isNotEmpty) {
                final topMatch = matches[0];
                final cardJson = topMatch['card'] as Map<String, dynamic>;
                final cardName = cardJson['name'] ?? 'Unknown';
                final similarity = topMatch['similarity'] as double;
                print('🎯 Top match: "$cardName" (similarity: ${(similarity * 100).toStringAsFixed(1)}%)');
              }

              results.add({
                'detection': detection,
                'matches': matches,
                'totalLatencyMs': totalStopwatch.elapsedMilliseconds,
              });
            }

            totalStopwatch.stop();
            print('🔧 Isolate: Total scan took ${totalStopwatch.elapsedMilliseconds}ms');

            // Send results back to main isolate with debug stats
            mainSendPort.send({
              'type': 'scanResult',
              'results': results,
              'debugStats': {
                'inputWidth': frameData.width,
                'inputHeight': frameData.height,
                'detectionCount': detections.length,
                'totalLatencyMs': totalStopwatch.elapsedMilliseconds,
              },
            });
            break;

          case 'dispose':
            // Clean up
            detectionModel?.close();
            embeddingModel?.close();
            print('🔧 ML isolate disposed');
            receivePort.close();
            break;
        }
      } catch (e) {
        print('🔧 ML isolate error: $e');
        mainSendPort.send({
          'type': 'error',
          'error': e.toString(),
        });
      }
    }
  }
}

// --- Private Helper Functions (run in isolate) ---

Future<List<Map<String, dynamic>>> _detectCards(
  FrameData frameData,
  Interpreter detectionModel,
  int detectionInputSize,
) async {
  final stopwatch = Stopwatch()..start();

  // Preprocess image and get padding info
  final preprocessResult = await _preprocessForDetection(frameData, detectionInputSize);
  final input = preprocessResult['input'] as List<List<List<List<double>>>>;
  final scale = preprocessResult['scale'] as double;
  final offsetX = preprocessResult['offsetX'] as int;
  final offsetY = preprocessResult['offsetY'] as int;

  // Run inference
  final outputShape = detectionModel.getOutputTensor(0).shape;
  print('🔍 Detection model output shape: $outputShape');

  final output = List.generate(
    outputShape[0],
    (_) => List.generate(
      outputShape[1],
      (_) => List.filled(outputShape[2], 0.0),
    ),
  );

  detectionModel.run(input, output);

  // Debug: Print first few raw output values
  if (output.isNotEmpty && output[0].isNotEmpty && output[0][0].isNotEmpty) {
    print('🔍 First detection raw values: ${output[0][0].take(10).toList()}');
    if (output[0].length > 1) {
      print('🔍 Second detection raw values: ${output[0][1].take(10).toList()}');
    }
  }

  // Parse detections with padding info
  final detections = _parseDetections(
    output,
    detectionInputSize,
    scale: scale,
    offsetX: offsetX,
    offsetY: offsetY,
    originalWidth: frameData.width,
    originalHeight: frameData.height,
  );

  stopwatch.stop();
  print('🔧 Isolate: Detection took ${stopwatch.elapsedMilliseconds}ms');

  return detections;
}

Future<List<double>> _generateEmbedding(
  Uint8List cardImage,
  Interpreter embeddingModel,
  int embeddingInputSize,
  int embeddingDim,
) async {
  final stopwatch = Stopwatch()..start();

  // Preprocess image
  final input = await _preprocessForEmbedding(cardImage, embeddingInputSize);

  // Run inference
  final output = List.filled(embeddingDim, 0.0).reshape([1, embeddingDim]);
  embeddingModel.run(input, output);

  // Normalize embedding
  final embedding = _normalizeEmbedding(output[0] as List<double>);

  stopwatch.stop();

  return embedding;
}

Future<List<Map<String, dynamic>>> _searchCards(
  List<double> embedding,
  List<List<double>> embeddingIndex,
  List<Card> cardDatabase,
  int embeddingDim,
) async {
  final stopwatch = Stopwatch()..start();

  // Compute cosine similarity with all cards
  final similarities = <MapEntry<int, double>>[];
  for (var i = 0; i < embeddingIndex.length; i++) {
    final sim = _cosineSimilarity(embedding, embeddingIndex[i]);
    similarities.add(MapEntry(i, sim));
  }

  // Sort by similarity and take top 5
  similarities.sort((a, b) => b.value.compareTo(a.value));
  final topMatches = similarities.take(5);

  stopwatch.stop();

  return topMatches.map((entry) {
    return {
      'card': cardDatabase[entry.key].toJson(),
      'similarity': entry.value,
      'latencyMs': stopwatch.elapsedMilliseconds,
    };
  }).toList();
}

Future<Map<String, dynamic>> _preprocessForDetection(
  FrameData frameData,
  int targetSize,
) async {
  img.Image image = img.Image.fromBytes(
    width: frameData.width,
    height: frameData.height,
    bytes: frameData.bytes.buffer,
    numChannels: 3,
  );

  // Calculate scale to fit image within 640x640
  final double scale = min(
    targetSize / image.width,
    targetSize / image.height,
  );

  final int newWidth = (image.width * scale).round();
  final int newHeight = (image.height * scale).round();

  // Resize image
  img.Image resized = img.copyResize(
    image,
    width: newWidth,
    height: newHeight,
    interpolation: img.Interpolation.linear,
  );

  // Create 640x640 canvas with gray padding
  img.Image padded = img.Image(
    width: targetSize,
    height: targetSize,
    numChannels: 3,
  );
  img.fill(padded, color: img.ColorRgb8(114, 114, 114));

  // Center the resized image
  final int offsetX = (targetSize - newWidth) ~/ 2;
  final int offsetY = (targetSize - newHeight) ~/ 2;
  img.compositeImage(padded, resized, dstX: offsetX, dstY: offsetY);

  // Convert to normalized tensor [1, 640, 640, 3]
  final tensor = List.generate(
    1,
    (_) => List.generate(
      targetSize,
      (y) => List.generate(
        targetSize,
        (x) {
          final pixel = padded.getPixel(x, y);
          return [
            pixel.r / 255.0,
            pixel.g / 255.0,
            pixel.b / 255.0,
          ];
        },
      ),
    ),
  );

  return {
    'input': tensor,
    'scale': scale,
    'offsetX': offsetX,
    'offsetY': offsetY,
  };
}

Future<List<List<List<List<double>>>>> _preprocessForEmbedding(
  Uint8List cardImage,
  int targetSize,
) async {
  img.Image? image = img.decodeImage(cardImage);
  if (image == null) {
    throw Exception('Failed to decode card image');
  }

  // IMPORTANT: Match training preprocessing exactly!
  // Training uses: Resize(256, 256) -> CenterCrop(224, 224)
  // This was causing misidentification by using direct resize to 224
  
  // Step 1: Resize to 256x256 (maintaining aspect ratio like training)
  const intermediateSize = 256;
  img.Image resized256 = img.copyResize(
    image,
    width: intermediateSize,
    height: intermediateSize,
    interpolation: img.Interpolation.linear,
  );

  // Step 2: Center crop to 224x224
  final cropOffset = (intermediateSize - targetSize) ~/ 2; // (256-224)/2 = 16
  img.Image cropped = img.copyCrop(
    resized256,
    x: cropOffset,
    y: cropOffset,
    width: targetSize,
    height: targetSize,
  );

  // ImageNet normalization (same as training)
  const imagenetMean = [0.485, 0.456, 0.406];
  const imagenetStd = [0.229, 0.224, 0.225];

  // NCHW format for embedding model
  final tensor = List.generate(
    1,
    (_) => List.generate(
      3,
      (c) => List.generate(
        targetSize,
        (y) => List.generate(
          targetSize,
          (x) {
            final pixel = cropped.getPixel(x, y);
            double value;

            if (c == 0) {
              value = pixel.r / 255.0;
            } else if (c == 1) {
              value = pixel.g / 255.0;
            } else {
              value = pixel.b / 255.0;
            }

            return (value - imagenetMean[c]) / imagenetStd[c];
          },
        ),
      ),
    ),
  );

  return tensor;
}

List<Map<String, dynamic>> _parseDetections(
  List<List<List<double>>> output,
  int detectionInputSize, {
  required double scale,
  required int offsetX,
  required int offsetY,
  required int originalWidth,
  required int originalHeight,
}) {
  final detections = <Map<String, dynamic>>[];

  if (output.isEmpty || output[0].isEmpty) {
    print('🔧 Isolate: No detection output');
    return detections;
  }

  // YOLOv8 TFLite outputs in transposed format: [1, 5, 8400]
  // output[0][0] = all x_centers, output[0][1] = all y_centers, etc.
  final numDetections = output[0][0].length;  // 8400
  var maxConfidence = 0.0;
  var countAboveThreshold = 0;

  print('🔍 Preprocessing info: scale=$scale, offsetX=$offsetX, offsetY=$offsetY, original=${originalWidth}x$originalHeight');

  for (var i = 0; i < numDetections; i++) {
    // Access transposed data: [feature_index][detection_index]
    // Model outputs are NORMALIZED (0-1) relative to 640x640 padded image
    final xCenterNorm = output[0][0][i];
    final yCenterNorm = output[0][1][i];
    final widthNorm = output[0][2][i];
    final heightNorm = output[0][3][i];
    final confidence = output[0][4][i];

    if (confidence > maxConfidence) maxConfidence = confidence;

    // Lower threshold back to 0.3 for better detection
    if (confidence < 0.3) continue;
    countAboveThreshold++;

    // Convert normalized (0-1) to pixel coordinates in 640x640 space
    final xCenterPx = xCenterNorm * detectionInputSize;
    final yCenterPx = yCenterNorm * detectionInputSize;
    final widthPx = widthNorm * detectionInputSize;
    final heightPx = heightNorm * detectionInputSize;

    // Convert to x1,y1,x2,y2 in 640x640 pixel space
    final x1Px640 = xCenterPx - widthPx / 2;
    final y1Px640 = yCenterPx - heightPx / 2;
    final x2Px640 = xCenterPx + widthPx / 2;
    final y2Px640 = yCenterPx + heightPx / 2;

    // Remove padding offset to get coordinates in resized image space
    final x1Resized = x1Px640 - offsetX;
    final y1Resized = y1Px640 - offsetY;
    final x2Resized = x2Px640 - offsetX;
    final y2Resized = y2Px640 - offsetY;

    // Calculate dimensions of resized image (before padding)
    final resizedWidth = (originalWidth * scale).round();
    final resizedHeight = (originalHeight * scale).round();

    // Skip detections that are mostly in the padding region
    if (x2Resized <= 0 || y2Resized <= 0 ||
        x1Resized >= resizedWidth || y1Resized >= resizedHeight) {
      continue;
    }

    // Normalize to 0-1 range relative to original image dimensions
    final normalizedBox = {
      'left': (x1Resized / resizedWidth).clamp(0.0, 1.0),
      'top': (y1Resized / resizedHeight).clamp(0.0, 1.0),
      'right': (x2Resized / resizedWidth).clamp(0.0, 1.0),
      'bottom': (y2Resized / resizedHeight).clamp(0.0, 1.0),
    };

    // Calculate box dimensions and filters
    final boxWidth = normalizedBox['right']! - normalizedBox['left']!;
    final boxHeight = normalizedBox['bottom']! - normalizedBox['top']!;
    final boxArea = boxWidth * boxHeight;
    final aspectRatio = boxHeight / boxWidth;

    // Debug: Log first few detections with full coordinate trace
    if (countAboveThreshold <= 3) {
      print('🔍 Detection $countAboveThreshold raw: center=(${xCenterNorm.toStringAsFixed(3)}, ${yCenterNorm.toStringAsFixed(3)}) size=(${widthNorm.toStringAsFixed(3)}, ${heightNorm.toStringAsFixed(3)}) conf=${confidence.toStringAsFixed(3)}');
      print('   640px space: (${x1Px640.toStringAsFixed(1)}, ${y1Px640.toStringAsFixed(1)}) - (${x2Px640.toStringAsFixed(1)}, ${y2Px640.toStringAsFixed(1)})');
      print('   After offset: (${x1Resized.toStringAsFixed(1)}, ${y1Resized.toStringAsFixed(1)}) - (${x2Resized.toStringAsFixed(1)}, ${y2Resized.toStringAsFixed(1)})');
      print('   Normalized: (${normalizedBox['left']!.toStringAsFixed(3)}, ${normalizedBox['top']!.toStringAsFixed(3)}) - (${normalizedBox['right']!.toStringAsFixed(3)}, ${normalizedBox['bottom']!.toStringAsFixed(3)})');
      print('   Area=${boxArea.toStringAsFixed(3)} aspectRatio=${aspectRatio.toStringAsFixed(3)}');
    }

    // TEMPORARILY DISABLE aggressive filtering to debug
    // Filter out overly large detections (likely false positives)
    if (boxArea > 0.8) {  // Relaxed from 0.7
      print('   ❌ Filtered: too large (area=${boxArea.toStringAsFixed(3)})');
      continue;
    }

    // Filter out very small detections (noise)
    if (boxArea < 0.02) {  // Relaxed from 0.03
      print('   ❌ Filtered: too small (area=${boxArea.toStringAsFixed(3)})');
      continue;
    }

    // Filter out detections with extreme aspect ratios
    // TCG cards have aspect ratio around 1.4 (height/width) ± variation
    if (aspectRatio < 0.4 || aspectRatio > 2.5) {  // Relaxed from 0.5-2.0
      print('   ❌ Filtered: bad aspect (${aspectRatio.toStringAsFixed(3)})');
      continue;
    }

    // Store as Map (not Rect) for isolate serialization
    detections.add({
      'boundingBox': normalizedBox,
      'confidence': confidence,
      'angle': 0.0,
    });
  }

  print('🔧 Isolate: Found $countAboveThreshold/$numDetections detections above 0.3 threshold (max confidence: ${maxConfidence.toStringAsFixed(3)})');
  print('🔧 Isolate: After filtering (size/aspect): ${detections.length} detections');

  // Apply very strict NMS threshold to eliminate overlapping detections
  // Lower IoU threshold = more aggressive (boxes with even small overlap get suppressed)
  final finalDetections = _applyNMS(detections, iouThreshold: 0.15);
  print('🔧 Isolate: After NMS: ${finalDetections.length} detections');
  return finalDetections;
}

List<Map<String, dynamic>> _applyNMS(
  List<Map<String, dynamic>> detections, {
  double iouThreshold = 0.45,
}) {
  if (detections.length <= 1) return detections;

  final sorted = List<Map<String, dynamic>>.from(detections)
    ..sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));

  final kept = <Map<String, dynamic>>[];

  for (final detection in sorted) {
    var shouldKeep = true;

    for (final keptDetection in kept) {
      final iou = _calculateIoU(
        detection['boundingBox'] as Map<String, dynamic>,
        keptDetection['boundingBox'] as Map<String, dynamic>,
      );

      if (iou > iouThreshold) {
        shouldKeep = false;
        break;
      }
    }

    if (shouldKeep) {
      kept.add(detection);
    }
  }

  return kept;
}

double _calculateIoU(Map<String, dynamic> box1, Map<String, dynamic> box2) {
  final intersectionLeft = max(box1['left'] as double, box2['left'] as double);
  final intersectionTop = max(box1['top'] as double, box2['top'] as double);
  final intersectionRight = min(box1['right'] as double, box2['right'] as double);
  final intersectionBottom = min(box1['bottom'] as double, box2['bottom'] as double);

  if (intersectionRight < intersectionLeft || intersectionBottom < intersectionTop) {
    return 0.0;
  }

  final intersectionWidth = intersectionRight - intersectionLeft;
  final intersectionHeight = intersectionBottom - intersectionTop;
  final intersectionArea = intersectionWidth * intersectionHeight;

  final box1Width = (box1['right'] as double) - (box1['left'] as double);
  final box1Height = (box1['bottom'] as double) - (box1['top'] as double);
  final box2Width = (box2['right'] as double) - (box2['left'] as double);
  final box2Height = (box2['bottom'] as double) - (box2['top'] as double);

  final box1Area = box1Width * box1Height;
  final box2Area = box2Width * box2Height;
  final unionArea = box1Area + box2Area - intersectionArea;

  if (unionArea == 0.0) return 0.0;

  return intersectionArea / unionArea;
}

List<double> _normalizeEmbedding(List<double> embedding) {
  final sumOfSquares = embedding.fold<double>(
    0.0,
    (sum, val) => sum + val * val,
  );
  final magnitude = sumOfSquares > 0 ? sqrt(sumOfSquares) : 1.0;

  return embedding.map((v) => v / magnitude).toList();
}

double _cosineSimilarity(List<double> a, List<double> b) {
  var dotProduct = 0.0;
  for (var i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
  }
  return dotProduct;
}

Future<Uint8List> _cropRegion(img.Image image, Map<String, dynamic> boundingBox) async {
  final left = boundingBox['left'] as double;
  final top = boundingBox['top'] as double;
  final right = boundingBox['right'] as double;
  final bottom = boundingBox['bottom'] as double;

  final rectWidth = right - left;
  final rectHeight = bottom - top;

  print('🔧 Isolate: Crop rect: left=$left, top=$top, right=$right, bottom=$bottom, width=$rectWidth, height=$rectHeight');
  print('🔧 Isolate: Image size: ${image.width}x${image.height}');

  final int x = (left * image.width).round().clamp(0, image.width - 1);
  final int y = (top * image.height).round().clamp(0, image.height - 1);
  final int width = (rectWidth * image.width).round();
  final int height = (rectHeight * image.height).round();

  final int cropWidth = min(width, image.width - x);
  final int cropHeight = min(height, image.height - y);

  print('🔧 Isolate: Crop params: x=$x, y=$y, width=$width, height=$height, cropWidth=$cropWidth, cropHeight=$cropHeight');

  if (cropWidth <= 0 || cropHeight <= 0) {
    throw Exception('Invalid crop dimensions: cropWidth=$cropWidth, cropHeight=$cropHeight');
  }

  img.Image cropped = img.copyCrop(
    image,
    x: x,
    y: y,
    width: cropWidth,
    height: cropHeight,
  );

  final croppedBytes = img.encodeJpg(cropped, quality: 95);

  return Uint8List.fromList(croppedBytes);
}
