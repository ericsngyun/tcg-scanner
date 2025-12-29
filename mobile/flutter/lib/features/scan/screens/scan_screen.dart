import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'dart:async';

import '../../../core/services/camera_service.dart';
import '../../../core/services/ml_service.dart';
import '../../../core/services/card_tracker.dart';
import '../../../core/services/recognition_stabilizer.dart';
import '../../../core/services/border_detector.dart';
import '../../../core/services/ml_debug_service.dart';
import '../../../core/models/frame_data.dart';
import '../../../core/models/recognition_result.dart';
import '../widgets/ml_overlay.dart';
import '../widgets/result_card_widget.dart';
import '../widgets/scan_instructions.dart';

/// Detection method options.
enum DetectionMethod {
  yolo,      // ML-based (may have accuracy issues)
  border,    // Edge-based (simpler but more robust)
}

/// Main scan screen with camera preview and ML overlay.
/// 
/// Integrates:
/// - Camera service with proper iOS rotation handling
/// - Card tracker for smooth bounding box transitions
/// - Recognition stabilizer for stable card identification
/// - Animated overlays with haptic feedback
class ScanScreen extends StatefulWidget {
  const ScanScreen({Key? key}) : super(key: key);

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with SingleTickerProviderStateMixin {
  final _cameraService = CameraService();
  final _mlService = MLService.instance;
  final _cardTracker = CardTracker();
  final _recognitionStabilizer = MultiCardRecognitionStabilizer();
  final _borderDetector = BorderDetector();
  final _debugService = MLDebugService.instance;

  List<TrackedCard> _trackedCards = [];
  List<ScanResult> _rawResults = [];
  bool _isInitializing = true;
  bool _isScanning = false;
  bool _isProcessingFrame = false;
  String? _error;
  String _initStatus = 'Starting...';
  DetectionMethod _detectionMethod = DetectionMethod.yolo;
  bool _debugMode = false;

  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _initializeServices();
  }

  Future<void> _initializeServices() async {
    try {
      setState(() {
        _isInitializing = true;
        _error = null;
        _initStatus = 'Loading ML models...';
      });

      // Initialize ML service
      await _mlService.initialize();
      
      setState(() => _initStatus = 'Starting camera...');

      // Initialize camera
      await _cameraService.initialize();

      // Set up frame processing callback
      _cameraService.onFrameAvailable = _processFrame;

      setState(() {
        _isInitializing = false;
      });

      // Start scanning automatically
      _startScanning();
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isInitializing = false;
      });
    }
  }

  void _startScanning() {
    if (!_isScanning && _cameraService.isInitialized) {
      // Use lower frame skip for more responsive tracking
      _cameraService.startProcessing(frameSkip: 5);
      setState(() => _isScanning = true);
    }
  }

  void _stopScanning() {
    if (_isScanning) {
      _cameraService.stopProcessing();
      _cardTracker.reset();
      _recognitionStabilizer.reset();
      setState(() {
        _isScanning = false;
        _trackedCards = [];
        _rawResults = [];
      });
    }
  }

  void _processFrame(FrameData frameData) {
    // Skip if already processing a frame
    if (_isProcessingFrame) return;

    _isProcessingFrame = true;

    // Process frame in background without blocking camera
    _mlService.scanFrame(frameData).then((results) {
      if (!mounted) return;
      
      _rawResults = results;
      
      // Extract detections for tracking
      final detections = results.map((r) => r.detection).toList();
      
      // Update card tracker with new detections
      final tracked = _cardTracker.update(detections);
      
      // Update recognition for each tracked card
      for (final card in tracked) {
        // Find the matching scan result for this tracked card
        final matchingResult = _findMatchingResult(card, results);
        if (matchingResult != null) {
          // Stabilize recognition for this card
          final stabilizer = _recognitionStabilizer.getStabilizer(card.id);
          final stableRecognition = stabilizer.stabilize(matchingResult.matches);
          card.recognition = stableRecognition;
        }
      }
      
      // Clean up stabilizers for removed cards
      _recognitionStabilizer.cleanup(tracked.map((c) => c.id).toSet());
      
      setState(() {
        _trackedCards = tracked;
      });
    }).catchError((e) {
      print('Frame processing error: $e');
    }).whenComplete(() {
      _isProcessingFrame = false;
    });
  }
  
  /// Find the scan result that matches a tracked card by IoU.
  ScanResult? _findMatchingResult(TrackedCard card, List<ScanResult> results) {
    double bestIou = 0.3;
    ScanResult? bestMatch;
    
    for (final result in results) {
      final iou = _calculateIoU(card.smoothedBox, result.detection.boundingBox);
      if (iou > bestIou) {
        bestIou = iou;
        bestMatch = result;
      }
    }
    
    return bestMatch;
  }
  
  double _calculateIoU(Rect a, Rect b) {
    final intersectionLeft = a.left > b.left ? a.left : b.left;
    final intersectionTop = a.top > b.top ? a.top : b.top;
    final intersectionRight = a.right < b.right ? a.right : b.right;
    final intersectionBottom = a.bottom < b.bottom ? a.bottom : b.bottom;

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

  @override
  Widget build(BuildContext context) {
    if (_isInitializing) {
      return _buildLoadingScreen();
    }

    if (_error != null) {
      return _buildErrorScreen();
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Camera preview (full screen)
          _buildCameraPreview(),

          // Smooth overlay with bounding boxes
          if (_isScanning) _buildOverlay(),

          // Top bar with controls
          _buildTopBar(),

          // Bottom sheet with results
          if (_trackedCards.any((c) => c.recognition != null)) 
            _buildResultsSheet(),

          // Scan instructions (when no cards detected)
          if (_trackedCards.isEmpty && _isScanning)
            const ScanInstructions(),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    final controller = _cameraService.controller;

    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    // Get screen size
    final size = MediaQuery.of(context).size;

    // Calculate scale to fill screen
    final scale = size.aspectRatio * controller.value.aspectRatio;

    // Ensure camera preview fills the screen
    return Transform.scale(
      scale: scale < 1 ? 1 / scale : scale,
      child: Center(
        child: CameraPreview(controller),
      ),
    );
  }
  
  Widget _buildOverlay() {
    final size = MediaQuery.of(context).size;
    
    return MLOverlay(
      results: _rawResults,
      cameraController: _cameraService.controller,
    );
  }

  Widget _buildTopBar() {
    return SafeArea(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.black.withOpacity(0.6),
              Colors.transparent,
            ],
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            // Back button
            IconButton(
              icon: const Icon(Icons.arrow_back, color: Colors.white),
              onPressed: () => Navigator.of(context).pop(),
            ),

            // Title with status
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Scan Cards',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                if (_trackedCards.isNotEmpty)
                  Text(
                    '${_trackedCards.length} card${_trackedCards.length != 1 ? 's' : ''} detected',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.7),
                      fontSize: 12,
                    ),
                  ),
              ],
            ),

            // Scan toggle
            AnimatedBuilder(
              animation: _pulseController,
              builder: (context, child) {
                return IconButton(
                  icon: Icon(
                    _isScanning ? Icons.pause_circle : Icons.play_circle,
                    color: _isScanning
                        ? Colors.green.withOpacity(0.7 + _pulseController.value * 0.3)
                        : Colors.white,
                  ),
                  iconSize: 32,
                  onPressed: () {
                    if (_isScanning) {
                      _stopScanning();
                    } else {
                      _startScanning();
                    }
                  },
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsSheet() {
    // Get cards with recognition results
    final recognizedCards = _trackedCards
        .where((c) => c.recognition != null)
        .toList();
    
    if (recognizedCards.isEmpty) return const SizedBox.shrink();

    return DraggableScrollableSheet(
      initialChildSize: 0.25,
      minChildSize: 0.12,
      maxChildSize: 0.6,
      builder: (context, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: Theme.of(context).scaffoldBackgroundColor,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.3),
                blurRadius: 10,
                offset: const Offset(0, -5),
              ),
            ],
          ),
          child: Column(
            children: [
              // Drag handle
              Container(
                margin: const EdgeInsets.symmetric(vertical: 12),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),

              // Results header
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      '${recognizedCards.length} card${recognizedCards.length != 1 ? 's' : ''} identified',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      'Tap for details',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 8),

              // Results list
              Expanded(
                child: ListView.builder(
                  controller: scrollController,
                  itemCount: recognizedCards.length,
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemBuilder: (context, index) {
                    final card = recognizedCards[index];
                    return _buildResultCard(card);
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }
  
  Widget _buildResultCard(TrackedCard card) {
    final recognition = card.recognition!;
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          // Card image placeholder
          Container(
            width: 50,
            height: 70,
            decoration: BoxDecoration(
              color: Colors.grey[200],
              borderRadius: BorderRadius.circular(6),
            ),
            child: Icon(
              Icons.style,
              color: Colors.grey[400],
            ),
          ),
          const SizedBox(width: 12),
          
          // Card info
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  recognition.card.name,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: Colors.green.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        '${(recognition.similarity * 100).toStringAsFixed(0)}% match',
                        style: const TextStyle(
                          color: Colors.green,
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                    if (recognition.card.pricing?.marketPrice != null) ...[
                      const SizedBox(width: 8),
                      Text(
                        '\$${recognition.card.pricing!.marketPrice!.toStringAsFixed(2)}',
                        style: const TextStyle(
                          color: Colors.green,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
          
          // Action button
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () {
              HapticFeedback.lightImpact();
              // TODO: Add to collection
            },
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingScreen() {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
            ),
            const SizedBox(height: 24),
            Text(
              _initStatus,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'This may take a moment on first launch',
              style: TextStyle(
                color: Colors.grey[400],
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorScreen() {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.error_outline,
                color: Colors.red,
                size: 64,
              ),
              const SizedBox(height: 24),
              const Text(
                'Failed to initialize',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                _error ?? 'Unknown error',
                style: TextStyle(
                  color: Colors.grey[400],
                  fontSize: 14,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _error = null;
                  });
                  _initializeServices();
                },
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _cameraService.dispose();
    super.dispose();
  }
}
