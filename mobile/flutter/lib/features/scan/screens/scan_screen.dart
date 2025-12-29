import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';

import '../../../core/services/camera_service.dart';
import '../../../core/services/ml_service.dart';
import '../../../core/models/frame_data.dart';
import '../../../core/models/recognition_result.dart';
import '../widgets/ml_overlay.dart';
import '../widgets/result_card_widget.dart';
import '../widgets/scan_instructions.dart';

/// Main scan screen with camera preview and ML overlay.
class ScanScreen extends StatefulWidget {
  const ScanScreen({Key? key}) : super(key: key);

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with SingleTickerProviderStateMixin {
  final _cameraService = CameraService();
  final _mlService = MLService.instance;

  List<ScanResult> _scanResults = [];
  bool _isInitializing = true;
  bool _isScanning = false;
  bool _isProcessingFrame = false;
  String? _error;

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
      });

      // Initialize ML service
      await _mlService.initialize();

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
      _cameraService.startProcessing(frameSkip: 10); // ~3 FPS at 30 FPS camera (non-blocking processing)
      setState(() => _isScanning = true);
    }
  }

  void _stopScanning() {
    if (_isScanning) {
      _cameraService.stopProcessing();
      setState(() => _isScanning = false);
    }
  }

  void _processFrame(FrameData frameData) {
    // Skip if already processing a frame
    if (_isProcessingFrame) return;

    _isProcessingFrame = true;

    // Process frame in background without blocking camera
    _mlService.scanFrame(frameData).then((results) {
      // Update UI with results
      if (mounted) {
        setState(() {
          _scanResults = results;
        });
      }
    }).catchError((e) {
      print('Frame processing error: $e');
    }).whenComplete(() {
      _isProcessingFrame = false;
    });
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

          // ML overlay with bounding boxes
          if (_isScanning) MLOverlay(results: _scanResults),

          // Top bar with controls
          _buildTopBar(),

          // Bottom sheet with results
          if (_scanResults.isNotEmpty) _buildResultsSheet(),

          // Scan instructions (when no cards detected)
          if (_scanResults.isEmpty && _isScanning)
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

            // Title
            Text(
              'Scan Cards',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
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
    return DraggableScrollableSheet(
      initialChildSize: 0.3,
      minChildSize: 0.15,
      maxChildSize: 0.7,
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
                      '${_scanResults.length} card${_scanResults.length != 1 ? 's' : ''} detected',
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      'Tap for details',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 12),

              // Results list
              Expanded(
                child: ListView.builder(
                  controller: scrollController,
                  itemCount: _scanResults.length,
                  itemBuilder: (context, index) {
                    return ResultCardWidget(
                      result: _scanResults[index],
                      onTap: () {
                        // Navigate to card detail
                        // TODO: Implement navigation
                      },
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
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
            const Text(
              'Initializing ML models...',
              style: TextStyle(
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
