import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:ui' as ui;
import 'dart:math' as math;

import '../../../core/models/recognition_result.dart';

/// Overlay showing ML detection results with animated bounding boxes.
///
/// Handles proper coordinate transformation from camera image space to
/// screen space, accounting for camera preview scaling and aspect ratio.
class MLOverlay extends StatefulWidget {
  final List<ScanResult> results;
  final CameraController? cameraController;

  const MLOverlay({
    Key? key,
    required this.results,
    this.cameraController,
  }) : super(key: key);

  @override
  State<MLOverlay> createState() => _MLOverlayState();
}

class _MLOverlayState extends State<MLOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  late Animation<double> _opacityAnimation;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );

    _scaleAnimation = Tween<double>(begin: 0.92, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: Curves.easeOutCubic,
      ),
    );

    _opacityAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.0, 0.6, curve: Curves.easeIn),
      ),
    );

    _pulseAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: Curves.easeInOut,
      ),
    );

    _animationController.forward();

    // Continuous pulse animation
    _animationController.addStatusListener((status) {
      if (status == AnimationStatus.completed && mounted) {
        _animationController.reverse();
      } else if (status == AnimationStatus.dismissed && mounted) {
        _animationController.forward();
      }
    });
  }

  @override
  void didUpdateWidget(MLOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);

    // Animate when results change
    if (widget.results.length != oldWidget.results.length) {
      _animationController.reset();
      _animationController.forward();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.results.isEmpty || widget.cameraController == null) {
      return const SizedBox.shrink();
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        return Stack(
          children: widget.results.map((result) {
            return _buildBoundingBox(
              context,
              result,
              constraints.biggest,
            );
          }).toList(),
        );
      },
    );
  }

  Widget _buildBoundingBox(
    BuildContext context,
    ScanResult result,
    Size screenSize,
  ) {
    final detection = result.detection;
    final bestMatch = result.bestMatch;
    final cameraController = widget.cameraController!;

    // Get camera preview size - note: on iOS, previewSize is in landscape orientation
    // but our camera_service now rotates frames to portrait, so normalized coords
    // are relative to portrait orientation
    final previewSize = cameraController.value.previewSize!;
    
    // Since we rotate frames in camera_service, the effective preview is portrait
    // The ML model receives portrait-oriented frames, so normalized coords are portrait
    // We just need to scale to screen size (which is also portrait)
    
    // Calculate how the camera preview fills the screen
    // In scan_screen.dart, we use: scale = size.aspectRatio * controller.value.aspectRatio
    // And then Transform.scale with scale < 1 ? 1/scale : scale
    final cameraAspect = previewSize.height / previewSize.width; // Raw sensor aspect
    final screenAspect = screenSize.width / screenSize.height; // Screen aspect (portrait)
    
    // After rotation, the camera produces portrait frames
    // The preview is scaled to fill the screen width, potentially cropping top/bottom
    final previewAspect = screenAspect; // Assume the rotated preview matches screen orientation
    final scale = screenSize.aspectRatio * cameraAspect;

    // Since the normalized coordinates (0-1) are relative to the rotated (portrait) frame,
    // and the camera preview fills the screen, we can map directly
    // The only adjustment needed is for any scaling/cropping applied to fill the screen
    
    double scaleX, scaleY, offsetX, offsetY;
    
    if (scale >= 1) {
      // Preview is scaled up horizontally, cropped on sides
      scaleX = screenSize.width * scale;
      scaleY = screenSize.height;
      offsetX = (scaleX - screenSize.width) / 2;
      offsetY = 0;
    } else {
      // Preview is scaled up vertically, cropped on top/bottom
      scaleX = screenSize.width;
      scaleY = screenSize.height / scale;
      offsetX = 0;
      offsetY = (scaleY - screenSize.height) / 2;
    }

    // Convert normalized coordinates (0-1) to screen coordinates
    // The normalized coords are relative to the ML-processed frame (portrait after rotation)
    final boxLeft = detection.boundingBox.left * scaleX - offsetX;
    final boxTop = detection.boundingBox.top * scaleY - offsetY;
    final boxWidth = detection.boundingBox.width * scaleX;
    final boxHeight = detection.boundingBox.height * scaleY;

    final box = ui.Rect.fromLTWH(boxLeft, boxTop, boxWidth, boxHeight);

    // Clamp to screen bounds
    final clampedBox = ui.Rect.fromLTRB(
      math.max(0, box.left),
      math.max(0, box.top),
      math.min(screenSize.width, box.right),
      math.min(screenSize.height, box.bottom),
    );

    // Determine color based on confidence
    final color = bestMatch != null && bestMatch.isConfident
        ? const Color(0xFF10B981) // Emerald green
        : const Color(0xFFFFAA00); // Amber

    return AnimatedBuilder(
      animation: _animationController,
      builder: (context, child) {
        return Positioned(
          left: clampedBox.left,
          top: clampedBox.top,
          width: clampedBox.width,
          height: clampedBox.height,
          child: Opacity(
            opacity: _opacityAnimation.value,
            child: Transform.scale(
              scale: _scaleAnimation.value,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  // Bounding box with corners
                  CustomPaint(
                    painter: BoundingBoxPainter(
                      color: color,
                      confidence: detection.confidence,
                      pulseValue: _pulseAnimation.value,
                    ),
                    size: Size(clampedBox.width, clampedBox.height),
                  ),

                  // Card info label
                  if (bestMatch != null && clampedBox.top > 60)
                    Positioned(
                      top: -52,
                      left: 0,
                      right: 0,
                      child: _buildCardLabel(bestMatch, color),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildCardLabel(RecognitionResult match, Color borderColor) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Colors.black.withOpacity(0.92),
            Colors.black.withOpacity(0.88),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: borderColor.withOpacity(0.8),
          width: 2.5,
        ),
        boxShadow: [
          BoxShadow(
            color: borderColor.withOpacity(0.4),
            blurRadius: 12,
            spreadRadius: 1,
            offset: const Offset(0, 2),
          ),
          BoxShadow(
            color: Colors.black.withOpacity(0.5),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Card name
          Text(
            match.card.name,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.3,
              height: 1.2,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),

          const SizedBox(height: 6),

          // Confidence and price row
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Confidence indicator with modern styling
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: borderColor.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(
                    color: borderColor.withOpacity(0.5),
                    width: 1,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.verified,
                      color: borderColor,
                      size: 13,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      '${(match.similarity * 100).toStringAsFixed(0)}%',
                      style: TextStyle(
                        color: borderColor,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.2,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 10),

              // Price with modern badge
              if (match.card.pricing?.marketPrice != null)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: const Color(0xFF10B981).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                      color: const Color(0xFF10B981).withOpacity(0.5),
                      width: 1,
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.monetization_on,
                        color: Color(0xFF10B981),
                        size: 13,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        '\$${match.card.pricing!.marketPrice!.toStringAsFixed(2)}',
                        style: const TextStyle(
                          color: Color(0xFF10B981),
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }
}

/// Custom painter for bounding box with corner accents and pulse animation.
class BoundingBoxPainter extends CustomPainter {
  final Color color;
  final double confidence;
  final double pulseValue;

  BoundingBoxPainter({
    required this.color,
    required this.confidence,
    required this.pulseValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Pulsing effect for stroke width
    final baseStrokeWidth = 3.5;
    final pulseStrokeWidth = baseStrokeWidth + (pulseValue * 0.8);

    final paint = Paint()
      ..color = color
      ..strokeWidth = pulseStrokeWidth
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    // Corner length with pulse effect
    final cornerLength = 28.0 + (pulseValue * 4);
    final cornerThickness = 4.0;

    // Draw thick corner brackets
    void drawCorner(Offset corner, bool isLeft, bool isTop) {
      final horizontalLength = cornerLength;
      final verticalLength = cornerLength;

      final horizontalStart = isLeft ? corner : Offset(corner.dx - horizontalLength, corner.dy);
      final horizontalEnd = isLeft ? Offset(corner.dx + horizontalLength, corner.dy) : corner;

      final verticalStart = isTop ? corner : Offset(corner.dx, corner.dy - verticalLength);
      final verticalEnd = isTop ? Offset(corner.dx, corner.dy + verticalLength) : corner;

      canvas.drawLine(horizontalStart, horizontalEnd, paint);
      canvas.drawLine(verticalStart, verticalEnd, paint);
    }

    // Draw all four corners
    drawCorner(const Offset(0, 0), true, true); // Top-left
    drawCorner(Offset(size.width, 0), false, true); // Top-right
    drawCorner(Offset(0, size.height), true, false); // Bottom-left
    drawCorner(Offset(size.width, size.height), false, false); // Bottom-right

    // Draw semi-transparent fill with pulse
    final fillOpacity = 0.08 + (pulseValue * 0.04);
    final fillPaint = Paint()
      ..color = color.withOpacity(fillOpacity)
      ..style = PaintingStyle.fill;

    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      fillPaint,
    );

    // Draw subtle glow effect
    final glowPaint = Paint()
      ..color = color.withOpacity(0.15 * pulseValue)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      glowPaint,
    );
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return oldDelegate.color != color ||
        oldDelegate.confidence != confidence ||
        oldDelegate.pulseValue != pulseValue;
  }
}
