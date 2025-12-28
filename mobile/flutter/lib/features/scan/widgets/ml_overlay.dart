import 'package:flutter/material.dart';
import 'dart:ui' as ui;

import '../../../core/models/recognition_result.dart';

/// Overlay showing ML detection results with animated bounding boxes.
class MLOverlay extends StatefulWidget {
  final List<ScanResult> results;

  const MLOverlay({
    Key? key,
    required this.results,
  }) : super(key: key);

  @override
  State<MLOverlay> createState() => _MLOverlayState();
}

class _MLOverlayState extends State<MLOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  late Animation<double> _opacityAnimation;

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );

    _scaleAnimation = Tween<double>(begin: 0.95, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeOut),
    );

    _opacityAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeIn),
    );

    _animationController.forward();
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
    if (widget.results.isEmpty) {
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

    // Convert normalized coordinates to screen coordinates
    final box = ui.Rect.fromLTWH(
      detection.boundingBox.left * screenSize.width,
      detection.boundingBox.top * screenSize.height,
      detection.boundingBox.width * screenSize.width,
      detection.boundingBox.height * screenSize.height,
    );

    // Determine color based on confidence
    final color = bestMatch != null && bestMatch.isConfident
        ? Colors.green
        : Colors.yellow;

    return AnimatedBuilder(
      animation: _animationController,
      builder: (context, child) {
        return Positioned(
          left: box.left,
          top: box.top,
          width: box.width,
          height: box.height,
          child: Opacity(
            opacity: _opacityAnimation.value,
            child: Transform.scale(
              scale: _scaleAnimation.value,
              child: Stack(
                children: [
                  // Bounding box with corners
                  CustomPaint(
                    painter: BoundingBoxPainter(
                      color: color,
                      confidence: detection.confidence,
                    ),
                    size: Size(box.width, box.height),
                  ),

                  // Card info label
                  if (bestMatch != null)
                    Positioned(
                      top: -40,
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
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.8),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: borderColor, width: 2),
        boxShadow: [
          BoxShadow(
            color: borderColor.withOpacity(0.3),
            blurRadius: 8,
            spreadRadius: 2,
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
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),

          const SizedBox(height: 4),

          // Confidence and price
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Confidence indicator
              Icon(
                Icons.check_circle,
                color: borderColor,
                size: 14,
              ),
              const SizedBox(width: 4),
              Text(
                '${(match.similarity * 100).toStringAsFixed(1)}%',
                style: TextStyle(
                  color: borderColor,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),

              const SizedBox(width: 12),

              // Price
              if (match.card.pricing?.marketPrice != null) ...[
                const Icon(
                  Icons.attach_money,
                  color: Colors.greenAccent,
                  size: 14,
                ),
                Text(
                  match.card.pricing!.marketPrice!.toStringAsFixed(2),
                  style: const TextStyle(
                    color: Colors.greenAccent,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
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

/// Custom painter for bounding box with corner accents.
class BoundingBoxPainter extends CustomPainter {
  final Color color;
  final double confidence;

  BoundingBoxPainter({
    required this.color,
    required this.confidence,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    // Draw corners (L-shaped markers)
    final cornerLength = 20.0;

    // Top-left corner
    canvas.drawLine(
      const Offset(0, 0),
      Offset(cornerLength, 0),
      paint,
    );
    canvas.drawLine(
      const Offset(0, 0),
      Offset(0, cornerLength),
      paint,
    );

    // Top-right corner
    canvas.drawLine(
      Offset(size.width, 0),
      Offset(size.width - cornerLength, 0),
      paint,
    );
    canvas.drawLine(
      Offset(size.width, 0),
      Offset(size.width, cornerLength),
      paint,
    );

    // Bottom-left corner
    canvas.drawLine(
      Offset(0, size.height),
      Offset(cornerLength, size.height),
      paint,
    );
    canvas.drawLine(
      Offset(0, size.height),
      Offset(0, size.height - cornerLength),
      paint,
    );

    // Bottom-right corner
    canvas.drawLine(
      Offset(size.width, size.height),
      Offset(size.width - cornerLength, size.height),
      paint,
    );
    canvas.drawLine(
      Offset(size.width, size.height),
      Offset(size.width, size.height - cornerLength),
      paint,
    );

    // Optional: Draw semi-transparent fill
    final fillPaint = Paint()
      ..color = color.withOpacity(0.1)
      ..style = PaintingStyle.fill;

    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      fillPaint,
    );
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return oldDelegate.color != color || oldDelegate.confidence != confidence;
  }
}
