import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../../../core/models/recognition_result.dart';
import '../../../../core/services/card_tracker.dart';

/// Animated detection overlay with smooth bounding box transitions.
/// 
/// Supports two modes:
/// 1. With TrackedCards (new API) - provides smoothing and identity persistence
/// 2. With DetectionResults (legacy API) - basic display without smoothing
class DetectionOverlay extends StatefulWidget {
  /// List of tracked cards with smoothed positions (new API).
  final List<TrackedCard>? trackedCards;
  
  /// List of raw detection results (legacy API).
  final List<DetectionResult>? detections;
  
  /// Screen size for coordinate mapping.
  final Size screenSize;
  
  /// Callback when a card is tapped.
  final Function(TrackedCard)? onCardTap;

  const DetectionOverlay({
    super.key,
    this.trackedCards,
    this.detections,
    required this.screenSize,
    this.onCardTap,
  }) : assert(trackedCards != null || detections != null, 
         'Either trackedCards or detections must be provided');

  @override
  State<DetectionOverlay> createState() => _DetectionOverlayState();
}

class _DetectionOverlayState extends State<DetectionOverlay> {
  // Internal tracker for legacy mode
  final CardTracker _legacyTracker = CardTracker();
  
  @override
  Widget build(BuildContext context) {
    if (widget.screenSize == Size.zero) {
      return const SizedBox.shrink();
    }

    // Use provided tracked cards or convert detections
    final cards = widget.trackedCards ?? _getTrackedCardsFromDetections();
    
    if (cards.isEmpty) {
      return const SizedBox.shrink();
    }

    return Stack(
      children: cards.map((card) {
        return _AnimatedBoundingBox(
          key: ValueKey(card.id),
          trackedCard: card,
          screenSize: widget.screenSize,
          onTap: widget.onCardTap != null ? () => widget.onCardTap!(card) : null,
        );
      }).toList(),
    );
  }
  
  List<TrackedCard> _getTrackedCardsFromDetections() {
    if (widget.detections == null || widget.detections!.isEmpty) {
      return [];
    }
    return _legacyTracker.update(widget.detections!);
  }
  
  @override
  void dispose() {
    _legacyTracker.reset();
    super.dispose();
  }
}

/// Individual animated bounding box for a tracked card.
class _AnimatedBoundingBox extends StatefulWidget {
  final TrackedCard trackedCard;
  final Size screenSize;
  final VoidCallback? onTap;

  const _AnimatedBoundingBox({
    super.key,
    required this.trackedCard,
    required this.screenSize,
    this.onTap,
  });

  @override
  State<_AnimatedBoundingBox> createState() => _AnimatedBoundingBoxState();
}

class _AnimatedBoundingBoxState extends State<_AnimatedBoundingBox>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);

    // Haptic feedback on first appear
    if (widget.trackedCard.framesTracked == 1) {
      HapticFeedback.lightImpact();
    }
  }

  @override
  void didUpdateWidget(_AnimatedBoundingBox oldWidget) {
    super.didUpdateWidget(oldWidget);
    
    // Haptic feedback when recognition becomes confident
    final oldRecog = oldWidget.trackedCard.recognition;
    final newRecog = widget.trackedCard.recognition;
    if (oldRecog == null && newRecog != null && newRecog.isConfident) {
      HapticFeedback.mediumImpact();
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final card = widget.trackedCard;
    final box = card.smoothedBox;
    final size = widget.screenSize;

    // Convert normalized coordinates to screen coordinates
    final left = box.left * size.width;
    final top = box.top * size.height;
    final width = box.width * size.width;
    final height = box.height * size.height;

    // Determine color based on state
    final color = _getColorForState(card);
    
    // Calculate opacity based on tracking stability
    final opacity = card.isStable ? 1.0 : 0.6;

    return AnimatedPositioned(
      duration: const Duration(milliseconds: 80),
      curve: Curves.easeOut,
      left: left,
      top: top,
      width: width,
      height: height,
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 150),
          opacity: opacity,
          child: AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) {
              return Stack(
                clipBehavior: Clip.none,
                children: [
                  // Bounding box
                  CustomPaint(
                    painter: _BoundingBoxPainter(
                      color: color,
                      pulseValue: _pulseController.value,
                      isStable: card.isStable,
                    ),
                    size: Size(width, height),
                  ),
                  
                  // Card info label (if recognized)
                  if (card.recognition != null && top > 70)
                    Positioned(
                      top: -60,
                      left: 0,
                      right: 0,
                      child: _CardLabel(
                        recognition: card.recognition!,
                        color: color,
                      ),
                    ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Color _getColorForState(TrackedCard card) {
    if (card.recognition != null && card.recognition!.isConfident) {
      return const Color(0xFF10B981); // Emerald green - confident match
    } else if (card.isStable) {
      return const Color(0xFFFFAA00); // Amber - detecting, not identified
    } else {
      return const Color(0xFF60A5FA); // Blue - new detection
    }
  }
}

/// Custom painter for the bounding box with corner accents.
class _BoundingBoxPainter extends CustomPainter {
  final Color color;
  final double pulseValue;
  final bool isStable;

  _BoundingBoxPainter({
    required this.color,
    required this.pulseValue,
    required this.isStable,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final baseStrokeWidth = isStable ? 3.0 : 2.0;
    final strokeWidth = baseStrokeWidth + (pulseValue * 0.5);

    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    // Corner length proportional to box size
    final cornerLength = (size.shortestSide * 0.15).clamp(15.0, 35.0);

    // Draw corner brackets
    _drawCorner(canvas, Offset.zero, cornerLength, true, true, paint);
    _drawCorner(canvas, Offset(size.width, 0), cornerLength, false, true, paint);
    _drawCorner(canvas, Offset(0, size.height), cornerLength, true, false, paint);
    _drawCorner(canvas, Offset(size.width, size.height), cornerLength, false, false, paint);

    // Semi-transparent fill with pulse
    final fillOpacity = 0.05 + (pulseValue * 0.03);
    final fillPaint = Paint()
      ..color = color.withOpacity(fillOpacity)
      ..style = PaintingStyle.fill;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), fillPaint);

    // Subtle glow effect
    if (isStable) {
      final glowPaint = Paint()
        ..color = color.withOpacity(0.1 * pulseValue)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), glowPaint);
    }
  }

  void _drawCorner(
    Canvas canvas,
    Offset corner,
    double length,
    bool isLeft,
    bool isTop,
    Paint paint,
  ) {
    final hDir = isLeft ? 1.0 : -1.0;
    final vDir = isTop ? 1.0 : -1.0;

    canvas.drawLine(
      corner,
      corner + Offset(length * hDir, 0),
      paint,
    );
    canvas.drawLine(
      corner,
      corner + Offset(0, length * vDir),
      paint,
    );
  }

  @override
  bool shouldRepaint(_BoundingBoxPainter oldDelegate) {
    return oldDelegate.color != color ||
        oldDelegate.pulseValue != pulseValue ||
        oldDelegate.isStable != isStable;
  }
}

/// Card info label shown above the bounding box.
class _CardLabel extends StatelessWidget {
  final RecognitionResult recognition;
  final Color color;

  const _CardLabel({
    required this.recognition,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.85),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.7), width: 2),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            recognition.card.name,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.bold,
              height: 1.2,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Confidence badge
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.check_circle, color: color, size: 12),
                    const SizedBox(width: 3),
                    Text(
                      '${(recognition.similarity * 100).toStringAsFixed(0)}%',
                      style: TextStyle(
                        color: color,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
              
              // Price badge
              if (recognition.card.pricing?.marketPrice != null) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFF10B981).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    '\$${recognition.card.pricing!.marketPrice!.toStringAsFixed(2)}',
                    style: const TextStyle(
                      color: Color(0xFF10B981),
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }
}
