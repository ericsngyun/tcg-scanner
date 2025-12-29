import 'package:flutter/material.dart';

import '../../../core/services/ml_debug_service.dart';

/// Debug overlay that displays ML pipeline stats.
/// 
/// Shows:
/// - Frame dimensions and tensor shape 
/// - Detection counts (raw → filtered)
/// - Processing latency
/// - Max confidence
class DebugOverlay extends StatelessWidget {
  final DebugStats? stats;
  final bool isEnabled;
  final VoidCallback onToggle;
  final VoidCallback onSaveFrame;

  const DebugOverlay({
    super.key,
    this.stats,
    required this.isEnabled,
    required this.onToggle,
    required this.onSaveFrame,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 100,
      left: 16,
      right: 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Debug toggle button
          Row(
            children: [
              IconButton(
                icon: Icon(
                  isEnabled ? Icons.bug_report : Icons.bug_report_outlined,
                  color: isEnabled ? Colors.amber : Colors.white54,
                ),
                onPressed: onToggle,
              ),
              if (isEnabled) ...[
                const SizedBox(width: 8),
                ElevatedButton.icon(
                  icon: const Icon(Icons.save, size: 16),
                  label: const Text('Save Frame'),
                  onPressed: onSaveFrame,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.amber.withOpacity(0.8),
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                  ),
                ),
              ],
            ],
          ),
          
          // Debug stats panel
          if (isEnabled && stats != null)
            Container(
              margin: const EdgeInsets.only(top: 8),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.85),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.amber.withOpacity(0.5)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHeader('ML Debug Stats'),
                  const SizedBox(height: 8),
                  _buildStatRow('Input', '${stats!.inputWidth}×${stats!.inputHeight}'),
                  _buildStatRow('Tensor', stats!.inputShape.join('×')),
                  _buildStatRow('Output', stats!.outputShape.join('×')),
                  const Divider(color: Colors.white24, height: 16),
                  _buildStatRow('Raw detections', '${stats!.rawDetectionCount}'),
                  _buildStatRow('After NMS', '${stats!.afterNmsCount}'),
                  _buildStatRow('Final', '${stats!.afterFilterCount}', 
                      highlight: stats!.afterFilterCount > 0),
                  const Divider(color: Colors.white24, height: 16),
                  _buildStatRow('Max confidence', 
                      '${(stats!.maxConfidence * 100).toStringAsFixed(1)}%',
                      highlight: stats!.maxConfidence > 0.5),
                  _buildStatRow('Latency', '${stats!.processingTime.inMilliseconds}ms'),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildHeader(String text) {
    return Text(
      text,
      style: const TextStyle(
        color: Colors.amber,
        fontSize: 14,
        fontWeight: FontWeight.bold,
      ),
    );
  }

  Widget _buildStatRow(String label, String value, {bool highlight = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 12,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              color: highlight ? Colors.greenAccent : Colors.white,
              fontSize: 12,
              fontWeight: highlight ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }
}
