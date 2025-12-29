import 'dart:typed_data';

/// Raw RGB frame data from camera with dimensions.
class FrameData {
  final Uint8List bytes;
  final int width;
  final int height;

  FrameData({
    required this.bytes,
    required this.width,
    required this.height,
  });
}
