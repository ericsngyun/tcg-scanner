# TCG Scanner - Flutter App

Real-time Trading Card Game card recognition app with on-device ML inference.

## Prerequisites

- Flutter SDK 3.2+
- Xcode 15+ (for iOS)
- Android Studio / Android SDK (for Android)
- Trained ML models (see below)

## Setup

### 1. Install Dependencies

```bash
cd mobile/flutter
flutter pub get
```

### 2. Add ML Models (After Training)

After training the models in Google Colab, copy them to the assets folder:

```
mobile/flutter/
├── assets/
│   ├── models/
│   │   ├── detection.tflite      # YOLOv8-nano detection model
│   │   └── embedding.tflite      # FastViT-T12 embedding model
│   ├── indices/
│   │   └── riftbound.bin         # Annoy vector index (converted)
│   └── data/
│       └── cards.json            # Card metadata with pricing
```

### 3. iOS Setup

```bash
cd ios
pod install
```

Open `ios/Runner.xcworkspace` in Xcode and configure:
- Bundle identifier
- Signing & Capabilities
- Camera usage description (already configured)

### 4. Android Setup

Ensure minimum SDK is 24+ in `android/app/build.gradle`.

## Running the App

```bash
# iOS Simulator
flutter run -d ios

# Android Emulator
flutter run -d android

# Physical device
flutter run
```

## Architecture

```
lib/
├── main.dart                    # Entry point
├── app.dart                     # App widget & navigation
├── core/
│   ├── models/                  # Data models (Card, Recognition results)
│   ├── services/                # ML inference service
│   └── theme/                   # App theming
└── features/
    ├── scanner/                 # Camera & card recognition
    │   └── presentation/
    │       ├── pages/
    │       ├── widgets/
    │       └── providers/
    └── collection/              # User's card collection
        └── presentation/
            └── pages/
```

## ML Pipeline (On-Device)

1. **Camera Frame** → YUV420 image from camera stream
2. **Detection** → YOLOv8-nano finds card bounding boxes (~30ms)
3. **Preprocessing** → Crop, perspective correct, normalize
4. **Embedding** → FastViT-T12 generates 384-dim vector (~50ms)
5. **Search** → Annoy index finds top-5 matches (~10ms)
6. **Display** → Show card info with confidence score

**Total Target Latency: <200ms**

## Models Required

Before the app can scan cards, you need:

| Model | Source | Size |
|-------|--------|------|
| `detection.tflite` | Train in Colab → Export TFLite | ~3MB |
| `embedding.tflite` | Train in Colab → Export TFLite | ~4MB |
| `riftbound.bin` | Build from embeddings | ~4MB |
| `cards.json` | From training manifest | ~250KB |

## Development Status

- [x] Project structure
- [x] Camera integration
- [x] Detection overlay UI
- [x] Result sheet UI
- [x] ML service skeleton
- [ ] TFLite model integration (awaiting trained models)
- [ ] Vector search implementation
- [ ] Collection persistence
- [ ] Price sync
