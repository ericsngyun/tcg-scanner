#!/bin/bash

# TCG Scanner iOS Setup Script
# This script sets up the iOS platform for the Flutter app

set -e

echo "🚀 Setting up iOS platform for TCG Scanner..."

# Step 1: Create iOS platform files
echo ""
echo "📱 Step 1: Creating iOS platform files..."
flutter create . --platforms=ios

# Step 2: Add camera permissions to Info.plist
echo ""
echo "🔐 Step 2: Adding camera permissions to Info.plist..."

PLIST_FILE="ios/Runner/Info.plist"

# Check if permissions already exist
if grep -q "NSCameraUsageDescription" "$PLIST_FILE"; then
    echo "✅ Camera permissions already configured"
else
    echo "Adding camera permissions..."

    # Backup original file
    cp "$PLIST_FILE" "${PLIST_FILE}.backup"

    # Add camera permissions before closing </dict>
    # Using perl for cross-platform compatibility
    perl -i -pe 's{(</dict>\s*</plist>)}{
    <!-- Camera Permission -->
    <key>NSCameraUsageDescription</key>
    <string>Camera access is required to scan trading cards</string>

    <!-- Photo Library Permission (optional) -->
    <key>NSPhotoLibraryUsageDescription</key>
    <string>Photo library access to save scanned cards</string>

    <!-- Prevent rotation for better scanning -->
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
    </array>

    $1}' "$PLIST_FILE"

    echo "✅ Camera permissions added"
fi

# Step 3: Install Flutter dependencies
echo ""
echo "📦 Step 3: Installing Flutter dependencies..."
flutter pub get

# Step 4: Verify assets
echo ""
echo "📂 Step 4: Verifying assets..."

ASSETS_OK=true

if [ ! -f "assets/models/detection.tflite" ]; then
    echo "⚠️  Missing: assets/models/detection.tflite"
    ASSETS_OK=false
fi

if [ ! -f "assets/models/embedding.tflite" ]; then
    echo "⚠️  Missing: assets/models/embedding.tflite"
    ASSETS_OK=false
fi

if [ ! -f "assets/data/cards.json" ]; then
    echo "⚠️  Missing: assets/data/cards.json"
    ASSETS_OK=false
fi

if [ ! -f "assets/indices/riftbound.bin" ]; then
    echo "⚠️  Missing: assets/indices/riftbound.bin"
    echo "   → Run the Colab notebook to generate this file"
    ASSETS_OK=false
fi

if [ "$ASSETS_OK" = true ]; then
    echo "✅ All required assets present"
fi

# Done
echo ""
echo "✅ iOS setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. If riftbound.bin is missing, run the Colab notebook (generate_embeddings.ipynb)"
echo "   2. Connect your iPhone via USB"
echo "   3. Run: flutter run"
echo "   4. Grant camera permission when prompted"
echo "   5. Start scanning cards!"
echo ""
