#!/usr/bin/env python3
"""
Verify all ML assets are in place and ready for Flutter deployment.

This script checks:
1. TFLite models are present
2. Card database is properly formatted
3. Sample images are available
4. File sizes are reasonable
"""

import sys
import json
from pathlib import Path


def check_file(path: Path, expected_min_size_kb: int = 0) -> bool:
    """Check if file exists and meets minimum size requirement."""
    if not path.exists():
        print(f"  ❌ Missing: {path}")
        return False

    size_kb = path.stat().st_size / 1024
    if size_kb < expected_min_size_kb:
        print(f"  ❌ Too small: {path} ({size_kb:.1f} KB < {expected_min_size_kb} KB)")
        return False

    print(f"  ✅ {path.name}: {size_kb:.1f} KB")
    return True


def verify_models(models_dir: Path) -> bool:
    """Verify TFLite models."""
    print("\n" + "=" * 60)
    print("CHECKING ML MODELS")
    print("=" * 60)

    detection_model = models_dir / 'detection.tflite'
    embedding_model = models_dir / 'embedding.tflite'

    all_good = True
    all_good &= check_file(detection_model, expected_min_size_kb=10000)  # ~10 MB
    all_good &= check_file(embedding_model, expected_min_size_kb=20000)  # ~20 MB

    return all_good


def verify_card_database(cards_db: Path) -> bool:
    """Verify card database is properly formatted."""
    print("\n" + "=" * 60)
    print("CHECKING CARD DATABASE")
    print("=" * 60)

    if not check_file(cards_db, expected_min_size_kb=100):
        return False

    try:
        with open(cards_db) as f:
            cards = json.load(f)

        print(f"\n  Total cards: {len(cards)}")

        # Check first card has required fields
        if cards:
            first_card = cards[0]
            required_fields = ['product_id', 'name', 'image_path', 'card_type']

            print(f"\n  Sample card: {first_card['name']}")
            for field in required_fields:
                if field in first_card:
                    print(f"    ✅ {field}: {first_card[field]}")
                else:
                    print(f"    ❌ Missing field: {field}")
                    return False

        # Count by set
        sets = {}
        for card in cards:
            set_name = card.get('group_name', 'Unknown')
            sets[set_name] = sets.get(set_name, 0) + 1

        print(f"\n  Cards by set:")
        for set_name, count in sorted(sets.items(), key=lambda x: -x[1]):
            print(f"    {set_name}: {count}")

        return True

    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON: {e}")
        return False


def verify_images(images_dir: Path) -> bool:
    """Verify sample images are available."""
    print("\n" + "=" * 60)
    print("CHECKING SAMPLE IMAGES")
    print("=" * 60)

    if not images_dir.exists():
        print(f"  ❌ Images directory not found: {images_dir}")
        return False

    # Count images
    images = list(images_dir.glob('**/*.jpg'))
    print(f"\n  Total images: {len(images)}")

    if len(images) == 0:
        print(f"  ❌ No images found")
        return False

    # Show samples
    print(f"\n  Sample images:")
    for img in images[:5]:
        size_kb = img.stat().st_size / 1024
        print(f"    {img.name}: {size_kb:.1f} KB")

    return True


def verify_flutter_setup(flutter_dir: Path) -> bool:
    """Verify Flutter project structure."""
    print("\n" + "=" * 60)
    print("CHECKING FLUTTER PROJECT")
    print("=" * 60)

    pubspec = flutter_dir / 'pubspec.yaml'

    if not pubspec.exists():
        print(f"  ❌ pubspec.yaml not found")
        return False

    print(f"  ✅ pubspec.yaml found")

    # Check key files
    ml_service = flutter_dir / 'lib' / 'core' / 'services' / 'ml_service.dart'
    card_model = flutter_dir / 'lib' / 'core' / 'models' / 'card.dart'

    check_file(ml_service, expected_min_size_kb=10)
    check_file(card_model, expected_min_size_kb=1)

    return True


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent

    print("=" * 60)
    print("TCG Scanner - ML Assets Verification")
    print("=" * 60)
    print(f"Project root: {project_root}")

    # Define paths
    models_dir = project_root / 'mobile' / 'flutter' / 'assets' / 'models'
    cards_db = project_root / 'mobile' / 'flutter' / 'assets' / 'data' / 'cards.json'
    images_dir = project_root / 'ml' / 'data' / 'images' / 'riftbound'
    flutter_dir = project_root / 'mobile' / 'flutter'

    # Run verifications
    all_passed = True
    all_passed &= verify_models(models_dir)
    all_passed &= verify_card_database(cards_db)
    all_passed &= verify_images(images_dir)
    all_passed &= verify_flutter_setup(flutter_dir)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("=" * 60)
        print("\nYour ML pipeline is ready to test in Flutter!")
        print("\nNext steps:")
        print("1. Open Flutter project: cd mobile/flutter")
        print("2. Run app: flutter run")
        print("3. The MLService should initialize successfully")
        print("4. Check debug console for: '✅ Loaded 723 cards from database'")
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before testing.")
        sys.exit(1)


if __name__ == '__main__':
    main()
