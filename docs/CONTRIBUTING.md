# Contributing to TCG Card Scanner

## Development Setup

### ML Environment

```bash
cd ml
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### iOS Development

Requirements:
- macOS 13+
- Xcode 15+
- CocoaPods or Swift Package Manager

### Android Development

Requirements:
- Android Studio Hedgehog+
- JDK 17+
- Android SDK 34+

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Format with `black` and `isort`
- Lint with `ruff`

### Swift
- Follow Swift API Design Guidelines
- Use SwiftFormat

### Kotlin
- Follow Kotlin Coding Conventions
- Use ktlint

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

## Pull Requests

1. Create a feature branch from `main`
2. Make changes with clear, atomic commits
3. Add tests for new functionality
4. Update documentation as needed
5. Open PR with description of changes

## Testing

### ML Models
```bash
cd ml
pytest tests/
```

### Mobile Apps
- iOS: `xcodebuild test`
- Android: `./gradlew test`
