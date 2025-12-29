import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app.dart';
import 'core/services/ml_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize ML models
  await MLService.instance.initialize();

  // Note: Drift database initialization will be handled by the database service
  // when first accessed (lazy initialization pattern)

  runApp(
    const ProviderScope(
      child: TCGScannerApp(),
    ),
  );
}
