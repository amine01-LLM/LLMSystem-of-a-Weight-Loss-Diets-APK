// vision_engine.dart
// ──────────────────
// Flutter equivalent of vision_engine.py
// Runs YOLO ONNX food detection on device using onnxruntime.
//
// Setup:
//   flutter pub add onnxruntime image
//
// Usage:
//   await VisionEngine.init('/path/to/models/best_ep12.onnx');
//   final result = await VisionEngine.detectFood(File('meal.jpg'));
//   print(result.foodItems);      // ['pizza', 'salad']
//   print(result.allDetections);  // {'pizza': 2, 'salad': 1}

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

// ── Config ────────────────────────────────────────────────────────────────────
const double _confidenceThreshold = 0.20;  // matches vision_engine.py
const int    _inputSize           = 640;   // YOLO input resolution
const int    _maxDetections       = 5;     // top N foods to return

// ── Detection result ──────────────────────────────────────────────────────────

class DetectionResult {
  final List<String>        foodItems;
  final Map<String, int>    allDetections;

  const DetectionResult({
    required this.foodItems,
    required this.allDetections,
  });
}

class VisionEngine {
  static OrtSession?   _session;
  static List<String>? _classNames;

  // ── Init ──────────────────────────────────────────────────────────────────
  // Call once at app startup after models are downloaded.

  static Future<void> init({
    String? modelPath,
    required List<String> classNames,
  }) async {
    if (_session != null) return;

    _classNames = classNames;

    String path;
    if (modelPath != null) {
      path = modelPath;
    } else {
      final dir = await getApplicationDocumentsDirectory();
      path = p.join(dir.path, 'models', 'best_ep12.onnx');
    }

    OrtEnv.instance.init();
    final opts = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1);

    _session = OrtSession.fromFile(path, opts);
  }

  static bool get isReady => _session != null && _classNames != null;

  // ── Preprocess image ──────────────────────────────────────────────────────
  // Matches vision_engine.py exactly:
  //   - Resize to 640x640
  //   - Normalize to [0, 1]
  //   - Channel-first layout: [1, 3, 640, 640]

  static Float32List _preprocessImage(File imageFile) {
    final bytes   = imageFile.readAsBytesSync();
    final image   = img.decodeImage(bytes)!;
    final resized = img.copyResize(image, width: _inputSize, height: _inputSize);

    final input = Float32List(1 * 3 * _inputSize * _inputSize);
    int idx = 0;

    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < _inputSize; y++) {
        for (int x = 0; x < _inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          final val   = c == 0 ? pixel.r : c == 1 ? pixel.g : pixel.b;
          input[idx++] = val / 255.0;
        }
      }
    }

    return input;
  }

  // ── Sigmoid ───────────────────────────────────────────────────────────────

  static double _sigmoid(double x) => 1.0 / (1.0 + exp(-x));

  // ── Detect food ───────────────────────────────────────────────────────────
  // Returns top N food items detected in the image.
  // Matches vision_engine.py: sigmoid on class scores, NMS via best-per-class.

  static Future<DetectionResult> detectFood(File imageFile) async {
    assert(isReady, 'VisionEngine.init() must be called first');

    // 1. Preprocess
    final inputData   = _preprocessImage(imageFile);
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, _inputSize, _inputSize],
    );

    // 2. Run inference
    final outputs = await _session!.runAsync(
      OrtRunOptions(),
      {'images': inputTensor},
    );

    inputTensor.release();

    // 3. Parse detections
    // YOLO output shape: [1, num_classes + 4, num_anchors]
    // First 4 values: bbox (x, y, w, h) — we only care about class scores
    final raw = outputs[0]!.value as List;
    final Map<int, double> bestPerClass = {};

    for (final detection in raw[0] as List) {
      final det         = (detection as List).cast<double>();
      final classScores = det.sublist(4).map(_sigmoid).toList();
      final maxScore    = classScores.reduce(max);

      if (maxScore >= _confidenceThreshold) {
        final classId = classScores.indexOf(maxScore);
        if (!bestPerClass.containsKey(classId) ||
            bestPerClass[classId]! < maxScore) {
          bestPerClass[classId] = maxScore;
        }
      }
    }

    for (final output in outputs) {
      output?.release();
    }

    // 4. Sort by confidence, take top N
    final sorted = bestPerClass.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final topN = sorted.take(_maxDetections).toList();

    // 5. Build result
    final foodItems = topN
        .map((e) => _classNames![e.key])
        .toList();

    final allDetections = <String, int>{};
    for (final e in topN) {
      final name = _classNames![e.key];
      allDetections[name] = (allDetections[name] ?? 0) + 1;
    }

    return DetectionResult(
      foodItems:     foodItems,
      allDetections: allDetections,
    );
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  static void dispose() {
    _session?.release();
    _session    = null;
    _classNames = null;
    OrtEnv.instance.release();
  }
}
