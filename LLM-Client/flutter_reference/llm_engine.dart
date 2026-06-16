// llm_engine.dart
// ───────────────
// Flutter equivalent of llm_engine.py
// Runs Qwen2.5-1.5B ONNX INT4 on device using onnxruntime_genai.
//
// Setup:
//   flutter pub add onnxruntime_genai
//
// Usage:
//   await LLMEngine.init('/path/to/qwen25_onnx');
//   final response = await LLMEngine.generate(system: '...', user: '...');
//   await for (final chunk in LLMEngine.generateStream(system: '...', user: '...')) {
//     print(chunk);
//   }

import 'package:onnxruntime_genai/onnxruntime_genai.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

// ── Config ────────────────────────────────────────────────────────────────────
const double _temperature   = 0.7;
const String _stopToken     = '<|im_end|>';
const int    _defaultMaxTok = 120;

class LLMEngine {
  static OgModel?     _model;
  static OgTokenizer? _tokenizer;
  static String?      _modelPath;

  // ── Init ──────────────────────────────────────────────────────────────────
  // Call once at app startup after models are downloaded.

  static Future<void> init([String? modelPath]) async {
    if (_model != null) return;

    if (modelPath != null) {
      _modelPath = modelPath;
    } else {
      final dir  = await getApplicationDocumentsDirectory();
      _modelPath = p.join(dir.path, 'qwen25_onnx');
    }

    _model     = await OgModel.create(_modelPath!);
    _tokenizer = await OgTokenizer.create(_model!);
  }

  static bool get isReady => _model != null && _tokenizer != null;

  // ── Build ChatML prompt ───────────────────────────────────────────────────
  // Must match llm_engine.py exactly — Qwen2.5 uses ChatML format.

  static String buildPrompt(String system, String user) {
    return '<|im_start|>system\n'
        '$system'
        '<|im_end|>\n'
        '<|im_start|>user\n'
        '$user'
        '<|im_end|>\n'
        '<|im_start|>assistant\n';
  }

  // ── Dynamic max_tokens ────────────────────────────────────────────────────
  // Matches _smart_max_tokens() in nutrition_llm.py

  static int smartMaxTokens(String message) {
    final words = message.trim().split(' ').length;
    if (words < 5)  return 100;
    if (words < 15) return 120;
    if (words > 30) return 160;
    return _defaultMaxTok;
  }

  // ── Generate (full response, non-streaming) ───────────────────────────────
  // Used for summaries and meal analysis.

  static Future<String> generate({
    required String system,
    required String user,
    int? maxTokens,
    double temperature = _temperature,
  }) async {
    assert(isReady, 'LLMEngine.init() must be called first');

    final prompt    = buildPrompt(system, user);
    final sequences = _tokenizer!.encode(prompt);
    final maxTok    = maxTokens ?? smartMaxTokens(user);

    final params = await OgGeneratorParams.create(_model!);
    params.setSearchOption('max_length',   sequences.sequenceCount + maxTok);
    params.setSearchOption('temperature',  temperature);
    params.setSearchOption('do_sample',    temperature > 0 ? 1 : 0);
    params.setInput(sequences);

    final generator = await OgGenerator.create(_model!, params);
    final buffer    = StringBuffer();

    while (!generator.isDone()) {
      await generator.computeLogits();
      await generator.generateNextToken();
      final token = generator.getSequence(0).last;
      final text  = _tokenizer!.decode([token]);
      if (text.contains(_stopToken)) break;
      buffer.write(text);
    }

    generator.dispose();
    return buffer.toString().trim();
  }

  // ── Generate stream ───────────────────────────────────────────────────────
  // Used for coach chat and meal planner — yields tokens as they arrive.

  static Stream<String> generateStream({
    required String system,
    required String user,
    int? maxTokens,
    double temperature = _temperature,
  }) async* {
    assert(isReady, 'LLMEngine.init() must be called first');

    final prompt    = buildPrompt(system, user);
    final sequences = _tokenizer!.encode(prompt);
    final maxTok    = maxTokens ?? smartMaxTokens(user);

    final params = await OgGeneratorParams.create(_model!);
    params.setSearchOption('max_length',   sequences.sequenceCount + maxTok);
    params.setSearchOption('temperature',  temperature);
    params.setSearchOption('do_sample',    temperature > 0 ? 1 : 0);
    params.setInput(sequences);

    final generator = await OgGenerator.create(_model!, params);

    while (!generator.isDone()) {
      await generator.computeLogits();
      await generator.generateNextToken();
      final token = generator.getSequence(0).last;
      final text  = _tokenizer!.decode([token]);
      if (text.contains(_stopToken)) break;
      yield text;
    }

    generator.dispose();
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  static void dispose() {
    _model?.dispose();
    _tokenizer?.dispose();
    _model     = null;
    _tokenizer = null;
  }
}
