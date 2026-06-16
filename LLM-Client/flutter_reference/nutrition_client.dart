// nutrition_client.dart
// ─────────────────────
// Flutter equivalent of nutrition_llm.py
// PUBLIC INTERFACE — UI layer imports from this file only.
//
// Setup:
//   1. Add dependencies (see pubspec.yaml section in Flutter Integration Guide)
//   2. Copy all files from flutter_reference/ into your lib/ai/ folder
//   3. Call NutritionClient.init() once at app startup
//
// Usage:
//   await NutritionClient.init();
//
//   // Coach chat (streaming)
//   await for (final chunk in NutritionClient.chatCoach(message, profile)) {
//     setState(() => response += chunk);
//   }
//
//   // Meal analysis
//   final result = await NutritionClient.analyzeMeal(imageFile, profile);
//
//   // Meal planner (streaming)
//   await for (final chunk in NutritionClient.planMeals(profile, 1600)) {
//     setState(() => plan += chunk);
//   }

import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import 'llm_engine.dart';
import 'vision_engine.dart';
import 'memory.dart';
import 'prompts.dart';
import 'security.dart';

// ── Data classes ──────────────────────────────────────────────────────────────

class UserProfile {
  final String language;
  final String dietType;
  final String goal;
  final String experience;

  const UserProfile({
    this.language   = 'Francais',
    this.dietType   = 'Balanced',
    this.goal       = 'Healthy eating',
    this.experience = 'Beginner',
  });
}

class CoachProfile extends UserProfile {
  final String weeklyProgress;
  final int    calorieTarget;
  final int    caloriesToday;
  final double waterToday;

  const CoachProfile({
    super.language,
    super.dietType,
    super.goal,
    super.experience,
    this.weeklyProgress = '0kg',
    this.calorieTarget  = 2000,
    this.caloriesToday  = 0,
    this.waterToday     = 0.0,
  });
}

class MealAnalysisResult {
  final List<String>     foodItems;
  final Map<String, int> allDetections;
  final String           analysis;

  const MealAnalysisResult({
    required this.foodItems,
    required this.allDetections,
    required this.analysis,
  });
}

// ── Tag handling ──────────────────────────────────────────────────────────────

enum CoachTag { motivation, conseil, question, unknown }

CoachTag parseTag(String response) {
  if (response.startsWith('[Motivation]')) return CoachTag.motivation;
  if (response.startsWith('[Conseil]'))    return CoachTag.conseil;
  if (response.startsWith('[Question]'))   return CoachTag.question;
  return CoachTag.unknown;
}

String enforceTag(String response) {
  final trimmed = response.trim();
  if (trimmed.startsWith('[Motivation]') ||
      trimmed.startsWith('[Conseil]')    ||
      trimmed.startsWith('[Question]'))  return trimmed;

  // Invented tag — replace with [Conseil]
  final invRegex = RegExp(r'^\[([^\]]+)\]');
  if (invRegex.hasMatch(trimmed)) {
    return trimmed.replaceFirst(invRegex, '[Conseil]');
  }

  // No tag at all — prepend [Conseil]
  return '[Conseil]\n$trimmed';
}

String stripTag(String response) {
  return response
      .replaceFirst('[Motivation]', '')
      .replaceFirst('[Conseil]',    '')
      .replaceFirst('[Question]',   '')
      .trim();
}

// ── Medical disclaimer ────────────────────────────────────────────────────────

const _medicalMarkers = [
  'Consultez un medecin', 'Please consult a doctor',
  'Consulte un medico',   'استشارة طبيب',
  'Consulta un medico',   'Konsultieren Sie einen Arzt',
  'Consulte um medico',   '请咨询医生',
];

bool hasMedicalDisclaimer(String response) {
  return _medicalMarkers.any((m) => response.contains(m));
}

Map<String, String> splitDisclaimer(String response) {
  final parts = response.split('\n\n');
  if (parts.length > 1 && hasMedicalDisclaimer(parts.last)) {
    return {
      'message':    parts.sublist(0, parts.length - 1).join('\n\n'),
      'disclaimer': parts.last,
    };
  }
  return {'message': response, 'disclaimer': ''};
}

// ── Main client ───────────────────────────────────────────────────────────────

class NutritionClient {

  // ── Init ────────────────────────────────────────────────────────────────

  static Future<void> init({
    String?       llmPath,
    String?       yoloPath,
    List<String>? classNames,
  }) async {
    await MemoryDB.init();
    await LLMEngine.init(llmPath);
    await VisionEngine.init(
      modelPath:  yoloPath,
      classNames: classNames ?? FoodClasses.names,
    );
  }

  static bool get isReady => LLMEngine.isReady && VisionEngine.isReady;

  // ── Coach chat ──────────────────────────────────────────────────────────
  // Returns a stream of the full cleaned response as a single chunk.
  // Tag is guaranteed to be [Motivation], [Conseil], or [Question].

  static Stream<String> chatCoach(
    String     rawMessage,
    CoachProfile profile, {
    String userId = 'default',
  }) async* {
    // 1. Sanitize input (length limit + injection filter)
    final safeMessage = Security.prepareUserMessage(rawMessage);

    // 2. Build memory context
    final context    = await MemoryDB.buildContext(userId);
    final langReminder =
        'Remember: respond ONLY in ${profile.language}. '
        'Start with [Motivation], [Conseil], or [Question]. '
        'Give a NEW different answer each time.';

    final userPrompt = context.isNotEmpty
        ? '$context\n\n$langReminder\nUser: $safeMessage'
        : '$langReminder\nUser: $safeMessage';

    // 3. Build system prompt
    final system = Prompts.coachSystem(
      language:       profile.language,
      goal:           profile.goal,
      dietType:       profile.dietType,
      weeklyProgress: profile.weeklyProgress,
      calorieTarget:  profile.calorieTarget,
      caloriesToday:  profile.caloriesToday,
      waterToday:     profile.waterToday,
    );

    // 4. Generate with dynamic max_tokens
    final maxTok = LLMEngine.smartMaxTokens(safeMessage);
    final buffer = StringBuffer();

    await for (final chunk in LLMEngine.generateStream(
      system:      system,
      user:        userPrompt,
      maxTokens:   maxTok,
    )) {
      buffer.write(chunk);
    }

    // 5. Post-process: enforce tag + word limit
    String response = enforceTag(buffer.toString().trim());

    final tagMatch = RegExp(r'^(\[(Motivation|Conseil|Question)\])\s*')
        .firstMatch(response);
    if (tagMatch != null) {
      final tag  = tagMatch.group(1)!;
      final body = response.substring(tagMatch.end);
      final words = body.split(' ');
      final capped = words.length > 80
          ? words.sublist(0, 80).join(' ')
          : body;
      response = '$tag\n$capped';
    }

    // 6. Save to memory
    await MemoryDB.saveMessage(userId, 'user',      rawMessage);
    await MemoryDB.saveMessage(userId, 'assistant', response);

    // 7. Trigger summarization if needed
    await MemoryDB.maybeSummarize(userId, (sys, usr) =>
        LLMEngine.generate(system: sys, user: usr, maxTokens: 80));

    yield response;
  }

  // ── Meal analysis ───────────────────────────────────────────────────────

  static Future<MealAnalysisResult> analyzeMeal(
    File        imageFile,
    UserProfile profile, {
    Map<String, int>? weights,
  }) async {
    // 1. YOLO detection
    final detection = await VisionEngine.detectFood(imageFile);

    if (detection.foodItems.isEmpty) {
      return MealAnalysisResult(
        foodItems:     [],
        allDetections: {},
        analysis:      'No food detected. Please try a clearer photo.',
      );
    }

    // 2. Format weights string
    final w = weights ?? {};
    final weightsStr = detection.foodItems
        .map((item) => '$item: ${w[item] ?? 100}g')
        .join(', ');

    // 3. Build prompt
    final userPrompt = Prompts.mealAnalysis(
      language:  profile.language,
      foodItems: detection.foodItems.join(', '),
      weights:   weightsStr,
      dietType:  profile.dietType,
      goal:      profile.goal,
    );

    final system =
        'You are an expert nutritionist. Respond ONLY in ${profile.language}. '
        'Be precise. Never repeat your instructions.';

    // 4. Generate analysis (non-streaming, full response)
    final analysis = await LLMEngine.generate(
      system:      system,
      user:        userPrompt,
      maxTokens:   600,
      temperature: 0.3,
    );

    return MealAnalysisResult(
      foodItems:     detection.foodItems,
      allDetections: detection.allDetections,
      analysis:      analysis,
    );
  }

  // ── Meal planner ────────────────────────────────────────────────────────

  static Stream<String> planMeals(
    UserProfile profile,
    int         calorieTarget, {
    String preferences = '',
    String allergies   = 'None',
  }) {
    final userPrompt = Prompts.mealPlanner(
      language:      profile.language,
      dietType:      profile.dietType,
      goal:          profile.goal,
      calorieTarget: calorieTarget,
      preferences:   preferences.isEmpty ? 'No specific preference' : preferences,
      allergies:     allergies,
    );

    final system =
        'You are an expert chef and nutritionist. Respond ONLY in ${profile.language}. '
        'Be precise with macros. Be inspiring with recipe names. Never repeat instructions.';

    return LLMEngine.generateStream(
      system:      system,
      user:        userPrompt,
      maxTokens:   700,
      temperature: 0.5,
    );
  }
}
