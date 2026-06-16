// memory.dart
// ───────────
// Flutter equivalent of memory.py
// Persistent conversation memory using sqflite (SQLite on device).
//
// Setup:
//   flutter pub add sqflite path
//
// Usage:
//   await MemoryDB.init();
//   await MemoryDB.saveMessage('user_42', 'user', 'I want to lose weight');
//   String context = await MemoryDB.buildContext('user_42');

import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

// ── Config ────────────────────────────────────────────────────────────────────
const int summaryEvery = 20;  // summarize after N assistant messages
const int maxHistory   = 4;   // messages passed as recent context

class MemoryDB {
  static Database? _db;

  // ── Init ──────────────────────────────────────────────────────────────────

  static Future<void> init() async {
    if (_db != null) return;

    final dbPath = await getDatabasesPath();
    final path   = join(dbPath, 'conversations.db');

    _db = await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at TEXT    NOT NULL
          )
        ''');

        await db.execute('''
          CREATE TABLE summaries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT    NOT NULL,
            summary    TEXT    NOT NULL,
            created_at TEXT    NOT NULL
          )
        ''');

        await db.execute('CREATE INDEX idx_messages_user ON messages(user_id)');
        await db.execute('CREATE INDEX idx_summaries_user ON summaries(user_id)');
      },
    );
  }

  // ── Save message ──────────────────────────────────────────────────────────

  static Future<void> saveMessage(
    String userId,
    String role,
    String content,
  ) async {
    await _db!.insert('messages', {
      'user_id':    userId,
      'role':       role,
      'content':    content,
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  // ── Load recent messages ──────────────────────────────────────────────────

  static Future<List<Map<String, String>>> loadRecentMessages(
    String userId, {
    int limit = maxHistory,
  }) async {
    final rows = await _db!.rawQuery('''
      SELECT role, content FROM messages
      WHERE user_id = ?
      ORDER BY id DESC LIMIT ?
    ''', [userId, limit]);

    return rows.reversed
        .map((r) => {
              'role':    r['role'] as String,
              'content': r['content'] as String,
            })
        .toList();
  }

  // ── Load latest summary ───────────────────────────────────────────────────

  static Future<String?> loadLatestSummary(String userId) async {
    final rows = await _db!.rawQuery('''
      SELECT summary FROM summaries
      WHERE user_id = ?
      ORDER BY id DESC LIMIT 1
    ''', [userId]);

    if (rows.isEmpty) return null;
    return rows.first['summary'] as String;
  }

  // ── Save summary ──────────────────────────────────────────────────────────

  static Future<void> saveSummary(String userId, String summary) async {
    await _db!.insert('summaries', {
      'user_id':    userId,
      'summary':    summary,
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  // ── Count assistant messages ──────────────────────────────────────────────

  static Future<int> countAssistantMessages(String userId) async {
    final result = await _db!.rawQuery('''
      SELECT COUNT(*) as cnt FROM messages
      WHERE user_id = ? AND role = 'assistant'
    ''', [userId]);
    return result.first['cnt'] as int;
  }

  // ── Get messages for summary ──────────────────────────────────────────────

  static Future<List<Map<String, String>>> getMessagesForSummary(
    String userId, {
    int limit = 6,
  }) async {
    final rows = await _db!.rawQuery('''
      SELECT role, content FROM messages
      WHERE user_id = ?
      ORDER BY id DESC LIMIT ?
    ''', [userId, limit]);

    return rows.reversed
        .map((r) => {
              'role':    r['role'] as String,
              'content': r['content'] as String,
            })
        .toList();
  }

  // ── Build context for LLM prompt ─────────────────────────────────────────

  static Future<String> buildContext(String userId) async {
    final parts = <String>[];

    // 1. Long-term memory (summary)
    final summary = await loadLatestSummary(userId);
    if (summary != null && summary.isNotEmpty) {
      parts.add('## User Profile (from past sessions)\n$summary');
    }

    // 2. Recent conversation history
    final recent  = await loadRecentMessages(userId, limit: maxHistory + 1);
    final history = recent.length > 1
        ? recent.sublist(0, recent.length - 1)
        : <Map<String, String>>[];

    if (history.isNotEmpty) {
      final historyText = history.map((m) {
        final label   = m['role'] == 'user' ? 'User' : 'Coach';
        final content = m['content']!.length > 100
            ? m['content']!.substring(0, 100)
            : m['content']!;
        return '$label: $content';
      }).join('\n');
      parts.add('## Chat history (do NOT repeat these)\n$historyText');
    }

    return parts.join('\n\n');
  }

  // ── Maybe summarize ───────────────────────────────────────────────────────
  // Call this after every assistant message is saved.
  // Pass your LLM generate function as a callback.

  static Future<void> maybeSummarize(
    String userId,
    Future<String> Function(String system, String prompt) generateFn,
  ) async {
    final count = await countAssistantMessages(userId);
    if (count > 0 && count % summaryEvery == 0) {
      final messages = await getMessagesForSummary(userId);
      if (messages.isEmpty) return;

      final conversationText = messages.map((m) {
        final role    = m['role'] == 'user' ? 'U' : 'C';
        final content = m['content']!.length > 80
            ? m['content']!.substring(0, 80)
            : m['content']!;
        return '$role: $content';
      }).join('\n');

      const system = 'Extract a user health profile in under 50 words. English only. Be very brief.';
      final prompt = 'Conversation:\n$conversationText\n\nProfile (goal, diet, progress, struggles):';

      final summary = await generateFn(system, prompt);
      if (summary.isNotEmpty) {
        await saveSummary(userId, summary.trim());
      }
    }
  }

  // ── Clear user history ────────────────────────────────────────────────────

  static Future<void> clearUserHistory(String userId) async {
    await _db!.delete('messages',  where: 'user_id = ?', whereArgs: [userId]);
    await _db!.delete('summaries', where: 'user_id = ?', whereArgs: [userId]);
  }
}
