# -*- coding: utf-8 -*-
"""
memory.py

Persistent conversation memory for BH Coach.
Uses SQLite (fully offline, no server needed) + LLM-generated summaries.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Config
DB_PATH = Path(__file__).resolve().parent / "data" / "conversations.db"
SUMMARY_EVERY = 20   # summarize after every N assistant messages
MAX_HISTORY = 4      # messages passed to LLM as recent context


# ---------------------------------------------------------------------
# Database Connection
# ---------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database and enable WAL mode."""
    with _get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
        CREATE INDEX IF NOT EXISTS idx_summaries_user ON summaries(user_id);
        """)

    logger.info("Memory DB ready at %s", DB_PATH)


# ---------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------

def save_message(user_id: str, role: str, content: str) -> None:
    """Save a single message."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, datetime.utcnow().isoformat())
        )


def load_recent_messages(user_id: str, limit: int = MAX_HISTORY) -> list[dict]:
    """Load recent messages for context."""
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT role, content FROM messages
               WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()

    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def load_latest_summary(user_id: str) -> str | None:
    """Load most recent summary."""
    with _get_conn() as conn:
        row = conn.execute(
            """SELECT summary FROM summaries
               WHERE user_id = ?
               ORDER BY id DESC LIMIT 1""",
            (user_id,)
        ).fetchone()

    return row["summary"] if row else None


def save_summary(user_id: str, summary: str) -> None:
    """Save summary."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO summaries (user_id, summary, created_at) VALUES (?, ?, ?)",
            (user_id, summary, datetime.utcnow().isoformat())
        )

    logger.info("Summary saved for user %s", user_id)


def count_assistant_messages(user_id: str) -> int:
    """Count assistant messages."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE user_id = ? AND role = 'assistant'",
            (user_id,)
        ).fetchone()

    return row["cnt"] if row else 0


def get_all_messages_for_summary(user_id: str, limit: int = 30) -> list[dict]:
    """Load last messages for summarization."""
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT role, content FROM messages
               WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()

    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_user_history(user_id: str) -> None:
    """Clear all user data."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))

    logger.info("Cleared history for user %s", user_id)


# ---------------------------------------------------------------------
# Summary Generation
# ---------------------------------------------------------------------

def generate_summary(user_id: str, language: str) -> str:
    """
    Generate short user health profile summary using LLM.
    """
    from llm_engine import _generate

    messages = get_all_messages_for_summary(user_id, limit=6)
    if not messages:
        return ""

    conversation_text = "\n".join(
        f"{'U' if m['role'] == 'user' else 'C'}: {m['content'][:80]}"
        for m in messages
    )

    system_prompt = (
        "Extract a user health profile in under 50 words. "
        "English only. Be very brief."
    )

    user_prompt = (
        f"Conversation:\n{conversation_text}\n\n"
        "Profile (goal, diet, progress, struggles):"
    )

    summary = _generate(
        system_prompt,
        user_prompt,
        max_tokens=80,
        temperature=0.2,
        stream=False
    )

    return summary.strip() if summary else ""


def maybe_summarize(user_id: str, language: str) -> None:
    """Trigger summary every N assistant messages."""
    count = count_assistant_messages(user_id)

    if count > 0 and count % SUMMARY_EVERY == 0:
        logger.info("Generating summary for user %s (message #%d)", user_id, count)
        summary = generate_summary(user_id, language)
        if summary:
            save_summary(user_id, summary)


# ---------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------

def build_context(user_id: str) -> str:
    """
    Build context injected into LLM system prompt.
    Combines summary + recent chat history.
    """
    parts = []

    # Long-term memory
    summary = load_latest_summary(user_id)
    if summary:
        parts.append(f"## User Profile (from past sessions)\n{summary}")

    # Recent conversation history
    recent = load_recent_messages(user_id, limit=MAX_HISTORY + 1)

    if recent:
        recent = recent[:-1]  # drop current user message

    if recent:
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Coach'}: {m['content'][:100]}"
            for m in recent
        )
        parts.append(f"## Chat history (do NOT repeat these)\n{history_text}")

    return "\n\n".join(parts)