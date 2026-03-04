"""Persistent conversation storage using SQLite."""

import json
import logging
import sqlite3
from pathlib import Path

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = Path(DATA_DIR) / "conversations.db"

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Get or create the SQLite connection (singleton)."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
            """
        )
        _conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
            """
        )
        _conn.commit()
        logger.info("Conversation database initialized at %s", DB_PATH)
    return _conn


def create_conversation(title: str = "New Chat") -> int:
    """Create a new conversation and return its ID."""
    conn = _get_conn()
    cursor = conn.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
    conn.commit()
    return cursor.lastrowid


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> int:
    """Add a message to a conversation. Returns the message ID."""
    conn = _get_conn()
    sources_json = json.dumps(sources) if sources else None
    cursor = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, sources_json),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
        (conversation_id,),
    )
    conn.commit()
    return cursor.lastrowid


def get_messages(conversation_id: int) -> list[dict]:
    """Get all messages for a conversation."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, sources, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    messages = []
    for row in rows:
        msg = {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        if row["sources"]:
            msg["sources"] = json.loads(row["sources"])
        messages.append(msg)
    return messages


def list_conversations() -> list[dict]:
    """List all conversations ordered by most recent."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT c.id, c.title, c.created_at, c.updated_at, "
        "COUNT(m.id) as message_count "
        "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
        "GROUP BY c.id ORDER BY c.updated_at DESC",
    ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(conversation_id: int) -> None:
    """Delete a conversation and its messages."""
    conn = _get_conn()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()


def rename_conversation(conversation_id: int, title: str) -> None:
    """Rename a conversation."""
    conn = _get_conn()
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (title, conversation_id),
    )
    conn.commit()


def close() -> None:
    """Close the database connection."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
