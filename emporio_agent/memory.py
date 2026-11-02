"""Conversation persistence (SQLite, stdlib).

A customer on WhatsApp writes, leaves, and comes back hours later ("e aquele violão que você me
mostrou?"), so history is kept per session id (in production: the customer's phone number).

Only user/assistant *text* is stored, not tool results: prices and stock change, so facts are
re-fetched by tools on every turn instead of being trusted from an old transcript.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import VAR_DIR


class ConversationStore:
    def __init__(self, path: Path | str = VAR_DIR / "conversations.db", max_turns: int = 20):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.max_turns = max_turns
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
            " role TEXT NOT NULL CHECK (role IN ('user','assistant')), content TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
        self.conn.commit()

    def history(self, session_id: str) -> list[dict]:
        """Last `max_turns` messages, oldest first (bounded context window and cost)."""
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, self.max_turns),
        ).fetchall()
        history = [{"role": r, "content": c} for r, c in reversed(rows)]
        while history and history[0]["role"] != "user":  # a window must start with a user turn
            history.pop(0)
        return history

    def append(self, session_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def clear(self, session_id: str) -> None:
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.conn.commit()
