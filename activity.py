"""Persistent, content-free audit events for MCP tool calls."""
import functools
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def log_path():
    return Path(os.getenv("VECTORVAULT_DATA_DIR", "data/vectorvault")) / "activity.sqlite3"


def record(tool, status, detail=""):
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10) as db:
        db.execute("CREATE TABLE IF NOT EXISTS activity (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, tool TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL)")
        db.execute("INSERT INTO activity (timestamp, tool, status, detail) VALUES (?, ?, ?, ?)", (datetime.now(timezone.utc).isoformat(), tool, status, detail))


def audit(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            try:
                record(fn.__name__, "error", type(exc).__name__)
            except Exception:
                pass
            raise
        try:
            record(fn.__name__, "success")
        except Exception:
            pass
        return result
    return wrapped


def recent(limit=200):
    path = log_path()
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=10) as db:
        db.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in db.execute("SELECT id, timestamp, tool, status, detail FROM activity ORDER BY id DESC LIMIT ?", (limit,))]
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise
