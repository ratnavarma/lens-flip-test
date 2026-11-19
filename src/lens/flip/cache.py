"""Call cache: sqlite, one row per distinct request, key = sha256 of the request."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_PATH = "results/cache/calls.sqlite"


def make_key(model: str, prompt_version: str, messages: list[dict[str, str]], options: dict[str, Any],
             extra: dict[str, Any] | None = None) -> str:
    blob = json.dumps(
        {"model": model, "prompt_version": prompt_version, "messages": messages, "options": options,
         "extra": extra or {}},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


class CallCache:
    """Stores the full raw reply text per key. Counts hits and stores."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls ("
            "key TEXT PRIMARY KEY, model TEXT NOT NULL, prompt_version TEXT NOT NULL, "
            "reply TEXT NOT NULL, created REAL NOT NULL)"
        )
        self.db.commit()
        self.hits = 0
        self.stores = 0

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT reply FROM calls WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        self.hits += 1
        return row[0]

    def put(self, key: str, model: str, prompt_version: str, reply: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO calls (key, model, prompt_version, reply, created) VALUES (?, ?, ?, ?, ?)",
            (key, model, prompt_version, reply, time.time()),
        )
        self.db.commit()
        self.stores += 1

    def close(self) -> None:
        self.db.close()
