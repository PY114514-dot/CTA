"""Small local session memory for the FOF agent.

This is intentionally an audit aid, not a source of financial facts: each run
is still calculated only from the NAV series supplied in its request.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FofSessionMemory:
    """Append-only JSONL memory with bounded retrieval by session id."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "sessions.jsonl"

    def recall(self, session_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        matches: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("session_id") == session_id:
                matches.append(item)
        return matches[-3:]

    def remember(self, session_id: str, summary: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {
            "session_id": session_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
