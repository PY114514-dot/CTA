"""Durable, user-owned Agent preferences backed by immutable snapshots."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DataSnapshot
from app.services import product_store as store


def load_preferences(session: Session, session_id: str | None) -> dict[str, Any]:
    """Return only the latest explicit preference contract for this chat."""
    if not session_id:
        return {}
    snapshot = session.execute(
        select(DataSnapshot)
        .where(DataSnapshot.label == f"agent-memory:{session_id}")
        .order_by(DataSnapshot.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return dict((snapshot.content or {}).get("preferences") or {}) if snapshot else {}


def remember_preferences(session: Session, session_id: str | None, preferences: dict[str, Any]) -> None:
    """Append an auditable preference version; never overwrite past intent."""
    if not session_id or not preferences:
        return
    store.create_snapshot(
        session,
        label=f"agent-memory:{session_id}",
        content={"preferences": preferences, "scope": "当前 Agent 对话"},
    )
