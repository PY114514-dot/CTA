"""Analysis history CRUD endpoints.

Previously the research archive was stored only in browser localStorage.
These endpoints let the frontend sync history records to the backend so
that data survives cache clears and is accessible across devices.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import AnalysisHistory

router = APIRouter(prefix="/api/history", tags=["研究档案"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HistoryRecordCreate(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    product_name: str = Field(default="未命名产品", max_length=256)
    frequency: str = Field(min_length=1, max_length=20)
    nav_count: int = Field(default=0, ge=0)
    nav_text: str = Field(default="")
    metrics: dict[str, Any]
    source_text: str | None = None
    strategy_profile: dict[str, Any] | None = None
    ai_report: dict[str, Any] | None = None


class HistoryRecordResponse(BaseModel):
    id: str
    product_name: str
    frequency: str
    nav_count: int
    nav_text: str
    metrics: dict[str, Any]
    source_text: str | None = None
    strategy_profile: dict[str, Any] | None = None
    ai_report: dict[str, Any] | None = None
    saved_at: str


class HistoryBatchCreate(BaseModel):
    """Bulk-sync request from the frontend localStorage."""
    records: list[HistoryRecordCreate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(record: AnalysisHistory) -> HistoryRecordResponse:
    return HistoryRecordResponse(
        id=record.id,
        product_name=record.product_name,
        frequency=record.frequency,
        nav_count=record.nav_count,
        nav_text=record.nav_text,
        metrics=record.metrics,
        source_text=record.source_text,
        strategy_profile=record.strategy_profile,
        ai_report=record.ai_report,
        saved_at=record.saved_at.strftime("%Y/%m/%d %H:%M:%S") if record.saved_at else "",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[HistoryRecordResponse])
def list_history(session: Session = Depends(get_session)) -> list[HistoryRecordResponse]:
    """Return all history records, newest first."""
    records = (
        session.query(AnalysisHistory)
        .order_by(AnalysisHistory.saved_at.desc())
        .limit(50)
        .all()
    )
    return [_to_response(r) for r in records]


@router.post("", response_model=HistoryRecordResponse)
def create_history(
    body: HistoryRecordCreate,
    session: Session = Depends(get_session),
) -> HistoryRecordResponse:
    """Save a new analysis history record."""
    record = AnalysisHistory(
        id=body.id,
        product_name=body.product_name,
        frequency=body.frequency,
        nav_count=body.nav_count,
        nav_text=body.nav_text,
        metrics=body.metrics,
        source_text=body.source_text,
        strategy_profile=body.strategy_profile,
        ai_report=body.ai_report,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _to_response(record)


@router.post("/sync", response_model=list[HistoryRecordResponse])
def sync_history(
    body: HistoryBatchCreate,
    session: Session = Depends(get_session),
) -> list[HistoryRecordResponse]:
    """Bulk-sync frontend localStorage records to the backend.

    Uses a simple strategy: replace all backend records with the incoming batch.
    This ensures the backend mirrors the frontend's authoritative state.
    """
    session.query(AnalysisHistory).delete()
    results: list[AnalysisHistory] = []
    for item in body.records:
        record = AnalysisHistory(
            id=item.id,
            product_name=item.product_name,
            frequency=item.frequency,
            nav_count=item.nav_count,
            nav_text=item.nav_text,
            metrics=item.metrics,
            source_text=item.source_text,
            strategy_profile=item.strategy_profile,
            ai_report=item.ai_report,
        )
        session.add(record)
        results.append(record)
    session.commit()
    for r in results:
        session.refresh(r)
    return [_to_response(r) for r in results]


@router.delete("/{record_id}")
def delete_history(record_id: str, session: Session = Depends(get_session)) -> dict[str, str]:
    """Delete a single history record."""
    record = session.get(AnalysisHistory, record_id)
    if record is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Record not found")
    session.delete(record)
    session.commit()
    return {"status": "deleted"}


@router.delete("")
def clear_history(session: Session = Depends(get_session)) -> dict[str, str]:
    """Delete all history records."""
    session.query(AnalysisHistory).delete()
    session.commit()
    return {"status": "cleared"}
