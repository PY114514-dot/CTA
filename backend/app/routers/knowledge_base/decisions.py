"""Snapshot and decision route handlers for the knowledge-base API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_session
from app.services import product_store as store
from app.services.fof_allocation_research import _portfolio_performance
from app.services.quant_screen import AllocationItem

from .schemas import DecisionCreate, DecisionReview, SnapshotCreate

router = APIRouter()


# ---------------------------------------------------------------------------
# Snapshot & Decision endpoints
# ---------------------------------------------------------------------------


@router.post("/snapshots")
def create_snapshot(body: SnapshotCreate, session: Session = Depends(get_session)):
    snapshot = store.create_snapshot(session, label=body.label, content=body.content)
    return {"id": snapshot.id, "label": snapshot.label}


@router.post("/decisions")
def create_decision(body: DecisionCreate, session: Session = Depends(get_session)):
    decision = store.create_decision(session, **body.model_dump())
    return {"id": decision.id, "decision_type": decision.decision_type, "status": decision.status}


@router.patch("/decisions/{decision_id}/review")
def review_decision(decision_id: str, body: DecisionReview, session: Session = Depends(get_session)):
    decision = store.review_decision(
        session, decision_id,
        status=body.status, reviewer=body.reviewer,
        veto_reason=body.veto_reason, adjustment=body.adjustment,
    )
    if decision is None:
        raise HTTPException(404, "Decision not found")
    return {"id": decision.id, "status": decision.status, "reviewer": decision.reviewer}


@router.get("/decisions")
def list_decisions(status: str | None = None, limit: int = 20, offset: int = 0, session: Session = Depends(get_session)):
    decisions = [
        decision for decision in store.list_decisions(session, status=status, limit=limit, offset=offset)
        if decision.decision_type == "allocation"
    ]
    repaired = False
    for decision in decisions:
        content = decision.content or {}
        allocations = content.get("allocations") or []
        if content.get("portfolio") or not allocations or len(allocations) > 20:
            continue
        portfolio = _portfolio_performance(session, [
            AllocationItem(
                product_id=item["product_id"],
                product_name=item.get("product_name") or item.get("name") or item["product_id"],
                weight=float(item["weight"]),
                score=float(item.get("score", 0)),
                rationale=item.get("rationale", ""),
            )
            for item in allocations
        ])
        if portfolio is not None:
            decision.content = {**content, "portfolio": portfolio}
            repaired = True
    if repaired:
        session.commit()
    return [
        {
            "id": d.id, "decision_type": d.decision_type, "title": d.title,
            "status": d.status, "reviewer": d.reviewer,
            "veto_reason": d.veto_reason,
            "allocation_count": len((d.content or {}).get("allocations") or []),
            "portfolio_metrics": ((d.content or {}).get("portfolio") or {}).get("metrics"),
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in decisions
    ]
