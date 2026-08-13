"""Investment-committee and outer-loop tracking HTTP interface (P4).

Endpoints for IC approval/veto, research memo export, post-recommendation
tracking and methodology review tasks.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.services import investment_committee as ic

router = APIRouter(prefix="/api/kb", tags=["投委会与外环追踪"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ApproveRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=128)
    adjustment: dict[str, Any] | None = None
    comment: str | None = None


class VetoRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2000)


class TrackingRequest(BaseModel):
    review_date: date | None = None
    return_deviation_threshold: float = Field(default=0.05, ge=0, le=1)
    drawdown_ratio_threshold: float = Field(default=1.5, ge=1, le=10)
    abs_drawdown_floor: float = Field(default=0.20, ge=0, le=1)
    auto_create_task: bool = True


class ReviewTaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str | None = None
    decision_id: str | None = None
    task_type: str = "manual"
    priority: str = "normal"
    assignee: str | None = None
    proposal: str | None = None


class ReviewTaskUpdate(BaseModel):
    status: str | None = None
    assignee: str | None = None
    priority: str | None = None


class ReviewTaskResolve(BaseModel):
    resolution: str = Field(min_length=1, max_length=4000)
    resolved_by: str = Field(min_length=1, max_length=128)
    status: str = "resolved"


# ---------------------------------------------------------------------------
# IC decision endpoints
# ---------------------------------------------------------------------------


@router.get("/decisions/{decision_id}/detail")
def decision_detail(decision_id: str, session: Session = Depends(get_session)):
    detail = ic.get_decision_detail(session, decision_id)
    if detail is None:
        raise HTTPException(404, "Decision not found")
    return detail


@router.post("/decisions/{decision_id}/approve")
def approve(decision_id: str, body: ApproveRequest, session: Session = Depends(get_session)):
    decision = ic.approve_decision(
        session, decision_id,
        reviewer=body.reviewer, adjustment=body.adjustment, comment=body.comment,
    )
    if decision is None:
        raise HTTPException(404, "Decision not found")
    return {"id": decision.id, "status": decision.status, "reviewer": decision.reviewer}


@router.post("/decisions/{decision_id}/veto")
def veto(decision_id: str, body: VetoRequest, session: Session = Depends(get_session)):
    decision = ic.veto_decision(session, decision_id, reviewer=body.reviewer, reason=body.reason)
    if decision is None:
        raise HTTPException(404, "Decision not found")
    return {"id": decision.id, "status": decision.status, "veto_reason": decision.veto_reason}


@router.post("/decisions/{decision_id}/reopen")
def reopen(decision_id: str, session: Session = Depends(get_session)):
    decision = ic.reopen_decision(session, decision_id)
    if decision is None:
        raise HTTPException(404, "Decision not found")
    return {"id": decision.id, "status": decision.status}


@router.get("/decisions/{decision_id}/memo")
def memo(decision_id: str, session: Session = Depends(get_session)):
    result = ic.export_investment_memo(session, decision_id)
    if result is None:
        raise HTTPException(404, "Decision not found")
    return result


@router.get("/decisions/{decision_id}/memo.md", response_class=PlainTextResponse)
def memo_markdown(decision_id: str, session: Session = Depends(get_session)):
    result = ic.export_investment_memo(session, decision_id)
    if result is None:
        raise HTTPException(404, "Decision not found")
    return result["markdown"]


# ---------------------------------------------------------------------------
# Outer-loop tracking endpoints
# ---------------------------------------------------------------------------


@router.post("/decisions/{decision_id}/track")
def track(decision_id: str, body: TrackingRequest, session: Session = Depends(get_session)):
    result = ic.record_tracking(
        session, decision_id,
        review_date=body.review_date,
        return_deviation_threshold=body.return_deviation_threshold,
        drawdown_ratio_threshold=body.drawdown_ratio_threshold,
        abs_drawdown_floor=body.abs_drawdown_floor,
        auto_create_task=body.auto_create_task,
    )
    if result is None:
        raise HTTPException(404, "Decision not found or has no allocations")
    return result


@router.get("/decisions/{decision_id}/tracking")
def tracking_history(decision_id: str, session: Session = Depends(get_session)):
    return ic.list_tracking(session, decision_id)


# ---------------------------------------------------------------------------
# Review task endpoints
# ---------------------------------------------------------------------------


@router.get("/review-tasks")
def list_review_tasks(
    status: str | None = None,
    decision_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    return ic.list_review_tasks(
        session, status=status, decision_id=decision_id, limit=limit, offset=offset
    )


@router.post("/review-tasks")
def create_review_task(body: ReviewTaskCreate, session: Session = Depends(get_session)):
    task = ic.create_review_task(session, **body.model_dump())
    return {"id": task.id, "title": task.title, "status": task.status}


@router.patch("/review-tasks/{task_id}")
def update_review_task(task_id: str, body: ReviewTaskUpdate, session: Session = Depends(get_session)):
    task = ic.update_review_task(
        session, task_id, status=body.status, assignee=body.assignee, priority=body.priority
    )
    if task is None:
        raise HTTPException(404, "Review task not found")
    return {"id": task.id, "status": task.status, "assignee": task.assignee}


@router.post("/review-tasks/{task_id}/resolve")
def resolve_review_task(task_id: str, body: ReviewTaskResolve, session: Session = Depends(get_session)):
    task = ic.resolve_review_task(
        session, task_id,
        resolution=body.resolution, resolved_by=body.resolved_by, status=body.status,
    )
    if task is None:
        raise HTTPException(404, "Review task not found")
    return {"id": task.id, "status": task.status, "resolved_by": task.resolved_by}
