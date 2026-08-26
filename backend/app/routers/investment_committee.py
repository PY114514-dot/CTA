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
from app.dependencies import get_report_config
from app.models import DecisionRecord, DecisionStatus
from app.services import investment_committee as ic
from app.services import product_store as store
from app.services.allocation_agent_graph import complete_allocation_approval, request_allocation_approval
from app.routers.chat import _synthesize_with_llm

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


@router.delete("/decisions/{decision_id}")
def delete_allocation_draft(decision_id: str, session: Session = Depends(get_session)):
    """Delete only disposable configuration records, never approved audit history."""
    decision = session.get(DecisionRecord, decision_id)
    if decision is None or decision.decision_type != "allocation":
        raise HTTPException(404, "未找到配置草案")
    if decision.status not in {DecisionStatus.DRAFT, DecisionStatus.VETOED}:
        raise HTTPException(409, "仅草稿或已否决配置可以删除；待审核请先否决，已通过版本会保留审计记录")
    session.delete(decision)
    session.commit()
    return {"id": decision_id, "deleted": True}


@router.post("/decisions/{decision_id}/interpretation")
def interpret_allocation(decision_id: str, session: Session = Depends(get_session)):
    """Generate optional prose only from a frozen, deterministic allocation."""
    decision = session.get(DecisionRecord, decision_id)
    if decision is None or decision.decision_type != "allocation":
        raise HTTPException(404, "未找到配置草案")
    config = get_report_config()
    if not config.enabled:
        raise HTTPException(409, "尚未启用 LLM，无法生成配置解读")
    content = dict(decision.content or {})
    portfolio = content.get("portfolio") or {}
    metrics = portfolio.get("metrics") or {}
    allocations = content.get("allocations") or []
    raw = "\n".join([
        "配置产品：" + "；".join(f"{item.get('product_name', item.get('product_id'))} {item.get('weight', 0):.1%}" for item in allocations),
        f"年化收益：{metrics.get('annualized_return', 0):.2%}；年化波动：{metrics.get('annualized_volatility', 0):.2%}；最大回撤：{metrics.get('maximum_drawdown', 0):.2%}；夏普：{metrics.get('sharpe_ratio')}",
        "约束：" + "；".join(content.get("constraints") or []),
        "风险提示：" + "；".join(content.get("risk_warnings") or []),
    ])
    try:
        interpretation = _synthesize_with_llm(
            raw, "请解释这份已计算的 FOF 配置：说明产品角色、主要取舍和需关注的风险，不得改写数值或提出新的权重。",
            [], config, scope_note="只解释已保存的配置草案；不得选择产品、计算指标或改变权重。",
        )
    except Exception as error:
        raise HTTPException(502, f"配置解读生成失败：{error}") from error
    content["llm_interpretation"] = interpretation
    decision.content = content
    session.commit()
    return {"decision_id": decision.id, "interpretation": interpretation}


@router.post("/decisions/{decision_id}/approve")
def approve(decision_id: str, body: ApproveRequest, session: Session = Depends(get_session)):
    existing = session.get(DecisionRecord, decision_id)
    if existing is not None and existing.decision_type == "allocation":
        if existing.status != DecisionStatus.PENDING_REVIEW:
            raise HTTPException(409, "配置草案尚未提交人工确认，或已完成审批")
        result = complete_allocation_approval(decision_id, approved=True, content=existing.content or {})
        if not result.approved or result.score is None or result.version is None:
            raise HTTPException(500, "审批工作流未完成确定性评分")
        content = dict(existing.content or {})
        content["deterministic_score"] = result.score
        content["workflow"] = {"stage": "approved", "tracking_enabled": True}
        version_snapshot = store.create_snapshot(
            session,
            label=f"FOF 配置版本 {existing.id} v{result.version['number']}",
            content={"source_draft_snapshot_id": content.get("source_draft_snapshot_id"), "decision_id": existing.id, "version": result.version, "content": content},
        )
        existing.content = content
        existing.data_snapshot_id = version_snapshot.id
        session.commit()
    decision = ic.approve_decision(
        session, decision_id,
        reviewer=body.reviewer, adjustment=body.adjustment, comment=body.comment,
    )
    if decision is None:
        raise HTTPException(404, "Decision not found")
    return {"id": decision.id, "status": decision.status, "reviewer": decision.reviewer}


@router.post("/decisions/{decision_id}/veto")
def veto(decision_id: str, body: VetoRequest, session: Session = Depends(get_session)):
    existing = session.get(DecisionRecord, decision_id)
    if existing is not None and existing.decision_type == "allocation" and existing.status == DecisionStatus.PENDING_REVIEW:
        complete_allocation_approval(decision_id, approved=False, content=existing.content or {})
    decision = ic.veto_decision(session, decision_id, reviewer=body.reviewer, reason=body.reason)
    if decision is None:
        raise HTTPException(404, "Decision not found")
    return {"id": decision.id, "status": decision.status, "veto_reason": decision.veto_reason}


@router.post("/decisions/{decision_id}/reopen")
def reopen(decision_id: str, session: Session = Depends(get_session)):
    decision = ic.reopen_decision(session, decision_id)
    if decision is None:
        raise HTTPException(404, "Decision not found")
    if decision.decision_type == "allocation":
        if not request_allocation_approval(decision.id, decision.content or {}).waiting_for_human:
            raise HTTPException(500, "审批工作流未能重新进入人工确认节点")
        content = dict(decision.content or {})
        content["workflow"] = {"stage": "awaiting_human_approval", "tracking_enabled": False}
        decision.content = content
        session.commit()
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
