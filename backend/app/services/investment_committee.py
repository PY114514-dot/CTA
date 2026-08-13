"""Investment-committee workflow and outer-loop tracking service (P4).

Implements the master design's P4 requirements:
- IC approval / veto with auditable human decisions (reviewer, reason, adjustment).
- Research memo export bound to the decision's data snapshot and citations.
- Post-recommendation tracking: recompute actual portfolio performance since the
  decision date and compare against the frozen expectation; when deviation
  exceeds threshold, create a methodology review task.
- Methodology review tasks: the outer loop only *proposes* changes; resolution
  is human-driven and recorded for audit.

All calculations are deterministic. The outer loop never mutates scoring
weights, tool logic, skills or prompts automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AgentRun,
    DataSnapshot,
    DecisionRecord,
    DecisionStatus,
    NavObservation,
    ProductEntity,
    RecommendationTracking,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTaskType,
    TrackingStatus,
)


# ---------------------------------------------------------------------------
# Default outer-loop thresholds
# ---------------------------------------------------------------------------

DEFAULT_RETURN_DEVIATION_THRESHOLD = 0.05  # |actual - expected| > 5pp → breach
DEFAULT_DRAWDOWN_RATIO_THRESHOLD = 1.5  # actual DD > 1.5× expected weighted DD → breach
DEFAULT_ABS_DRAWDDOWN_FLOOR = 0.20  # actual DD > 20% absolute → breach regardless


# ---------------------------------------------------------------------------
# IC approval / veto
# ---------------------------------------------------------------------------


def approve_decision(
    session: Session,
    decision_id: str,
    *,
    reviewer: str,
    adjustment: dict[str, Any] | None = None,
    comment: str | None = None,
) -> DecisionRecord | None:
    """Approve a pending recommendation. Optional human adjustment is recorded."""
    decision = session.get(DecisionRecord, decision_id)
    if decision is None:
        return None
    decision.status = DecisionStatus.APPROVED
    decision.reviewer = reviewer
    decision.reviewed_at = datetime.now()
    decision.veto_reason = None
    if adjustment is not None:
        decision.adjustment = adjustment
    if comment:
        existing = decision.content or {}
        existing = dict(existing)
        existing["ic_comment"] = comment
        decision.content = existing
    session.commit()
    session.refresh(decision)
    return decision


def veto_decision(
    session: Session,
    decision_id: str,
    *,
    reviewer: str,
    reason: str,
) -> DecisionRecord | None:
    """Veto a recommendation. A veto reason is mandatory for audit."""
    decision = session.get(DecisionRecord, decision_id)
    if decision is None:
        return None
    decision.status = DecisionStatus.VETOED
    decision.reviewer = reviewer
    decision.reviewed_at = datetime.now()
    decision.veto_reason = reason
    session.commit()
    session.refresh(decision)
    return decision


def reopen_decision(session: Session, decision_id: str) -> DecisionRecord | None:
    """Return an approved/vetoed decision to pending review (re-deliberation)."""
    decision = session.get(DecisionRecord, decision_id)
    if decision is None:
        return None
    decision.status = DecisionStatus.PENDING_REVIEW
    decision.veto_reason = None
    session.commit()
    session.refresh(decision)
    return decision


def get_decision_detail(session: Session, decision_id: str) -> dict[str, Any] | None:
    """Full decision bundle: record + snapshot + agent run + tracking + tasks."""
    decision = session.get(DecisionRecord, decision_id)
    if decision is None:
        return None

    snapshot = None
    if decision.data_snapshot_id:
        snap = session.get(DataSnapshot, decision.data_snapshot_id)
        if snap is not None:
            snapshot = {"id": snap.id, "label": snap.label, "content": snap.content,
                        "created_at": snap.created_at.isoformat() if snap.created_at else None}

    run = None
    if decision.run_id:
        r = session.get(AgentRun, decision.run_id)
        if r is not None:
            run = {
                "id": r.id, "user_query": r.user_query, "phase": r.phase,
                "tools_used": r.tools_used, "citations": r.citations,
                "answer": r.answer,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }

    tracking = [
        _tracking_to_dict(t)
        for t in session.execute(
            select(RecommendationTracking)
            .where(RecommendationTracking.decision_id == decision_id)
            .order_by(RecommendationTracking.review_date)
        ).scalars().all()
    ]

    tasks = [
        _task_to_dict(t)
        for t in session.execute(
            select(ReviewTask)
            .where(ReviewTask.decision_id == decision_id)
            .order_by(ReviewTask.created_at.desc())
        ).scalars().all()
    ]

    return {
        "id": decision.id,
        "decision_type": decision.decision_type,
        "title": decision.title,
        "content": decision.content,
        "status": decision.status,
        "reviewer": decision.reviewer,
        "reviewed_at": decision.reviewed_at.isoformat() if decision.reviewed_at else None,
        "veto_reason": decision.veto_reason,
        "adjustment": decision.adjustment,
        "created_at": decision.created_at.isoformat() if decision.created_at else None,
        "snapshot": snapshot,
        "run": run,
        "tracking": tracking,
        "review_tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Research memo export
# ---------------------------------------------------------------------------


def export_investment_memo(session: Session, decision_id: str) -> dict[str, Any] | None:
    """Generate a markdown research memo bound to the decision snapshot.

    The memo aggregates the recommendation, evidence citations from the agent
    run, risk warnings, due-diligence gaps and the IC decision, so any memo can
    be reproduced from the same snapshot.
    """
    detail = get_decision_detail(session, decision_id)
    if detail is None:
        return None

    content = detail.get("content") or {}
    allocations = content.get("allocations", [])
    constraints = content.get("constraints", {})
    expected = content.get("expected_metrics", {})
    run = detail.get("run") or {}
    citations = run.get("citations") or []

    lines: list[str] = []
    lines.append(f"# 研究备忘录：{detail.get('title') or 'FOF 配置建议'}")
    lines.append("")
    lines.append(f"- 决策编号：`{detail['id']}`")
    lines.append(f"- 状态：{detail['status']}")
    lines.append(f"- 生成时间：{detail.get('created_at')}")
    if detail.get("snapshot"):
        lines.append(f"- 数据快照：`{detail['snapshot']['id']}`（{detail['snapshot'].get('label')}）")
    if constraints:
        lines.append(f"- 约束：单产品上限 {constraints.get('max_single_weight', '—')}，"
                     f"无风险利率 {constraints.get('risk_free_rate', '—')}")
    lines.append("")

    # Allocation table.
    lines.append("## 一、配置建议")
    lines.append("")
    if allocations:
        lines.append("| 产品 | 权重 | 评分 | 预期年化 | 预期回撤 |")
        lines.append("|---|---|---|---|---|")
        for a in allocations:
            pid = a.get("product_id")
            exp = expected.get(pid, {}) if isinstance(expected, dict) else {}
            ann = exp.get("annualized_return")
            mdd = exp.get("maximum_drawdown")
            ann_s = f"{ann:.2%}" if isinstance(ann, (int, float)) else "—"
            mdd_s = f"{mdd:.2%}" if isinstance(mdd, (int, float)) else "—"
            lines.append(f"| {a.get('name', pid)} | {a.get('weight', 0):.1%} | "
                         f"{a.get('score', 0):.1f} | {ann_s} | {mdd_s} |")
    else:
        lines.append("_无配置项。_")
    lines.append("")

    # Excluded count.
    excluded_count = content.get("excluded_count")
    if excluded_count:
        lines.append(f"剔除候选：{excluded_count} 个（详见决策记录与快照）。")
        lines.append("")

    # Evidence citations.
    lines.append("## 二、证据引用")
    lines.append("")
    if citations:
        for c in citations[:12]:
            fname = c.get("filename") or c.get("file_id") or "未知来源"
            page = c.get("page_number")
            page_s = f" p.{page}" if page is not None else ""
            snippet = (c.get("snippet") or "").strip().replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:80] + "…"
            conf = c.get("confidence")
            conf_s = f"（置信 {conf:.2f}）" if isinstance(conf, (int, float)) else ""
            lines.append(f"- {fname}{page_s}{conf_s}：{snippet}" if snippet else f"- {fname}{page_s}")
    else:
        lines.append("_本次推荐未绑定材料引用（仅基于净值数据）。_")
    lines.append("")

    # IC decision.
    lines.append("## 三、投委会决议")
    lines.append("")
    lines.append(f"- 决议状态：{detail['status']}")
    if detail.get("reviewer"):
        lines.append(f"- 审核人：{detail['reviewer']}")
    if detail.get("reviewed_at"):
        lines.append(f"- 审核时间：{detail['reviewed_at']}")
    if detail.get("veto_reason"):
        lines.append(f"- 否决理由：{detail['veto_reason']}")
    if detail.get("adjustment"):
        lines.append(f"- 人工调整：{detail['adjustment']}")
    ic_comment = content.get("ic_comment")
    if ic_comment:
        lines.append(f"- 审核意见：{ic_comment}")
    lines.append("")
    lines.append("---")
    lines.append("_本备忘录由 FOF Agent 生成，绑定数据快照，可复现。推荐仅为研究参考，不构成投资建议。_")

    markdown = "\n".join(lines)
    return {
        "decision_id": decision_id,
        "title": detail.get("title"),
        "snapshot_id": (detail.get("snapshot") or {}).get("id"),
        "markdown": markdown,
        "generated_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Outer loop: post-recommendation tracking
# ---------------------------------------------------------------------------


@dataclass
class TrackingComputation:
    """Intermediate result of recomputing portfolio performance since decision."""

    review_date: date
    periods_elapsed: int
    expected_return: float | None
    actual_return: float | None
    return_deviation: float | None
    actual_max_drawdown: float | None
    per_product: dict[str, dict[str, Any]] = field(default_factory=dict)
    breach_reasons: list[str] = field(default_factory=list)


def _nav_after(session: Session, product_id: str, since: date) -> list[tuple[date, float]]:
    """NAV observations strictly after `since`, ordered by date."""
    rows = session.execute(
        select(NavObservation.observation_date, NavObservation.nav)
        .where(NavObservation.product_id == product_id)
        .order_by(NavObservation.observation_date)
    ).all()
    return [(d, v) for d, v in rows if d > since]


def _max_drawdown(values: list[float]) -> float:
    """Maximum drawdown (negative number) from a NAV/value series."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = arr / running_max - 1.0
    dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=-1.0)
    return float(dd.min())


def compute_tracking(
    session: Session,
    decision_id: str,
    *,
    review_date: date | None = None,
    return_deviation_threshold: float = DEFAULT_RETURN_DEVIATION_THRESHOLD,
    drawdown_ratio_threshold: float = DEFAULT_DRAWDOWN_RATIO_THRESHOLD,
    abs_drawdown_floor: float = DEFAULT_ABS_DRAWDDOWN_FLOOR,
) -> TrackingComputation | None:
    """Recompute actual portfolio performance since the decision date.

    Compares weighted actual return/drawdown against the expectation frozen in
    the decision content. Returns a computation object (not yet persisted).
    """
    decision = session.get(DecisionRecord, decision_id)
    if decision is None:
        return None

    content = decision.content or {}
    allocations = content.get("allocations", [])
    if not allocations:
        return None

    expected_metrics = content.get("expected_metrics", {}) or {}
    as_of = review_date or date.today()

    # Determine decision (baseline) date.
    decision_date_raw = content.get("decision_date")
    if decision_date_raw:
        try:
            baseline = date.fromisoformat(decision_date_raw)
        except ValueError:
            baseline = decision.created_at.date() if decision.created_at else as_of
    else:
        baseline = decision.created_at.date() if decision.created_at else as_of

    per_product: dict[str, dict[str, Any]] = {}
    weighted_actual_return = 0.0
    weighted_expected_return = 0.0
    weighted_expected_dd = 0.0
    total_weight = 0.0
    computed_weight = 0.0  # weight of products with enough post-decision data
    max_periods = 0

    for alloc in allocations:
        pid = alloc.get("product_id")
        weight = float(alloc.get("weight", 0.0))
        if not pid or weight <= 0:
            continue

        series = _nav_after(session, pid, baseline)
        exp = expected_metrics.get(pid, {}) if isinstance(expected_metrics, dict) else {}
        exp_return = exp.get("annualized_return")
        exp_dd = exp.get("maximum_drawdown")

        entry: dict[str, Any] = {
            "product_id": pid,
            "weight": weight,
            "observations": len(series),
            "actual_return": None,
            "actual_max_drawdown": None,
            "expected_return": exp_return,
        }

        if len(series) >= 2:
            first_nav = series[0][1]
            last_nav = series[-1][1]
            if first_nav > 0:
                actual_return = last_nav / first_nav - 1.0
                actual_dd = _max_drawdown([v for _, v in series])
                entry["actual_return"] = round(actual_return, 6)
                entry["actual_max_drawdown"] = round(actual_dd, 6)
                entry["start_date"] = series[0][0].isoformat()
                entry["end_date"] = series[-1][0].isoformat()
                weighted_actual_return += weight * actual_return
                computed_weight += weight
                max_periods = max(max_periods, len(series))

        if isinstance(exp_return, (int, float)):
            weighted_expected_return += weight * exp_return
        if isinstance(exp_dd, (int, float)):
            weighted_expected_dd += weight * exp_dd

        total_weight += weight
        per_product[pid] = entry

    if total_weight <= 0:
        return None

    # Normalize by computed weight (products with actual data). If no product
    # had enough post-decision observations, actual_return is None — this is
    # "insufficient data", NOT a zero return.
    actual_return = weighted_actual_return / computed_weight if computed_weight > 0 else None
    expected_return = weighted_expected_return / total_weight if total_weight else None
    deviation = (actual_return - expected_return) if (
        actual_return is not None and expected_return is not None
    ) else None

    # Portfolio-level actual max drawdown: weighted per-product DD (proxy).
    weighted_actual_dd = 0.0
    dd_weight = 0.0
    for pid, entry in per_product.items():
        if entry.get("actual_max_drawdown") is not None:
            weighted_actual_dd += entry["weight"] * entry["actual_max_drawdown"]
            dd_weight += entry["weight"]
    actual_dd = weighted_actual_dd / dd_weight if dd_weight > 0 else None

    # Breach detection.
    breach_reasons: list[str] = []
    if deviation is not None and abs(deviation) > return_deviation_threshold:
        breach_reasons.append(
            f"实际收益 {actual_return:.2%} 与预期 {expected_return:.2%} 偏差 "
            f"{deviation:+.2%}，超过阈值 ±{return_deviation_threshold:.0%}。"
        )
    if actual_dd is not None:
        if abs(actual_dd) > abs_drawdown_floor:
            breach_reasons.append(
                f"组合实际最大回撤 {actual_dd:.2%} 超过绝对底线 {abs_drawdown_floor:.0%}。"
            )
        if weighted_expected_dd < 0 and abs(actual_dd) > drawdown_ratio_threshold * abs(weighted_expected_dd):
            breach_reasons.append(
                f"实际回撤 {actual_dd:.2%} 超过预期加权回撤 {weighted_expected_dd:.2%} 的 "
                f"{drawdown_ratio_threshold:.1f} 倍。"
            )

    return TrackingComputation(
        review_date=as_of,
        periods_elapsed=max_periods,
        expected_return=round(expected_return, 6) if expected_return is not None else None,
        actual_return=round(actual_return, 6) if actual_return is not None else None,
        return_deviation=round(deviation, 6) if deviation is not None else None,
        actual_max_drawdown=round(actual_dd, 6) if actual_dd is not None else None,
        per_product=per_product,
        breach_reasons=breach_reasons,
    )


def record_tracking(
    session: Session,
    decision_id: str,
    *,
    review_date: date | None = None,
    return_deviation_threshold: float = DEFAULT_RETURN_DEVIATION_THRESHOLD,
    drawdown_ratio_threshold: float = DEFAULT_DRAWDOWN_RATIO_THRESHOLD,
    abs_drawdown_floor: float = DEFAULT_ABS_DRAWDDOWN_FLOOR,
    auto_create_task: bool = True,
) -> dict[str, Any] | None:
    """Persist a tracking review point; create a methodology task on breach.

    The created ReviewTask only carries a non-binding proposal; methodology
    changes require human resolution.
    """
    comp = compute_tracking(
        session, decision_id,
        review_date=review_date,
        return_deviation_threshold=return_deviation_threshold,
        drawdown_ratio_threshold=drawdown_ratio_threshold,
        abs_drawdown_floor=abs_drawdown_floor,
    )
    if comp is None:
        return None

    # Determine status: insufficient data takes precedence over breach/normal.
    if comp.actual_return is None:
        status = TrackingStatus.INSUFFICIENT_DATA
        breached = False
    else:
        breached = len(comp.breach_reasons) > 0
        status = TrackingStatus.BREACH if breached else TrackingStatus.NORMAL

    tracking = RecommendationTracking(
        decision_id=decision_id,
        review_date=comp.review_date,
        periods_elapsed=comp.periods_elapsed,
        expected_return=comp.expected_return,
        actual_return=comp.actual_return,
        return_deviation=comp.return_deviation,
        actual_max_drawdown=comp.actual_max_drawdown,
        per_product=comp.per_product,
        status=status,
        breach_reasons=comp.breach_reasons or None,
    )
    session.add(tracking)
    session.commit()
    session.refresh(tracking)

    triggered_task: ReviewTask | None = None
    if breached and auto_create_task:
        decision = session.get(DecisionRecord, decision_id)
        title_base = decision.title if decision else f"决策 {decision_id}"
        proposal = (
            "外环检测到推荐表现偏离预期。建议复核：评分权重、筛选阈值、"
            "相关性约束与候选池范围。任何方法论调整须经人工审核、版本化并回测后生效。"
        )
        triggered_task = ReviewTask(
            decision_id=decision_id,
            tracking_id=tracking.id,
            task_type=ReviewTaskType.METHODOLOGY,
            title=f"方法论复核：{title_base}（{comp.review_date.isoformat()}）",
            description="；".join(comp.breach_reasons),
            status=ReviewTaskStatus.OPEN,
            priority="high",
            proposal=proposal,
        )
        session.add(triggered_task)
        session.commit()
        session.refresh(triggered_task)

        tracking.triggered_task_id = triggered_task.id
        session.commit()
        session.refresh(tracking)

    result = _tracking_to_dict(tracking)
    result["triggered_task"] = _task_to_dict(triggered_task) if triggered_task else None
    return result


def list_tracking(
    session: Session, decision_id: str
) -> list[dict[str, Any]]:
    rows = session.execute(
        select(RecommendationTracking)
        .where(RecommendationTracking.decision_id == decision_id)
        .order_by(RecommendationTracking.review_date)
    ).scalars().all()
    return [_tracking_to_dict(t) for t in rows]


# ---------------------------------------------------------------------------
# Review tasks
# ---------------------------------------------------------------------------


def create_review_task(
    session: Session,
    *,
    title: str,
    description: str | None = None,
    decision_id: str | None = None,
    task_type: str = ReviewTaskType.MANUAL,
    priority: str = "normal",
    assignee: str | None = None,
    proposal: str | None = None,
) -> ReviewTask:
    task = ReviewTask(
        decision_id=decision_id,
        task_type=task_type,
        title=title,
        description=description,
        priority=priority,
        assignee=assignee,
        proposal=proposal,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def update_review_task(
    session: Session,
    task_id: str,
    *,
    status: str | None = None,
    assignee: str | None = None,
    priority: str | None = None,
) -> ReviewTask | None:
    task = session.get(ReviewTask, task_id)
    if task is None:
        return None
    if status is not None:
        task.status = status
    if assignee is not None:
        task.assignee = assignee
    if priority is not None:
        task.priority = priority
    session.commit()
    session.refresh(task)
    return task


def resolve_review_task(
    session: Session,
    task_id: str,
    *,
    resolution: str,
    resolved_by: str,
    status: str = ReviewTaskStatus.RESOLVED,
) -> ReviewTask | None:
    """Record the human resolution of a review task (audit requirement)."""
    task = session.get(ReviewTask, task_id)
    if task is None:
        return None
    task.status = status
    task.resolution = resolution
    task.resolved_by = resolved_by
    task.resolved_at = datetime.now()
    session.commit()
    session.refresh(task)
    return task


def list_review_tasks(
    session: Session,
    *,
    status: str | None = None,
    decision_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    stmt = select(ReviewTask).order_by(ReviewTask.created_at.desc())
    if status:
        stmt = stmt.where(ReviewTask.status == status)
    if decision_id:
        stmt = stmt.where(ReviewTask.decision_id == decision_id)
    rows = session.execute(stmt.offset(offset).limit(limit)).scalars().all()
    return [_task_to_dict(t) for t in rows]


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _tracking_to_dict(t: RecommendationTracking) -> dict[str, Any]:
    return {
        "id": t.id,
        "decision_id": t.decision_id,
        "review_date": t.review_date.isoformat() if t.review_date else None,
        "periods_elapsed": t.periods_elapsed,
        "expected_return": t.expected_return,
        "actual_return": t.actual_return,
        "return_deviation": t.return_deviation,
        "actual_max_drawdown": t.actual_max_drawdown,
        "per_product": t.per_product,
        "status": t.status,
        "breach_reasons": t.breach_reasons,
        "triggered_task_id": t.triggered_task_id,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


def _task_to_dict(t: ReviewTask) -> dict[str, Any]:
    return {
        "id": t.id,
        "decision_id": t.decision_id,
        "tracking_id": t.tracking_id,
        "task_type": t.task_type,
        "title": t.title,
        "description": t.description,
        "status": t.status,
        "priority": t.priority,
        "assignee": t.assignee,
        "proposal": t.proposal,
        "resolution": t.resolution,
        "resolved_by": t.resolved_by,
        "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }
