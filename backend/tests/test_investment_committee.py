"""Tests for P4: investment-committee workflow and outer-loop tracking."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import DecisionStatus, ReviewTaskStatus, TrackingStatus
from app.services import investment_committee as ic
from app.services import product_store as store


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        yield s


def _make_product_with_nav(
    session: Session,
    name: str,
    navs: list[float],
    start: date = date(2025, 1, 31),
    freq: str = "monthly",
) -> str:
    """Create a confirmed product with a monthly NAV series; return its id."""
    product = store.create_product(session, standard_name=name, strategy="commodity_cta")
    store.confirm_product(session, product.id, "analyst")
    points = []
    for i, nav in enumerate(navs):
        # Roughly monthly dates.
        d = start + timedelta(days=30 * i)
        points.append({"observation_date": d, "nav": nav})
    store.add_nav_observations(session, product.id, points, frequency=freq)
    return product.id


def _make_decision(session: Session, allocations: list[dict], expected: dict) -> str:
    """Create a pending recommendation decision with baseline metrics."""
    snapshot = store.create_snapshot(
        session, label="FOF 推荐快照",
        content={"expected_metrics": expected, "scores": {}},
    )
    decision = store.create_decision(
        session,
        decision_type="recommendation",
        title="FOF 初始配置建议",
        content={
            "allocations": allocations,
            "expected_metrics": expected,
            "decision_date": "2025-06-30",
            "constraints": {"max_single_weight": 0.35, "risk_free_rate": 0.015},
        },
        data_snapshot_id=snapshot.id,
        status="pending_review",
    )
    return decision.id


# ---------------------------------------------------------------------------
# IC approval / veto
# ---------------------------------------------------------------------------


class TestICDecisions:
    def test_approve_records_reviewer(self, session: Session):
        pid = _make_product_with_nav(session, "CTA A", [1.0, 1.05, 1.10])
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA A", "weight": 1.0, "score": 80.0}],
            {pid: {"annualized_return": 0.10, "maximum_drawdown": -0.05}},
        )
        result = ic.approve_decision(session, decision_id, reviewer="ic_chair", comment="同意")
        assert result is not None
        assert result.status == DecisionStatus.APPROVED
        assert result.reviewer == "ic_chair"
        assert result.content["ic_comment"] == "同意"

    def test_veto_requires_and_stores_reason(self, session: Session):
        pid = _make_product_with_nav(session, "CTA B", [1.0, 1.02])
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA B", "weight": 1.0, "score": 50.0}],
            {pid: {"annualized_return": 0.02, "maximum_drawdown": -0.03}},
        )
        result = ic.veto_decision(session, decision_id, reviewer="ic_chair", reason="流动性不足")
        assert result is not None
        assert result.status == DecisionStatus.VETOED
        assert result.veto_reason == "流动性不足"

    def test_reopen_returns_to_pending(self, session: Session):
        pid = _make_product_with_nav(session, "CTA C", [1.0, 1.03])
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA C", "weight": 1.0, "score": 60.0}],
            {pid: {"annualized_return": 0.03, "maximum_drawdown": -0.02}},
        )
        ic.approve_decision(session, decision_id, reviewer="ic_chair")
        reopened = ic.reopen_decision(session, decision_id)
        assert reopened is not None
        assert reopened.status == DecisionStatus.PENDING_REVIEW

    def test_decision_detail_bundles_snapshot_and_run(self, session: Session):
        pid = _make_product_with_nav(session, "CTA D", [1.0, 1.04])
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA D", "weight": 1.0, "score": 70.0}],
            {pid: {"annualized_return": 0.04, "maximum_drawdown": -0.02}},
        )
        detail = ic.get_decision_detail(session, decision_id)
        assert detail is not None
        assert detail["snapshot"] is not None
        assert detail["snapshot"]["content"]["expected_metrics"]
        assert detail["tracking"] == []
        assert detail["review_tasks"] == []


# ---------------------------------------------------------------------------
# Research memo
# ---------------------------------------------------------------------------


class TestInvestmentMemo:
    def test_memo_contains_allocation_and_decision(self, session: Session):
        pid = _make_product_with_nav(session, "CTA Memo", [1.0, 1.08])
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA Memo", "weight": 0.6, "score": 82.0}],
            {pid: {"annualized_return": 0.12, "maximum_drawdown": -0.06}},
        )
        ic.approve_decision(session, decision_id, reviewer="ic_chair")
        memo = ic.export_investment_memo(session, decision_id)
        assert memo is not None
        md = memo["markdown"]
        assert "CTA Memo" in md
        assert "60.0%" in md
        assert memo["snapshot_id"] is not None
        assert "投委会决议" in md
        assert "approved" in md


# ---------------------------------------------------------------------------
# Outer-loop tracking
# ---------------------------------------------------------------------------


class TestOuterLoopTracking:
    def test_normal_tracking_no_breach(self, session: Session):
        # NAV rises modestly after decision date (2025-06-30), close to expectation.
        pid = _make_product_with_nav(
            session, "CTA Steady",
            [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],  # last ~2025-08
            start=date(2025, 1, 31),
        )
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA Steady", "weight": 1.0, "score": 75.0}],
            {pid: {"annualized_return": 0.05, "maximum_drawdown": -0.02}},
        )
        result = ic.record_tracking(session, decision_id, review_date=date(2025, 9, 1))
        assert result is not None
        assert result["status"] == TrackingStatus.NORMAL
        assert result["actual_return"] is not None
        assert result["triggered_task"] is None

    def test_breach_creates_methodology_task(self, session: Session):
        # NAV collapses after decision date → large negative deviation + deep DD.
        pid = _make_product_with_nav(
            session, "CTA Crash",
            [1.0, 1.02, 1.03, 1.0, 0.85, 0.70, 0.65, 0.60],
            start=date(2025, 1, 31),
        )
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA Crash", "weight": 1.0, "score": 78.0}],
            {pid: {"annualized_return": 0.15, "maximum_drawdown": -0.05}},
        )
        result = ic.record_tracking(session, decision_id, review_date=date(2025, 9, 1))
        assert result is not None
        assert result["status"] == TrackingStatus.BREACH
        assert result["breach_reasons"]
        assert result["triggered_task"] is not None
        assert result["triggered_task"]["task_type"] == "methodology"
        assert result["triggered_task"]["status"] == ReviewTaskStatus.OPEN

    def test_tracking_history_accumulates(self, session: Session):
        pid = _make_product_with_nav(
            session, "CTA Hist",
            [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
            start=date(2025, 1, 31),
        )
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA Hist", "weight": 1.0, "score": 75.0}],
            {pid: {"annualized_return": 0.05, "maximum_drawdown": -0.02}},
        )
        ic.record_tracking(session, decision_id, review_date=date(2025, 8, 1))
        ic.record_tracking(session, decision_id, review_date=date(2025, 9, 1))
        history = ic.list_tracking(session, decision_id)
        assert len(history) == 2

    def test_insufficient_data_no_false_breach(self, session: Session):
        """When NAV series ends before decision_date, tracking reports insufficient_data."""
        # Only 3 monthly points: 2025-01-31, 2025-03-02, 2025-04-01 — all before 2025-06-30.
        pid = _make_product_with_nav(
            session, "CTA Old",
            [1.0, 1.02, 1.04],
            start=date(2025, 1, 31),
        )
        # Decision date is 2025-06-30 — well after the last NAV point.
        decision_id = _make_decision(
            session,
            [{"product_id": pid, "name": "CTA Old", "weight": 1.0, "score": 80.0}],
            {pid: {"annualized_return": 0.15, "maximum_drawdown": -0.03}},
        )
        result = ic.record_tracking(session, decision_id, review_date=date(2025, 7, 30))
        assert result is not None
        assert result["status"] == TrackingStatus.INSUFFICIENT_DATA
        assert result["actual_return"] is None
        assert result["return_deviation"] is None
        assert result["triggered_task"] is None
        assert not result["breach_reasons"]


# ---------------------------------------------------------------------------
# Review tasks
# ---------------------------------------------------------------------------


class TestReviewTasks:
    def test_manual_task_and_resolution(self, session: Session):
        task = ic.create_review_task(
            session, title="复核评分权重", description="人工发起", priority="high"
        )
        assert task.status == ReviewTaskStatus.OPEN

        resolved = ic.resolve_review_task(
            session, task.id, resolution="维持现有权重，补充回撤约束", resolved_by="ic_chair"
        )
        assert resolved is not None
        assert resolved.status == ReviewTaskStatus.RESOLVED
        assert resolved.resolved_by == "ic_chair"
        assert resolved.resolved_at is not None

    def test_list_filter_by_status(self, session: Session):
        t1 = ic.create_review_task(session, title="任务一")
        t2 = ic.create_review_task(session, title="任务二")
        ic.resolve_review_task(session, t2.id, resolution="完成", resolved_by="analyst")

        open_tasks = ic.list_review_tasks(session, status=ReviewTaskStatus.OPEN)
        assert any(t["id"] == t1.id for t in open_tasks)
        assert not any(t["id"] == t2.id for t in open_tasks)
