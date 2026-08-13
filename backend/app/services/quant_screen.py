"""Quantitative screening, comparison, correlation and FOF allocation tools.

All calculations are deterministic and traceable. Results bind to a data
snapshot so any recommendation can be reproduced from the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.models import NavObservation, ProductEntity, ReviewStatus
from app.schemas import DataFrequency, NavAnalysisRequest, NetAssetValuePoint
from app.services.nav_metrics import calculate_nav_analysis


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProductMetrics:
    """Computed metrics for one product over the aligned window."""

    product_id: str
    product_name: str
    observation_count: int
    start_date: str
    end_date: str
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float | None
    maximum_drawdown: float
    calmar_ratio: float | None


@dataclass
class ScreenResult:
    """One product's screening outcome with pass/fail and reasons."""

    product_id: str
    product_name: str
    passed: bool
    metrics: ProductMetrics | None = None
    exclusion_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple products."""

    products: list[ProductMetrics] = field(default_factory=list)
    aligned_window: str = ""
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class AllocationItem:
    """One product's weight in the FOF portfolio."""

    product_id: str
    product_name: str
    weight: float
    score: float
    rationale: str


@dataclass
class RecommendationResult:
    """Full recommendation output with audit trail."""

    allocations: list[AllocationItem] = field(default_factory=list)
    excluded: list[ScreenResult] = field(default_factory=list)
    total_candidates: int = 0
    passed_candidates: int = 0
    constraints: dict[str, Any] = field(default_factory=dict)
    risk_warnings: list[str] = field(default_factory=list)
    due_diligence_gaps: list[str] = field(default_factory=list)
    data_snapshot: dict[str, Any] = field(default_factory=dict)
    expected_metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core: load aligned NAV series
# ---------------------------------------------------------------------------


def _load_nav_array(session: Session, product_id: str) -> tuple[list[str], list[float]]:
    """Load NAV observations as (dates, values) sorted by date."""
    observations = list(
        session.query(NavObservation)
        .filter(
            NavObservation.product_id == product_id,
            NavObservation.review_status == ReviewStatus.REVIEWED,
        )
        .order_by(NavObservation.observation_date)
        .all()
    )
    dates = [o.observation_date.isoformat() for o in observations]
    values = [o.nav for o in observations]
    return dates, values


def _compute_returns(values: list[float]) -> np.ndarray:
    """Simple period-over-period returns."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) < 2:
        return np.array([])
    return arr[1:] / arr[:-1] - 1.0


# ---------------------------------------------------------------------------
# 1. Product comparison
# ---------------------------------------------------------------------------


def compare_products(
    session: Session,
    product_ids: list[str],
    *,
    risk_free_rate: float = 0.015,
) -> ComparisonResult:
    """Compute metrics for each product and pairwise return correlation."""
    products: list[ProductMetrics] = []
    returns_map: dict[str, np.ndarray] = {}

    for pid in product_ids:
        entity = session.get(ProductEntity, pid)
        name = entity.standard_name if entity else pid
        dates, values = _load_nav_array(session, pid)

        if len(values) < 2:
            products.append(ProductMetrics(
                product_id=pid, product_name=name, observation_count=len(values),
                start_date=dates[0] if dates else "", end_date=dates[-1] if dates else "",
                cumulative_return=0, annualized_return=0, annualized_volatility=0,
                sharpe_ratio=None, maximum_drawdown=0, calmar_ratio=None,
            ))
            continue

        frequency = "monthly"  # default; could be inferred from dates
        obs = (
            session.query(NavObservation)
            .filter(
                NavObservation.product_id == pid,
                NavObservation.review_status == ReviewStatus.REVIEWED,
            )
            .first()
        )
        if obs and obs.frequency:
            frequency = obs.frequency

        try:
            nav_points = [
                NetAssetValuePoint(observation_date=d, net_asset_value=v)
                for d, v in zip(dates, values)
            ]
            result = calculate_nav_analysis(NavAnalysisRequest(
                nav_points=nav_points,
                frequency=DataFrequency(frequency),
                annual_risk_free_rate=risk_free_rate,
            ))
            m = result.metrics
            products.append(ProductMetrics(
                product_id=pid, product_name=name, observation_count=len(values),
                start_date=dates[0], end_date=dates[-1],
                cumulative_return=m.cumulative_return,
                annualized_return=m.annualized_return,
                annualized_volatility=m.annualized_volatility,
                sharpe_ratio=m.sharpe_ratio,
                maximum_drawdown=m.maximum_drawdown,
                calmar_ratio=m.calmar_ratio,
            ))
        except Exception:
            products.append(ProductMetrics(
                product_id=pid, product_name=name, observation_count=len(values),
                start_date=dates[0], end_date=dates[-1],
                cumulative_return=0, annualized_return=0, annualized_volatility=0,
                sharpe_ratio=None, maximum_drawdown=0, calmar_ratio=None,
            ))

        returns_map[pid] = _compute_returns(values)

    # Pairwise correlation (on overlapping length).
    correlation: dict[str, dict[str, float]] = {}
    for pid_a in product_ids:
        correlation[pid_a] = {}
        for pid_b in product_ids:
            if pid_a == pid_b:
                correlation[pid_a][pid_b] = 1.0
                continue
            ra, rb = returns_map.get(pid_a, np.array([])), returns_map.get(pid_b, np.array([]))
            min_len = min(len(ra), len(rb))
            if min_len < 3:
                correlation[pid_a][pid_b] = 0.0
            else:
                a, b = ra[-min_len:], rb[-min_len:]
                if np.std(a) == 0 or np.std(b) == 0:
                    correlation[pid_a][pid_b] = 0.0
                else:
                    correlation[pid_a][pid_b] = round(float(np.corrcoef(a, b)[0, 1]), 4)

    window = ""
    if products:
        starts = [p.start_date for p in products if p.start_date]
        ends = [p.end_date for p in products if p.end_date]
        if starts and ends:
            window = f"{max(starts)} ~ {min(ends)}"

    return ComparisonResult(products=products, aligned_window=window, correlation_matrix=correlation)


# ---------------------------------------------------------------------------
# 2. Rule-based screening
# ---------------------------------------------------------------------------


@dataclass
class ScreenRules:
    """Explicit screening rules. Products failing any rule are excluded."""

    min_observations: int = 8
    max_drawdown: float | None = None  # e.g. 0.15 means exclude if |MDD| > 15%
    min_sharpe: float | None = None
    min_annualized_return: float | None = None
    max_annualized_volatility: float | None = None
    confirmed_only: bool = True


def screen_products(
    session: Session,
    product_ids: list[str],
    rules: ScreenRules | None = None,
    *,
    risk_free_rate: float = 0.015,
) -> list[ScreenResult]:
    """Screen products against explicit rules, returning pass/fail with reasons."""
    rules = rules or ScreenRules()
    comparison = compare_products(session, product_ids, risk_free_rate=risk_free_rate)
    results: list[ScreenResult] = []

    for pm in comparison.products:
        entity = session.get(ProductEntity, pm.product_id)
        exclusion_reasons: list[str] = []
        warnings: list[str] = []

        # Rule: minimum observations
        if pm.observation_count < rules.min_observations:
            exclusion_reasons.append(f"净值观测仅 {pm.observation_count} 期，不足 {rules.min_observations} 期最低要求。")

        # Rule: confirmation status
        if rules.confirmed_only and entity and entity.confirmation_status != "confirmed":
            exclusion_reasons.append(f"产品确认状态为 {entity.confirmation_status}，未经人工确认。")

        # Rule: max drawdown
        if rules.max_drawdown is not None and abs(pm.maximum_drawdown) > rules.max_drawdown:
            exclusion_reasons.append(f"最大回撤 {pm.maximum_drawdown:.2%} 超过阈值 {rules.max_drawdown:.2%}。")

        # Rule: min sharpe
        if rules.min_sharpe is not None:
            if pm.sharpe_ratio is None:
                exclusion_reasons.append("夏普比率无法计算（波动率为零或数据不足）。")
            elif pm.sharpe_ratio < rules.min_sharpe:
                exclusion_reasons.append(f"夏普比率 {pm.sharpe_ratio:.2f} 低于阈值 {rules.min_sharpe:.2f}。")

        # Rule: min return
        if rules.min_annualized_return is not None and pm.annualized_return < rules.min_annualized_return:
            exclusion_reasons.append(f"年化收益 {pm.annualized_return:.2%} 低于阈值 {rules.min_annualized_return:.2%}。")

        # Rule: max volatility
        if rules.max_annualized_volatility is not None and pm.annualized_volatility > rules.max_annualized_volatility:
            exclusion_reasons.append(f"年化波动率 {pm.annualized_volatility:.2%} 超过阈值 {rules.max_annualized_volatility:.2%}。")

        # Warnings (non-excluding)
        if pm.observation_count < 24:
            warnings.append("历史期不足 24 期，指标稳定性有限。")
        if abs(pm.maximum_drawdown) > 0.20:
            warnings.append("历史回撤超过 20%，需关注流动性与止损机制。")

        results.append(ScreenResult(
            product_id=pm.product_id,
            product_name=pm.product_name,
            passed=len(exclusion_reasons) == 0,
            metrics=pm,
            exclusion_reasons=exclusion_reasons,
            warnings=warnings,
        ))

    return results


# ---------------------------------------------------------------------------
# 3. Constrained FOF allocation
# ---------------------------------------------------------------------------


def optimize_allocation(
    session: Session,
    product_ids: list[str],
    *,
    max_single_weight: float = 0.35,
    risk_free_rate: float = 0.015,
    screen_rules: ScreenRules | None = None,
) -> RecommendationResult:
    """Full recommendation pipeline: screen → score → allocate → verify."""
    rules = screen_rules or ScreenRules()
    screening = screen_products(session, product_ids, rules, risk_free_rate=risk_free_rate)

    passed = [s for s in screening if s.passed]
    excluded = [s for s in screening if not s.passed]

    # Score passed products (transparent: return + sharpe + drawdown control + depth).
    scored: list[tuple[ScreenResult, float]] = []
    for s in passed:
        if s.metrics is None:
            continue
        m = s.metrics
        ret_score = _scale(m.annualized_return, -0.20, 0.30)
        sharpe_score = _scale(m.sharpe_ratio if m.sharpe_ratio is not None else -0.5, -0.5, 2.0)
        dd_score = _scale(0.40 - abs(m.maximum_drawdown), 0.0, 0.40)
        depth_score = min(m.observation_count / 52.0, 1.0)
        total = ret_score * 25 + sharpe_score * 30 + dd_score * 30 + depth_score * 15
        scored.append((s, round(total, 2)))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Inverse-volatility allocation with score tilt.
    allocations: list[AllocationItem] = []
    if scored:
        raw_weights: dict[str, float] = {}
        for s, score in scored:
            m = s.metrics
            assert m is not None
            vol = max(m.annualized_volatility, 0.03)
            raw_weights[s.product_id] = max(score, 1.0) / vol

        total_raw = sum(raw_weights.values())
        weights = {pid: w / total_raw for pid, w in raw_weights.items()}

        # Cap and redistribute.
        weights = _cap_weights(weights, max_single_weight)

        score_map = {s.product_id: score for s, score in scored}
        name_map = {s.product_id: s.product_name for s, _ in scored}
        for pid, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            allocations.append(AllocationItem(
                product_id=pid,
                product_name=name_map[pid],
                weight=round(weight, 6),
                score=score_map[pid],
                rationale=f"综合评分 {score_map[pid]:.1f}，逆波动率配置，单产品上限 {max_single_weight:.0%}。",
            ))

    # Risk warnings and due diligence gaps.
    risk_warnings: list[str] = []
    dd_gaps: list[str] = []

    if len(allocations) < 2:
        risk_warnings.append("可推荐产品少于 2 个，无法形成有效分散化。")
    if any(abs(s.metrics.maximum_drawdown) > 0.25 for s, _ in scored if s.metrics):
        risk_warnings.append("部分产品历史回撤超过 25%，需压力测试。")

    # Check correlation concentration.
    if len(allocations) >= 2:
        pids = [a.product_id for a in allocations]
        comp = compare_products(session, pids, risk_free_rate=risk_free_rate)
        high_corr_pairs = []
        for i, pid_a in enumerate(pids):
            for pid_b in pids[i + 1:]:
                corr = comp.correlation_matrix.get(pid_a, {}).get(pid_b, 0)
                if corr > 0.7:
                    high_corr_pairs.append(f"{comp.products[i].product_name} ↔ {pid_b} (ρ={corr:.2f})")
        if high_corr_pairs:
            risk_warnings.append(f"高相关性产品对：{'; '.join(high_corr_pairs)}。分散化效果有限。")

    dd_gaps.append("未纳入申赎条款、锁定期和费率结构。")
    dd_gaps.append("未完成管理人运营尽调（合规、风控、IT）。")
    dd_gaps.append("未接入实时估值和持仓数据。")
    dd_gaps.append("推荐仅为研究参考，需投委会复核后方可执行。")

    # Expected-performance baseline (frozen for post-recommendation tracking).
    expected_metrics: dict[str, dict[str, float | None]] = {}
    for s, _score in scored:
        m = s.metrics
        if m is None:
            continue
        expected_metrics[s.product_id] = {
            "annualized_return": m.annualized_return,
            "maximum_drawdown": m.maximum_drawdown,
            "annualized_volatility": m.annualized_volatility,
            "sharpe_ratio": m.sharpe_ratio,
        }

    # Data snapshot for reproducibility.
    snapshot = {
        "product_ids": product_ids,
        "rules": {
            "min_observations": rules.min_observations,
            "max_drawdown": rules.max_drawdown,
            "min_sharpe": rules.min_sharpe,
            "confirmed_only": rules.confirmed_only,
        },
        "max_single_weight": max_single_weight,
        "risk_free_rate": risk_free_rate,
        "scores": {a.product_id: a.score for a in allocations},
        "expected_metrics": expected_metrics,
    }

    return RecommendationResult(
        allocations=allocations,
        excluded=excluded,
        total_candidates=len(product_ids),
        passed_candidates=len(passed),
        constraints={"max_single_weight": max_single_weight, "risk_free_rate": risk_free_rate},
        risk_warnings=risk_warnings,
        due_diligence_gaps=dd_gaps,
        data_snapshot=snapshot,
        expected_metrics=expected_metrics,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scale(value: float, floor: float, ceiling: float) -> float:
    return min(max((value - floor) / (ceiling - floor), 0.0), 1.0)


def _cap_weights(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Iteratively cap weights and redistribute excess."""
    uncapped = set(weights)
    while uncapped:
        over = {k: v for k, v in weights.items() if k in uncapped and v > cap}
        if not over:
            break
        for k in over:
            weights[k] = cap
            uncapped.discard(k)
        remaining = 1.0 - sum(weights[k] for k in weights if k not in uncapped)
        if not uncapped or remaining <= 0:
            break
        subtotal = sum(weights[k] for k in uncapped)
        if subtotal <= 0:
            break
        for k in list(uncapped):
            weights[k] = remaining * weights[k] / subtotal
    return weights
