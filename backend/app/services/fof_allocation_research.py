"""Deterministic FOF allocation research orchestration.

This service owns screening, allocation, evidence snapshots and decision
records.  HTTP routes may format its result, but must not reimplement the
financial workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from time import perf_counter
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models import NavObservation, ProductEntity, ReviewStatus
from app.services import product_store as store
from app.services import quant_screen
from app.services.nav_metrics import calculate_nav_analysis
from app.services.fof_allocation_intent import (
    AllocationIntent,
    is_demo_product,
    is_screening_candidate,
    parse_allocation_intent,
    strategy_matches,
)
from app.schemas import DataFrequency, NavAnalysisRequest, NetAssetValuePoint


@dataclass
class AllocationResearchResult:
    content: str
    tool_duration_ms: float | None = None
    tool_summary: str | None = None
    draft: dict[str, Any] | None = None
    citations: list[dict[str, Any]] | None = None


def _portfolio_performance(session: Session, allocations: list[quant_screen.AllocationItem]) -> dict[str, Any] | None:
    """Calculate the realized historical NAV path of the proposed weights."""
    if not allocations:
        return None
    raw_series = store.get_nav_series_bulk(session, [item.product_id for item in allocations], reviewed_only=True)
    observations = [point for points in raw_series.values() for point in points]
    if not observations:
        return None
    frequency = "monthly" if any(point.frequency == "monthly" for point in observations) else "weekly"

    def period_end(observation_date: date) -> str:
        if frequency == "monthly":
            return observation_date.replace(day=1).isoformat()
        return (observation_date + timedelta(days=4 - observation_date.weekday())).isoformat()

    # 周频净值的披露日常不一致；按自然周取最后一个净值对齐，避免把周一和周五的
    # 同一周误判为没有共同样本。
    series = {
        product_id: {period_end(point.observation_date): point.nav for point in points}
        for product_id, points in raw_series.items()
    }
    common_dates = sorted(set.intersection(*(set(points) for points in series.values()))) if series else []
    if len(common_dates) < 2:
        return None
    weights = {item.product_id: item.weight for item in allocations}
    nav_values = [1.0]
    for previous, current in zip(common_dates, common_dates[1:]):
        period_return = sum(weights[product_id] * (points[current] / points[previous] - 1.0) for product_id, points in series.items())
        nav_values.append(nav_values[-1] * (1.0 + period_return))
    analysis = calculate_nav_analysis(NavAnalysisRequest(
        nav_points=[NetAssetValuePoint(observation_date=day, net_asset_value=value) for day, value in zip(common_dates, nav_values)],
        frequency=DataFrequency(frequency),
    ))
    return {
        "nav": [{"date": day, "nav": value} for day, value in zip(common_dates, nav_values)],
        "metrics": analysis.metrics.model_dump(),
    }


def build_allocation_research(
    session: Session,
    query: str,
    intent: AllocationIntent | None = None,
) -> AllocationResearchResult:
    """Create an auditable initial FOF research allocation from the reviewed product library."""
    intent = intent or parse_allocation_intent(query)
    candidate_products = list(session.execute(
        select(ProductEntity)
        .join(NavObservation, NavObservation.product_id == ProductEntity.id)
        .where(ProductEntity.confirmation_status == "confirmed")
        .group_by(ProductEntity.id)
        .having(func.count(NavObservation.id) >= 8)
        .having(func.sum(case((NavObservation.review_status == ReviewStatus.REVIEWED, 1), else_=0)) == func.count(NavObservation.id))
        .having(func.sum(case((NavObservation.source_file_id.is_not(None), 1), else_=0)) == func.count(NavObservation.id))
    ).scalars())
    candidate_ids = [product.id for product in candidate_products if not is_demo_product(product)]

    if intent.strategy_keywords:
        before_filter = len(candidate_ids)
        candidate_ids = [
            product.id for product in candidate_products
            if product.id in candidate_ids and strategy_matches(product.strategy, intent.strategy_keywords)
        ]
        if not candidate_ids:
            label = "商品 CTA" if intent.strategy_keywords == ("commodity_cta",) else "CTA"
            return AllocationResearchResult(
                content=(
                    f"当前没有已确认、净值已复核且策略标注为「{label}」的候选产品。"
                    f"已检查 {before_filter} 个可研究产品；请先在产品信息中补充策略标签，或选择候选产品后重试。"
                )
            )

    if not candidate_ids:
        return AllocationResearchResult(
            content="当前没有可用于配置的已确认产品。请先完成产品身份与净值复核；Agent 不会把待确认或示例数据纳入配置。"
        )

    rules = quant_screen.ScreenRules(
        min_observations=8,
        max_drawdown=intent.max_drawdown,
        min_annualized_return=intent.min_annualized_return,
        max_annualized_volatility=intent.max_annualized_volatility,
        confirmed_only=True,
    )
    started = perf_counter()
    max_products = intent.max_products or 5
    result = quant_screen.optimize_allocation(
        session,
        candidate_ids,
        max_single_weight=intent.max_single_weight,
        max_products=max_products,
        screen_rules=rules,
    )
    duration_ms = (perf_counter() - started) * 1000
    portfolio = _portfolio_performance(session, result.allocations)

    constraints = ["最低 8 期已复核净值", f"单产品上限 {intent.max_single_weight:.0%}"]
    if intent.max_drawdown is not None:
        constraints.append(f"历史最大回撤不超过 {intent.max_drawdown:.0%}")
    if intent.strategy_keywords:
        constraints.append("策略标签匹配「商品 CTA」" if intent.strategy_keywords == ("commodity_cta",) else "策略标签匹配「CTA」")
    if intent.min_annualized_return is not None:
        constraints.append(f"年化收益不低于 {intent.min_annualized_return:.0%}")
    if intent.max_annualized_volatility is not None:
        constraints.append(f"年化波动不高于 {intent.max_annualized_volatility:.0%}")
    constraints.append(
        f"组合产品数不超过 {max_products} 只"
        + ("（系统默认）" if intent.max_products is None else "")
    )
    soft_preferences = (
        ["表达了回撤控制偏好，但未给出百分比上限，未设为硬性筛选条件"]
        if intent.drawdown_preference and intent.max_drawdown is None
        else (["优先匹配用户表达的策略方向"] if intent.strategy_keywords else [])
    )
    lines: list[str] = [f"已从产品库 {len(candidate_ids)} 个已复核产品中完成初始 FOF 计算。"]

    if result.allocations:
        lines.extend([f"**FOF 初始配置建议（{len(result.allocations)} 个产品）：**\n", "| 产品 | 权重 | 评分 | 依据 |", "|---|---|---|---|"])
        lines.extend(
            f"| {item.product_name} | {item.weight:.1%} | {item.score:.1f} | {item.rationale} |"
            for item in result.allocations
        )
        if len(result.allocations) > max_products:
            raise RuntimeError("配置结果超过产品数上限，已阻止返回该结果。")
    else:
        lines.append("无产品通过筛选，无法生成配置建议。")
    if result.risk_warnings:
        lines.append("\n**风险提示：**")
        lines.extend(f"  ⚠ {item}" for item in result.risk_warnings)
    if portfolio:
        metrics = portfolio["metrics"]
        sharpe = metrics["sharpe_ratio"]
        sharpe_text = f"{sharpe:.2f}" if sharpe is not None else "—"
        lines.append(
            f"\n组合历史表现：累计收益 {metrics['cumulative_return']:.2%} · 年化收益 {metrics['annualized_return']:.2%} · "
            f"年化波动 {metrics['annualized_volatility']:.2%} · 夏普 {sharpe_text} · 最大回撤 {metrics['maximum_drawdown']:.2%}"
        )

    evidence_snapshot, citations = store.build_research_evidence_snapshot(session, candidate_ids)
    snapshot = store.create_snapshot(
        session,
        label="FOF 配置研究快照",
        content={
            **result.data_snapshot,
            "user_query": query,
            "evidence": evidence_snapshot,
            "interpreted_constraints": {
                "max_drawdown": intent.max_drawdown,
                "min_annualized_return": intent.min_annualized_return,
                "max_annualized_volatility": intent.max_annualized_volatility,
                "max_products": max_products,
                "max_products_source": "user" if intent.max_products is not None else "system_default",
                "strategy_keywords": list(intent.strategy_keywords),
                "drawdown_preference": intent.drawdown_preference,
                "max_single_weight": intent.max_single_weight,
            },
            "draft": {
                "allocations": [{"product_id": item.product_id, "product_name": item.product_name, "weight": item.weight, "score": item.score, "rationale": item.rationale} for item in result.allocations],
                "constraints": constraints,
                "portfolio": portfolio,
                "risk_warnings": result.risk_warnings,
            },
        },
    )
    return AllocationResearchResult(
        content="\n".join(lines),
        tool_duration_ms=duration_ms,
        tool_summary=f"{result.passed_candidates}/{result.total_candidates} 通过筛选，{len(result.allocations)} 个配置；单产品上限 {intent.max_single_weight:.0%}",
        draft={
            "goal": "生成可复核的初始 FOF 研究配置",
            "draft_id": snapshot.id,
            "hard_constraints": constraints,
            "soft_preferences": soft_preferences,
            "allocations": [{"product_id": item.product_id, "product_name": item.product_name, "weight": item.weight, "score": item.score, "rationale": item.rationale} for item in result.allocations],
            "exclusions": [],
            "risk_warnings": result.risk_warnings,
            "portfolio": portfolio,
        },
        citations=citations,
    )
