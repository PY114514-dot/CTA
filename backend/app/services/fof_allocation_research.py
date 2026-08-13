"""Deterministic FOF allocation research orchestration.

This service owns screening, allocation, evidence snapshots and decision
records.  HTTP routes may format its result, but must not reimplement the
financial workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import ProductEntity
from app.services import product_store as store
from app.services import quant_screen
from app.services.fof_allocation_intent import (
    is_screening_candidate,
    parse_allocation_intent,
    strategy_matches,
)


@dataclass
class AllocationResearchResult:
    content: str
    tool_duration_ms: float | None = None
    tool_summary: str | None = None
    draft: dict[str, Any] | None = None
    citations: list[dict[str, Any]] | None = None


def build_allocation_research(
    session: Session, query: str, product_ids: list[str]
) -> AllocationResearchResult:
    """Create an auditable initial FOF research allocation from confirmed data."""
    intent = parse_allocation_intent(query)
    candidate_ids = list(dict.fromkeys(product_ids))
    if not candidate_ids:
        candidate_ids = [
            product.id
            for product in session.query(ProductEntity).all()
            if is_screening_candidate(product)
        ]

    if intent.strategy_keywords:
        before_filter = len(candidate_ids)
        candidate_ids = [
            product_id for product_id in candidate_ids
            if (product := session.get(ProductEntity, product_id)) is not None
            and strategy_matches(product.strategy, intent.strategy_keywords)
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
        confirmed_only=True,
    )
    started = perf_counter()
    result = quant_screen.optimize_allocation(
        session,
        candidate_ids,
        max_single_weight=intent.max_single_weight,
        screen_rules=rules,
    )
    duration_ms = (perf_counter() - started) * 1000

    constraints = ["最低 8 期已复核净值", f"单产品上限 {intent.max_single_weight:.0%}"]
    if intent.max_drawdown is not None:
        constraints.append(f"历史最大回撤不超过 {intent.max_drawdown:.0%}")
    if intent.strategy_keywords:
        constraints.append("策略标签匹配「商品 CTA」" if intent.strategy_keywords == ("commodity_cta",) else "策略标签匹配「CTA」")
    soft_preferences = (
        ["表达了回撤控制偏好，但未给出百分比上限，未设为硬性筛选条件"]
        if intent.drawdown_preference and intent.max_drawdown is None
        else (["优先匹配用户表达的策略方向"] if intent.strategy_keywords else [])
    )
    lines: list[str] = ["**已识别目标**：生成可复核的初始 FOF 研究配置。", f"**硬约束**：{'；'.join(constraints)}。"]
    if soft_preferences:
        lines.append(f"**软偏好**：{'；'.join(soft_preferences)}。")

    if result.allocations:
        lines.extend([f"**FOF 初始配置建议（{len(result.allocations)} 个产品）：**\n", "| 产品 | 权重 | 评分 | 依据 |", "|---|---|---|---|"])
        lines.extend(
            f"| {item.product_name} | {item.weight:.1%} | {item.score:.1f} | {item.rationale} |"
            for item in result.allocations
        )
    else:
        lines.append("无产品通过筛选，无法生成配置建议。")
    if result.excluded:
        lines.append(f"\n**剔除产品（{len(result.excluded)} 个）：**")
        lines.extend(f"  {item.product_name}：{'；'.join(item.exclusion_reasons)}" for item in result.excluded)
    if result.risk_warnings:
        lines.append("\n**风险提示：**")
        lines.extend(f"  ⚠ {item}" for item in result.risk_warnings)
    if result.due_diligence_gaps:
        lines.append("\n**待补尽调项：**")
        lines.extend(f"  □ {item}" for item in result.due_diligence_gaps)

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
                "strategy_keywords": list(intent.strategy_keywords),
                "drawdown_preference": intent.drawdown_preference,
                "max_single_weight": intent.max_single_weight,
            },
        },
    )
    store.create_decision(
        session,
        decision_type="recommendation",
        title="FOF 初始配置研究建议",
        content={
            "allocations": [{"product_id": item.product_id, "name": item.product_name, "weight": item.weight, "score": item.score} for item in result.allocations],
            "excluded_count": len(result.excluded),
            "constraints": result.constraints,
            "expected_metrics": result.expected_metrics,
            "decision_date": date.today().isoformat(),
            "risk_free_rate": result.constraints.get("risk_free_rate", 0.015),
            "method_provenance": {
                "screening": {"method": "本地确定性统计筛选", "detail": "仅使用已确认、净值已复核且满足最少样本数的产品；不由 LLM 自主筛选。"},
                "allocation": {"method": "规则状态机 + 本地统计工具", "detail": "按风险约束、单产品上限和评分确定初始权重；结果需投委会人工审核。"},
            },
        },
        data_snapshot_id=snapshot.id,
        status="pending_review",
    )
    lines.append("\n*已创建决策记录（待投委会审核）并绑定数据快照。*")
    return AllocationResearchResult(
        content="\n".join(lines),
        tool_duration_ms=duration_ms,
        tool_summary=f"{result.passed_candidates}/{result.total_candidates} 通过筛选，{len(result.allocations)} 个配置；单产品上限 {intent.max_single_weight:.0%}",
        draft={
            "goal": "生成可复核的初始 FOF 研究配置",
            "hard_constraints": constraints,
            "soft_preferences": soft_preferences,
            "allocations": [{"product_id": item.product_id, "product_name": item.product_name, "weight": item.weight, "score": item.score, "rationale": item.rationale} for item in result.allocations],
            "exclusions": [{"product_id": item.product_id, "product_name": item.product_name, "reasons": item.exclusion_reasons} for item in result.excluded],
            "risk_warnings": result.risk_warnings,
            "due_diligence_gaps": result.due_diligence_gaps,
            "snapshot_id": snapshot.id,
        },
        citations=citations,
    )
