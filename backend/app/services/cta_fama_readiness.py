"""Readiness audit for the post-production CTA-Fama research design."""

from __future__ import annotations

from typing import Any

from app.schemas import CtaFamaReadinessRequest


MIN_CROSS_SECTION = 30
MIN_HISTORY_BY_FREQUENCY = {"daily": 252, "weekly": 52, "monthly": 36}
MIN_OOS_STATE_SEGMENTS = 3


def _gate(status: str, observed: Any, required: Any, reason: str) -> dict[str, Any]:
    return {"status": status, "observed": observed, "required": required, "reason": reason}


def evaluate_cta_fama_readiness(request: CtaFamaReadinessRequest) -> dict[str, Any]:
    frequencies = {str(product.frequency) for product in request.products}
    common_frequency = next(iter(frequencies)) if len(frequencies) == 1 else None
    history_requirement = MIN_HISTORY_BY_FREQUENCY.get(common_frequency or "weekly", 52)
    long_history_count = sum(len(product.nav_points) - 1 >= history_requirement for product in request.products) if common_frequency else 0
    exit_count = sum(product.exit_date is not None for product in request.products)
    point_in_time_count = sum(product.point_in_time_verified for product in request.products)
    source_group_counts: dict[str, int] = {}
    for product in request.products:
        source_group_counts[product.source_group] = source_group_counts.get(product.source_group, 0) + 1
    duplicate_source_groups = sorted(group for group, count in source_group_counts.items() if count > 1)

    gates = {
        "cross_section": _gate(
            "passed" if len(request.products) >= MIN_CROSS_SECTION and common_frequency else "failed",
            {"product_count": len(request.products), "frequency": common_frequency},
            {"minimum_products": MIN_CROSS_SECTION, "one_common_frequency": True},
            "横截面数量和频率达到研究门槛。" if len(request.products) >= MIN_CROSS_SECTION and common_frequency else "产品横截面不足或频率不统一。",
        ),
        "history": _gate(
            "passed" if common_frequency and long_history_count >= MIN_CROSS_SECTION else "failed",
            {"long_history_products": long_history_count, "history_requirement": history_requirement},
            {"minimum_long_history_products": MIN_CROSS_SECTION},
            "历史长度达到同频率研究门槛。" if common_frequency and long_history_count >= MIN_CROSS_SECTION else "可用长期历史产品数不足。",
        ),
        "exit_history": _gate(
            "passed" if request.exit_history_complete and exit_count > 0 else "failed",
            {"exit_products": exit_count, "audit_declared": request.exit_history_complete},
            {"requires_exit_product": True, "requires_complete_audit": True},
            "退出/清盘产品已纳入并完成审计。" if request.exit_history_complete and exit_count > 0 else "未证明包含退出/清盘产品，不能排除幸存者偏差。",
        ),
        "point_in_time": _gate(
            "passed" if point_in_time_count == len(request.products) else "failed",
            {"verified_products": point_in_time_count},
            {"all_products_verified": True},
            "产品特征、净值和状态均有 point-in-time 证明。" if point_in_time_count == len(request.products) else "仍有产品缺少 point-in-time 可得性证明。",
        ),
        "bias_controls": _gate(
            "passed" if request.survivorship_audit_passed and request.backfill_audit_passed and request.same_source_deduplicated else "failed",
            {
                "survivorship_audit_passed": request.survivorship_audit_passed,
                "backfill_audit_passed": request.backfill_audit_passed,
                "same_source_deduplicated": request.same_source_deduplicated,
                "duplicate_source_groups": duplicate_source_groups,
            },
            {"survivorship": True, "backfill": True, "same_source_deduplicated": True},
            "幸存者、回填和同源重复审计均通过。" if request.survivorship_audit_passed and request.backfill_audit_passed and request.same_source_deduplicated else "偏差审计尚未全部通过。",
        ),
        "oos_validation": _gate(
            "passed" if request.oos_state_segments >= MIN_OOS_STATE_SEGMENTS else "failed",
            {"state_segments": request.oos_state_segments},
            {"minimum_state_segments": MIN_OOS_STATE_SEGMENTS},
            "多个市场状态的滚动样本外段可执行。" if request.oos_state_segments >= MIN_OOS_STATE_SEGMENTS else "多个市场状态的滚动样本外验证不足。",
        ),
    }
    passed = all(item["status"] == "passed" for item in gates.values())
    warnings = [
        "CTA-Fama 研究门只判断数据是否适合开展实验，不生成生产因子或产品评分。",
    ]
    if not passed:
        warnings.extend(item["reason"] for item in gates.values() if item["status"] != "passed")
    return {
        "status": "ready_for_research" if passed else "blocked",
        "production_ready": False,
        "policy": "research_design_only",
        "common_frequency": common_frequency,
        "gates": gates,
        "warnings": warnings,
    }
