"""Choose the attribution contract before any factor calculation."""

from __future__ import annotations

from typing import Any

from app.services.multi_asset_factor_contracts import CONTRACTS, contract_metadata


_CTA_TOKENS = ("cta", "商品", "期货", "趋势", "套利", "跨期", "管理期货")
_EQUITY_TOKENS = ("股票", "股指", "沪深", "中证", "权益", "指数增强", "量化选股")
_OPTION_TOKENS = ("期权", "波动率", "volatility", " vix", "vix")
CONFIRMED_CTA_STRATEGIES = frozenset({
    "commodity_cta", "trend_cta", "arbitrage_cta", "mixed_cta",
    "commodity_arbitrage", "commodity_spread_arbitrage",
})


def _contains(value: str, tokens: tuple[str, ...]) -> bool:
    return any(token in value.lower() for token in tokens)


def is_confirmed_cta_strategy(strategy: str | None) -> bool:
    """Return whether a product may receive a formal CTA-specific result."""
    normalized = (strategy or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in CONFIRMED_CTA_STRATEGIES


def select_attribution_contract(product: Any, reviewed_nav_count: int, *, as_of_date=None) -> dict[str, Any]:
    """Return the only model contract currently allowed for a product.

    A strategy field is explicit; the name is only a discovery signal. A
    name-labelled CTA can be observed but cannot be presented as confirmed.
    """
    strategy = (getattr(product, "strategy", None) or "").strip()
    name = (getattr(product, "standard_name", None) or "").strip()
    strategy_value = strategy.lower().replace("-", "_").replace(" ", "_")
    combined = f"{name} {strategy}"
    coverage = None
    confirmed_cta = is_confirmed_cta_strategy(strategy)

    if confirmed_cta:
        pool, source, status = "commodity_cta", "confirmed_strategy", "applicable"
        reason = "已确认策略属于商品 CTA，采用商品 CTA 因子合同。"
    elif _contains(combined, _OPTION_TOKENS):
        pool, source, status = "options_volatility", "initial_name_label", "not_covered"
        coverage = contract_metadata(pool, as_of_date=as_of_date)
        reason = "初步标签为期权/波动率；当前缺少完整、带可得日期的专属因子数据，不能使用商品 CTA 因子。"
    elif _contains(combined, _EQUITY_TOKENS):
        pool, source, status = "equity_quant", "initial_name_label", "not_covered"
        coverage = contract_metadata(pool, as_of_date=as_of_date)
        reason = "初步标签为股票/股指；当前缺少完整、带可得日期的专属因子数据，不能使用商品 CTA 因子。"
    elif _contains(combined, _CTA_TOKENS):
        pool, source, status = "commodity_cta", "initial_name_label", "observe_only"
        reason = "名称初步识别为 CTA；可作观察性归因，请补充已确认策略后再形成正式结论。"
    else:
        pool, source, status = "mixed_or_unconfirmed", "insufficient_evidence", "not_covered"
        reason = "尚无法确认产品池；仅保留业绩与风险分析，不运行专属因子归因。"

    if reviewed_nav_count < 20 and status in {"applicable", "observe_only"}:
        status, reason = "insufficient_data", "已审核净值少于 20 条，暂不运行因子归因。"

    factor_contract = "commodity_cta_v1" if pool == "commodity_cta" else CONTRACTS[pool]["version"] if pool in CONTRACTS else None
    if pool in {"equity_quant", "options_volatility"} and coverage["status"] == "available":
        status = "ready_for_baseline"
        reason = f"{coverage['label']}已覆盖；可进入基础归因。"
    return {
        "product_pool": pool, "classification_source": source, "status": status,
        "factor_contract": factor_contract,
        "factor_coverage": coverage,
        "reason": reason,
        "next_step": (
            "补充已确认策略" if source == "initial_name_label" and pool == "commodity_cta"
            else "等待该产品池的专属因子合同接入" if status == "not_covered"
            else "补充策略或披露信息" if pool == "mixed_or_unconfirmed"
            else "补充并审核更多净值"
        ),
    }
