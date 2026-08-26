"""Deterministic FOF allocation intent and candidate eligibility rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import ProductEntity


_DEMO_PRODUCT_NAMES = {
    "cta alpha", "cta beta", "cta gamma", "cta steady no.1", "cta steady no 1", "cta growth",
}


@dataclass(frozen=True)
class AllocationIntent:
    """User-stated constraints translated into explicit, reviewable rules."""

    max_drawdown: float | None = None
    strategy_keywords: tuple[str, ...] = ()
    drawdown_preference: bool = False
    max_single_weight: float = 0.35
    max_products: int | None = None
    min_annualized_return: float | None = None
    max_annualized_volatility: float | None = None


def is_demo_product(product: ProductEntity) -> bool:
    """Exclude historical fixtures from user-facing research and allocation."""
    return (
        product.standard_name.strip().lower() in _DEMO_PRODUCT_NAMES
        or (product.manager_name or "").strip().lower() in {"alphacap", "betafund", "gammaam", "testmanager"}
    )


def is_screening_candidate(product: ProductEntity) -> bool:
    """Allow only confirmed, human-reviewed, source-linked products."""
    observations = product.nav_observations
    return (
        product.confirmation_status == "confirmed"
        and not is_demo_product(product)
        and len(observations) >= 2
        and all(observation.source_file_id is not None for observation in observations)
        and all(observation.review_status == "reviewed" for observation in observations)
    )


def parse_allocation_intent(query: str, interpreted: dict[str, object] | None = None) -> AllocationIntent:
    """Extract only unambiguous allocation constraints from plain Chinese."""
    normalized = query.lower()

    def chinese_percent(match: re.Match[str]) -> str:
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        token = match.group(1)
        if "十" in token:
            before, after = token.split("十", 1)
            value = (digits[before] if before else 1) * 10 + (digits[after] if after else 0)
        else:
            value = digits.get(token, 0)
        return f"{value}%"

    normalized = re.sub(r"百分之([零一二两三四五六七八九十]+)", chinese_percent, normalized)
    max_drawdown: float | None = None
    drawdown_preference = "回撤" in normalized or "drawdown" in normalized
    match = re.search(r"(?:回撤|drawdown)[^0-9]{0,16}(\d{1,2}(?:\.\d+)?)\s*%", normalized)
    if match:
        candidate = float(match.group(1)) / 100
        if 0 < candidate <= 0.80:
            max_drawdown = candidate

    strategy_keywords: tuple[str, ...] = ()
    if "商品" in normalized and "cta" in normalized:
        strategy_keywords = ("commodity_cta",)
    elif "cta" in normalized:
        strategy_keywords = ("cta",)

    interpreted = interpreted or {}

    def percentage(name: str) -> float | None:
        value = interpreted.get(name)
        if isinstance(value, (int, float)) and 0 < float(value) <= 1:
            return float(value)
        return None

    def positive_count(name: str) -> int | None:
        value = interpreted.get(name)
        if isinstance(value, int) and 1 <= value <= 30:
            return value
        return None

    # The optional semantic interpretation is authoritative when available.
    # The local parser is only a safe fallback for simple explicit percentages.
    max_drawdown = percentage("max_drawdown") or max_drawdown
    min_annualized_return = percentage("min_annualized_return")
    max_annualized_volatility = percentage("max_annualized_volatility")
    max_products = positive_count("max_products")
    if min_annualized_return is None:
        match = re.search(r"(?:年化(?:收益)?|收益)[^0-9]{0,16}(\d{1,2}(?:\.\d+)?)\s*%", normalized)
        if match:
            min_annualized_return = float(match.group(1)) / 100
    if max_annualized_volatility is None:
        match = re.search(r"(?:年化)?波动[^0-9]{0,16}(\d{1,2}(?:\.\d+)?)\s*%", normalized)
        if match:
            max_annualized_volatility = float(match.group(1)) / 100
    if max_products is None:
        match = re.search(r"(?:最多|不超过|至多|最好不超过|尽量不超过)\s*(\d+)\s*(?:只|个|款)?", normalized)
        if match and 1 <= int(match.group(1)) <= 30:
            max_products = int(match.group(1))
    max_single_weight = 0.30 if max_drawdown is not None and max_drawdown <= 0.10 else 0.35
    return AllocationIntent(
        max_drawdown=max_drawdown,
        strategy_keywords=strategy_keywords,
        drawdown_preference=drawdown_preference,
        max_single_weight=max_single_weight,
        max_products=max_products,
        min_annualized_return=min_annualized_return,
        max_annualized_volatility=max_annualized_volatility,
    )


def strategy_matches(product_strategy: str | None, intent_keywords: tuple[str, ...]) -> bool:
    """Match user-facing strategy language to persisted strategy identifiers."""
    if not intent_keywords:
        return True
    normalized = (product_strategy or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return False

    aliases: dict[str, tuple[str, ...]] = {
        "commodity_cta": ("commodity_cta", "商品cta", "商品_cta", "商品期货cta"),
        "cta": ("cta", "commodity_cta", "trend_cta", "arbitrage_cta", "mixed_cta", "商品cta", "趋势cta"),
    }
    return all(any(candidate in normalized for candidate in aliases.get(keyword, (keyword,))) for keyword in intent_keywords)
