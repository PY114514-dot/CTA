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


def parse_allocation_intent(query: str) -> AllocationIntent:
    """Extract only unambiguous allocation constraints from plain Chinese."""
    normalized = query.lower()
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

    max_single_weight = 0.30 if max_drawdown is not None and max_drawdown <= 0.10 else 0.35
    return AllocationIntent(
        max_drawdown=max_drawdown,
        strategy_keywords=strategy_keywords,
        drawdown_preference=drawdown_preference,
        max_single_weight=max_single_weight,
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
