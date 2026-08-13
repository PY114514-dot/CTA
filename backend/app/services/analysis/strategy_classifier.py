"""Step 1: Strategy-type classification.

Determines whether a product's NAV dynamics are more consistent with a
commodity CTA or an equity quantitative strategy by comparing its return
series against major benchmark indices.

Methods:
1. Full-sample Pearson & Spearman correlation with equity/commodity indices.
2. Rolling-window correlation stability (does the dominant correlation persist?).
3. Up/down market segment beta asymmetry.
4. Composite scoring with conservative thresholds.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.services.market_data.provider import MarketDataProvider
from app.services.market_data.index_registry import INDEX_BENCHMARKS
from app.services.analysis._stats import safe_pearson
from app.services.analysis._pipeline_utils import align_product_and_market_returns

logger = logging.getLogger(__name__)

# Indices used for classification, grouped by asset class
_EQUITY_INDICES = ["hs300", "zz500", "zz1000"]
_COMMODITY_INDICES = ["nh_commodity", "nh_industrial", "nh_agriculture", "nh_metal", "nh_energy"]

# Thresholds
_MIN_OBSERVATIONS = 8  # fewer than this → insufficient_data
_HIGH_CONF_THRESHOLD = 0.5
_MEDIUM_CONF_THRESHOLD = 0.3


@dataclass
class ClassificationEvidence:
    """One piece of evidence supporting the classification."""

    description: str
    value: float | None = None


@dataclass
class StrategyClassification:
    """Result of strategy-type classification."""

    strategy_type: str  # commodity CTA | equity/index direction (CTA or quant) | mixed | insufficient_data
    confidence_pct: float  # 0-100
    confidence_label: str  # "高" | "中" | "低"
    evidence: list[str] = field(default_factory=list)
    correlations: dict[str, float] = field(default_factory=dict)
    details: dict = field(default_factory=dict)


def classify_strategy(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    provider: MarketDataProvider,
    strategy_hint: str = "",
    strategy_confirmation: str = "auto",
) -> StrategyClassification:
    """Classify a product's strategy type based on its return series.

    Args:
        product_returns: Array of periodic returns (not NAV levels).
        product_dates: Corresponding observation dates.
        frequency: "daily" | "weekly" | "monthly".
        provider: Market data provider for fetching benchmarks.
        strategy_hint: Optional OCR/VLM or user-supplied disclosure text. It
            is recorded as a candidate and never overrides NAV statistics.
        strategy_confirmation: Optional explicit user confirmation. When set
            to a concrete scope, it is recorded separately and may override
            the automatic routing decision while preserving statistical evidence.

    Returns:
        StrategyClassification with type, confidence, and evidence.
    """
    disclosure = _classify_disclosure_hint(strategy_hint)
    n = len(product_returns)
    if n < _MIN_OBSERVATIONS:
        result = StrategyClassification(
            strategy_type="insufficient_data",
            confidence_pct=0.0,
            confidence_label="低",
            evidence=[f"样本量仅 {n} 期，不足 {_MIN_OBSERVATIONS} 期最低要求，无法进行有效分类"],
        )
        return _finalize_classification(result, disclosure, strategy_confirmation)

    # Determine date range for fetching benchmarks (with buffer)
    start = min(product_dates)
    end = max(product_dates)

    # Fetch benchmark returns aligned to product frequency
    equity_corrs = _fetch_and_correlate(
        product_returns, product_dates, _EQUITY_INDICES, start, end, frequency, provider, is_index=True
    )
    commodity_corrs = _fetch_and_correlate(
        product_returns, product_dates, _COMMODITY_INDICES, start, end, frequency, provider, is_index=True
    )

    all_corrs = {**equity_corrs, **commodity_corrs}

    if not equity_corrs and not commodity_corrs:
        result = StrategyClassification(
            strategy_type="insufficient_data",
            confidence_pct=0.0,
            confidence_label="低",
            evidence=["无法获取任何基准指数数据，请检查网络连接或上传 CSV 行情数据"],
            correlations=all_corrs,
        )
        return _finalize_classification(result, disclosure, strategy_confirmation)

    # Compute group-level scores
    equity_score = _group_score(equity_corrs)
    commodity_score = _group_score(commodity_corrs)

    evidence: list[str] = []
    details: dict = {}

    # Evidence: top correlations
    for symbol, corr in sorted(all_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        name = INDEX_BENCHMARKS.get(symbol, None)
        display_name = name.name if name else symbol
        evidence.append(f"与{display_name}相关系数 {corr:.3f}")

    # Rolling stability check
    rolling_stability = _rolling_correlation_stability(
        product_returns, product_dates, start, end, frequency, provider
    )
    details["rolling_stability"] = rolling_stability
    if rolling_stability:
        best_group = rolling_stability.get("dominant_group", "")
        stability_pct = rolling_stability.get("stability_pct", 0)
        evidence.append(f"滚动窗口中 {stability_pct:.0f}% 的时段{best_group}相关性占优")

    # Up/down segment analysis
    segment_analysis = _segment_beta_analysis(
        product_returns, product_dates, start, end, frequency, provider
    )
    details["segment_analysis"] = segment_analysis
    if segment_analysis:
        up_beta = segment_analysis.get("equity_up_beta", 0)
        down_beta = segment_analysis.get("equity_down_beta", 0)
        if abs(up_beta) > 0.3 or abs(down_beta) > 0.3:
            evidence.append(f"股票市场上涨段 beta={up_beta:.2f}，下跌段 beta={down_beta:.2f}")

    # Final classification
    score_diff = commodity_score - equity_score
    max_score = max(abs(commodity_score), abs(equity_score))

    if max_score < _MEDIUM_CONF_THRESHOLD:
        strategy_type = "insufficient_data"
        confidence_pct = max_score * 100
        confidence_label = "低"
        evidence.append(f"最高相关性仅 {max_score:.3f}，低于有效分类阈值 {_MEDIUM_CONF_THRESHOLD}")
    elif abs(score_diff) < 0.15 and max_score >= _MEDIUM_CONF_THRESHOLD:
        strategy_type = "mixed"
        confidence_pct = max_score * 100
        confidence_label = "中" if max_score >= _HIGH_CONF_THRESHOLD else "低"
        evidence.append(f"商品得分 {commodity_score:.3f} 与股票得分 {equity_score:.3f} 接近，判定为混合型")
    elif commodity_score > equity_score:
        strategy_type = "commodity_cta"
        confidence_pct = min(commodity_score * 100 + abs(score_diff) * 30, 100)
        confidence_label = _confidence_label(confidence_pct)
        evidence.append(f"商品类相关性（{commodity_score:.3f}）显著高于股票类（{equity_score:.3f}）")
    else:
        # Correlation can identify an equity/index return direction, but it
        # cannot distinguish a stock-quant product from an equity-index CTA.
        # Keep the stable API value for compatibility and expose the ambiguity
        # in the evidence/report instead of claiming "股票量化".
        strategy_type = "equity_quant"
        confidence_pct = min(equity_score * 100 + abs(score_diff) * 30, 100)
        confidence_label = _confidence_label(confidence_pct)
        evidence.append(f"股票/股指类相关性（{equity_score:.3f}）显著高于商品类（{commodity_score:.3f}）；仅凭净值无法区分股指 CTA 与股票量化")

    result = StrategyClassification(
        strategy_type=strategy_type,
        confidence_pct=round(confidence_pct, 1),
        confidence_label=confidence_label,
        evidence=evidence,
        correlations=all_corrs,
        details=details,
    )
    return _finalize_classification(result, disclosure, strategy_confirmation)


def _classify_disclosure_hint(text: str) -> dict:
    """Extract a conservative strategy candidate from OCR/user disclosure.

    The text is deliberately treated as a management disclosure, not as a
    substitute for NAV statistics.  A bare ``CTA`` mention is kept as a
    generic candidate because it cannot identify the asset class.
    """
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized:
        return {
            "detected_type": None,
            "label": "",
            "evidence": [],
            "explicit_cta": False,
            "source": "",
        }

    has_cta = bool(re.search(r"CTA|cta", normalized))
    has_equity = bool(re.search(r"股指(?:期货|CTA|策略)?|指数期货|股票指数|沪深300|中证500|中证1000", normalized))
    has_commodity = bool(re.search(r"商品(?:期货|CTA|策略)?|商品趋势|期货CTA|期货趋势", normalized))
    has_mixed = bool(
        re.search(r"多资产|混合策略|跨资产|商品.{0,18}(?:股指|国债|股票)|(?:股指|国债|股票).{0,18}商品", normalized)
    )

    if has_mixed or (has_equity and has_commodity):
        detected_type = "mixed"
        label = "多资产/混合 CTA"
        evidence = ["披露文本同时出现商品与股指/股票/国债等资产类别"]
    elif has_commodity:
        detected_type = "commodity_cta"
        label = "商品 CTA"
        evidence = ["披露文本提示商品 CTA 或商品期货策略"]
    elif has_equity:
        detected_type = "equity_cta"
        label = "股指 CTA"
        evidence = ["披露文本提示股指 CTA、股指期货或指数期货策略"]
    elif has_cta:
        detected_type = None
        label = "CTA（资产类别未披露）"
        evidence = ["披露文本提到 CTA，但未明确商品、股指或混合资产类别"]
    else:
        detected_type = None
        label = ""
        evidence = []

    return {
        "detected_type": detected_type,
        "label": label,
        "evidence": evidence,
        "explicit_cta": has_cta,
        "source": text[:5000],
    }


def _attach_disclosure_evidence(result: StrategyClassification, disclosure: dict) -> StrategyClassification:
    """Attach disclosure-vs-statistics comparison without overriding stats."""
    if not disclosure.get("label"):
        return result

    disclosed_type = disclosure.get("detected_type")
    statistical_type = result.strategy_type
    equity_equivalent = statistical_type == "equity_quant" and disclosed_type == "equity_cta"
    aligned = bool(disclosed_type and (statistical_type == disclosed_type or equity_equivalent))
    # A generic CTA mention has no asset-class assertion, so it cannot conflict.
    conflict = bool(disclosed_type and statistical_type not in ("insufficient_data",) and not aligned)
    status = "aligned" if aligned else "conflict" if conflict else "disclosure_only"

    result.details["disclosure_hint"] = {
        "detected_type": disclosed_type,
        "label": disclosure.get("label", ""),
        "evidence": disclosure.get("evidence", []),
        "explicit_cta": bool(disclosure.get("explicit_cta")),
        "status": status,
        "conflict": conflict,
        "statistical_type": statistical_type,
    }
    result.evidence.extend(
        [
            f"管理人披露候选：{disclosure['label']}",
            "披露文本仅作为候选假设，未视为实际持仓或统计确认",
        ]
    )
    if conflict:
        result.evidence.append(
            f"披露候选与净值统计分类不一致（统计：{statistical_type}），已避免强行套用单一资产因子"
        )
    elif statistical_type == "insufficient_data":
        result.evidence.append("净值统计证据不足，暂不依据披露文本自动运行单一资产因子模型")
    else:
        result.evidence.append("披露候选与统计方向一致，但仍不等同于实际持仓确认")
    return result


def _finalize_classification(
    result: StrategyClassification,
    disclosure: dict,
    strategy_confirmation: str,
) -> StrategyClassification:
    """Attach disclosure evidence and apply an explicit user scope, if any."""
    result = _attach_disclosure_evidence(result, disclosure)
    requested = (strategy_confirmation or "auto").strip().lower()
    if requested in ("", "auto"):
        return result
    if requested not in {"commodity_cta", "equity_cta", "mixed"}:
        result.evidence.append(f"忽略未知的策略确认值：{requested}")
        return result

    statistical_type = result.strategy_type
    model_type = "equity_quant" if requested == "equity_cta" else requested
    statistical_equivalent = (
        statistical_type == model_type
        or (statistical_type == "equity_quant" and model_type == "equity_quant")
    )
    statistical_conflict = statistical_type not in {"insufficient_data"} and not statistical_equivalent
    disclosed_type = disclosure.get("detected_type")
    disclosure_equivalent = (
        disclosed_type is None
        or disclosed_type == requested
        or (disclosed_type == "equity_cta" and requested == "equity_cta")
    )
    disclosure_conflict = bool(disclosed_type and not disclosure_equivalent)
    result.details["user_confirmation"] = {
        "confirmed": True,
        "requested_type": requested,
        "model_type": model_type,
        "statistical_type": statistical_type,
        "statistical_conflict": statistical_conflict,
        "disclosure_conflict": disclosure_conflict,
    }
    result.strategy_type = model_type
    result.evidence.append(f"用户已确认研究范围：{requested}；因子路由按该确认执行")
    if statistical_type == "insufficient_data":
        result.evidence.append("净值统计证据不足；当前因子结论属于用户确认范围下的条件性分析")
    elif statistical_conflict:
        result.evidence.append(f"用户确认与净值统计分类不一致（统计：{statistical_type}），报告将保留该冲突")
    if disclosure_conflict:
        result.evidence.append(f"用户确认范围与资料披露候选不完全一致（披露：{disclosure.get('label', disclosed_type)}），报告将保留该差异")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_and_correlate(
    product_returns: np.ndarray,
    product_dates: list[date],
    symbols: list[str],
    start: date,
    end: date,
    frequency: str,
    provider: MarketDataProvider,
    is_index: bool = True,
) -> dict[str, float]:
    """Fetch benchmark data and compute correlation with product returns."""
    correlations = {}
    for symbol in symbols:
        try:
            bench_returns = provider.get_returns(symbol, start, end, is_index=is_index)
            if bench_returns.empty or len(bench_returns) < _MIN_OBSERVATIONS:
                continue
            aligned_corr = _align_and_correlate(product_returns, product_dates, bench_returns, frequency)
            if aligned_corr is not None:
                correlations[symbol] = aligned_corr
        except Exception as exc:
            logger.debug("Skipping %s: %s", symbol, exc)
    return correlations


def _align_and_correlate(
    product_returns: np.ndarray,
    product_dates: list[date],
    bench_returns: pd.Series,
    frequency: str,
) -> float | None:
    """Align product and benchmark returns to the same dates and correlate.

    For weekly/monthly products, aggregates daily benchmark returns into
    the corresponding period returns before correlating.
    """
    if bench_returns.empty:
        return None

    aligned = align_product_and_market_returns(product_returns, product_dates, bench_returns, frequency)
    if aligned is None:
        return None
    p, b = aligned

    # Zero-variance guard is handled inside safe_pearson (issue #6).
    return safe_pearson(p, b)


def _group_score(correlations: dict[str, float]) -> float:
    """Compute a group-level score from individual correlations.

    Uses the maximum absolute correlation with a slight bonus for consistency
    across multiple indices in the group.
    """
    if not correlations:
        return 0.0
    abs_corrs = [abs(v) for v in correlations.values()]
    max_corr = max(abs_corrs)
    mean_corr = np.mean(abs_corrs)
    # Weighted: 70% max + 30% mean (rewards consistency)
    return 0.7 * max_corr + 0.3 * mean_corr


def _rolling_correlation_stability(
    product_returns: np.ndarray,
    product_dates: list[date],
    start: date,
    end: date,
    frequency: str,
    provider: MarketDataProvider,
) -> dict | None:
    """Check whether the dominant correlation group is stable over time."""
    # Use representative indices for speed
    equity_rep = "hs300"
    commodity_rep = "nh_commodity"

    try:
        eq_returns = provider.get_returns(equity_rep, start, end, is_index=True)
        cm_returns = provider.get_returns(commodity_rep, start, end, is_index=True)
    except Exception:
        return None

    if eq_returns.empty or cm_returns.empty:
        return None

    # Determine rolling window size based on frequency
    window = {"daily": 60, "weekly": 12, "monthly": 6}.get(frequency, 12)

    if len(product_returns) < window * 2:
        return None  # not enough data for rolling analysis

    aligned_eq = align_product_and_market_returns(product_returns, product_dates, eq_returns, frequency)
    aligned_cm = align_product_and_market_returns(product_returns, product_dates, cm_returns, frequency)
    if aligned_eq is None or aligned_cm is None:
        return None
    p_eq, eq = aligned_eq
    p_cm, cm = aligned_cm
    # Both benchmark pairs must cover the same trailing observations.  This is
    # conservative; when one is sparse we decline to manufacture a comparison.
    min_len = min(len(p_eq), len(p_cm), len(eq), len(cm))
    if min_len < window * 2:
        return None
    p = p_eq[-min_len:]
    eq = eq[-min_len:]
    cm = cm[-min_len:]

    equity_wins = 0
    commodity_wins = 0
    total_windows = 0

    for i in range(0, min_len - window + 1, window // 2):
        p_win = p[i:i + window]
        eq_win = eq[i:i + window]
        cm_win = cm[i:i + window]

        if np.std(p_win) == 0:
            continue

        eq_corr = abs(safe_pearson(p_win, eq_win))
        cm_corr = abs(safe_pearson(p_win, cm_win))

        if cm_corr > eq_corr:
            commodity_wins += 1
        else:
            equity_wins += 1
        total_windows += 1

    if total_windows == 0:
        return None

    dominant = "商品" if commodity_wins >= equity_wins else "股票"
    stability = max(commodity_wins, equity_wins) / total_windows * 100

    return {
        "dominant_group": dominant,
        "stability_pct": round(stability, 1),
        "total_windows": total_windows,
        "window_size": window,
    }


def _segment_beta_analysis(
    product_returns: np.ndarray,
    product_dates: list[date],
    start: date,
    end: date,
    frequency: str,
    provider: MarketDataProvider,
) -> dict | None:
    """Analyze beta in up-market vs down-market segments for equity indices."""
    try:
        eq_returns = provider.get_returns("hs300", start, end, is_index=True)
    except Exception:
        return None

    if eq_returns.empty:
        return None

    aligned = align_product_and_market_returns(product_returns, product_dates, eq_returns, frequency)
    if aligned is None:
        return None
    p, eq = aligned

    # Split into up/down segments based on equity market direction
    up_mask = eq > 0
    down_mask = eq < 0

    result = {}

    if up_mask.sum() >= 4:
        result["equity_up_beta"] = safe_pearson(p[up_mask], eq[up_mask])
    else:
        result["equity_up_beta"] = 0.0

    if down_mask.sum() >= 4:
        result["equity_down_beta"] = safe_pearson(p[down_mask], eq[down_mask])
    else:
        result["equity_down_beta"] = 0.0

    result["up_periods"] = int(up_mask.sum())
    result["down_periods"] = int(down_mask.sum())

    return result


def _confidence_label(pct: float) -> str:
    """Convert percentage to a coarse label."""
    if pct >= 70:
        return "高"
    if pct >= 40:
        return "中"
    return "低"
