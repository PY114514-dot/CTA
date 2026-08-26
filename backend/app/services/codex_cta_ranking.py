"""Deep, read-only CODEX CTA product ranking implementation.

The public seam is intentionally small: rank_cta_products accepts one
validated request and returns one auditable response.  The implementation
keeps feature engineering, robust cross-sectional scoring, and ranking-local
warnings behind that seam.  It never creates allocation weights or mutates
the product/configuration store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
from typing import Iterable

import numpy as np

from app.schemas import (
    CtaRankingAttributionEvidence,
    CtaRankingDimensionScore,
    CtaRankingMarketSeries,
    CtaRankingProductInput,
    CtaRankingProductResult,
    CtaRankingRequest,
    CtaRankingResponse,
)


MODEL_VERSION = "codex-cta-score-v1.1"
WINDOWS = (13, 26, 52)
WINDOW_WEIGHTS = {13: 0.20, 26: 0.30, 52: 0.50}
MIN_FORMAL_RETURNS = 26
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}

DIMENSION_DEFINITIONS: dict[str, dict[str, object]] = {
    "absolute": {
        "label": "绝对收益质量",
        "weight": 25.0,
        "metrics": {
            "annualized_return": 0.35,
            "rolling_return_median": 0.25,
            "positive_period_ratio": 0.15,
            "gain_loss_asymmetry": 0.25,
        },
    },
    "risk_adjusted": {
        "label": "风险调整收益",
        "weight": 25.0,
        "metrics": {
            "sharpe_ratio": 0.25,
            "sortino_ratio": 0.25,
            "calmar_ratio": 0.30,
            "tail_loss_protection": 0.20,
        },
    },
    "trend_regime": {
        "label": "趋势暴露与状态适应",
        "weight": 20.0,
        "metrics": {
            "trend_beta": 0.35,
            "trend_alignment": 0.25,
            "reversal_resilience": 0.20,
            "exposure_stability": 0.20,
        },
    },
    "tail": {
        "label": "极端环境表现",
        "weight": 15.0,
        "metrics": {
            "stress_hit_rate": 0.30,
            "left_tail_capture": 0.30,
            "recovery_speed": 0.20,
            "tail_dependence_protection": 0.20,
        },
    },
    "robustness": {
        "label": "稳健性与持续性",
        "weight": 15.0,
        "metrics": {
            "rolling_rank_stability": 0.35,
            "coefficient_stability": 0.25,
            "bootstrap_persistence": 0.25,
            "model_sensitivity": 0.10,
            # This is a validation-quality reference, not nonlinear return
            # uplift.  Its weight is deliberately modest and is renormalized
            # per product when the frozen evidence is absent.
            "oos_validation": 0.05,
        },
    },
}


@dataclass
class _PreparedProduct:
    """Internal feature container; callers never depend on this structure."""

    product: CtaRankingProductInput
    dates: list[date]
    nav_values: np.ndarray
    return_dates: list[date]
    returns: np.ndarray
    return_map: dict[date, float]
    metrics_by_window: dict[int, dict[str, float]]
    trend_by_window: dict[int, dict[str, float]]
    tail_by_window: dict[int, dict[str, float]]
    robustness_values: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    eligible: bool = False
    status: str = "insufficient_data"
    dimension_window_scores: dict[str, dict[int, float]] = field(default_factory=dict)
    dimension_raw_scores: dict[str, float] = field(default_factory=dict)
    dimension_adjusted_scores: dict[str, float] = field(default_factory=dict)
    dimension_metric_scores: dict[str, dict[str, float]] = field(default_factory=dict)


def rank_cta_products(request: CtaRankingRequest) -> CtaRankingResponse:
    """Calculate one reproducible, ranking-only CODEX CTA snapshot.

    Interface invariants:
    - NAV points are already reviewed by the caller; this function does not
      promote candidates or write to the product store.
    - as_of_date is a hard information cutoff.  Points after it are ignored,
      so a later backfill cannot leak into an earlier ranking.
    - A product needs at least 26 periodic returns for a formal rank.  Short
      samples are returned with an explicit status and no score.
    """

    as_of_date = request.as_of_date or _latest_input_date(request)
    prepared = [_prepare_product(item, as_of_date, request.annual_risk_free_rate) for item in request.products]
    market_proxy, market_data_end, market_warnings = _build_market_proxy(
        prepared, request.market_series, as_of_date
    )

    for item in prepared:
        item.trend_by_window = {
            window: _trend_features(item, market_proxy, window)
            for window in WINDOWS
            if len(item.returns) >= min(window, MIN_FORMAL_RETURNS)
        }
        item.tail_by_window = {
            window: _tail_features(item, market_proxy, window)
            for window in WINDOWS
            if len(item.returns) >= min(window, MIN_FORMAL_RETURNS)
        }

    _score_primary_dimensions(prepared)
    _populate_robustness_metrics(
        prepared,
        request.attribution_evidence,
        as_of_date,
    )
    _score_robustness_dimension(prepared)
    _apply_robust_shrinkage(prepared)

    eligible = [item for item in prepared if item.eligible and item.dimension_adjusted_scores]
    _assign_total_scores_and_ranks(eligible)

    rankings = [_to_result(item) for item in sorted(
        prepared,
        key=lambda item: (
            not item.eligible,
            -(getattr(item, "total_score", -1.0)),
            item.product.product_id,
        ),
    )]

    all_warnings = list(dict.fromkeys(market_warnings + [
        warning
        for item in prepared
        for warning in item.warnings
    ]))
    frequency = request.products[0].frequency.value
    return CtaRankingResponse(
        model_version=request.model_version or MODEL_VERSION,
        ranking_only=True,
        as_of_date=as_of_date,
        market_data_end_date=market_data_end,
        universe_size=len(prepared),
        eligible_count=len(eligible),
        dimension_weights={
            key: float(definition["weight"])
            for key, definition in DIMENSION_DEFINITIONS.items()
        },
        rankings=rankings,
        warnings=all_warnings,
        nav_fingerprint=_fingerprint_products(prepared, as_of_date),
        attribution_evidence_fingerprint=_fingerprint_attribution_evidence(
            request.attribution_evidence,
            as_of_date,
        ),
        method_provenance={
            "model": {
                "method": "CODEX-CTA 多窗口状态条件评分",
                "version": request.model_version or MODEL_VERSION,
                "dimensions": {
                    key: {
                        "label": str(definition["label"]),
                        "weight": float(definition["weight"]),
                        "metrics": definition["metrics"],
                    }
                    for key, definition in DIMENSION_DEFINITIONS.items()
                },
            },
            "normalization": {
                "method": "5%/95% 截尾后的横截面分位分数",
                "windows": list(WINDOWS),
                "window_weights": WINDOW_WEIGHTS,
            },
            "data": {
                "frequency": frequency,
                "as_of_date": as_of_date.isoformat(),
                "market_proxy": "provided" if market_proxy else "not_covered",
                "market_data_end_date": market_data_end.isoformat() if market_data_end else None,
            },
            "ranking": {
                "ranking_only": True,
                "allocation_written": False,
                "fof_marginal_diversification_calculated": False,
                "nav_completeness_scored": False,
                "update_timeliness_scored": False,
            },
            "robustness": _robustness_provenance(
                prepared,
                request.attribution_evidence,
                as_of_date,
            ),
        },
    )


def _latest_input_date(request: CtaRankingRequest) -> date:
    dates = [
        point.observation_date
        for product in request.products
        for point in product.nav_points
    ]
    dates.extend(
        point.observation_date
        for series in request.market_series
        for point in series.points
    )
    return max(dates)


def _fingerprint_attribution_evidence(
    evidence: dict[str, CtaRankingAttributionEvidence],
    as_of_date: date,
) -> str:
    payload = [
        {
            "product_id": product_id,
            "source": item.source,
            "snapshot_id": item.snapshot_id,
            "model_version": item.model_version,
            "nav_value_signature": item.nav_value_signature,
            "as_of_date": item.as_of_date.isoformat(),
            "observation_count": item.observation_count,
            "status": item.status,
            "evaluated_segments": item.evaluated_segments,
            "minimum_segments": item.minimum_segments,
            "stable_improvement": item.stable_improvement,
            "sensitivity_stable": item.sensitivity_stable,
            "selected_from_sensitivity": item.selected_from_sensitivity,
        }
        for product_id, item in sorted(evidence.items())
        if item.as_of_date <= as_of_date
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _prepare_product(
    product: CtaRankingProductInput,
    as_of_date: date,
    annual_risk_free_rate: float,
) -> _PreparedProduct:
    points = [point for point in product.nav_points if point.observation_date <= as_of_date]
    dates = [point.observation_date for point in points]
    nav_values = np.asarray([float(point.net_asset_value) for point in points], dtype=float)
    return_dates = dates[1:]
    returns = nav_values[1:] / nav_values[:-1] - 1.0 if len(nav_values) >= 2 else np.asarray([], dtype=float)
    item = _PreparedProduct(
        product=product,
        dates=dates,
        nav_values=nav_values,
        return_dates=return_dates,
        returns=np.asarray(returns, dtype=float),
        return_map={day: float(value) for day, value in zip(return_dates, returns)},
        metrics_by_window={},
        trend_by_window={},
        tail_by_window={},
    )
    if len(returns) < MIN_FORMAL_RETURNS:
        item.warnings.append(
            f"样本不足：截至 {as_of_date.isoformat()} 只有 {len(returns)} 个收益周期，正式排名至少需要 {MIN_FORMAL_RETURNS} 个。"
        )
        item.status = "insufficient_data"
        return item
    item.eligible = True
    item.status = "short_sample" if len(returns) < 52 else "eligible"
    if len(returns) < 52:
        item.warnings.append("短样本：少于 52 个收益周期，排名稳定性有限。")

    periods_per_year = PERIODS_PER_YEAR[product.frequency.value]
    for window in WINDOWS:
        if len(returns) >= window:
            item.metrics_by_window[window] = _window_metrics(
                returns[-window:],
                nav_values[-(window + 1):],
                periods_per_year,
                annual_risk_free_rate,
            )
    item.metrics_by_window[0] = _window_metrics(
        returns,
        nav_values,
        periods_per_year,
        annual_risk_free_rate,
    )
    return item


def _window_metrics(
    returns: np.ndarray,
    nav_values: np.ndarray,
    periods_per_year: int,
    annual_risk_free_rate: float,
) -> dict[str, float]:
    if returns.size == 0:
        return {}
    returns = np.asarray(returns, dtype=float)
    nav_values = np.asarray(nav_values, dtype=float)
    n = returns.size
    compound = float(np.prod(1.0 + returns))
    annualized_return = float(compound ** (periods_per_year / n) - 1.0) if compound > 0 else -1.0
    volatility = float(np.std(returns, ddof=1) * math.sqrt(periods_per_year)) if n > 1 else 0.0
    sharpe = (
        float((annualized_return - annual_risk_free_rate) / volatility)
        if volatility > 1e-12 else None
    )
    periodic_rf = annual_risk_free_rate / periods_per_year
    downside = np.minimum(returns - periodic_rf, 0.0)
    downside_deviation = float(math.sqrt(np.mean(downside ** 2)) * math.sqrt(periods_per_year))
    sortino = (
        float((annualized_return - annual_risk_free_rate) / downside_deviation)
        if downside_deviation > 1e-12 else None
    )
    running_peaks = np.maximum.accumulate(nav_values)
    drawdowns = nav_values / running_peaks - 1.0
    maximum_drawdown = float(np.min(drawdowns)) if drawdowns.size else 0.0
    calmar = (
        float(annualized_return / abs(maximum_drawdown))
        if maximum_drawdown < -1e-12 else None
    )
    positive_ratio = float(np.mean(returns > 0.0))
    positive = returns[returns > 0.0]
    negative = returns[returns < 0.0]
    if negative.size == 0:
        gain_loss_asymmetry = 3.0
    elif positive.size == 0:
        gain_loss_asymmetry = 0.0
    else:
        gain_loss_asymmetry = float(
            np.median(positive) / max(abs(float(np.median(negative))), 1e-12)
        )
    tail_cutoff = float(np.quantile(returns, 0.10))
    tail_returns = returns[returns <= tail_cutoff]
    # Higher means the left-tail slice is less damaging (or positive).
    tail_loss_protection = float(np.mean(tail_returns)) if tail_returns.size else 0.0
    rolling_window = min(13, n)
    rolling_values = [
        float(np.prod(1.0 + returns[index - rolling_window + 1:index + 1]) - 1.0)
        for index in range(rolling_window - 1, n)
    ]
    rolling_return_median = float(np.median(rolling_values)) if rolling_values else 0.0
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar,
        "positive_period_ratio": positive_ratio,
        "gain_loss_asymmetry": gain_loss_asymmetry,
        "tail_loss_protection": tail_loss_protection,
        "rolling_return_median": rolling_return_median,
        "recovery_speed": _recovery_speed(nav_values),
    }


def _build_market_proxy(
    prepared: list[_PreparedProduct],
    market_series: list[CtaRankingMarketSeries],
    as_of_date: date,
) -> tuple[dict[date, float], date | None, list[str]]:
    warnings: list[str] = []
    if market_series:
        by_date: dict[date, list[float]] = {}
        for series in market_series:
            for point in series.points:
                if point.observation_date <= as_of_date:
                    by_date.setdefault(point.observation_date, []).append(float(point.return_value))
        proxy = {day: float(np.mean(values)) for day, values in by_date.items() if values}
        end_date = max(proxy) if proxy else None
        if not proxy:
            warnings.append("提供的市场代理在排名截点前没有可用收益，已回退到产品横截面代理。")
        else:
            return proxy, end_date, warnings

    warnings.append("未提供可用市场代理；趋势、状态和尾部相关指标未覆盖，不使用产品横截面代理。")
    return {}, None, warnings


def _trend_features(
    item: _PreparedProduct,
    market_proxy: dict[date, float],
    window: int,
) -> dict[str, float]:
    aligned = [(day, item.return_map[day], market_proxy[day]) for day in item.return_dates if day in market_proxy]
    if len(aligned) < 10:
        return {}
    aligned = aligned[-window:]
    product = np.asarray([row[1] for row in aligned], dtype=float)
    market = np.asarray([row[2] for row in aligned], dtype=float)
    n = product.size
    beta_values: list[float] = []
    alignment_values: list[float] = []
    state_signs: list[int] = []
    for horizon in (4, 12, 26):
        if n <= horizon + 2:
            continue
        previous = np.asarray([
            np.sum(market[index - horizon:index])
            for index in range(horizon, n)
        ], dtype=float)
        current = product[horizon:]
        state = np.sign(previous)
        signal = state * np.abs(market[horizon:])
        beta = _safe_beta(current, signal)
        if beta is not None:
            beta_values.append(beta)
        alignment_values.extend((current * state).tolist())
        state_signs.extend(state.astype(int).tolist())
    if not beta_values:
        return {}
    reversal_mask = np.asarray([
        state_signs[index] != state_signs[index - 1]
        for index in range(1, len(state_signs))
    ], dtype=bool)
    aligned_product = np.asarray(alignment_values, dtype=float)
    reversal_returns = aligned_product[1:][reversal_mask] if reversal_mask.any() else np.asarray([])
    overall_beta = float(np.mean(beta_values))
    stability = _beta_stability(product, market)
    return {
        "trend_beta": overall_beta,
        "trend_alignment": float(np.mean(aligned_product)) if aligned_product.size else 0.0,
        "reversal_resilience": float(np.mean(reversal_returns)) if reversal_returns.size else 0.0,
        "exposure_stability": stability,
    }


def _tail_features(
    item: _PreparedProduct,
    market_proxy: dict[date, float],
    window: int,
) -> dict[str, float]:
    aligned = [(day, item.return_map[day], market_proxy[day]) for day in item.return_dates if day in market_proxy]
    if len(aligned) < 10:
        return {}
    aligned = aligned[-window:]
    product = np.asarray([row[1] for row in aligned], dtype=float)
    market = np.asarray([row[2] for row in aligned], dtype=float)
    cutoff = float(np.quantile(market, 0.10))
    stress_mask = market <= cutoff
    if stress_mask.sum() < 2:
        stress_mask = np.zeros_like(market, dtype=bool)
        stress_mask[np.argsort(market)[: min(2, market.size)]] = True
    stress_returns = product[stress_mask]
    # Higher means the product captures less of the market's left-tail loss.
    left_tail_capture = float(np.mean(stress_returns)) if stress_returns.size else 0.0
    tail_dependence = _safe_correlation(product[stress_mask], market[stress_mask])
    if tail_dependence is not None:
        tail_dependence = -tail_dependence
    else:
        tail_dependence = 0.0
    nav_slice = item.nav_values[-(min(window, len(item.nav_values))):]
    return {
        "stress_hit_rate": float(np.mean(stress_returns > 0.0)) if stress_returns.size else 0.0,
        "left_tail_capture": left_tail_capture,
        "recovery_speed": _recovery_speed(nav_slice),
        "tail_dependence_protection": float(tail_dependence),
    }


def _score_primary_dimensions(prepared: list[_PreparedProduct]) -> None:
    """Score absolute/risk/trend/tail dimensions via robust percentiles."""

    for dimension in ("absolute", "risk_adjusted", "trend_regime", "tail"):
        definition = DIMENSION_DEFINITIONS[dimension]
        metric_weights = definition["metrics"]
        assert isinstance(metric_weights, dict)
        metric_scores_by_window = {
            metric: _score_metric_by_window(prepared, dimension, metric)
            for metric in metric_weights
        }
        for item in prepared:
            if not item.eligible:
                continue
            item.dimension_metric_scores.setdefault(dimension, {})
            item.dimension_window_scores.setdefault(dimension, {})
            for metric, metric_scores in metric_scores_by_window.items():
                score = _weighted_available(
                    {
                        window: metric_scores.get(item.product.product_id, {}).get(window)
                        for window in WINDOWS
                    },
                    WINDOW_WEIGHTS,
                )
                if score is not None:
                    item.dimension_metric_scores[dimension][metric] = score
            window_scores: dict[int, float] = {}
            for window in WINDOWS:
                window_metrics = {
                    metric: metric_scores_by_window[metric]
                    .get(item.product.product_id, {})
                    .get(window)
                    for metric in metric_weights
                }
                score = _weighted_available(window_metrics, metric_weights)
                if score is not None:
                    window_scores[window] = score
            item.dimension_window_scores[dimension] = window_scores
            score = _weighted_available(window_scores, WINDOW_WEIGHTS)
            if score is not None:
                item.dimension_raw_scores[dimension] = score


def _dimension_metric_windows(
    item: _PreparedProduct,
    dimension: str,
    metric: str,
) -> dict[int, float]:
    if dimension in {"absolute", "risk_adjusted"}:
        return {
            window: values[metric]
            for window, values in item.metrics_by_window.items()
            if window in WINDOWS and metric in values and values[metric] is not None
        }
    if dimension == "trend_regime":
        return {
            window: values[metric]
            for window, values in item.trend_by_window.items()
            if metric in values
        }
    if dimension == "tail":
        return {
            window: values[metric]
            for window, values in item.tail_by_window.items()
            if metric in values
        }
    return {}


def _score_metric_by_window(
    prepared: list[_PreparedProduct],
    dimension: str,
    metric: str,
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {item.product.product_id: {} for item in prepared}
    for window in WINDOWS:
        values: dict[str, float] = {}
        for item in prepared:
            if not item.eligible:
                continue
            metric_values = _dimension_metric_windows(item, dimension, metric)
            if window in metric_values and np.isfinite(metric_values[window]):
                values[item.product.product_id] = float(metric_values[window])
        scores = _cross_sectional_scores(values)
        for product_id, score in scores.items():
            result[product_id][window] = score
    return result


def _populate_robustness_metrics(
    prepared: list[_PreparedProduct],
    attribution_evidence: dict[str, CtaRankingAttributionEvidence],
    as_of_date: date,
) -> None:
    for item in prepared:
        if not item.eligible:
            continue
        primary_scores = [
            score
            for dimension in ("absolute", "risk_adjusted", "trend_regime", "tail")
            for score in item.dimension_window_scores.get(dimension, {}).values()
        ]
        rolling_rank_stability = max(
            0.0,
            100.0 - 2.0 * float(np.std(primary_scores, ddof=0))
        ) if primary_scores else 50.0
        coefficient_stability = (
            float(item.trend_by_window.get(52, item.trend_by_window.get(26, {})).get("exposure_stability", 0.5)) * 100.0
        )
        bootstrap_persistence = _bootstrap_persistence(item.returns, item.product.product_id)
        full_return = item.metrics_by_window.get(0, {}).get("annualized_return", 0.0)
        recent_return = item.metrics_by_window.get(26, {}).get("annualized_return", full_return)
        model_sensitivity = 1.0 / (1.0 + abs(float(full_return) - float(recent_return)))
        item.robustness_values = {
            "rolling_rank_stability": rolling_rank_stability,
            "coefficient_stability": coefficient_stability,
            "bootstrap_persistence": bootstrap_persistence,
            "model_sensitivity": model_sensitivity,
        }
        evidence = attribution_evidence.get(item.product.product_id)
        oos_score, evidence_warning = _phase_d_oos_validation_score(evidence, as_of_date)
        if oos_score is not None:
            item.robustness_values["oos_validation"] = oos_score
        if evidence_warning:
            item.warnings.append(evidence_warning)


def _score_robustness_dimension(prepared: list[_PreparedProduct]) -> None:
    dimension = "robustness"
    metric_weights = DIMENSION_DEFINITIONS[dimension]["metrics"]
    assert isinstance(metric_weights, dict)
    metric_scores_by_metric: dict[str, dict[str, float]] = {}
    for metric in metric_weights:
        raw_values = {
            item.product.product_id: item.robustness_values[metric]
            for item in prepared
            if item.eligible and metric in item.robustness_values
        }
        if metric == "oos_validation":
            # The Phase-D value is already a bounded evidence-quality score:
            # 100 means stable base-case OOS support and 50 is deliberately
            # neutral.  Ranking it again cross-sectionally would turn a
            # neutral 50 into a zero whenever another product scores 100.
            metric_scores_by_metric[metric] = raw_values
        else:
            metric_scores_by_metric[metric] = _cross_sectional_scores(raw_values)
    for item in prepared:
        if not item.eligible:
            continue
        metric_scores = {
            metric: metric_scores_by_metric.get(metric, {}).get(item.product.product_id)
            for metric in metric_weights
        }
        item.dimension_metric_scores[dimension] = {
            metric: float(score)
            for metric, score in metric_scores.items()
            if score is not None
        }
        score = _weighted_available(metric_scores, metric_weights)
        if score is not None:
            item.dimension_window_scores[dimension] = {52: score}
            item.dimension_raw_scores[dimension] = score


def _phase_d_oos_validation_score(
    evidence: CtaRankingAttributionEvidence | None,
    ranking_as_of_date: date,
) -> tuple[float | None, str | None]:
    """Translate frozen Phase-D quality into a neutral robustness reference.

    The only positive value is reserved for an available, sufficiently sized
    base-case result whose improvement survives sensitivity diagnostics.  A
    valid but unstable or non-improving result remains neutral (50), while
    missing/invalid evidence is omitted entirely so the other robustness
    metrics are reweighted rather than treating missing evidence as a failure.
    """

    if evidence is None:
        return None, "Phase D 证据缺失：未将非线性 OOS 结果计入稳健性指标。"
    if evidence.status != "available":
        return None, (
            f"Phase D 证据不可用（状态 {evidence.status}）："
            "未将其计入稳健性指标。"
        )
    if evidence.evaluated_segments < evidence.minimum_segments:
        return None, (
            f"Phase D 测试段不足（{evidence.evaluated_segments}/"
            f"{evidence.minimum_segments}）：未将其计入稳健性指标。"
        )
    if evidence.selected_from_sensitivity:
        return None, "Phase D 结果来自敏感性选择：未将其计入稳健性指标。"
    if evidence.as_of_date > ranking_as_of_date:
        return None, "Phase D 证据晚于排名截点：未将未来信息计入稳健性指标。"
    if evidence.sensitivity_stable and evidence.stable_improvement:
        return 100.0, None
    return 50.0, None


def _robustness_provenance(
    prepared: list[_PreparedProduct],
    attribution_evidence: dict[str, CtaRankingAttributionEvidence],
    as_of_date: date,
) -> dict[str, object]:
    valid_count = 0
    included_count = 0
    considered_count = 0
    for item in prepared:
        evidence = attribution_evidence.get(item.product.product_id)
        if evidence is not None and evidence.as_of_date <= as_of_date:
            considered_count += 1
        score, _warning = _phase_d_oos_validation_score(evidence, as_of_date)
        if evidence is not None and score is not None:
            valid_count += 1
        if "oos_validation" in item.robustness_values:
            included_count += 1
    return {
        "phase_d_evidence_source": "immutable_phase_d_snapshot",
        "phase_d_evidence_products": considered_count,
        "phase_d_valid_evidence_products": valid_count,
        "phase_d_included_products": included_count,
        "oos_validation_metric": "validation_quality_only",
        "nonlinear_increment_direct_score": False,
        "missing_phase_d_evidence_policy": "omit_metric",
        "invalid_phase_d_evidence_policy": "omit_metric",
        "per_product_metric_weights_renormalized": True,
        "as_of_date_cutoff": as_of_date.isoformat(),
    }


def _apply_robust_shrinkage(prepared: list[_PreparedProduct]) -> None:
    for item in prepared:
        if not item.eligible:
            continue
        robustness = item.dimension_raw_scores.get("robustness", 50.0)
        shrinkage = min(1.0, max(0.5, 0.5 + 0.5 * robustness / 100.0))
        for dimension, raw_score in item.dimension_raw_scores.items():
            item.dimension_adjusted_scores[dimension] = _clip(
                50.0 + shrinkage * (raw_score - 50.0),
                0.0,
                100.0,
            )


def _assign_total_scores_and_ranks(eligible: list[_PreparedProduct]) -> None:
    if not eligible:
        return
    for item in eligible:
        available = [
            (score, float(DIMENSION_DEFINITIONS[dimension]["weight"]))
            for dimension, score in item.dimension_adjusted_scores.items()
        ]
        item.total_score = float(
            sum(score * weight for score, weight in available) / sum(weight for _, weight in available)
        ) if available else 0.0
    ordered = sorted(eligible, key=lambda item: (-item.total_score, item.product.product_id))
    for index, item in enumerate(ordered, start=1):
        item.rank = index
        item.percentile = 100.0 if len(ordered) == 1 else 100.0 * (len(ordered) - index) / (len(ordered) - 1)


def _to_result(item: _PreparedProduct) -> CtaRankingProductResult:
    all_metrics = dict(item.metrics_by_window.get(0, {}))
    all_metrics.update(item.robustness_values)
    dimensions: list[CtaRankingDimensionScore] = []
    for dimension, definition in DIMENSION_DEFINITIONS.items():
        if not item.eligible:
            continue
        metric_weights = definition["metrics"]
        assert isinstance(metric_weights, dict)
        metric_scores = item.dimension_metric_scores.get(dimension, {})
        metric_values = {
            metric: _latest_metric_value(item, dimension, metric)
            for metric in metric_weights
        }
        dimensions.append(CtaRankingDimensionScore(
            dimension=dimension,
            label=str(definition["label"]),
            weight=float(definition["weight"]),
            raw_score=round(item.dimension_raw_scores[dimension], 4) if dimension in item.dimension_raw_scores else None,
            adjusted_score=round(item.dimension_adjusted_scores[dimension], 4) if dimension in item.dimension_adjusted_scores else None,
            metric_count=len(metric_weights),
            covered_metric_count=len(metric_scores),
            metric_scores={key: round(value, 4) for key, value in metric_scores.items()},
            metric_values={
                key: round(value, 8) if value is not None else None
                for key, value in metric_values.items()
                if value is not None or not (dimension == "robustness" and key == "oos_validation")
            },
        ))
    return CtaRankingProductResult(
        product_id=item.product.product_id,
        product_name=item.product.product_name,
        rank=getattr(item, "rank", None),
        score=round(getattr(item, "total_score", 0.0), 4) if item.eligible else None,
        percentile=round(getattr(item, "percentile", 0.0), 4) if item.eligible else None,
        eligible=item.eligible,
        status=item.status,
        nav_start_date=item.dates[0] if item.dates else None,
        nav_end_date=item.dates[-1] if item.dates else None,
        observation_count=len(item.dates),
        dimension_scores={
            key: round(value, 4)
            for key, value in item.dimension_adjusted_scores.items()
        },
        dimensions=dimensions,
        metrics={
            key: round(value, 8) if value is not None else None
            for key, value in all_metrics.items()
        },
        warnings=list(dict.fromkeys(item.warnings)),
    )


def _latest_metric_value(item: _PreparedProduct, dimension: str, metric: str) -> float | None:
    if dimension == "robustness":
        return item.robustness_values.get(metric)
    source = item.metrics_by_window if dimension in {"absolute", "risk_adjusted"} else (
        item.trend_by_window if dimension == "trend_regime" else item.tail_by_window
    )
    for window in (52, 26, 13, 0):
        if metric in source.get(window, {}):
            return source[window][metric]
    return None


def _weighted_available(
    values: dict[int | str, float | None],
    weights: dict[int | str, float],
) -> float | None:
    available = [
        (float(value), float(weights[key]))
        for key, value in values.items()
        if value is not None and np.isfinite(value) and key in weights
    ]
    if not available:
        return None
    numerator = sum(value * weight for value, weight in available)
    denominator = sum(weight for _, weight in available)
    return float(numerator / denominator) if denominator else None


def _cross_sectional_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    finite = {key: float(value) for key, value in values.items() if np.isfinite(value)}
    if not finite:
        return {}
    lower = float(np.quantile(list(finite.values()), 0.05))
    upper = float(np.quantile(list(finite.values()), 0.95))
    clipped = {key: _clip(value, lower, upper) for key, value in finite.items()}
    if len(clipped) == 1 or max(clipped.values()) - min(clipped.values()) < 1e-12:
        return {key: 50.0 for key in clipped}
    ordered = sorted(clipped.items(), key=lambda pair: (pair[1], pair[0]))
    scores: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and abs(ordered[end][1] - ordered[index][1]) < 1e-12:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        score = 100.0 * (average_rank - 1.0) / (len(ordered) - 1)
        for key, _ in ordered[index:end]:
            scores[key] = score
        index = end
    return scores


def _safe_beta(y: np.ndarray, x: np.ndarray) -> float | None:
    if y.size < 3 or x.size != y.size:
        return None
    variance = float(np.var(x))
    if variance <= 1e-12:
        return None
    covariance = float(np.cov(x, y, ddof=1)[0, 1])
    result = covariance / variance
    return float(result) if np.isfinite(result) else None


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or x.size != y.size:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    result = float(np.corrcoef(x, y)[0, 1])
    return result if np.isfinite(result) else None


def _beta_stability(product: np.ndarray, market: np.ndarray) -> float:
    if product.size < 12:
        return 0.5
    chunks = np.array_split(np.arange(product.size), 3)
    betas = [
        _safe_beta(product[indexes], market[indexes])
        for indexes in chunks
        if len(indexes) >= 4
    ]
    betas = [value for value in betas if value is not None]
    if len(betas) < 2:
        return 0.5
    signs = [np.sign(value) for value in betas if abs(value) > 1e-12]
    sign_consistency = max(
        sum(1 for sign in signs if sign == np.sign(betas[-1])) / max(len(signs), 1),
        0.5,
    )
    dispersion = float(np.std(betas) / (np.mean(np.abs(betas)) + 1e-8))
    return _clip(0.5 * sign_consistency + 0.5 / (1.0 + dispersion), 0.0, 1.0)


def _recovery_speed(nav_values: np.ndarray) -> float:
    if nav_values.size < 2:
        return 0.0
    peak = float(nav_values[0])
    trough_index = 0
    trough_value = peak
    recovery_periods: list[int] = []
    for index, value in enumerate(nav_values[1:], start=1):
        value = float(value)
        if value >= peak:
            if trough_value < peak:
                recovery_periods.append(max(index - trough_index, 1))
            peak = value
            trough_index = index
            trough_value = value
        elif value < trough_value:
            trough_value = value
            trough_index = index
    if trough_value < peak:
        recovery_periods.append(max(len(nav_values) - 1 - trough_index + 1, 1))
    return float(1.0 / np.mean(recovery_periods)) if recovery_periods else 1.0


def _bootstrap_persistence(returns: np.ndarray, product_id: str) -> float:
    if returns.size < MIN_FORMAL_RETURNS:
        return 50.0
    digest = hashlib.sha256(product_id.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    block_size = 4
    draws = 48
    # 向量化重写，与旧版逐块循环的结果逐位一致：
    # start 取值范围是 [0, n-4]（high = n - block_size + 1），切片长度恒为 4，
    # 因此每次抽取恰好消耗 ceil(n/4) 个随机数，且按 draw 顺序消费，
    # 与旧版 while 循环逐块调用 rng.integers 的顺序完全相同。
    n = int(returns.size)
    blocks_per_draw = (n + block_size - 1) // block_size
    high = max(n - block_size + 1, 1)
    starts = rng.integers(0, high, size=(draws, blocks_per_draw))
    indices = (starts[..., None] + np.arange(block_size)).reshape(draws, blocks_per_draw * block_size)[:, :n]
    sampled = returns[indices]
    compounds = np.prod(1.0 + sampled, axis=1)
    annualized = np.where(compounds > 0, compounds ** (52 / n) - 1.0, -1.0)
    successes = int(np.sum((annualized > 0) & (np.mean(sampled, axis=1) > 0)))
    return float(100.0 * successes / draws)


def _fingerprint_products(prepared: Iterable[_PreparedProduct], as_of_date: date) -> str:
    payload = [
        {
            "product_id": item.product.product_id,
            "points": [
                (day.isoformat(), round(float(value), 8))
                for day, value in zip(item.dates, item.nav_values)
                if day <= as_of_date
            ],
        }
        for item in sorted(prepared, key=lambda value: value.product.product_id)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))
