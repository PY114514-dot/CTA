"""Separate CTA quality, conclusion confidence, and allocation value."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

import numpy as np

from app.schemas import CtaProductScoreRequest, CtaRankingRequest, CtaRankingProductInput
from app.services.codex_cta_ranking import rank_cta_products
from app.services.cta_factor_bundle import CTA_FACTOR_NAMES, get_cta_factor_bundle, get_factor_series
from app.services.cta_return_risk import (
    analyze_returns,
    analyze_risk_exposure,
    score_return_analysis,
    score_risk_exposure,
)
from app.services.factor_library.factor_regression import RegressionResult, assess_oos_applicability, run_factor_regression


QUALITY_MODEL_VERSION = "codex-cta-score-v1.1"
PRODUCT_SCORE_MODEL_VERSION = "cta-product-score-v3.0"
_MAX_QUALITY_HISTORY_POINTS = 6
_MIN_HISTORY_RETURNS = 26
_MAX_ATTRIBUTION_WORKERS = 4


def _returns(product: CtaRankingProductInput, as_of_date: date | None) -> dict[date, float]:
    points = [point for point in product.nav_points if as_of_date is None or point.observation_date <= as_of_date]
    points.sort(key=lambda point: point.observation_date)
    return {
        right.observation_date: float(right.net_asset_value / left.net_asset_value - 1.0)
        for left, right in zip(points, points[1:], strict=False)
    }


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(min(max(value, lower), upper))


def _logistic(value: float) -> float:
    """数值稳定的 logistic 函数，用于把无界指标映射到 0-1。"""
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exp_value = float(np.exp(value))
    return exp_value / (1.0 + exp_value)


def _quality_history_dates(
    products: list[CtaRankingProductInput],
    cutoff: date,
) -> list[date]:
    """Select a small, deterministic set of usable historical cutoffs."""
    candidates = {
        point.observation_date
        for product in products
        for point in product.nav_points
        if point.observation_date <= cutoff
    }
    candidates.add(cutoff)
    usable = sorted(
        day for day in candidates
        if any(len(_returns(product, day)) >= _MIN_HISTORY_RETURNS for product in products)
    )
    if len(usable) <= _MAX_QUALITY_HISTORY_POINTS:
        return usable
    indexes = np.linspace(0, len(usable) - 1, _MAX_QUALITY_HISTORY_POINTS, dtype=int)
    return [usable[index] for index in indexes]


def _quality_history_for_product(
    product_id: str,
    rankings: list[Any],
) -> list[dict[str, Any]]:
    """Explain score changes by comparing adjacent frozen ranking cutoffs."""
    history: list[dict[str, Any]] = []
    previous_dimensions: dict[str, float] = {}
    previous_score: float | None = None
    for snapshot in sorted(rankings, key=lambda item: item.as_of_date):
        item = next(result for result in snapshot.rankings if result.product_id == product_id)
        current_dimensions = dict(item.dimension_scores)
        score_delta = (
            round(float(item.score - previous_score), 4)
            if item.score is not None and previous_score is not None
            else None
        )
        reasons: list[dict[str, Any]] = []
        if not history:
            reasons.append({"dimension": "历史起点", "delta": None, "reason": "历史轨迹起点。"})
        else:
            for dimension in item.dimensions:
                previous = previous_dimensions.get(dimension.dimension)
                if previous is None or dimension.adjusted_score is None:
                    continue
                delta = round(float(dimension.adjusted_score - previous), 4)
                if abs(delta) < 0.01:
                    continue
                direction = "上升" if delta > 0 else "下降"
                reasons.append({
                    "dimension": dimension.label,
                    "delta": delta,
                    "reason": f"{dimension.label}{direction}{abs(delta):.2f} 分。",
                })
            reasons.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
            reasons = reasons[:3]
            if not reasons:
                if item.status not in {"eligible", "short_sample"}:
                    reason = item.warnings[0] if item.warnings else "当前截点样本不足，暂无可比较维度。"
                else:
                    reason = "相邻截点各维度没有可见变化。"
                reasons.append({"dimension": "整体", "delta": score_delta, "reason": reason})
        history.append({
            "as_of_date": snapshot.as_of_date.isoformat(),
            "quality_score": item.score,
            "quality_status": item.status,
            "dimension_scores": current_dimensions,
            "score_delta": score_delta,
            "change_reasons": reasons,
            "warnings": list(item.warnings),
        })
        previous_dimensions = current_dimensions
        previous_score = item.score
    return history


def _confidence(
    product: CtaRankingProductInput,
    returns: dict[date, float],
    *,
    factor_bundle: dict[str, Any],
    regression: RegressionResult,
) -> dict[str, Any]:
    """结论置信度 v2：成熟产品不再全体满分。

    v1 的分量几乎全是「覆盖/饱和」型（样本≥门槛即 100 分），长历史周频
    产品七个分量常全部触顶，置信分对成熟宇宙失去区分度（真实数据曾普遍
    100.0）。v2 改用响应曲线：样本长度用 log2 比值 logistic、beta 不确定
    性用因子 |t| 均值 logistic、OOS 稳定性用样本外 R² 连续打分，任何分量
    都不再轻易触顶；仅数据覆盖与因子对齐覆盖保留覆盖型语义。
    """
    frequency = str(product.frequency)
    minimum_observations = {"daily": 252, "weekly": 52, "monthly": 36}[frequency]
    ratio = len(returns) / minimum_observations if minimum_observations else 0.0
    sample_length = 100.0 * _logistic(np.log2(max(ratio, 1e-9)) - 0.5)
    finite_count = sum(np.isfinite(value) for value in returns.values())
    data_coverage = 100.0 * finite_count / len(returns) if returns else 0.0
    t_values = [
        abs(float(factor.t_stat))
        for factor in regression.factors
        if factor.t_stat is not None and np.isfinite(factor.t_stat)
    ]
    beta_uncertainty = (
        100.0 * _logistic(float(np.mean(t_values)) - 2.0) if t_values else None
    )
    factor_alignment_coverage = (
        _clip(100.0 * regression.n_observations / len(returns))
        if regression.factors and returns
        else None
    )
    negative_threshold = float(np.quantile(list(returns.values()), 0.20)) if returns else 0.0
    crisis_count = sum(value <= negative_threshold for value in returns.values())
    state_sample = 100.0 * float(np.sqrt(min(crisis_count / 12.0, 1.0)))
    out_of_sample = regression.out_of_sample
    oos_r2 = out_of_sample.get("r_squared")
    oos_segments = [
        item for item in out_of_sample.get("segments", [])
        if item.get("r_squared") is not None
    ]
    positive_fraction = out_of_sample.get("positive_segment_fraction")
    if oos_r2 is None or len(oos_segments) < 3:
        oos_score = None
    else:
        # OOS R² 在 0~0.10 区间给足区分度；负 R² 平滑降分而非一刀切 0。
        r2_component = 100.0 * _logistic((float(oos_r2) - 0.03) / 0.05)
        fraction_component = (
            100.0 * float(positive_fraction) if positive_fraction is not None else 50.0
        )
        oos_score = 0.5 * r2_component + 0.5 * fraction_component
    factors = factor_bundle.get("factors", [])
    input_version = _clip(100.0 * sum(item.get("status") == "available" for item in factors) / max(len(factors), 1))
    components: dict[str, float | None] = {
        "sample_length": round(sample_length, 4),
        "data_coverage": round(data_coverage, 4),
        "beta_uncertainty": round(beta_uncertainty, 4) if beta_uncertainty is not None else None,
        "factor_alignment_coverage": round(factor_alignment_coverage, 4) if factor_alignment_coverage is not None else None,
        "state_sample": round(state_sample, 4),
        "oos_stability": round(oos_score, 4) if oos_score is not None else None,
        "input_version_completeness": round(input_version, 4),
    }
    weights = {
        "sample_length": 0.20,
        "data_coverage": 0.10,
        "beta_uncertainty": 0.15,
        "factor_alignment_coverage": 0.20,
        "state_sample": 0.05,
        "oos_stability": 0.20,
        "input_version_completeness": 0.10,
    }
    available_weights = {name: weight for name, weight in weights.items() if components[name] is not None}
    weight_total = sum(available_weights.values())
    effective_weights = {name: weight / weight_total for name, weight in available_weights.items()}
    score = sum(float(components[name]) * effective_weights[name] for name in effective_weights)
    warnings = []
    if not regression.factors:
        warnings.append("CTA 因子回归不可用，因子关联不确定性不纳入置信度。")
    elif t_values and float(np.mean(t_values)) < 1.5:
        warnings.append(
            f"因子 beta 显著性整体偏低（平均 |t|={np.mean(t_values):.2f}），"
            "归因相关结论置信度已下调。"
        )
    elif factor_alignment_coverage is not None and factor_alignment_coverage < 80:
        warnings.append(f"CTA 因子仅对齐 {regression.n_observations}/{len(returns)} 个收益期，归因相关结论置信度已下调。")
    if oos_score is None:
        warnings.append("CTA 因子回归的样本外测试段不足，OOS 稳定性不纳入置信度。")
    if input_version < 100.0:
        warnings.append("cta_factor_bundle_v1 仍有未覆盖因子，输入版本完整性未满分。")
    return {
        "score": round(float(score), 4),
        "band": "high" if score >= 75 else "medium" if score >= 50 else "low",
        "components": components,
        "weights": effective_weights,
        "method": "independent_conclusion_confidence_v2",
        "warnings": warnings,
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _maximum_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    nav = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(nav)
    return float(np.min(nav / peaks - 1.0))


def _cosine_overlap(left: dict[str, float], right: dict[str, float]) -> float | None:
    names = sorted(set(left) | set(right))
    if not names:
        return None
    left_values = np.asarray([left.get(name, 0.0) for name in names], dtype=float)
    right_values = np.asarray([right.get(name, 0.0) for name in names], dtype=float)
    left_norm = float(np.linalg.norm(left_values))
    right_norm = float(np.linalg.norm(right_values))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return float(abs(np.dot(left_values, right_values) / (left_norm * right_norm)))


def _allocation_score(
    request: CtaProductScoreRequest,
    by_product: dict[str, dict[date, float]],
    confidence_by_product: dict[str, dict[str, Any]],
    factor_exposures_by_product: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not request.current_portfolio:
        return {
            "status": "not_available",
            "score": None,
            "reason": "未提供当前组合及权重；配置分不会对单产品场景生成。",
            "metrics": {},
            "components": {},
            "warnings": ["配置分是组合条件指标，不与质量分或置信度合并。"],
        }
    assert request.candidate_product_id is not None
    assert request.candidate_weight is not None
    candidate_id = request.candidate_product_id
    all_ids = [position.product_id for position in request.current_portfolio] + [candidate_id]
    common_dates = set(by_product[all_ids[0]])
    for product_id in all_ids[1:]:
        common_dates &= set(by_product[product_id])
    dates = sorted(common_dates)
    if len(dates) < 12:
        return {
            "status": "insufficient",
            "score": None,
            "candidate_product_id": candidate_id,
            "reason": "当前组合与候选产品的共同净值期少于 12 期。",
            "metrics": {},
            "components": {},
            "warnings": ["共同日期不足，拒绝生成配置分。"],
        }
    current = np.asarray([
        sum(position.weight * by_product[position.product_id][day] for position in request.current_portfolio)
        for day in dates
    ], dtype=float)
    candidate = np.asarray([by_product[candidate_id][day] for day in dates], dtype=float)
    after = (1.0 - request.candidate_weight) * current + request.candidate_weight * candidate
    current_vol = float(np.std(current, ddof=1))
    after_vol = float(np.std(after, ddof=1))
    current_drawdown = _maximum_drawdown(current)
    after_drawdown = _maximum_drawdown(after)
    full_corr = _correlation(candidate, current)
    crisis_threshold = float(np.quantile(current, 0.20))
    crisis_mask = current <= crisis_threshold
    crisis_corr = _correlation(candidate[crisis_mask], current[crisis_mask]) if int(crisis_mask.sum()) >= 3 else None
    sign_changes = np.r_[False, np.sign(current[1:]) != np.sign(current[:-1])]
    whipsaw_mask = sign_changes
    candidate_factor = factor_exposures_by_product[candidate_id]
    current_factor: dict[str, float] = {}
    for position in request.current_portfolio:
        exposure = factor_exposures_by_product[position.product_id]
        for name, value in exposure.items():
            current_factor[name] = current_factor.get(name, 0.0) + position.weight * value
    factor_overlap = _cosine_overlap(candidate_factor, current_factor)
    metrics: dict[str, float | None] = {
        "full_sample_correlation": round(full_corr, 8) if full_corr is not None else None,
        "crisis_correlation": round(crisis_corr, 8) if crisis_corr is not None else None,
        "marginal_volatility_change": round(after_vol - current_vol, 8),
        "marginal_maximum_drawdown_change": round(after_drawdown - current_drawdown, 8),
        "crisis_complementarity": round(float(np.mean(candidate[crisis_mask]) - np.mean(current[crisis_mask])), 8) if crisis_mask.any() else None,
        "whipsaw_complementarity": round(float(np.mean(candidate[whipsaw_mask]) - np.mean(current[whipsaw_mask])), 8) if whipsaw_mask.any() else None,
        "factor_exposure_overlap": round(factor_overlap, 8) if factor_overlap is not None else None,
    }
    complementarity_values = [
        float(np.mean(candidate[mask]) - np.mean(current[mask]))
        for mask in (crisis_mask, whipsaw_mask)
        if mask.any()
    ]
    components: dict[str, float] = {
        "correlation": _clip(50.0 * (1.0 - (full_corr if full_corr is not None else 0.0))),
        "volatility": _clip(50.0 + 500.0 * (current_vol - after_vol)),
        "maximum_drawdown": _clip(50.0 + 500.0 * (after_drawdown - current_drawdown)),
        "crisis_whipsaw_complementarity": _clip(50.0 + 500.0 * float(np.mean(complementarity_values)) if complementarity_values else 50.0),
        "factor_overlap": _clip(100.0 * (1.0 - factor_overlap)) if factor_overlap is not None else 50.0,
        "confidence": confidence_by_product[candidate_id]["score"],
    }
    component_weights = {
        "correlation": 0.20,
        "volatility": 0.20,
        "maximum_drawdown": 0.20,
        "crisis_whipsaw_complementarity": 0.15,
        "factor_overlap": 0.10,
        "confidence": 0.15,
    }
    score = sum(components[name] * weight for name, weight in component_weights.items())
    warnings = ["配置分不生成或写入任何自动权重。"]
    if crisis_corr is None:
        warnings.append("危机期共同样本少于 3 期，危机相关性不纳入解释。")
    if factor_overlap is None:
        warnings.append("CTA 因子回归不可用，因子重合度使用中性分。")
    return {
        "status": "available",
        "score": round(float(score), 4),
        "candidate_product_id": candidate_id,
        "candidate_weight": request.candidate_weight,
        "current_portfolio": [position.model_dump(mode="json") for position in request.current_portfolio],
        "metrics": metrics,
        "components": {name: round(value, 4) for name, value in components.items()},
        "component_weights": component_weights,
        "warnings": warnings,
    }


def _attribution_detail(result: RegressionResult) -> dict[str, Any]:
    """Compact, UI-ready subset of the regression output for drill-down.

    Returns a status-bearing envelope so the frontend can distinguish
    "attribution ran but is weak" from "attribution could not run".
    """
    if result.n_observations == 0:
        return {
            "status": "unavailable",
            "warnings": list(result.warnings) or ["因子库未构建，归因不可用。"],
        }
    factors = [
        {
            "name": factor.name,
            "display_name": factor.display_name,
            "beta": factor.beta,
            "t_stat": factor.t_stat,
            "p_value": factor.p_value,
            "significant": factor.significant,
            "factor_group": factor.factor_group,
            "contribution_pct": factor.contribution_pct,
            "mean_return_contribution": factor.mean_return_contribution,
        }
        for factor in result.factors
    ]
    return {
        "status": "available",
        "inference_method": result.inference_method,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "n_observations": result.n_observations,
        "r_squared": result.r_squared,
        "adj_r_squared": result.adj_r_squared,
        "annualized_alpha": result.annualized_alpha,
        "alpha_t_stat": result.alphas_t_stat,
        "alpha_p_value": result.alpha_p_value,
        "alpha_bootstrap_ci": {
            "low": result.annualized_alpha_bootstrap_ci_low,
            "high": result.annualized_alpha_bootstrap_ci_high,
        },
        "factors": factors,
        "factor_group_mean_return_contributions": result.factor_group_mean_return_contributions,
        "return_reconciliation": {
            "mean_product_return": result.mean_product_return,
            "mean_factor_explained_return": result.mean_factor_explained_return,
            "mean_intercept_return": result.intercept,
            "mean_residual_return": (
                round(result.mean_product_return - result.intercept - result.mean_factor_explained_return, 8)
                if all(value is not None for value in (result.mean_product_return, result.intercept, result.mean_factor_explained_return))
                else None
            ),
        },
        "out_of_sample": result.out_of_sample,
        "regime": result.regime,
        "attribution_quality": result.attribution_quality,
        "warnings": list(result.warnings),
    }


def _product_summary(
    item: Any,
    confidence: dict[str, Any],
    regression: RegressionResult,
    returns_analysis: dict[str, Any],
    risk_exposure: dict[str, Any],
    return_score: dict[str, Any],
    risk_score: dict[str, Any],
) -> dict[str, Any]:
    """One-row headline block: the few scores a user sees before drill-down.

    评判标准主线在 v3 扩展为三组：收益分 / 风险暴露分（绝对尺度）与
    质量分 / 置信分 / 归因质量分（证据可信度）。headline 补充收益与
    风险关键值，让第一眼就能读到产品的收益与风险，而不只是回归质量。
    """
    attribution_quality = regression.attribution_quality
    regime = regression.regime
    has_regression = regression.n_observations > 0
    return_metrics = returns_analysis.get("metrics", {})
    risk_statistical = risk_exposure.get("statistical", {})
    return {
        "quality_score": item.score,
        "quality_status": item.status,
        "confidence_score": confidence["score"],
        "confidence_band": confidence["band"],
        "attribution_quality_score": attribution_quality["score"] if attribution_quality else None,
        "attribution_quality_band": attribution_quality["band"] if attribution_quality else None,
        "return_score": return_score["score"],
        "return_band": return_score["band"],
        "risk_score": risk_score["score"],
        "risk_band": risk_score["band"],
        "headline": {
            "annualized_return": return_metrics.get("annualized_return"),
            "cumulative_return": return_metrics.get("cumulative_return"),
            "maximum_drawdown": return_metrics.get("maximum_drawdown"),
            "annualized_volatility": return_metrics.get("annualized_volatility"),
            "var_95": risk_statistical.get("var_95"),
            "r_squared": regression.r_squared if has_regression else None,
            "oos_r_squared": regression.out_of_sample.get("r_squared") if has_regression else None,
            "annualized_alpha": regression.annualized_alpha if has_regression else None,
            "alpha_t_stat": regression.alphas_t_stat if has_regression else None,
            "trending_regime_r_squared": (
                regime["trending"]["r_squared"] if regime else None
            ),
            "choppy_regime_r_squared": (
                regime["choppy"]["r_squared"] if regime else None
            ),
        },
    }


def build_product_score_report(request: CtaProductScoreRequest) -> dict[str, Any]:
    """Build one report whose products expose summary scores plus drill-down detail.

    The report keeps the existing quality/confidence/allocation semantics and
    adds one factor regression per product.  Each product now carries a
    ``summary`` block (the few headline scores) and a ``detail`` block
    (quality explanation, confidence, return analysis, risk exposure,
    attribution, and quality history).  Return analysis and risk exposure
    supply the two absolute-scale evaluation criteria (收益分 / 风险暴露分)
    that the v2 report lacked.
    """
    ranking_request = CtaRankingRequest(
        products=request.products,
        market_series=request.market_series,
        as_of_date=request.as_of_date,
        annual_risk_free_rate=request.annual_risk_free_rate,
        model_version=QUALITY_MODEL_VERSION,
    )
    ranking = rank_cta_products(ranking_request)
    cutoff = request.as_of_date or ranking.as_of_date
    history_dates = _quality_history_dates(request.products, cutoff)
    rankings_by_date: dict[date, Any] = {ranking.as_of_date: ranking}
    for history_date in history_dates:
        if history_date in rankings_by_date:
            continue
        rankings_by_date[history_date] = rank_cta_products(CtaRankingRequest(
            products=request.products,
            market_series=request.market_series,
            as_of_date=history_date,
            annual_risk_free_rate=request.annual_risk_free_rate,
            model_version=QUALITY_MODEL_VERSION,
        ))
    factor_bundle = get_cta_factor_bundle(as_of_date=cutoff)
    by_product = {
        product.product_id: _returns(product, cutoff)
        for product in request.products
    }
    def build_attribution(product: CtaRankingProductInput) -> tuple[str, RegressionResult]:
        returns_by_date = by_product[product.product_id]
        dates = sorted(returns_by_date)
        return product.product_id, run_factor_regression(
            product_returns=np.asarray([returns_by_date[day] for day in dates], dtype=float),
            product_dates=dates,
            frequency=str(product.frequency),
            factor_names=list(CTA_FACTOR_NAMES),
            factor_series_loader=lambda name: get_factor_series(name, as_of_date=cutoff),
            # 批次 13（精度）：bootstrap 索引生成已向量化（200 次约 0.02s），
            # 置信区间从 200 次提到 1000 次，分位数估计更稳定，成本可忽略。
            bootstrap_reps=1000,
            rolling_window=None,
            oos_train_window={"daily": 60, "weekly": 26, "monthly": 24}[str(product.frequency)],
            random_seed=20260811,
        )
    with ThreadPoolExecutor(max_workers=min(_MAX_ATTRIBUTION_WORKERS, len(request.products))) as executor:
        attribution_by_product = dict(executor.map(build_attribution, request.products))
    confidence_by_product = {
        product.product_id: _confidence(
            product,
            by_product[product.product_id],
            factor_bundle=factor_bundle,
            regression=attribution_by_product[product.product_id],
        )
        for product in request.products
    }
    products: list[dict[str, Any]] = []
    inputs_by_id = {product.product_id: product for product in request.products}
    for item in ranking.rankings:
        confidence = confidence_by_product[item.product_id]
        regression = attribution_by_product[item.product_id]
        product_input = inputs_by_id[item.product_id]
        nav_points = [
            point for point in product_input.nav_points
            if point.observation_date <= cutoff
        ]
        nav_points.sort(key=lambda point: point.observation_date)
        nav_values = [point.net_asset_value for point in nav_points]
        nav_dates = [point.observation_date for point in nav_points]
        returns_by_date = by_product[item.product_id]
        returns_array = np.asarray(
            [returns_by_date[day] for day in sorted(returns_by_date)],
            dtype=float,
        )
        returns_analysis = analyze_returns(nav_values, nav_dates, str(product_input.frequency))
        risk_exposure = analyze_risk_exposure(returns_array, regression, str(product_input.frequency))
        return_score = score_return_analysis(returns_analysis)
        risk_score = score_risk_exposure(risk_exposure, frequency=str(product_input.frequency))
        products.append({
            "product_id": item.product_id,
            "product_name": item.product_name,
            "summary": _product_summary(
                item,
                confidence,
                regression,
                returns_analysis,
                risk_exposure,
                return_score,
                risk_score,
            ),
            "detail": {
                "quality_explanation": {
                    "model_version": ranking.model_version,
                    "dimension_weights": ranking.dimension_weights,
                    "dimensions": [dimension.model_dump(mode="json") for dimension in item.dimensions],
                    "warnings": item.warnings,
                },
                "confidence": confidence,
                "returns_analysis": returns_analysis,
                "risk_exposure": risk_exposure,
                "return_score": return_score,
                "risk_score": risk_score,
                "attribution": _attribution_detail(regression),
                "quality_history": _quality_history_for_product(
                    item.product_id,
                    list(rankings_by_date.values()),
                ),
            },
        })
    confidence_map = {item["product_id"]: item["detail"]["confidence"] for item in products}
    allocation = _allocation_score(
        request,
        by_product,
        confidence_map,
        {
            product_id: {factor.name: float(factor.beta) for factor in regression.factors}
            for product_id, regression in attribution_by_product.items()
        },
    )
    return {
        "model_version": PRODUCT_SCORE_MODEL_VERSION,
        "quality_model_version": ranking.model_version,
        "as_of_date": ranking.as_of_date,
        "nav_fingerprint": ranking.nav_fingerprint,
        "universe_size": ranking.universe_size,
        "products": products,
        "allocation": allocation,
        "factor_bundle": factor_bundle,
        "warnings": list(ranking.warnings),
    }
