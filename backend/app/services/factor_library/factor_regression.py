"""Factor regression engine — L1 quantitative profiling.

Takes a product's return series and regresses it against the factor library's
return series to produce statistically testable factor exposures (beta, t, R²)
and an alpha estimate (residual).

This replaces the old "LLM guesses factors" approach with real regression.
The LLM's new role is to *interpret* these results, not to generate them.

Methods:
- Full-sample OLS (statsmodels)
- Rolling-window OLS (time-varying exposures)
- LASSO (factor selection under multicollinearity)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_OBS = 20  # minimum observations for meaningful regression


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FactorBeta:
    """Regression result for a single factor."""

    name: str
    display_name: str
    beta: float
    std_error: float
    t_stat: float
    p_value: float
    contribution_pct: float  # Generic-library display metric; CTA views use mean_return_contribution instead.
    mean_return_contribution: float  # beta * mean(factor_ret), in the product return frequency.
    significant: bool  # |t| > 2
    factor_group: str = "other"
    ordinary_std_error: float = 0.0
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None


@dataclass
class RollingSnapshot:
    """One point in the rolling regression time series."""

    date: str
    r_squared: float
    betas: dict[str, float]


@dataclass
class RegressionResult:
    """Complete L1 factor attribution output."""

    # Full-sample OLS
    intercept: float  # daily alpha
    annualized_alpha: float  # intercept * 252
    alphas_t_stat: float
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    n_observations: int

    # Per-factor details
    factors: list[FactorBeta]

    # Residual analysis
    residual_annual_vol: float
    residual_skew: float
    residual_kurtosis: float

    # Rolling regression
    rolling_r_squared: list[RollingSnapshot] = field(default_factory=list)

    # LASSO factor selection
    lasso_selected: list[str] = field(default_factory=list)

    # Metadata
    frequency: str = "daily"
    start_date: str = ""
    end_date: str = ""
    warnings: list[str] = field(default_factory=list)

    # Statistical reliability and attribution provenance
    inference_method: str = "OLS"
    hac_max_lags: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    bootstrap: dict[str, Any] = field(default_factory=dict)
    factor_groups: dict[str, list[str]] = field(default_factory=dict)
    factor_group_contributions: dict[str, float] = field(default_factory=dict)
    factor_group_mean_return_contributions: dict[str, float] = field(default_factory=dict)
    factor_risk_contributions: dict[str, float | None] = field(default_factory=dict)
    annualized_alpha_bootstrap_ci_low: float | None = None
    annualized_alpha_bootstrap_ci_high: float | None = None
    alpha_p_value: float = 1.0
    joint_hac: dict[str, Any] = field(default_factory=dict)
    collinearity: dict[str, Any] = field(default_factory=dict)
    out_of_sample: dict[str, Any] = field(default_factory=dict)

    # Trend/choppy regime decomposition and the composite attribution-quality
    # score.  Both are None when the aligned sample cannot support them (no
    # trend factor, insufficient observations, or no available components).
    regime: dict[str, Any] | None = None
    attribution_quality: dict[str, Any] | None = None
    mean_product_return: float | None = None
    mean_factor_explained_return: float | None = None


# ---------------------------------------------------------------------------
# Frequency aggregation
# ---------------------------------------------------------------------------

_FREQUENCY_ORDER = {"daily": 0, "weekly": 1, "monthly": 2}


def _aggregate_returns(returns: pd.Series, frequency: str) -> pd.Series:
    """Aggregate daily returns to weekly or monthly frequency.

    Uses compound return: (1+r1)(1+r2)...(1+rn) - 1
    """
    if frequency == "daily":
        return returns

    if frequency == "weekly":
        # Use a PeriodIndex so a Thursday-labelled product observation and a
        # Friday-labelled factor observation still represent the same week.
        periods = returns.index.to_period("W-SUN")
        return returns.groupby(periods).apply(lambda x: (1 + x).prod() - 1).dropna()

    if frequency == "monthly":
        periods = returns.index.to_period("M")
        return returns.groupby(periods).apply(lambda x: (1 + x).prod() - 1).dropna()

    return returns


def _frequency_index(index: pd.Index, frequency: str) -> pd.Index:
    """Normalize product and factor dates to the same economic period."""
    dates = pd.to_datetime(index)
    if frequency == "weekly":
        return dates.to_period("W-SUN")
    if frequency == "monthly":
        return dates.to_period("M")
    return dates.normalize()


def _default_hac_max_lags(frequency: str, n_observations: int) -> int:
    """Choose a conservative HAC lag length for the observation frequency."""
    defaults = {"daily": 5, "weekly": 4, "monthly": 3}
    return min(defaults.get(frequency, 5), max(0, n_observations - 2))


def _default_bootstrap_block_length(frequency: str) -> int:
    """Return a transparent default block length for weakly dependent returns."""
    return {"daily": 20, "weekly": 12, "monthly": 12}.get(frequency, 12)


def _stationary_bootstrap_parameters(
    y: pd.Series,
    X: pd.DataFrame,
    reps: int,
    block_length: int,
    random_seed: int,
) -> np.ndarray:
    """Estimate bootstrap parameter draws using stationary row resampling.

    批次 13（精度）：原实现逐 rep、逐位置用两层 Python 循环生成索引，
    200 次重抽样约 1.1s/产品。这里把索引生成全部向量化——块起始位置用
    np.maximum.accumulate 前向传播，随后 (起始值 + 位置偏移) % n 一次
    算出 (reps, n) 的索引矩阵；最小二乘仍逐 rep 调用 lstsq，保持与原
    实现完全相同的 SVD 数值语义。200 次约 0.02s，1000 次约 0.06s。
    随机流改为批量抽取，同一 seed 下结果确定，但具体数值与旧版本不再
    逐位一致（测试只约束同 seed 可复现）。
    """
    if reps <= 0:
        return np.empty((0, X.shape[1] + 1), dtype=float)

    rng = np.random.default_rng(random_seed)
    y_values = y.to_numpy(dtype=float)
    x_values = X.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(X)), x_values])
    n = len(y_values)
    probability = 1.0 / max(1, block_length)

    # 块起始标记 (reps, n)：p=0 恒为块起始，其 start_values 即首索引。
    new_block = rng.random(size=(reps, n)) < probability
    new_block[:, 0] = True
    positions = np.arange(n)
    # 每个位置所属块的最新起始位置（axis=1 上的 forward-fill）。
    block_start_pos = np.maximum.accumulate(
        np.where(new_block, positions, 0), axis=1
    )
    start_values = rng.integers(0, n, size=(reps, n))
    indices = (
        np.take_along_axis(start_values, block_start_pos, axis=1)
        + (positions - block_start_pos)
    ) % n

    draws: list[np.ndarray] = []
    for rep_indices in indices:
        try:
            params, _, _, _ = np.linalg.lstsq(
                design[rep_indices], y_values[rep_indices], rcond=None
            )
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(params).all():
            draws.append(params)
    return np.asarray(draws, dtype=float).reshape((-1, design.shape[1]))


def _percentile_interval(values: np.ndarray) -> tuple[float | None, float | None]:
    """Return a 95% percentile interval, or nulls for unavailable draws."""
    if values.size == 0:
        return None, None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _residual_diagnostics(residuals: pd.Series) -> dict[str, Any]:
    """Run residual tests without allowing a diagnostic failure to abort attribution."""
    diagnostics: dict[str, Any] = {
        "ljung_box_pvalue_6": None,
        "ljung_box_pvalue_12": None,
        "arch_lm_pvalue_6": None,
        "jarque_bera_pvalue": None,
        "residual_autocorrelation_detected": False,
        "volatility_clustering_detected": False,
        "non_normality_detected": False,
    }
    values = residuals.to_numpy(dtype=float)
    n = len(values)
    if n < 12:
        return diagnostics

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lags = [lag for lag in (6, 12) if lag < n]
        if lags:
            table = acorr_ljungbox(values, lags=lags, return_df=True)
            for lag in lags:
                key = f"ljung_box_pvalue_{lag}"
                diagnostics[key] = float(table.loc[lag, "lb_pvalue"])
            diagnostics["residual_autocorrelation_detected"] = any(
                diagnostics[f"ljung_box_pvalue_{lag}"] is not None
                and diagnostics[f"ljung_box_pvalue_{lag}"] < 0.05
                for lag in lags
            )
    except Exception as exc:  # noqa: BLE001 - diagnostics are supplementary
        diagnostics["ljung_box_error"] = str(exc)

    try:
        from statsmodels.stats.diagnostic import het_arch

        arch_lags = min(6, max(1, n // 10))
        _, pvalue, _, _ = het_arch(values, nlags=arch_lags)
        diagnostics["arch_lm_pvalue_6"] = float(pvalue)
        diagnostics["volatility_clustering_detected"] = float(pvalue) < 0.05
    except Exception as exc:  # noqa: BLE001
        diagnostics["arch_lm_error"] = str(exc)

    try:
        from statsmodels.stats.stattools import jarque_bera

        _, pvalue, _, _ = jarque_bera(values)
        diagnostics["jarque_bera_pvalue"] = float(pvalue)
        diagnostics["non_normality_detected"] = float(pvalue) < 0.05
    except Exception as exc:  # noqa: BLE001
        diagnostics["jarque_bera_error"] = str(exc)

    return diagnostics


def _infer_factor_group(name: str, category: str = "") -> str:
    """Map factor keys to stable economic groups for grouped attribution."""
    if name in {"trend", "short_term_trend_20"}:
        return "trend"
    if name in {"cross_section_mom", "mean_reversion_5d"}:
        return "momentum_reversal"
    if name in {"basis_carry", "term_structure_carry"}:
        return "carry"
    if name in {"profit_margin", "warehouse_receipt", "inventory"}:
        return "fundamental"
    if name in {"volume_price_corr", "skewness"}:
        return "volume_price"
    return category or "other"


def _collinearity_diagnostics(X: pd.DataFrame) -> dict[str, Any]:
    """Summarise factor overlap without treating it as product exposure."""
    if X.empty:
        return {"factor_names": [], "condition_number": None, "max_abs_correlation": None, "high_correlation_pairs": []}
    values = X.to_numpy(dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    try:
        condition_number = float(np.linalg.cond(centered))
    except np.linalg.LinAlgError:
        condition_number = None
    correlation = X.corr().fillna(0.0)
    high_pairs: list[str] = []
    max_abs = 0.0
    for left_index, left in enumerate(X.columns):
        for right in X.columns[left_index + 1:]:
            value = float(correlation.loc[left, right])
            max_abs = max(max_abs, abs(value))
            if abs(value) >= 0.80:
                high_pairs.append(f"{left} ↔ {right} ({value:.2f})")
    return {
        "factor_names": list(X.columns),
        "condition_number": round(condition_number, 4) if condition_number is not None else None,
        "max_abs_correlation": round(max_abs, 4),
        "high_correlation_pairs": high_pairs,
        "correlation_matrix": correlation.round(6).to_dict(),
    }


def _joint_hac_test(robust_model: Any, n_factors: int) -> dict[str, Any]:
    """Test whether all factor slopes are jointly zero under HAC covariance."""
    if n_factors <= 0:
        return {"statistic": None, "p_value": None, "degrees_of_freedom": 0}
    restriction = np.zeros((n_factors, n_factors + 1), dtype=float)
    restriction[:, 1:] = np.eye(n_factors)
    try:
        test = robust_model.wald_test(restriction, use_f=True)
        statistic = float(np.asarray(test.statistic).reshape(-1)[0])
        p_value = float(np.asarray(test.pvalue).reshape(-1)[0])
        return {
            "statistic": round(statistic, 6),
            "p_value": round(p_value, 6),
            "degrees_of_freedom": n_factors,
        }
    except Exception as exc:  # noqa: BLE001 - joint test is supplementary
        return {"statistic": None, "p_value": None, "degrees_of_freedom": n_factors, "error": str(exc)}


def _expanding_walk_forward(
    y: pd.Series,
    X: pd.DataFrame,
    train_window: int,
    annualization_factor: int,
) -> dict[str, Any]:
    """Predict each later observation using only data available beforehand."""
    n = len(y)
    if train_window < _MIN_OBS or n <= train_window:
        return {
            "method": "expanding_window",
            "train_window": train_window,
            "observations": 0,
            "r_squared": None,
            "correlation": None,
            "tracking_error_annual": None,
            "warnings": ["样本外训练窗口不足"],
        }

    y_values = y.to_numpy(dtype=float)
    x_values = X.to_numpy(dtype=float)
    predictions: list[float] = []
    realised: list[float] = []
    for index in range(train_window, n):
        design = np.column_stack([np.ones(index), x_values[:index]])
        try:
            coefficients, _, _, _ = np.linalg.lstsq(design, y_values[:index], rcond=None)
        except np.linalg.LinAlgError:
            continue
        predictions.append(float(np.r_[1.0, x_values[index]] @ coefficients))
        realised.append(float(y_values[index]))

    if not realised:
        return {
            "method": "expanding_window",
            "train_window": train_window,
            "observations": 0,
            "r_squared": None,
            "correlation": None,
            "tracking_error_annual": None,
            "warnings": ["没有可用样本外预测"],
        }

    actual = np.asarray(realised)
    predicted = np.asarray(predictions)
    benchmark_error = np.sum((actual - actual.mean()) ** 2)
    residual_error = np.sum((actual - predicted) ** 2)
    oos_r2 = 1.0 - residual_error / benchmark_error if benchmark_error > 1e-15 else 0.0
    correlation = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else None
    tracking_error = float(np.std(actual - predicted, ddof=1) * np.sqrt(annualization_factor)) if len(actual) > 1 else 0.0
    segments: list[dict[str, Any]] = []
    # Keep chronological order: no random folds and no cherry-picking the
    # best window. Three equal contiguous pieces are the minimum admission
    # evidence for a strong attribution conclusion.
    if len(actual) >= 9:
        for number, positions in enumerate(np.array_split(np.arange(len(actual)), 3), start=1):
            segment_actual = actual[positions]
            segment_predicted = predicted[positions]
            baseline = np.sum((segment_actual - segment_actual.mean()) ** 2)
            segment_r2 = 1.0 - np.sum((segment_actual - segment_predicted) ** 2) / baseline if baseline > 1e-15 else None
            direction = float(np.mean(np.sign(segment_actual) == np.sign(segment_predicted)))
            dates = y.index[train_window:][positions]
            segments.append({
                "segment": number,
                "test_start_date": dates[0].strftime("%Y-%m-%d"),
                "test_end_date": dates[-1].strftime("%Y-%m-%d"),
                "observations": len(positions),
                "r_squared": round(float(segment_r2), 6) if segment_r2 is not None else None,
                "directional_accuracy": round(direction, 6),
            })
    return {
        "method": "expanding_window",
        "train_window": train_window,
        "observations": len(actual),
        "r_squared": round(float(oos_r2), 6),
        "correlation": round(correlation, 6) if correlation is not None else None,
        "tracking_error_annual": round(tracking_error, 6),
        "segments": segments,
        "positive_segment_fraction": round(float(np.mean([item["r_squared"] > 0 for item in segments if item["r_squared"] is not None])), 6) if any(item["r_squared"] is not None for item in segments) else None,
        "warnings": [],
    }


def assess_oos_applicability(out_of_sample: dict[str, Any]) -> dict[str, Any]:
    """Turn OOS diagnostics into one conservative, user-facing conclusion."""
    r_squared = out_of_sample.get("r_squared")
    segments = [item for item in out_of_sample.get("segments", []) if item.get("r_squared") is not None]
    positive_fraction = out_of_sample.get("positive_segment_fraction")
    if r_squared is None or len(segments) < 3:
        return {"status": "observe_only", "reason": "样本外测试段不足 3 段，基础模型只供观察。", "segments_passed": len(segments), "segments_required": 3}
    if r_squared < 0:
        return {"status": "not_applicable", "reason": "样本外 R² 为负，当前 CTA 因子模型不适用。", "segments_passed": len(segments), "segments_required": 3}
    if positive_fraction is None or positive_fraction < 2 / 3:
        return {"status": "not_applicable", "reason": "多数连续样本外测试段未改善基准预测，当前 CTA 因子模型不适用。", "segments_passed": len(segments), "segments_required": 3}
    return {"status": "applicable", "reason": "样本外整体与多数连续测试段均优于历史均值基准。", "segments_passed": len(segments), "segments_required": 3}


# ---------------------------------------------------------------------------
# Regime decomposition and attribution quality
# ---------------------------------------------------------------------------

def _trend_regime_column(X: pd.DataFrame) -> str | None:
    """Pick the factor column that drives the trend/choppy regime split."""
    for preferred in ("trend", "short_term_trend_20"):
        if preferred in X.columns:
            return preferred
    for col in X.columns:
        if _infer_factor_group(col) == "trend":
            return col
    return None


def _regime_trend_indicator(
    trend_returns: pd.Series,
    frequency: str,
) -> tuple[pd.Series, int]:
    """Build a trending-market dummy from a trend factor's trailing |cumulative return|.

    In trending regimes a trend factor accumulates same-sign returns so its
    trailing absolute cumulative return is large; in choppy regimes the
    returns cancel and the value stays small.  A median split turns this
    into a regime dummy using only trailing-window information.
    """
    window = {"daily": 60, "weekly": 13, "monthly": 6}.get(frequency, 60)
    trailing = trend_returns.rolling(window=window, min_periods=1).sum().abs()
    valid = trailing.dropna()
    if valid.empty:
        return pd.Series(dtype=float), window
    median = float(valid.median())
    return (trailing >= median).astype(float), window


def _subset_ols_stats(
    y: pd.Series,
    X: pd.DataFrame,
    annualization_factor: int,
) -> dict[str, Any]:
    """Compact OLS statistics for one regime subset via numpy lstsq."""
    design = np.column_stack([np.ones(len(y)), X.to_numpy(dtype=float)])
    try:
        params, _, _, _ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        return {}
    fitted = design @ params
    residuals = y.to_numpy(dtype=float) - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y.to_numpy(dtype=float) - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return {
        "observations": int(len(y)),
        "r_squared": round(float(np.clip(r_squared, 0.0, 1.0)), 6),
        "annualized_alpha": round(float(params[0]) * annualization_factor, 6),
    }


def _regime_decomposition(
    y: pd.Series,
    X: pd.DataFrame,
    trend_column: str,
    frequency: str,
    min_regime_observations: int = 12,
) -> dict[str, Any] | None:
    """Split the aligned sample into trending and choppy regimes.

    Reports subset R² and the R² gain of adding the regime dummy to the full
    model.  A high trending-regime R² with a low choppy-regime R² is the
    classic CTA payoff shape (trend models earn in trends, give back in
    chops) and should be read as a regime story, not a broken model.
    """
    indicator, window = _regime_trend_indicator(X[trend_column], frequency)
    trending_mask = indicator > 0.5
    choppy_mask = ~trending_mask
    if int(trending_mask.sum()) < min_regime_observations or int(choppy_mask.sum()) < min_regime_observations:
        return None

    annualization_factor = {"daily": 252, "weekly": 52, "monthly": 12}.get(frequency, 252)
    trending_stats = _subset_ols_stats(y[trending_mask], X[trending_mask], annualization_factor)
    choppy_stats = _subset_ols_stats(y[choppy_mask], X[choppy_mask], annualization_factor)
    if not trending_stats or not choppy_stats:
        return None

    base_stats = _subset_ols_stats(y, X, annualization_factor)
    augmented = X.copy()
    augmented["__regime_dummy"] = indicator
    augmented_stats = _subset_ols_stats(y, augmented, annualization_factor)
    gain = (
        float(augmented_stats.get("r_squared") or 0.0)
        - float(base_stats.get("r_squared") or 0.0)
        if augmented_stats
        else None
    )

    return {
        "method": "trailing_abs_cumulative_return_median_split",
        "regime_factor": trend_column,
        "window": window,
        "trending_share": round(float(trending_mask.mean()), 4),
        "trending": trending_stats,
        "choppy": choppy_stats,
        "augmented": {
            "r_squared": augmented_stats.get("r_squared"),
            "r_squared_gain": round(gain, 6) if gain is not None else None,
        },
    }


def _attribution_quality_score(
    *,
    alphas_t_stat: float,
    annualized_alpha_ci_low: float | None,
    annualized_alpha_ci_high: float | None,
    oos_r_squared: float | None,
    n_significant_factors: int,
    n_observations: int,
    frequency: str,
) -> dict[str, Any] | None:
    """Aggregate regression diagnostics into one 0-100 attribution-quality score.

    Components are renormalized over whatever evidence is available, so a
    missing bootstrap interval or an unavailable walk-forward sample degrades
    gracefully instead of aborting the report.
    """
    minimum_observations = {"daily": 252, "weekly": 52, "monthly": 36}.get(frequency, 252)

    # A statistically distinct alpha means the public factor model still
    # leaves a systematic part of return unexplained.  It must reduce—not
    # improve—the attribution-quality score.
    alpha_significance = 100.0 * float(np.clip(1.0 - abs(alphas_t_stat) / 2.0, 0.0, 1.0))

    # Where does zero sit inside the annualized bootstrap interval?  An
    # interval that includes zero supports the view that the public factor
    # model has not left a persistent alpha.  An interval excluding zero is
    # therefore scored 0.
    if (
        annualized_alpha_ci_low is not None
        and annualized_alpha_ci_high is not None
        and annualized_alpha_ci_high > annualized_alpha_ci_low
    ):
        if annualized_alpha_ci_low >= 0.0 or annualized_alpha_ci_high <= 0.0:
            alpha_bootstrap = 0.0
        else:
            position = annualized_alpha_ci_high / (annualized_alpha_ci_high - annualized_alpha_ci_low)
            alpha_bootstrap = 100.0 * float(np.clip(2.0 * min(position, 1.0 - position), 0.0, 1.0))
    else:
        alpha_bootstrap = None

    # An out-of-sample R² of 0.10 or above is strong for CTA products.
    oos_predictability = (
        100.0 * float(np.clip(float(oos_r_squared) / 0.10, 0.0, 1.0))
        if oos_r_squared is not None
        else None
    )
    factor_significance = 100.0 * float(min(1.0, n_significant_factors / 2.0))
    sample_coverage = 100.0 * float(np.clip(n_observations / minimum_observations, 0.0, 1.0))

    components: dict[str, float | None] = {
        "alpha_significance": round(alpha_significance, 4),
        "alpha_bootstrap": round(alpha_bootstrap, 4) if alpha_bootstrap is not None else None,
        "oos_predictability": round(oos_predictability, 4) if oos_predictability is not None else None,
        "factor_significance": round(factor_significance, 4),
        "sample_coverage": round(sample_coverage, 4),
    }
    weights = {
        "alpha_significance": 0.30,
        "alpha_bootstrap": 0.20,
        "oos_predictability": 0.20,
        "factor_significance": 0.15,
        "sample_coverage": 0.15,
    }
    available = {name: weight for name, weight in weights.items() if components[name] is not None}
    if not available:
        return None
    weight_total = sum(available.values())
    effective = {name: weight / weight_total for name, weight in available.items()}
    score = sum(float(components[name]) * effective[name] for name in effective)
    warnings = []
    if components["alpha_bootstrap"] is None:
        warnings.append("Bootstrap 置信区间不可用，Alpha 区间位置不纳入归因质量分。")
    if components["oos_predictability"] is None:
        warnings.append("样本外走步验证不可用，OOS 可预测性不纳入归因质量分。")
    band = "high" if score >= 75 else "medium" if score >= 50 else "low"
    return {
        "score": round(float(score), 4),
        "band": band,
        "components": components,
        "weights": {name: round(weight, 4) for name, weight in effective.items()},
        "method": "attribution_quality_v1",
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Core regression
# ---------------------------------------------------------------------------

def run_factor_regression(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str = "daily",
    factor_names: list[str] | None = None,
    factor_series_loader: Callable[[str], pd.Series | None] | None = None,
    rolling_window: int | None = None,
    hac_max_lags: int | None = None,
    bootstrap_reps: int = 1000,
    bootstrap_block_length: int | None = None,
    random_seed: int = 20260811,
    oos_train_window: int | None = None,
    minimum_observations: int = _MIN_OBS,
) -> RegressionResult:
    """Run L1 factor attribution regression.

    Parameters
    ----------
    product_returns : array of periodic returns
    product_dates : corresponding observation dates
    frequency : "daily" | "weekly" | "monthly"
    factor_names : which factors to include (None = all cached)
    factor_series_loader : optional public factor loader; when supplied, it
        replaces the legacy factor-library loader for the selected names
    rolling_window : if set, run rolling regression with this many periods
    hac_max_lags : Newey-West lag length; None selects a frequency-based default
    bootstrap_reps : stationary-bootstrap repetitions; zero disables bootstrap
    bootstrap_block_length : expected bootstrap block length
    random_seed : deterministic seed for the bootstrap confidence intervals
    oos_train_window : initial expanding-window training length; None uses a
        frequency-based default

    Returns
    -------
    RegressionResult with full-sample OLS, rolling, and LASSO results.
    """
    from app.services import factor_library

    warnings: list[str] = []
    if bootstrap_reps < 0:
        raise ValueError("bootstrap_reps cannot be negative")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2")
    if bootstrap_block_length is not None and bootstrap_block_length < 1:
        raise ValueError("bootstrap_block_length must be positive")
    if oos_train_window is not None and oos_train_window < _MIN_OBS:
        raise ValueError(f"oos_train_window must be >= {_MIN_OBS}")

    # Build product return series
    y_raw = pd.Series(
        product_returns,
        index=pd.to_datetime(product_dates),
        name="product",
    ).sort_index()

    # Load factor returns from cache
    all_factors = factor_library.get_all_factors()
    if factor_names is None:
        factor_names = [f["name"] for f in all_factors]

    # Metadata lookup also tells us whether a factor can be aligned to the
    # product frequency without inventing higher-frequency observations.
    metadata_by_name = {f["name"]: f for f in all_factors}
    display_names = {name: metadata["display_name"] for name, metadata in metadata_by_name.items()}

    # Load and aggregate each factor
    factor_series: dict[str, pd.Series] = {}
    for name in factor_names:
        factor_frequency = metadata_by_name.get(name, {}).get("frequency", "daily")
        if _FREQUENCY_ORDER.get(factor_frequency, 0) > _FREQUENCY_ORDER.get(frequency, 0):
            warnings.append(
                f"因子 '{name}' 为 {factor_frequency} 频率，不能用于 {frequency} 归因，已跳过。"
            )
            continue
        # Attribution intentionally uses baseline returns.  Risk overlays are
        # strategy-level execution choices and must not alter the economic
        # definition of a factor beta.
        raw = (
            factor_series_loader(name)
            if factor_series_loader is not None
            else factor_library.get_factor_series(name, risk_profile="baseline")
        )
        if raw is None or raw.empty:
            warnings.append(f"因子 '{name}' 未构建，跳过。请先调用 /api/factor-library/build")
            continue
        agg = _aggregate_returns(raw, frequency)
        if len(agg) > 0:
            factor_series[name] = agg

    if not factor_series:
        return RegressionResult(
            intercept=0, annualized_alpha=0, alphas_t_stat=0,
            r_squared=0, adj_r_squared=0, f_statistic=0, f_p_value=1,
            n_observations=0, factors=[], residual_annual_vol=0,
            residual_skew=0, residual_kurtosis=0,
            frequency=frequency, warnings=warnings,
            inference_method="OLS + HAC(Newey-West)",
            hac_max_lags=hac_max_lags,
            bootstrap={"reps": 0, "block_length": None, "seed": random_seed, "successful_reps": 0},
            joint_hac={"statistic": None, "p_value": None, "degrees_of_freedom": 0},
            collinearity={"factor_names": [], "condition_number": None, "max_abs_correlation": None, "high_correlation_pairs": []},
            out_of_sample={"method": "expanding_window", "train_window": oos_train_window or 0, "observations": 0, "r_squared": None, "correlation": None, "tracking_error_annual": None, "warnings": ["没有可用因子"]},
        )

    # Align by economic period.  Product NAV observations and market-factor
    # observations are often labelled on different weekdays/month-end dates;
    # matching raw timestamps would undercount the overlap for weekly/monthly
    # products.
    X_raw = pd.DataFrame(factor_series)
    # Compound product returns within each economic period.  Multiple NAV
    # observations can share one week/month and the earlier ones must not be
    # discarded by a last-value groupby.
    y_periodic = _aggregate_returns(y_raw, frequency)
    common_idx = y_periodic.index.intersection(X_raw.index)

    if len(common_idx) < minimum_observations:
        warnings.append(
            f"对齐后仅 {len(common_idx)} 个观测值（需 >= {minimum_observations}），"
            f"请确认产品净值日期范围与因子库重叠充分。"
        )
        return RegressionResult(
            intercept=0, annualized_alpha=0, alphas_t_stat=0,
            r_squared=0, adj_r_squared=0, f_statistic=0, f_p_value=1,
            n_observations=len(common_idx), factors=[],
            residual_annual_vol=0, residual_skew=0, residual_kurtosis=0,
            frequency=frequency, warnings=warnings,
            inference_method="OLS + HAC(Newey-West)",
            hac_max_lags=hac_max_lags,
            bootstrap={"reps": 0, "block_length": None, "seed": random_seed, "successful_reps": 0},
            joint_hac={"statistic": None, "p_value": None, "degrees_of_freedom": 0},
            collinearity={"factor_names": list(X_raw.columns), "condition_number": None, "max_abs_correlation": None, "high_correlation_pairs": []},
            out_of_sample={"method": "expanding_window", "train_window": oos_train_window or 0, "observations": 0, "r_squared": None, "correlation": None, "tracking_error_annual": None, "warnings": ["对齐后样本不足"]},
        )

    y = y_periodic.loc[common_idx]
    X = X_raw.loc[common_idx]

    # Drop factors that are all-NaN in the aligned window
    valid_cols = X.columns[X.notna().sum() >= minimum_observations]
    if len(valid_cols) < len(X.columns):
        dropped = set(X.columns) - set(valid_cols)
        warnings.append(f"以下因子在对齐窗口内数据不足，已排除: {', '.join(dropped)}")
        X = X[valid_cols]

    # Missing factor observations are not zero returns.  Remove incomplete
    # economic periods so the model never fabricates a factor value.
    missing_rows = X.isna().any(axis=1)
    if missing_rows.any():
        warnings.append(f"因子对齐窗口内有 {int(missing_rows.sum())} 个观测缺失，已剔除这些期间。")
        y = y.loc[~missing_rows]
        X = X.loc[~missing_rows]
    if len(y) < minimum_observations or X.empty:
        warnings.append(
            f"剔除缺失因子后仅剩 {len(y)} 个观测值（需 >= {minimum_observations}）。"
        )
        return RegressionResult(
            intercept=0, annualized_alpha=0, alphas_t_stat=0,
            r_squared=0, adj_r_squared=0, f_statistic=0, f_p_value=1,
            n_observations=len(y), factors=[], residual_annual_vol=0,
            residual_skew=0, residual_kurtosis=0,
            frequency=frequency, warnings=warnings,
            inference_method="OLS + HAC(Newey-West)",
            hac_max_lags=hac_max_lags,
            bootstrap={"reps": 0, "block_length": None, "seed": random_seed, "successful_reps": 0},
            joint_hac={"statistic": None, "p_value": None, "degrees_of_freedom": 0},
            collinearity={"factor_names": list(X.columns), "condition_number": None, "max_abs_correlation": None, "high_correlation_pairs": []},
            out_of_sample={"method": "expanding_window", "train_window": oos_train_window or 0, "observations": 0, "r_squared": None, "correlation": None, "tracking_error_annual": None, "warnings": ["缺失因子导致样本不足"]},
        )

    # Annualization factor
    ann_factor = {"daily": 252, "weekly": 52, "monthly": 12}.get(frequency, 252)
    effective_hac_lags = (
        _default_hac_max_lags(frequency, len(y))
        if hac_max_lags is None
        else min(int(hac_max_lags), max(0, len(y) - 2))
    )
    effective_block_length = bootstrap_block_length or _default_bootstrap_block_length(frequency)
    effective_oos_train_window = oos_train_window or {"daily": 60, "weekly": 26, "monthly": 24}.get(frequency, 60)

    # --- Full-sample OLS ---
    import statsmodels.api as sm

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    try:
        robust_model = model.get_robustcov_results(
            cov_type="HAC",
            maxlags=effective_hac_lags,
            use_correction=True,
        )
    except Exception as exc:  # noqa: BLE001 - retain OLS result if HAC is unavailable
        robust_model = model
        warnings.append(f"HAC 协方差估计失败，暂使用普通 OLS 标准误：{exc}")
        inference_method = "OLS"
    else:
        inference_method = "OLS + HAC(Newey-West)"

    parameter_names = list(X_const.columns)
    robust_bse = dict(zip(parameter_names, np.asarray(robust_model.bse, dtype=float)))
    robust_tvalues = dict(zip(parameter_names, np.asarray(robust_model.tvalues, dtype=float)))
    robust_pvalues = dict(zip(parameter_names, np.asarray(robust_model.pvalues, dtype=float)))

    intercept = model.params.get("const", 0.0)
    annualized_alpha = intercept * ann_factor

    factor_betas: list[FactorBeta] = []
    factor_groups: dict[str, list[str]] = {}
    mean_product_ret = y.mean()
    mean_factor_explained_return = 0.0
    for col in X.columns:
        beta_val = model.params[col]
        ordinary_se = float(model.bse[col])
        se = float(robust_bse.get(col, ordinary_se))
        t = float(robust_tvalues.get(col, model.tvalues[col]))
        p = float(robust_pvalues.get(col, model.pvalues[col]))
        # Contribution: how much of product's mean return this factor explains
        factor_mean = X[col].mean()
        if abs(mean_product_ret) > 1e-10:
            contrib = (beta_val * factor_mean) / mean_product_ret * 100
        else:
            contrib = 0.0
        mean_factor_explained_return += float(beta_val * factor_mean)

        metadata = metadata_by_name.get(col, {})
        factor_group = metadata.get("factor_group") or _infer_factor_group(
            col, metadata.get("category", "other")
        )
        factor_groups.setdefault(factor_group, []).append(col)
        factor_betas.append(FactorBeta(
            name=col,
            display_name=display_names.get(col, col),
            beta=round(float(beta_val), 6),
            std_error=round(float(se), 6),
            t_stat=round(float(t), 4),
            p_value=round(float(p), 6),
            contribution_pct=round(float(contrib), 2),
            mean_return_contribution=round(float(beta_val * factor_mean), 8),
            significant=abs(t) > 2.0,
            factor_group=factor_group,
            ordinary_std_error=round(ordinary_se, 6),
        ))

    # Residual analysis
    residuals = model.resid
    resid_vol = float(residuals.std() * np.sqrt(ann_factor))
    resid_skew = float(residuals.skew())
    resid_kurt = float(residuals.kurtosis())

    diagnostics = _residual_diagnostics(residuals)
    collinearity = _collinearity_diagnostics(X)
    joint_hac = _joint_hac_test(robust_model, len(X.columns))
    out_of_sample = _expanding_walk_forward(
        y,
        X,
        train_window=effective_oos_train_window,
        annualization_factor=ann_factor,
    )
    if diagnostics.get("residual_autocorrelation_detected"):
        warnings.append("残差存在显著自相关；已使用 HAC 标准误，静态 Beta 可能未捕捉时变暴露。")
    if diagnostics.get("volatility_clustering_detected"):
        warnings.append("残差存在波动聚集；Alpha 候选的稳定性需结合状态模型和压力测试。")
    if diagnostics.get("non_normality_detected"):
        warnings.append("残差未通过正态性检验；置信区间应优先参考区块 Bootstrap。")
    if collinearity.get("high_correlation_pairs"):
        warnings.append("因子之间存在较高相关性；单个 Beta 可能不稳定，应优先参考经济因子组贡献。")
    if out_of_sample.get("observations", 0) == 0:
        warnings.extend(out_of_sample.get("warnings", []))

    bootstrap_draws = _stationary_bootstrap_parameters(
        y,
        X,
        reps=bootstrap_reps,
        block_length=effective_block_length,
        random_seed=random_seed,
    )
    alpha_ci_low, alpha_ci_high = _percentile_interval(bootstrap_draws[:, 0] if len(bootstrap_draws) else np.array([]))
    if bootstrap_reps > 0 and len(bootstrap_draws) < max(20, bootstrap_reps // 2):
        warnings.append(
            f"区块 Bootstrap 仅成功 {len(bootstrap_draws)}/{bootstrap_reps} 次；置信区间可靠性有限。"
        )
    for index, factor in enumerate(factor_betas, start=1):
        low, high = _percentile_interval(bootstrap_draws[:, index] if len(bootstrap_draws) else np.array([]))
        factor.bootstrap_ci_low = round(low, 6) if low is not None else None
        factor.bootstrap_ci_high = round(high, 6) if high is not None else None

    group_contributions: dict[str, float] = {}
    group_mean_return_contributions: dict[str, float] = {}
    for factor in factor_betas:
        group_contributions[factor.factor_group] = round(
            group_contributions.get(factor.factor_group, 0.0) + factor.contribution_pct,
            2,
        )
        group_mean_return_contributions[factor.factor_group] = round(
            group_mean_return_contributions.get(factor.factor_group, 0.0) + factor.mean_return_contribution,
            8,
        )

    covariance = X.cov().to_numpy(dtype=float)
    beta_vector = np.asarray([float(model.params[column]) for column in X.columns])
    factor_variance = float(beta_vector @ covariance @ beta_vector)
    if factor_variance > 1e-18:
        factor_volatility = float(np.sqrt(factor_variance))
        marginal_risk = covariance @ beta_vector
        factor_risk_contributions = {
            column: round(float(beta * marginal / factor_volatility), 8)
            for column, beta, marginal in zip(X.columns, beta_vector, marginal_risk, strict=True)
        }
    else:
        factor_risk_contributions = {column: None for column in X.columns}

    # --- Regime decomposition and attribution quality ---
    regime_column = _trend_regime_column(X)
    regime: dict[str, Any] | None = None
    if regime_column is not None:
        regime = _regime_decomposition(y, X, regime_column, frequency)
        if regime is None:
            warnings.append("趋势/震荡期拆分因单侧样本不足 12 期未生成。")

    attribution_quality = _attribution_quality_score(
        alphas_t_stat=float(robust_tvalues.get("const", model.tvalues.get("const", 0))),
        annualized_alpha_ci_low=alpha_ci_low * ann_factor if alpha_ci_low is not None else None,
        annualized_alpha_ci_high=alpha_ci_high * ann_factor if alpha_ci_high is not None else None,
        oos_r_squared=out_of_sample.get("r_squared"),
        n_significant_factors=sum(1 for factor in factor_betas if factor.significant),
        n_observations=int(model.nobs),
        frequency=frequency,
    )

    result = RegressionResult(
        intercept=round(float(intercept), 8),
        annualized_alpha=round(float(annualized_alpha), 6),
        alphas_t_stat=round(float(robust_tvalues.get("const", model.tvalues.get("const", 0))), 4),
        r_squared=round(float(model.rsquared), 6),
        adj_r_squared=round(float(model.rsquared_adj), 6),
        f_statistic=round(float(getattr(robust_model, "fvalue", model.fvalue)), 4),
        f_p_value=round(float(getattr(robust_model, "f_pvalue", model.f_pvalue)), 6),
        n_observations=int(model.nobs),
        factors=factor_betas,
        residual_annual_vol=round(resid_vol, 6),
        residual_skew=round(resid_skew, 4),
        residual_kurtosis=round(resid_kurt, 4),
        frequency=frequency,
        start_date=common_idx[0].strftime("%Y-%m-%d"),
        end_date=common_idx[-1].strftime("%Y-%m-%d"),
        warnings=warnings,
        inference_method=inference_method,
        hac_max_lags=effective_hac_lags,
        diagnostics=diagnostics,
        bootstrap={
            "method": "stationary bootstrap",
            "reps": bootstrap_reps,
            "successful_reps": int(len(bootstrap_draws)),
            "block_length": effective_block_length,
            "seed": random_seed,
            "confidence_level": 0.95,
        },
        factor_groups=factor_groups,
        factor_group_contributions=group_contributions,
        factor_group_mean_return_contributions=group_mean_return_contributions,
        factor_risk_contributions=factor_risk_contributions,
        annualized_alpha_bootstrap_ci_low=round(alpha_ci_low * ann_factor, 6) if alpha_ci_low is not None else None,
        annualized_alpha_bootstrap_ci_high=round(alpha_ci_high * ann_factor, 6) if alpha_ci_high is not None else None,
        alpha_p_value=round(float(robust_pvalues.get("const", model.pvalues.get("const", 1.0))), 6),
        joint_hac=joint_hac,
        collinearity=collinearity,
        out_of_sample=out_of_sample,
        regime=regime,
        attribution_quality=attribution_quality,
        mean_product_return=round(float(mean_product_ret), 8),
        mean_factor_explained_return=round(float(mean_factor_explained_return), 8),
    )

    # --- Rolling regression ---
    if rolling_window and rolling_window >= _MIN_OBS and len(common_idx) >= rolling_window + 5:
        result.rolling_r_squared = _rolling_regression(
            y, X, rolling_window, display_names
        )

    # --- LASSO factor selection ---
    if len(X.columns) >= 3:
        result.lasso_selected = _lasso_selection(y, X)

    return result


def _rolling_regression(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
    display_names: dict[str, str],
) -> list[RollingSnapshot]:
    """Run rolling OLS and return time-varying R² and betas.

    Uses numpy lstsq directly (10-50x faster than statsmodels per window)
    since we only need coefficients and R², not p-values or diagnostics.
    """
    snapshots: list[RollingSnapshot] = []
    dates = y.index
    columns = list(X.columns)

    y_arr = y.values
    X_arr = X.values

    for i in range(window, len(dates)):
        y_win = y_arr[i - window:i]
        X_win = X_arr[i - window:i]

        # Design matrix with intercept
        X_design = np.column_stack([np.ones(window), X_win])

        try:
            beta_hat, _, _, _ = np.linalg.lstsq(X_design, y_win, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # R² = 1 - SS_res / SS_tot
        y_hat = X_design @ beta_hat
        ss_res = np.sum((y_win - y_hat) ** 2)
        ss_tot = np.sum((y_win - y_win.mean()) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        betas = {col: round(float(beta_hat[j + 1]), 4) for j, col in enumerate(columns)}
        snapshots.append(RollingSnapshot(
            date=dates[i].strftime("%Y-%m-%d"),
            r_squared=round(float(r_sq), 4),
            betas=betas,
        ))

    return snapshots


def _lasso_selection(y: pd.Series, X: pd.DataFrame) -> list[str]:
    """Use LassoCV to select the most relevant factors."""
    try:
        from sklearn.linear_model import LassoCV

        lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso.fit(X.values, y.values)

        selected = [
            col for col, coef in zip(X.columns, lasso.coef_)
            if abs(coef) > 1e-6
        ]
        return selected
    except Exception as exc:
        logger.debug("LASSO selection failed: %s", exc)
        return []
