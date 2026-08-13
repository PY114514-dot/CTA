"""Step 2: Factor exposure analysis.

Constructs factor proxy variables from market data and estimates the
product's exposure to each factor via OLS regression, rolling windows,
and segmented analysis.

Covered factors:
- Beta (market): broad market index returns
- Momentum / Trend-following: time-series momentum signal
- Mean Reversion: short-term reversal
- Carry / Term Structure: roll-yield proxy (commodity) or N/A (equity)
- Volatility: realized-vol regime change
- Fundamental (equity only): value/growth proxy (optional, future)
"""

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.services.market_data.provider import MarketDataProvider
from app.services.market_data.index_registry import INDEX_BENCHMARKS, FUTURES_VARIETIES
from app.services.analysis._stats import safe_pearson
from app.services.analysis._pipeline_utils import align_product_and_market_returns

logger = logging.getLogger(__name__)

_MIN_OBS = 12  # minimum observations for regression
_ROLLING_MIN = 20  # minimum window for rolling regression


@dataclass
class FactorExposureDetail:
    """Exposure estimate for a single factor."""

    factor_name: str
    factor_label: str  # Chinese display name
    exposure_beta: float  # OLS coefficient
    correlation: float  # simple Pearson correlation
    t_statistic: float
    p_value: float
    confidence_pct: float  # 0-100
    confidence_label: str  # "高" | "中" | "低"
    # Walk-forward direction agreement.  This is deliberately not called a
    # trading win rate: it tests whether a factor with a beta estimated only
    # from prior observations has the same sign as the realised product return.
    out_of_sample_hit_rate: float | None = None
    hit_rate_observations: int = 0
    hit_rate_p_value: float | None = None
    evidence: list[str] = field(default_factory=list)
    rolling_exposure: list[float] = field(default_factory=list)


@dataclass
class FactorCollinearityDiagnostics:
    """Numerical diagnostics for interpreting multi-factor estimates.

    These diagnostics describe the factor proxies, not the product's actual
    holdings.  When proxies move together, an individual beta can be unstable
    even when the model's overall fit looks reasonable.
    """

    factor_names: list[str] = field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    covariance_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    condition_number: float | None = None
    max_abs_correlation: float | None = None
    high_correlation_pairs: list[str] = field(default_factory=list)
    ridge_applied: bool = False
    ridge_alpha: float | None = None


@dataclass
class FactorRiskProfile:
    """Stress and variance diagnostics derived from factor proxy returns.

    This is a model-based risk portrait.  It does not identify actual
    positions, nor does it forecast the product's future return.
    """

    downside_betas: dict[str, float | None] = field(default_factory=dict)
    tail_correlations: dict[str, float | None] = field(default_factory=dict)
    regime_returns: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    variance_contributions_pct: dict[str, float | None] = field(default_factory=dict)
    principal_components: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FactorAnalysisResult:
    """Complete factor analysis output."""

    factors: list[FactorExposureDetail]
    r_squared: float
    adj_r_squared: float
    rolling_r_squared: list[float] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    collinearity: FactorCollinearityDiagnostics = field(default_factory=FactorCollinearityDiagnostics)
    risk_profile: FactorRiskProfile = field(default_factory=FactorRiskProfile)
    # Aligned observations used by the regression.  These are intentionally
    # returned as chart-ready primitives so the report can explain *how* a
    # factor contributed instead of only showing one beta number.
    explainability_series: list[dict] = field(default_factory=list)


# Factor display names
FACTOR_LABELS = {
    "beta": "市场Beta",
    "momentum": "动量/趋势跟踪",
    "mean_reversion": "均值回归",
    "carry": "Carry/期限结构",
    "volatility": "波动率",
}


def analyze_factors(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    strategy_type: str,
    provider: MarketDataProvider,
    rolling_window: int = 12,
) -> FactorAnalysisResult:
    """Run full factor exposure analysis.

    Args:
        product_returns: Periodic returns array.
        product_dates: Corresponding dates.
        frequency: "daily" | "weekly" | "monthly".
        strategy_type: From Step 1 classification.
        provider: Market data provider.
        rolling_window: Window size for rolling regression (in periods).

    Returns:
        FactorAnalysisResult with per-factor details.
    """
    n = len(product_returns)
    if n < _MIN_OBS:
        return FactorAnalysisResult(
            factors=[],
            r_squared=0.0,
            adj_r_squared=0.0,
            warnings=[f"样本量 {n} 期不足 {_MIN_OBS} 期，无法进行因子回归"],
        )

    start = min(product_dates)
    end = max(product_dates)

    # Buffer the start date back by one period so benchmark returns have
    # enough data points to align with the product return series.
    from datetime import timedelta
    buffer = {"daily": timedelta(days=5), "weekly": timedelta(weeks=2), "monthly": timedelta(days=35)}
    buffered_start = start - buffer.get(frequency, timedelta(days=5))

    # Build factor proxy matrix
    factor_matrix = _build_factor_matrix(
        product_returns, product_dates, frequency, strategy_type, provider, buffered_start, end
    )

    if factor_matrix.empty or factor_matrix.shape[1] == 0:
        return FactorAnalysisResult(
            factors=[],
            r_squared=0.0,
            adj_r_squared=0.0,
            warnings=["无法构建任何因子代理变量，请检查行情数据可用性"],
        )

    # The factor builder stores the matching product observations after an
    # actual date intersection.  Do not trim the two arrays positionally.
    aligned_product = factor_matrix.attrs.get("aligned_product")
    if aligned_product is None:
        return FactorAnalysisResult(
            factors=[], r_squared=0.0, adj_r_squared=0.0,
            warnings=["外部因子与产品净值没有足够的实际日期交集，已停止因子解释"],
        )
    aligned_product = np.asarray(aligned_product, dtype=float)
    factor_matrix = factor_matrix.drop(columns=["__product__"], errors="ignore")
    if len(aligned_product) < _MIN_OBS:
        return FactorAnalysisResult(
            factors=[],
            r_squared=0.0,
            adj_r_squared=0.0,
            warnings=["对齐后样本不足"],
        )

    # A covariance/correlation check happens before regression so coefficient
    # precision is not mistaken for evidence when factor proxies overlap.
    collinearity = _build_collinearity_diagnostics(factor_matrix)

    # OLS regression remains the full-sample statistical reference (including
    # t/p values).  Ridge is used only to stabilise the rolling trend below.
    factors_detail = _run_ols_regression(aligned_product, factor_matrix)
    risk_profile = _build_risk_profile(aligned_product, factor_matrix, factors_detail)

    # Rolling regression
    effective_window = max(rolling_window, _ROLLING_MIN)
    if collinearity.ridge_applied:
        rolling_results = _run_ridge_rolling_regression(
            aligned_product, factor_matrix, effective_window, collinearity.ridge_alpha or 1.0
        )
    else:
        rolling_results = _run_rolling_regression(aligned_product, factor_matrix, effective_window)

    # Attach rolling exposure to factor details
    for fd in factors_detail:
        if fd.factor_name in rolling_results:
            fd.rolling_exposure = rolling_results[fd.factor_name]

    # Overall model fit
    r_sq, adj_r_sq, rolling_r_sq = _compute_model_fit(aligned_product, factor_matrix, effective_window)

    explainability_series = _build_explainability_series(
        aligned_product,
        factor_matrix,
        factor_matrix.attrs.get("aligned_dates", []),
    )

    # Generate evidence summary
    evidence = _generate_evidence(factors_detail, r_sq)
    warnings = []
    warnings.extend(risk_profile.warnings)
    if n < 30:
        warnings.append(f"样本仅 {n} 期，因子暴露估计可能不稳定")
    if r_sq < 0.3:
        warnings.append(f"模型整体解释力较低（R²={r_sq:.3f}），可能存在未捕获的因子")
    if collinearity.ridge_applied:
        warnings.append(
            "因子代理存在较强共线性，单个 Beta 的归因不稳定；滚动风格趋势已采用岭回归稳定化，仅作参考。"
        )

    return FactorAnalysisResult(
        factors=factors_detail,
        r_squared=round(r_sq, 4),
        adj_r_squared=round(adj_r_sq, 4),
        rolling_r_squared=rolling_r_sq,
        evidence=evidence,
        warnings=warnings,
        collinearity=collinearity,
        risk_profile=risk_profile,
        explainability_series=explainability_series,
    )


# ---------------------------------------------------------------------------
# Factor proxy construction
# ---------------------------------------------------------------------------


def _build_factor_matrix(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    strategy_type: str,
    provider: MarketDataProvider,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Build factors from one dated benchmark series and exact period matches.

    Earlier code independently trimmed several arrays to their last ``N``
    values.  That silently paired different weeks when the market feed had a
    holiday or a missing date.  We now align the benchmark first, then derive
    all proxies from that already-aligned period series.
    """
    symbol = "nh_commodity" if strategy_type == "commodity_cta" else "hs300"
    try:
        raw_returns = provider.get_returns(symbol, start, end, is_index=True)
    except Exception as exc:
        logger.warning("factor benchmark fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()
    aligned = align_product_and_market_returns(product_returns, product_dates, raw_returns, frequency)
    if aligned is None:
        return pd.DataFrame()
    aligned_product, beta = aligned
    base = pd.Series(beta, dtype=float)
    frame = pd.DataFrame({"beta": base})

    periods = {"daily": 20, "weekly": 4, "monthly": 3}.get(frequency, 4)
    past = (1.0 + base).rolling(periods, min_periods=periods).apply(np.prod, raw=True) - 1.0
    frame["momentum"] = np.sign(past.shift(1)) * base
    frame["mean_reversion"] = -past.shift(1)
    realized = base.rolling(periods, min_periods=periods).std()
    frame["volatility"] = realized.shift(1)

    valid = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.DataFrame()
    frame = valid.reset_index(drop=True)
    frame.attrs["aligned_product"] = aligned_product[valid.index.to_numpy()]
    # Keep the actual product-period labels through the rolling-feature
    # warm-up.  Positional trimming here would silently mislabel holidays or
    # missing disclosure weeks.
    aligned_dates = _aligned_product_dates(product_returns, product_dates, raw_returns, frequency)
    if aligned_dates:
        frame.attrs["aligned_dates"] = [aligned_dates[i].isoformat() for i in valid.index.to_numpy() if i < len(aligned_dates)]
    return frame


def _aligned_product_dates(
    product_returns: np.ndarray,
    product_dates: list[date],
    market_returns: pd.Series,
    frequency: str,
) -> list[date]:
    """Return product observation dates at the exact market intersection.

    ``align_product_and_market_returns`` deliberately returns only arrays for
    backwards compatibility.  This companion helper mirrors its calendar
    bucketing solely to retain labels for the explainability chart.
    """
    if market_returns.empty or len(product_returns) != len(product_dates):
        return []
    product_index = pd.to_datetime(product_dates)
    product = pd.Series(product_index, index=product_index)
    market = market_returns.copy()
    market.index = pd.to_datetime(market.index)
    market = market[~market.index.duplicated(keep="last")].sort_index()
    if frequency == "weekly":
        product = product.groupby(product.index.to_period("W-SUN")).last()
        market_index = ((1.0 + market).resample("W-SUN").prod() - 1.0).index.to_period("W-SUN")
        common = product.index.intersection(market_index)
    elif frequency == "monthly":
        product = product.groupby(product.index.to_period("M")).last()
        market_index = ((1.0 + market).resample("ME").prod() - 1.0).index.to_period("M")
        common = product.index.intersection(market_index)
    else:
        product.index = product.index.normalize()
        market_index = market.index.normalize()
        common = product.index.intersection(market_index)
    return [pd.Timestamp(product.loc[item]).date() for item in common]


def _build_explainability_series(
    product_returns: np.ndarray,
    factor_matrix: pd.DataFrame,
    aligned_dates: list[str],
) -> list[dict]:
    """Build product/predicted/factor contribution series from the OLS fit."""
    y = np.asarray(product_returns, dtype=float)
    if len(y) == 0 or factor_matrix.empty or len(y) != len(factor_matrix):
        return []
    X = factor_matrix.to_numpy(dtype=float)
    try:
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    except Exception:
        return []
    intercept = float(coefficients[0])
    betas = coefficients[1:]
    predicted = intercept + X @ betas
    dates = aligned_dates if len(aligned_dates) == len(y) else [str(i + 1) for i in range(len(y))]
    product_curve = 1.0
    predicted_curve = 1.0
    cumulative_contribution = {name: 0.0 for name in factor_matrix.columns}
    series: list[dict] = []
    for idx, label in enumerate(dates):
        product_curve *= 1.0 + float(y[idx])
        predicted_curve *= 1.0 + float(predicted[idx])
        row = {
            "date": label,
            "product_return": round(float(y[idx]), 8),
            "predicted_return": round(float(predicted[idx]), 8),
            "product_index": round(product_curve, 8),
            "predicted_index": round(predicted_curve, 8),
            "residual": round(float(y[idx] - predicted[idx]), 8),
            "factor_contributions": {},
        }
        for factor_index, name in enumerate(factor_matrix.columns):
            contribution = float(betas[factor_index] * X[idx, factor_index])
            cumulative_contribution[name] += contribution
            row["factor_contributions"][name] = round(cumulative_contribution[name], 8)
        series.append(row)
    return series


def _get_beta_factor(
    strategy_type: str, dates: list[date], frequency: str,
    provider: MarketDataProvider, start: date, end: date,
) -> np.ndarray | None:
    """Market beta: use the primary index for the strategy type."""
    if strategy_type in ("commodity_cta", "mixed"):
        symbol = "nh_commodity"
    else:
        symbol = "hs300"

    returns = provider.get_returns(symbol, start, end, is_index=True)
    if returns.empty or len(returns) < _MIN_OBS:
        # Fallback: try the other index
        fallback = "hs300" if symbol == "nh_commodity" else "nh_commodity"
        returns = provider.get_returns(fallback, start, end, is_index=True)
    if returns.empty:
        return None
    return _resample_returns(returns, frequency, len(dates))


def _get_momentum_factor(
    strategy_type: str, dates: list[date], frequency: str,
    provider: MarketDataProvider, start: date, end: date,
) -> np.ndarray | None:
    """Time-series momentum: sign(past return) × current return.

    Uses a 20-period lookback for daily, 4 for weekly, 3 for monthly.
    """
    if strategy_type in ("commodity_cta", "mixed"):
        symbol = "nh_commodity"
    else:
        symbol = "hs300"

    returns = provider.get_returns(symbol, start, end, is_index=True)
    if returns.empty or len(returns) < 30:
        return None

    values = returns.values
    lookback = {"daily": 20, "weekly": 4, "monthly": 3}.get(frequency, 20)

    # TSM signal: sign of cumulative past return × current return
    momentum_signal = np.zeros(len(values))
    for i in range(lookback, len(values)):
        past_cum = np.sum(values[i - lookback:i])
        momentum_signal[i] = np.sign(past_cum) * values[i]

    # Trim the initial lookback period
    momentum_signal = momentum_signal[lookback:]
    return _resample_array(momentum_signal, frequency, len(dates))


def _get_mean_reversion_factor(
    strategy_type: str, dates: list[date], frequency: str,
    provider: MarketDataProvider, start: date, end: date,
) -> np.ndarray | None:
    """Short-term reversal: negative of past 5-period return."""
    if strategy_type in ("commodity_cta", "mixed"):
        symbol = "nh_commodity"
    else:
        symbol = "hs300"

    returns = provider.get_returns(symbol, start, end, is_index=True)
    if returns.empty or len(returns) < 10:
        return None

    values = returns.values
    lookback = {"daily": 5, "weekly": 2, "monthly": 1}.get(frequency, 5)

    reversal = np.zeros(len(values))
    for i in range(lookback, len(values)):
        past_cum = np.sum(values[i - lookback:i])
        reversal[i] = -past_cum  # negative past return = reversal bet

    reversal = reversal[lookback:]
    return _resample_array(reversal, frequency, len(dates))


def _get_carry_factor(
    dates: list[date], frequency: str,
    provider: MarketDataProvider, start: date, end: date,
) -> np.ndarray | None:
    """Carry proxy: average momentum of commodity futures (roll-yield proxy).

    True carry requires term structure data; as a proxy we use the average
    return of a commodity basket (positive carry → positive basket return).
    """
    basket = ["rb", "cu", "au", "m"]
    all_returns = []

    for symbol in basket:
        ret = provider.get_returns(symbol, start, end, is_index=False)
        if not ret.empty and len(ret) >= _MIN_OBS:
            all_returns.append(ret.values)

    if not all_returns:
        return None

    # Equal-weight average as carry proxy
    min_len = min(len(r) for r in all_returns)
    aligned = np.array([r[-min_len:] for r in all_returns])
    carry_proxy = aligned.mean(axis=0)

    return _resample_array(carry_proxy, frequency, len(dates))


def _get_volatility_factor(
    strategy_type: str, dates: list[date], frequency: str,
    provider: MarketDataProvider, start: date, end: date,
) -> np.ndarray | None:
    """Volatility regime: rolling realized vol (higher vol → vol factor exposure).

    Normalized by long-run average to create a regime indicator.
    """
    if strategy_type in ("commodity_cta", "mixed"):
        symbol = "nh_commodity"
    else:
        symbol = "hs300"

    returns = provider.get_returns(symbol, start, end, is_index=True)
    if returns.empty or len(returns) < 30:
        return None

    values = returns.values
    window = {"daily": 20, "weekly": 4, "monthly": 3}.get(frequency, 20)

    # Rolling realized vol
    vol_series = np.zeros(len(values))
    for i in range(window, len(values)):
        vol_series[i] = np.std(values[i - window:i]) * np.sqrt(252 if frequency == "daily" else 52 if frequency == "weekly" else 12)

    vol_series = vol_series[window:]

    # Normalize: vol / long-run mean vol - 1 (positive = high vol regime)
    mean_vol = np.mean(vol_series[vol_series > 0]) if np.any(vol_series > 0) else 1.0
    if mean_vol > 0:
        vol_factor = vol_series / mean_vol - 1.0
    else:
        vol_factor = vol_series

    return _resample_array(vol_factor, frequency, len(dates))


# ---------------------------------------------------------------------------
# Regression and statistics
# ---------------------------------------------------------------------------


def _build_collinearity_diagnostics(X: pd.DataFrame) -> FactorCollinearityDiagnostics:
    """Summarise factor-proxy overlap and decide whether rolling Ridge is needed."""
    names = list(X.columns)
    if not names:
        return FactorCollinearityDiagnostics()

    numeric = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(numeric) < 2:
        return FactorCollinearityDiagnostics(factor_names=names)

    covariance = numeric.cov().fillna(0.0)
    correlation = numeric.corr().fillna(0.0)
    for name in names:
        if name in correlation.index:
            correlation.loc[name, name] = 1.0

    matrix = correlation.to_numpy(dtype=float)
    try:
        # Correlation-space conditioning is scale independent and therefore
        # meaningful across market proxies with different return volatility.
        condition_number = float(np.linalg.cond(matrix + np.eye(len(names)) * 1e-10))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    pairs: list[str] = []
    max_abs_correlation = 0.0
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            value = float(correlation.loc[left, right])
            max_abs_correlation = max(max_abs_correlation, abs(value))
            if abs(value) >= 0.85:
                pairs.append(f"{FACTOR_LABELS.get(left, left)} / {FACTOR_LABELS.get(right, right)} ({value:.2f})")

    ridge_applied = bool(pairs or condition_number >= 100.0)
    return FactorCollinearityDiagnostics(
        factor_names=names,
        correlation_matrix={name: {other: round(float(correlation.loc[name, other]), 4) for other in names} for name in names},
        covariance_matrix={name: {other: round(float(covariance.loc[name, other]), 8) for other in names} for name in names},
        condition_number=round(condition_number, 2) if np.isfinite(condition_number) else None,
        max_abs_correlation=round(max_abs_correlation, 4),
        high_correlation_pairs=pairs,
        ridge_applied=ridge_applied,
        ridge_alpha=1.0 if ridge_applied else None,
    )


def _build_risk_profile(
    y: np.ndarray, X: pd.DataFrame, factor_details: list[FactorExposureDetail]
) -> FactorRiskProfile:
    """Calculate downside sensitivity, stress behaviour and model risk shares."""
    if X.empty or len(y) != len(X):
        return FactorRiskProfile()

    values = X.to_numpy(dtype=float)
    betas = np.array([
        next((detail.exposure_beta for detail in factor_details if detail.factor_name == name), 0.0)
        for name in X.columns
    ])
    downside_betas: dict[str, float | None] = {}
    tail_correlations: dict[str, float | None] = {}
    regime_returns: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(X.columns):
        factor = values[:, index]
        downside = factor < 0
        upside = factor >= 0
        downside_count = int(np.sum(downside))
        upside_count = int(np.sum(upside))
        if downside_count >= 6 and float(np.var(factor[downside])) > 1e-14:
            downside_beta = float(np.cov(y[downside], factor[downside], ddof=1)[0, 1] / np.var(factor[downside], ddof=1))
            downside_betas[name] = round(downside_beta, 4)
            tail_correlations[name] = round(safe_pearson(y[downside], factor[downside]), 4)
        else:
            downside_betas[name] = None
            tail_correlations[name] = None
        regime_returns[name] = {
            "downside_periods": downside_count,
            "product_mean_return_when_factor_down": round(float(np.mean(y[downside])), 5) if downside_count else None,
            "upside_periods": upside_count,
            "product_mean_return_when_factor_up": round(float(np.mean(y[upside])), 5) if upside_count else None,
        }

    contributions: dict[str, float | None] = {name: None for name in X.columns}
    try:
        covariance = np.cov(values, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        component_variance = betas * (covariance @ betas)
        total_variance = float(np.sum(component_variance))
        if abs(total_variance) > 1e-14:
            contributions = {
                name: round(float(component_variance[index] / total_variance * 100), 1)
                for index, name in enumerate(X.columns)
            }
    except (ValueError, np.linalg.LinAlgError):
        pass

    warnings: list[str] = []
    concentrated = [(name, value) for name, value in contributions.items() if value is not None and value > 65]
    if concentrated:
        labels = "、".join(FACTOR_LABELS.get(name, name) for name, _ in concentrated)
        warnings.append(f"模型波动风险主要由 {labels} 代理解释；该结论是风险贡献估计，不是持仓比例。")
    return FactorRiskProfile(
        downside_betas=downside_betas,
        tail_correlations=tail_correlations,
        regime_returns=regime_returns,
        variance_contributions_pct=contributions,
        principal_components=_principal_component_summary(y, X),
        warnings=warnings,
    )


def _principal_component_summary(y: np.ndarray, X: pd.DataFrame) -> list[dict]:
    """Describe unlabelled common return drivers using PCA.

    PCA is run on the product return plus available factor proxies after
    standardisation.  Component signs are arbitrary, so the UI must treat the
    loadings as a description of co-movement rather than a directional trade.
    """
    if len(y) < _MIN_OBS or X.empty:
        return []
    labels = ["__product__", *list(X.columns)]
    values = np.column_stack([y, X.to_numpy(dtype=float)])
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    valid = scale > 1e-12
    if int(np.sum(valid)) < 2:
        return []
    standardised = (values[:, valid] - mean[valid]) / scale[valid]
    valid_labels = [label for label, keep in zip(labels, valid) if keep]
    try:
        _, singular_values, right_vectors = np.linalg.svd(standardised, full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    total = float(np.sum(singular_values ** 2))
    if total <= 1e-14:
        return []
    result: list[dict] = []
    for component_index in range(min(3, len(singular_values))):
        loadings = right_vectors[component_index]
        loading_by_name = dict(zip(valid_labels, loadings))
        dominant = sorted(
            ((name, float(value)) for name, value in loading_by_name.items() if name != "__product__"),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:3]
        result.append({
            "component": component_index + 1,
            "explained_variance_pct": round(float(singular_values[component_index] ** 2 / total * 100), 1),
            "product_loading": round(float(loading_by_name.get("__product__", 0.0)), 3),
            "dominant_factors": [
                {"factor_name": name, "factor_label": FACTOR_LABELS.get(name, name), "loading": round(value, 3)}
                for name, value in dominant
            ],
        })
    return result


def _run_ols_regression(y: np.ndarray, X: pd.DataFrame) -> list[FactorExposureDetail]:
    """Run OLS and extract per-factor statistics."""
    X_arr = X.values
    n, k = X_arr.shape

    # Add intercept
    X_design = np.column_stack([np.ones(n), X_arr])

    try:
        # OLS: beta = (X'X)^-1 X'y
        beta_hat = np.linalg.lstsq(X_design, y, rcond=None)[0]
        residuals = y - X_design @ beta_hat
        dof = n - k - 1
        if dof <= 0:
            dof = 1
        sigma2 = np.sum(residuals ** 2) / dof
        var_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
        se_beta = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        # Singular matrix fallback: use pseudo-inverse
        beta_hat = np.linalg.lstsq(X_design, y, rcond=None)[0]
        residuals = y - X_design @ beta_hat
        se_beta = np.ones(k + 1) * 0.1

    results = []
    for i, col_name in enumerate(X.columns):
        idx = i + 1  # skip intercept
        beta_val = float(beta_hat[idx])
        se_val = float(se_beta[idx]) if se_beta[idx] > 0 else 0.01
        t_stat = beta_val / se_val
        p_val = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=max(n - k - 1, 1))))

        # Simple correlation (zero-variance guard handled inside safe_pearson)
        col_data = X_arr[:, i]
        corr = safe_pearson(y, col_data)

        # Confidence: combine t-stat significance and correlation strength
        conf_pct = _factor_confidence(t_stat, corr, n)
        conf_label = "高" if conf_pct >= 70 else "中" if conf_pct >= 40 else "低"

        # Evidence
        evidence = []
        label = FACTOR_LABELS.get(col_name, col_name)
        if abs(t_stat) > 2:
            evidence.append(f"回归系数 {beta_val:.3f}，t={t_stat:.2f}（p={p_val:.4f}），统计显著")
        else:
            evidence.append(f"回归系数 {beta_val:.3f}，t={t_stat:.2f}，统计上不显著")
        evidence.append(f"与产品收益简单相关系数 {corr:.3f}")

        hit_rate, hit_observations, hit_p_value = _walk_forward_direction_hit_rate(y, col_data)
        if hit_rate is not None:
            evidence.append(
                f"滚动样本外方向一致率 {hit_rate:.1%}（{hit_observations} 期，二项检验 p={hit_p_value:.4f}）"
            )

        results.append(FactorExposureDetail(
            factor_name=col_name,
            factor_label=label,
            exposure_beta=round(beta_val, 4),
            correlation=round(corr, 4),
            t_statistic=round(t_stat, 3),
            p_value=round(p_val, 5),
            confidence_pct=round(conf_pct, 1),
            confidence_label=conf_label,
            out_of_sample_hit_rate=round(hit_rate, 4) if hit_rate is not None else None,
            hit_rate_observations=hit_observations,
            hit_rate_p_value=round(hit_p_value, 5) if hit_p_value is not None else None,
            evidence=evidence,
        ))

    # Sort by absolute t-statistic descending
    results.sort(key=lambda f: abs(f.t_statistic), reverse=True)
    return results


def _walk_forward_direction_hit_rate(
    product_returns: np.ndarray,
    factor_returns: np.ndarray,
    min_train_periods: int = 20,
) -> tuple[float | None, int, float | None]:
    """Calculate an honest walk-forward direction agreement statistic.

    At each period ``t`` beta is fit with observations strictly before ``t``.
    The sign of ``beta × factor[t]`` is then compared with product return[t].
    Periods with a zero factor/product value are excluded rather than counted
    as wins.  It measures explanatory stability, not a tradable factor signal.
    """
    n = min(len(product_returns), len(factor_returns))
    if n < min_train_periods + 8:
        return None, 0, None

    hits = 0
    trials = 0
    eps = 1e-12
    for end in range(min_train_periods, n):
        y_train = product_returns[end - min_train_periods:end]
        x_train = factor_returns[end - min_train_periods:end]
        denominator = float(np.dot(x_train, x_train))
        if denominator <= eps:
            continue
        beta = float(np.dot(x_train, y_train) / denominator)
        predicted = beta * factor_returns[end]
        realised = product_returns[end]
        if abs(predicted) <= eps or abs(realised) <= eps:
            continue
        trials += 1
        hits += int(np.sign(predicted) == np.sign(realised))

    if trials < 8:
        return None, trials, None
    hit_rate = hits / trials
    p_value = float(sp_stats.binomtest(hits, trials, 0.5, alternative="two-sided").pvalue)
    return hit_rate, trials, p_value


def _run_rolling_regression(
    y: np.ndarray, X: pd.DataFrame, window: int
) -> dict[str, list[float]]:
    """Rolling OLS for each factor, returning time series of betas."""
    n = len(y)
    if n < window * 2:
        return {}

    X_arr = X.values
    k = X_arr.shape[1]
    rolling_betas: dict[str, list[float]] = {col: [] for col in X.columns}

    step = max(1, window // 4)  # overlap windows for smoother series
    for i in range(window, n + 1, step):
        y_win = y[i - window:i]
        X_win = X_arr[i - window:i]
        X_design = np.column_stack([np.ones(window), X_win])

        try:
            beta_hat = np.linalg.lstsq(X_design, y_win, rcond=None)[0]
            for j, col in enumerate(X.columns):
                rolling_betas[col].append(round(float(beta_hat[j + 1]), 4))
        except np.linalg.LinAlgError:
            for col in X.columns:
                rolling_betas[col].append(0.0)

    return rolling_betas


def _run_ridge_rolling_regression(
    y: np.ndarray, X: pd.DataFrame, window: int, alpha: float
) -> dict[str, list[float]]:
    """Rolling Ridge coefficients on standardised factor proxies.

    The output is deliberately labelled as a *stabilised reference trend* in
    the UI.  It is not mixed with OLS t-statistics because Ridge has a
    different inferential interpretation.
    """
    n = len(y)
    if n < window * 2:
        return {}

    values = X.to_numpy(dtype=float)
    results: dict[str, list[float]] = {col: [] for col in X.columns}
    step = max(1, window // 4)
    for end in range(window, n + 1, step):
        x_win = values[end - window:end]
        y_win = y[end - window:end]
        x_mean = x_win.mean(axis=0)
        x_scale = x_win.std(axis=0, ddof=0)
        x_scale[x_scale < 1e-12] = 1.0
        x_standardised = (x_win - x_mean) / x_scale
        y_centered = y_win - y_win.mean()
        try:
            coefficients = np.linalg.solve(
                x_standardised.T @ x_standardised + alpha * np.eye(x_standardised.shape[1]),
                x_standardised.T @ y_centered,
            )
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(
                x_standardised.T @ x_standardised + alpha * np.eye(x_standardised.shape[1])
            ) @ x_standardised.T @ y_centered
        for index, name in enumerate(X.columns):
            results[name].append(round(float(coefficients[index]), 4))
    return results


def _compute_model_fit(
    y: np.ndarray, X: pd.DataFrame, window: int
) -> tuple[float, float, list[float]]:
    """Compute R², adjusted R², and rolling R² series."""
    X_arr = X.values
    n, k = X_arr.shape
    X_design = np.column_stack([np.ones(n), X_arr])

    try:
        beta_hat = np.linalg.lstsq(X_design, y, rcond=None)[0]
        y_hat = X_design @ beta_hat
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / max(n - k - 1, 1)
    except Exception:
        r_sq, adj_r_sq = 0.0, 0.0

    # Rolling R²
    rolling_r_sq = []
    if n >= window * 2:
        step = max(1, window // 4)
        for i in range(window, n + 1, step):
            y_win = y[i - window:i]
            X_win = X_arr[i - window:i]
            X_d = np.column_stack([np.ones(window), X_win])
            try:
                b = np.linalg.lstsq(X_d, y_win, rcond=None)[0]
                y_h = X_d @ b
                ss_r = np.sum((y_win - y_h) ** 2)
                ss_t = np.sum((y_win - np.mean(y_win)) ** 2)
                rolling_r_sq.append(round(1 - ss_r / ss_t if ss_t > 0 else 0.0, 4))
            except Exception:
                rolling_r_sq.append(0.0)

    return r_sq, adj_r_sq, rolling_r_sq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _factor_confidence(t_stat: float, corr: float, n: int) -> float:
    """Combine statistical significance and effect size into a confidence score."""
    # t-stat contribution: |t| > 2 is significant
    t_score = min(abs(t_stat) / 3.0, 1.0) * 60  # max 60 points from t-stat

    # Correlation contribution
    corr_score = min(abs(corr) / 0.8, 1.0) * 30  # max 30 points from correlation

    # Sample size bonus (more data → more confidence)
    n_bonus = min(n / 100, 1.0) * 10  # max 10 points from sample size

    return min(t_score + corr_score + n_bonus, 100.0)


def _resample_returns(returns: pd.Series, frequency: str, target_len: int) -> np.ndarray | None:
    """Resample a daily returns Series to match the product frequency and length."""
    if returns.empty:
        return None

    # Normalize index
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    if frequency == "weekly":
        resampled = (1 + returns).resample("W").prod() - 1
    elif frequency == "monthly":
        resampled = (1 + returns).resample("ME").prod() - 1
    else:
        resampled = returns

    resampled = resampled.dropna()
    values = resampled.values

    if len(values) < target_len:
        return None
    return values[-target_len:]


def _resample_array(arr: np.ndarray, frequency: str, target_len: int) -> np.ndarray | None:
    """Trim or aggregate a daily array to match product frequency and length."""
    if len(arr) < target_len:
        return None

    if frequency == "daily":
        return arr[-target_len:]
    elif frequency == "weekly":
        # Aggregate every 5 days
        chunk_size = 5
        n_chunks = len(arr) // chunk_size
        if n_chunks < target_len:
            return None
        aggregated = np.array([arr[i * chunk_size:(i + 1) * chunk_size].sum() for i in range(n_chunks)])
        return aggregated[-target_len:]
    elif frequency == "monthly":
        chunk_size = 21
        n_chunks = len(arr) // chunk_size
        if n_chunks < target_len:
            return None
        aggregated = np.array([arr[i * chunk_size:(i + 1) * chunk_size].sum() for i in range(n_chunks)])
        return aggregated[-target_len:]
    return arr[-target_len:]


def _generate_evidence(factors: list[FactorExposureDetail], r_sq: float) -> list[str]:
    """Generate a summary evidence list from factor results."""
    evidence = [f"模型整体解释力 R²={r_sq:.3f}"]

    significant = [f for f in factors if abs(f.t_statistic) > 2]
    if significant:
        names = "、".join(f.factor_label for f in significant[:3])
        evidence.append(f"统计显著的因子：{names}")
    else:
        evidence.append("无统计显著的因子暴露（所有 |t| < 2）")

    # Strongest factor
    if factors:
        top = factors[0]
        evidence.append(f"最强因子：{top.factor_label}（beta={top.exposure_beta:.3f}，置信度 {top.confidence_pct}%）")

    return evidence
