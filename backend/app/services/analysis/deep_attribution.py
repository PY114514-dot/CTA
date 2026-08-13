"""Deep, evidence-first CTA attribution from a product NAV series.

The module is deliberately a *statistical attribution* engine.  It does not
try to reconstruct positions from a one-dimensional NAV series.  Instead it
combines three complementary views of the same dated observations:

* sparse linear exposures (Elastic Net) for a stable, signed reference;
* a time-ordered gradient-boosting model for nonlinear state interactions;
* block-bootstrap sign consistency and state-conditioned returns for
  reliability and alternative explanations.

The market-data provider exposes daily returns only, so the first version
uses price-only proxies.  Carry is reported as unavailable until an adapter
with spot/near/far-month data is supplied; calling a price momentum signal
"carry" would overstate what the data can prove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.services.market_data.provider import MarketDataProvider
from app.services.analysis._stats import safe_pearson


@dataclass(frozen=True)
class ProxySpec:
    """A small, reviewable market proxy used by the attribution engine."""

    symbol: str
    name: str
    group: str
    asset_class: str
    is_index: bool


@dataclass
class DeepAttributionResult:
    """Serializable result returned by the deep attribution module."""

    asset_class: dict[str, Any] = field(default_factory=dict)
    sector_exposures: list[dict[str, Any]] = field(default_factory=list)
    strategy_fingerprints: list[dict[str, Any]] = field(default_factory=list)
    state_analysis: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    method_provenance: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "深度归因是基于公开市场代理的统计候选，不是实际持仓重建；"
        "板块、策略和资产类别均需用持仓或管理人周报进一步验证。"
    )


_MIN_OBS = 24
_DEFAULT_BOOTSTRAPS = 80
_LOOKBACKS = {"daily": (5, 20, 60), "weekly": (2, 4, 12), "monthly": (1, 3, 6)}

_STRATEGY_LABELS = {
    "trend_short": "短周期时序趋势",
    "trend_medium": "中周期时序趋势",
    "trend_long": "长周期时序趋势",
    "cross_sectional_momentum": "截面动量",
    "short_reversal": "短期反转",
    "volatility_control": "波动率控制/动态杠杆",
    "dispersion": "市场分化/相对价值",
    "liquidity_stress_proxy": "流动性冲击防御代理",
    "carry_proxy": "Carry/期限结构",
}

_SECTOR_LABELS = {
    "黑色": "黑色",
    "有色": "有色",
    "贵金属": "贵金属",
    "能化": "能源化工",
    "农产品": "农产品",
    "大盘指数": "大盘指数",
    "中盘指数": "中盘指数",
    "小盘指数": "小盘指数",
    "股指期货": "股指期货",
    "利率对冲": "利率/债券对冲",
}


def deep_attribution(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    strategy_type: str,
    provider: MarketDataProvider,
    top_n: int = 8,
    bootstrap_samples: int = _DEFAULT_BOOTSTRAPS,
) -> DeepAttributionResult:
    """Run the multi-view CTA attribution model.

    The interface is intentionally small: callers provide the same dated
    product returns and market-data adapter used by the existing pipeline.
    All factor construction, time ordering, model comparison and confidence
    language stay behind this seam.
    """

    y = np.asarray(product_returns, dtype=float)
    if len(y) != len(product_dates):
        return DeepAttributionResult(warnings=["产品收益与日期长度不一致，已停止深度归因。"])
    if len(y) < _MIN_OBS:
        return DeepAttributionResult(
            warnings=[f"样本量 {len(y)} 期不足 {_MIN_OBS} 期，无法运行深度归因。"],
            diagnostics={"observation_count": int(len(y)), "reliability_label": "不可识别"},
        )
    if not np.isfinite(y).all():
        return DeepAttributionResult(warnings=["产品收益包含非有限值，已停止深度归因。"])

    specs = _proxy_specs(strategy_type)
    if not specs:
        return DeepAttributionResult(
            warnings=[f"策略范围 {strategy_type} 没有可用的市场代理。"],
            diagnostics={"observation_count": int(len(y)), "reliability_label": "不可识别"},
        )

    market_frame, available_specs, fetch_warnings = _load_market_frame(
        y, product_dates, frequency, specs, provider
    )
    if market_frame.empty or not available_specs:
        return DeepAttributionResult(
            warnings=fetch_warnings or ["没有获取到可对齐的市场代理数据，已停止深度归因。"],
            diagnostics={"observation_count": int(len(y)), "reliability_label": "不可识别"},
        )

    feature_frame, metadata = _build_features(market_frame, available_specs, frequency)
    aligned = feature_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["__product__"])
    feature_names = [name for name in feature_frame.columns if name != "__product__"]
    usable = [name for name in feature_names if aligned[name].notna().sum() >= _MIN_OBS and aligned[name].std() > 1e-12]
    if not usable:
        return DeepAttributionResult(
            warnings=fetch_warnings + ["市场代理存在，但没有足够变化和日期交集来构造深度因子。"],
            diagnostics={"observation_count": int(len(y)), "reliability_label": "不可识别"},
        )

    model_frame = aligned[["__product__", *usable]].dropna()
    if len(model_frame) < _MIN_OBS:
        return DeepAttributionResult(
            warnings=fetch_warnings + [f"因子构造后的有效样本仅 {len(model_frame)} 期，无法运行深度归因。"],
            diagnostics={"observation_count": int(len(model_frame)), "reliability_label": "不可识别"},
        )

    y_aligned = model_frame["__product__"].to_numpy(dtype=float)
    x_aligned = model_frame[usable].to_numpy(dtype=float)
    model = _fit_models(y_aligned, x_aligned, usable)
    bootstrap = _bootstrap_consistency(
        y_aligned,
        x_aligned,
        usable,
        samples=max(20, min(int(bootstrap_samples), 300)),
    )
    scores = _feature_scores(usable, metadata, model, bootstrap)
    sector_exposures = _sector_candidates(scores, metadata, top_n)
    asset_class = _asset_class_candidates(scores, metadata, strategy_type)
    strategy_fingerprints = _strategy_candidates(scores, metadata)
    state_analysis = _state_analysis(model_frame, available_specs, market_frame, frequency)
    diagnostics = _diagnostics(model, bootstrap, len(model_frame), len(available_specs), usable)
    warnings = fetch_warnings + _diagnostic_warnings(diagnostics, strategy_fingerprints)
    evidence = _evidence_summary(available_specs, model, diagnostics, sector_exposures, strategy_fingerprints)

    return DeepAttributionResult(
        asset_class=asset_class,
        sector_exposures=sector_exposures,
        strategy_fingerprints=strategy_fingerprints,
        state_analysis=state_analysis,
        diagnostics=diagnostics,
        evidence=evidence,
        warnings=warnings,
        method_provenance={
            "model": "Elastic Net + 时间切分梯度提升 + Block Bootstrap",
            "factor_construction": "多周期时序趋势、截面动量、反转、波动率控制、分化与冲击代理",
            "data_frequency": frequency,
            "proxy_count": len(available_specs),
            "factor_count": len(usable),
            "bootstrap_samples": max(20, min(int(bootstrap_samples), 300)),
            "github_references": [
                "20power/cta_attribution_system（CTA 因子与状态/漂移框架）",
                "brianbanna/commodity-curve-factors（Carry、板块、Bootstrap 思路）",
            ],
            "limitations": [
                "净值序列是一维结果，不能识别真实持仓或精确仓位。",
                "当前行情接口没有现货/近远月字段，Carry 只报告为不可识别。",
                "树模型用于状态交互发现，不把 feature importance 当作持仓比例。",
            ],
        },
    )


def _proxy_specs(strategy_type: str) -> list[ProxySpec]:
    """Return a deliberately small proxy universe to keep the request fast."""
    commodity = [
        ProxySpec("rb", "螺纹钢", "黑色", "commodity", False),
        ProxySpec("i", "铁矿石", "黑色", "commodity", False),
        ProxySpec("cu", "沪铜", "有色", "commodity", False),
        ProxySpec("al", "沪铝", "有色", "commodity", False),
        ProxySpec("au", "沪金", "贵金属", "commodity", False),
        ProxySpec("sc", "原油", "能化", "commodity", False),
        ProxySpec("ta", "PTA", "能化", "commodity", False),
        ProxySpec("m", "豆粕", "农产品", "commodity", False),
        ProxySpec("y", "豆油", "农产品", "commodity", False),
    ]
    equity = [
        ProxySpec("hs300", "沪深300", "大盘指数", "equity", True),
        ProxySpec("zz500", "中证500", "中盘指数", "equity", True),
        ProxySpec("zz1000", "中证1000", "小盘指数", "equity", True),
        ProxySpec("IF", "沪深300股指期货", "股指期货", "equity", False),
        ProxySpec("IC", "中证500股指期货", "股指期货", "equity", False),
        ProxySpec("T", "十年国债期货", "利率对冲", "equity", False),
    ]
    if strategy_type == "commodity_cta":
        return commodity
    if strategy_type == "equity_quant":
        return equity
    if strategy_type == "mixed":
        return commodity + equity
    return []


def _period_index(values: pd.Series, frequency: str) -> pd.Series:
    series = values.copy().astype(float)
    series.index = pd.to_datetime(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if frequency == "weekly":
        result = (1.0 + series).resample("W-SUN").prod() - 1.0
        result.index = result.index.to_period("W-SUN")
        return result
    if frequency == "monthly":
        result = (1.0 + series).resample("ME").prod() - 1.0
        result.index = result.index.to_period("M")
        return result
    series.index = series.index.normalize()
    return series


def _product_periods(values: np.ndarray, dates: list[date], frequency: str) -> pd.Series:
    raw = pd.Series(values, index=pd.to_datetime(dates), dtype=float)
    if frequency == "weekly":
        return raw.groupby(raw.index.to_period("W-SUN")).last()
    if frequency == "monthly":
        return raw.groupby(raw.index.to_period("M")).last()
    raw.index = raw.index.normalize()
    return raw


def _load_market_frame(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    specs: list[ProxySpec],
    provider: MarketDataProvider,
) -> tuple[pd.DataFrame, list[ProxySpec], list[str]]:
    """Fetch proxies once and align them by actual calendar periods."""
    buffer_days = {"daily": 120, "weekly": 550, "monthly": 1200}.get(frequency, 550)
    start = min(product_dates) - timedelta(days=buffer_days)
    end = max(product_dates)
    product = _product_periods(product_returns, product_dates, frequency)
    series: dict[str, pd.Series] = {}
    available: list[ProxySpec] = []
    warnings: list[str] = []
    for spec in specs:
        try:
            raw = provider.get_returns(spec.symbol, start, end, is_index=spec.is_index)
        except Exception:
            raw = pd.Series(dtype=float)
        if raw is None or raw.empty:
            warnings.append(f"未获取到{spec.name}行情，已从深度归因代理池排除。")
            continue
        period_series = _period_index(raw, frequency)
        if period_series.dropna().size < _MIN_OBS:
            warnings.append(f"{spec.name}有效行情不足，已从深度归因代理池排除。")
            continue
        series[spec.symbol] = period_series
        available.append(spec)
    if not series:
        return pd.DataFrame(), [], warnings
    frame = pd.DataFrame(series)
    frame["__product__"] = product.reindex(frame.index)
    return frame.sort_index(), available, warnings


def _build_features(
    frame: pd.DataFrame,
    specs: list[ProxySpec],
    frequency: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Construct lagged, price-only strategy and sector factor returns."""
    result = pd.DataFrame(index=frame.index)
    metadata: dict[str, dict[str, Any]] = {}
    by_group: dict[str, list[str]] = {}
    by_asset: dict[str, list[str]] = {}
    for spec in specs:
        if spec.symbol not in frame:
            continue
        by_group.setdefault(spec.group, []).append(spec.symbol)
        by_asset.setdefault(spec.asset_class, []).append(spec.symbol)
    for group, symbols in by_group.items():
        name = f"sector_{_slug(group)}"
        result[name] = frame[symbols].mean(axis=1)
        metadata[name] = {"kind": "sector", "group": group, "label": _SECTOR_LABELS.get(group, group), "asset_class": _asset_for_group(specs, group)}
    aggregate: dict[str, pd.Series] = {}
    for asset, symbols in by_asset.items():
        name = f"asset_class_{asset}"
        aggregate[asset] = frame[symbols].mean(axis=1)
        result[name] = aggregate[asset]
        metadata[name] = {"kind": "asset_class", "group": asset, "label": "商品" if asset == "commodity" else "股指/利率", "asset_class": asset}

    all_symbols = [spec.symbol for spec in specs if spec.symbol in frame]
    base = frame[all_symbols]
    market = base.mean(axis=1)
    lookbacks = _LOOKBACKS.get(frequency, _LOOKBACKS["weekly"])
    for index, lookback in enumerate(lookbacks):
        name = f"trend_{('short', 'medium', 'long')[index]}"
        result[name] = _trend_factor(base, lookback)
        metadata[name] = {"kind": "strategy", "group": name, "label": _STRATEGY_LABELS[name], "asset_class": "all"}
    result["cross_sectional_momentum"] = _cross_sectional_factor(base, lookbacks[-1])
    metadata["cross_sectional_momentum"] = {"kind": "strategy", "group": "cross_sectional_momentum", "label": _STRATEGY_LABELS["cross_sectional_momentum"], "asset_class": "all"}
    result["short_reversal"] = _reversal_factor(base, lookbacks[0])
    metadata["short_reversal"] = {"kind": "strategy", "group": "short_reversal", "label": _STRATEGY_LABELS["short_reversal"], "asset_class": "all"}
    result["volatility_control"] = _volatility_control_factor(market, lookbacks[1])
    metadata["volatility_control"] = {"kind": "strategy", "group": "volatility_control", "label": _STRATEGY_LABELS["volatility_control"], "asset_class": "all"}
    result["dispersion"] = _dispersion_factor(base, market)
    metadata["dispersion"] = {"kind": "strategy", "group": "dispersion", "label": _STRATEGY_LABELS["dispersion"], "asset_class": "all"}
    result["liquidity_stress_proxy"] = _liquidity_stress_factor(market, lookbacks[0])
    metadata["liquidity_stress_proxy"] = {"kind": "strategy", "group": "liquidity_stress_proxy", "label": _STRATEGY_LABELS["liquidity_stress_proxy"], "asset_class": "all"}
    return pd.concat([result, frame[["__product__"]]], axis=1), metadata


def _asset_for_group(specs: list[ProxySpec], group: str) -> str:
    classes = {spec.asset_class for spec in specs if spec.group == group}
    return next(iter(classes), "all")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower() or "group"


def _trend_factor(base: pd.DataFrame, lookback: int) -> pd.Series:
    past = (1.0 + base).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True).shift(1) - 1.0
    return (np.sign(past) * base).mean(axis=1)


def _cross_sectional_factor(base: pd.DataFrame, lookback: int) -> pd.Series:
    past = (1.0 + base).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True).shift(1) - 1.0
    ranks = past.rank(axis=1, pct=True) - 0.5
    weights = ranks.div(ranks.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return (weights * base).sum(axis=1)


def _reversal_factor(base: pd.DataFrame, lookback: int) -> pd.Series:
    previous = base.shift(1)
    scale = base.rolling(max(lookback * 4, 4), min_periods=max(lookback * 2, 3)).std().shift(1)
    weights = (-previous).div(scale.replace(0, np.nan))
    weights = weights.div(weights.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return (weights * base).sum(axis=1)


def _volatility_control_factor(market: pd.Series, lookback: int) -> pd.Series:
    lagged_vol = market.rolling(max(lookback * 3, 6), min_periods=max(lookback, 3)).std().shift(1)
    target = lagged_vol.median()
    if not isfinite(float(target)) or target <= 0:
        target = 0.01
    scale = (target / lagged_vol.replace(0, np.nan)).clip(0, 3)
    return market * scale


def _dispersion_factor(base: pd.DataFrame, market: pd.Series) -> pd.Series:
    return base.std(axis=1) * np.sign(market.shift(1))


def _liquidity_stress_factor(market: pd.Series, lookback: int) -> pd.Series:
    rolling_vol = market.rolling(max(lookback * 3, 6), min_periods=max(lookback, 3)).std().shift(1)
    shock = (market.abs() / rolling_vol.replace(0, np.nan)).clip(0, 5)
    return -np.sign(market.shift(1)) * shock * market.abs()


def _fit_models(y: np.ndarray, x: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Fit a time-ordered sparse reference and nonlinear state model."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    linear_name = "Elastic Net"
    try:
        splits = 3 if len(y) >= 30 else 2
        linear = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            alphas=np.logspace(-4, -1, 24),
            cv=TimeSeriesSplit(n_splits=splits),
            max_iter=20000,
            random_state=17,
        )
        linear.fit(x_scaled, y)
    except Exception:
        linear_name = "Ridge（Elastic Net 样本不足回退）"
        linear = Ridge(alpha=1.0).fit(x_scaled, y)

    split = max(8, int(len(y) * 0.7))
    split = min(split, len(y) - 8)
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]
    train_scaler = StandardScaler().fit(train_x)
    baseline = Ridge(alpha=1.0).fit(train_scaler.transform(train_x), train_y)
    linear_pred = baseline.predict(train_scaler.transform(test_x))
    linear_oos = _safe_r2(test_y, linear_pred)
    nonlinear_name = "HistGradientBoosting"
    nonlinear_oos: float | None = None
    permutation: dict[str, float] = {}
    if len(test_y) >= 8:
        try:
            nonlinear = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=7,
                min_samples_leaf=max(4, min(12, len(train_y) // 8)),
                l2_regularization=1.0,
                random_state=17,
            )
            nonlinear.fit(train_x, train_y)
            nonlinear_pred = nonlinear.predict(test_x)
            nonlinear_oos = _safe_r2(test_y, nonlinear_pred)
            if len(test_y) >= 10:
                importance = permutation_importance(
                    nonlinear,
                    test_x,
                    test_y,
                    n_repeats=8,
                    random_state=17,
                    scoring="r2",
                ).importances_mean
                permutation = {name: float(max(0.0, value)) for name, value in zip(names, importance)}
        except Exception:
            nonlinear_name = "HistGradientBoosting（样本不足）"
    return {
        "linear": linear,
        "linear_name": linear_name,
        "scaler": scaler,
        "coefficients": {name: float(value) for name, value in zip(names, linear.coef_)},
        "linear_oos_r2": linear_oos,
        "nonlinear_oos_r2": nonlinear_oos,
        "nonlinear_uplift": None if nonlinear_oos is None else nonlinear_oos - linear_oos,
        "permutation_importance": permutation,
        "split_index": split,
        "nonlinear_name": nonlinear_name,
    }


def _safe_r2(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2 or np.std(actual) < 1e-12:
        return None
    value = float(r2_score(actual, predicted))
    return round(value, 4) if isfinite(value) else None


def _bootstrap_consistency(y: np.ndarray, x: np.ndarray, names: list[str], samples: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(17)
    n = len(y)
    block = max(3, min(12, n // 8))
    values: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(samples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            indices.extend(range(start, min(start + block, n)))
        idx = np.asarray(indices[:n], dtype=int)
        for column, name in enumerate(names):
            values[name].append(safe_pearson(y[idx], x[idx, column]))
    return {
        name: {
            "median_correlation": round(float(np.median(vals)), 4),
            "sign_consistency": round(float(max(np.mean(np.asarray(vals) >= 0), np.mean(np.asarray(vals) < 0))), 4),
        }
        for name, vals in values.items()
    }


def _feature_scores(
    names: list[str],
    metadata: dict[str, dict[str, Any]],
    model: dict[str, Any],
    bootstrap: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    coefficients = model["coefficients"]
    importance = model["permutation_importance"]
    max_coef = max((abs(value) for value in coefficients.values()), default=1.0) or 1.0
    max_importance = max(importance.values(), default=1.0) or 1.0
    scored: dict[str, dict[str, Any]] = {}
    for name in names:
        corr = float(bootstrap.get(name, {}).get("median_correlation", 0.0))
        consistency = float(bootstrap.get(name, {}).get("sign_consistency", 0.0))
        coefficient = float(coefficients.get(name, 0.0))
        perm = float(importance.get(name, 0.0))
        consistency_component = consistency if abs(corr) >= 0.1 else consistency * 0.35
        score = 100 * (0.35 * min(1.0, abs(corr)) + 0.25 * abs(coefficient) / max_coef + 0.20 * perm / max_importance + 0.20 * consistency_component)
        direction_value = 0.6 * corr + 0.4 * np.sign(coefficient) * abs(corr)
        direction = "正向" if direction_value > 0.02 else "负向" if direction_value < -0.02 else "不稳定/不明确"
        evidence: list[str] = [f"Bootstrap 中相关性符号一致率 {consistency:.0%}"]
        if abs(corr) >= 0.2:
            evidence.append(f"Bootstrap 中位相关性 {corr:+.3f}")
        if abs(coefficient) > 1e-8:
            evidence.append(f"Elastic Net 标准化系数 {coefficient:+.3f}")
        if perm > 0:
            evidence.append(f"时间切分非线性模型置换重要性 {perm:.4f}")
        counter: list[str] = []
        if abs(corr) < 0.15:
            counter.append("线性相关性弱")
        if consistency < 0.6:
            counter.append("跨区间符号不稳定")
        if model.get("nonlinear_uplift") is not None and model["nonlinear_uplift"] <= 0:
            counter.append("非线性模型没有提升样本外解释度")
        scored[name] = {
            "name": name,
            "label": metadata.get(name, {}).get("label", name),
            "kind": metadata.get(name, {}).get("kind", "strategy"),
            "group": metadata.get(name, {}).get("group", name),
            "asset_class": metadata.get(name, {}).get("asset_class", "all"),
            "correlation": round(corr, 4),
            "coefficient": round(coefficient, 5),
            "permutation_importance": round(perm, 6),
            "sign_consistency": round(consistency, 4),
            "score": round(float(min(100.0, max(0.0, score))), 1),
            "direction": direction,
            "evidence": evidence,
            "counter_evidence": counter,
        }
    return scored


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    shifted = np.asarray(values, dtype=float) - max(values)
    weights = np.exp(np.clip(shifted / 20.0, -30, 30))
    total = float(weights.sum()) or 1.0
    return [round(float(value / total * 100), 1) for value in weights]


def _sector_candidates(scores: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    rows = [value for name, value in scores.items() if metadata.get(name, {}).get("kind") == "sector"]
    rows.sort(key=lambda value: value["score"], reverse=True)
    probabilities = _softmax([float(row["score"]) for row in rows])
    output: list[dict[str, Any]] = []
    for row, probability in zip(rows[:top_n], probabilities[:top_n]):
        output.append({
            "sector": row["group"],
            "label": row["label"],
            "candidate_probability_pct": probability,
            "score_pct": row["score"],
            "direction": row["direction"],
            "correlation": row["correlation"],
            "standardized_coefficient": row["coefficient"],
            "stability_pct": round(row["sign_consistency"] * 100, 1),
            "evidence": row["evidence"],
            "counter_evidence": row["counter_evidence"],
            "interpretation": "候选板块共同波动，不是实际仓位比例。",
        })
    return output


def _asset_class_candidates(scores: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]], strategy_type: str) -> dict[str, Any]:
    rows = [value for name, value in scores.items() if metadata.get(name, {}).get("kind") == "asset_class"]
    if not rows:
        return {"most_likely": "不可识别", "confidence_pct": 0.0, "candidates": []}
    probabilities = _softmax([float(row["score"]) for row in rows])
    candidates = []
    for row, probability in zip(rows, probabilities):
        candidates.append({
            "asset_class": row["group"],
            "label": row["label"],
            "candidate_probability_pct": probability,
            "score_pct": row["score"],
            "direction": row["direction"],
            "correlation": row["correlation"],
            "stability_pct": round(row["sign_consistency"] * 100, 1),
            "evidence": row["evidence"],
        })
    top = max(candidates, key=lambda row: row["candidate_probability_pct"])
    return {
        "most_likely": top["asset_class"],
        "most_likely_label": top["label"],
        "confidence_pct": top["candidate_probability_pct"],
        "requested_scope": strategy_type,
        "candidates": candidates,
        "interpretation": "资产大类概率是多模型相对候选分数的归一化，不是持仓比例。",
    }


def _strategy_candidates(scores: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [value for name, value in scores.items() if metadata.get(name, {}).get("kind") == "strategy"]
    output = [{
        "strategy": row["name"],
        "label": row["label"],
        "evidence_score_pct": row["score"],
        "direction": row["direction"],
        "stability_pct": round(row["sign_consistency"] * 100, 1),
        "evidence": row["evidence"],
        "counter_evidence": row["counter_evidence"],
        "status": "supported" if row["score"] >= 25 and row["sign_consistency"] >= 0.6 else "weak_or_unstable",
        "interpretation": "策略行为指纹，不等同于管理人实际交易规则。",
    } for row in rows]
    output.append({
        "strategy": "carry_proxy",
        "label": _STRATEGY_LABELS["carry_proxy"],
        "evidence_score_pct": 0.0,
        "direction": "不可识别",
        "stability_pct": 0.0,
        "evidence": ["当前市场数据接口只有价格收益，没有现货、近月/远月或期限结构字段。"],
        "counter_evidence": ["不能把价格动量或相关性改写为 Carry 证据。"],
        "status": "not_available",
        "interpretation": "需要补充期限结构或现货数据后再判断。",
    })
    output.sort(key=lambda row: float(row["evidence_score_pct"]), reverse=True)
    return output


def _state_analysis(frame: pd.DataFrame, specs: list[ProxySpec], market_frame: pd.DataFrame, frequency: str) -> list[dict[str, Any]]:
    symbols = [spec.symbol for spec in specs if spec.symbol in market_frame]
    if not symbols:
        return []
    market = market_frame[symbols].mean(axis=1)
    y = frame["__product__"]
    lookback = _LOOKBACKS.get(frequency, _LOOKBACKS["weekly"])[1]
    lagged_vol = market.rolling(max(lookback * 3, 6), min_periods=max(lookback, 3)).std().shift(1)
    threshold = lagged_vol.median()
    if not isfinite(float(threshold)) or threshold <= 0:
        return []
    trend = (1.0 + market).rolling(max(lookback, 2), min_periods=max(lookback, 2)).apply(np.prod, raw=True).shift(1) - 1.0
    states = [("低波动", lagged_vol <= threshold), ("高波动", lagged_vol > threshold), ("趋势向上", trend > 0), ("趋势向下", trend <= 0)]
    output = []
    for name, mask in states:
        valid = mask & y.notna()
        count = int(valid.sum())
        if count < 4:
            continue
        product_values = y[valid].to_numpy(dtype=float)
        market_values = market[valid].to_numpy(dtype=float)
        output.append({
            "state": name,
            "periods": count,
            "product_mean_return": round(float(product_values.mean()), 6),
            "product_positive_rate": round(float(np.mean(product_values > 0)), 4),
            "market_mean_return": round(float(market_values.mean()), 6),
            "interpretation": "条件样本表现，用于发现状态依赖，不是预测。",
        })
    return output


def _diagnostics(model: dict[str, Any], bootstrap: dict[str, dict[str, float]], observations: int, proxy_count: int, factors: list[str]) -> dict[str, Any]:
    consistency = float(np.mean([value["sign_consistency"] for value in bootstrap.values()])) if bootstrap else 0.0
    nonlinear = model.get("nonlinear_oos_r2")
    linear = model.get("linear_oos_r2")
    uplift = model.get("nonlinear_uplift")
    reliability = "高" if observations >= 80 and consistency >= 0.7 else "中" if observations >= 40 and consistency >= 0.55 else "低"
    if observations < _MIN_OBS or not factors:
        reliability = "不可识别"
    return {
        "observation_count": int(observations),
        "proxy_count": int(proxy_count),
        "factor_count": int(len(factors)),
        "linear_model": model.get("linear_name"),
        "nonlinear_model": model.get("nonlinear_name"),
        "linear_oos_r2": linear,
        "nonlinear_oos_r2": nonlinear,
        "nonlinear_uplift": uplift,
        "mean_bootstrap_sign_consistency": round(consistency, 4),
        "reliability_label": reliability,
        "feature_list": factors,
        "carry_data_available": False,
    }


def _diagnostic_warnings(diagnostics: dict[str, Any], strategies: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    n = int(diagnostics.get("observation_count", 0))
    if n < 60:
        warnings.append(f"深度归因仅有 {n} 个有效共同收益期，策略指纹只适合探索性参考。")
    if diagnostics.get("nonlinear_oos_r2") is None:
        warnings.append("时间切分非线性模型没有足够测试样本，未输出非线性样本外证据。")
    elif diagnostics.get("nonlinear_uplift") is not None and diagnostics["nonlinear_uplift"] <= 0:
        warnings.append("非线性模型没有改善样本外解释度，不能据此宣称存在稳定的复杂交互。")
    if float(diagnostics.get("mean_bootstrap_sign_consistency", 0.0)) < 0.55:
        warnings.append("Bootstrap 符号一致性偏低，候选暴露可能随窗口变化。")
    weak = [row["label"] for row in strategies if row.get("status") == "weak_or_unstable"]
    if weak:
        warnings.append(f"弱或不稳定的策略指纹：{'、'.join(weak[:4])}。")
    warnings.append("Carry/期限结构未识别：需要现货或近远月数据，不能用价格相关性替代。")
    return warnings


def _evidence_summary(specs: list[ProxySpec], model: dict[str, Any], diagnostics: dict[str, Any], sectors: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> list[str]:
    evidence = [f"使用 {len(specs)} 个公开市场代理、{diagnostics.get('observation_count', 0)} 个实际日期交集。"]
    evidence.append(f"线性参考模型：{model.get('linear_name', '未知')}；非线性模型：{model.get('nonlinear_name', '未知')}。")
    if model.get("nonlinear_uplift") is not None:
        evidence.append(f"非线性模型相对线性模型的样本外 R² 变化：{model['nonlinear_uplift']:+.4f}。")
    if sectors:
        evidence.append("候选板块最高分：" + "、".join(f"{row['label']}({row['candidate_probability_pct']}%)" for row in sectors[:3]) + "。")
    supported = [row["label"] for row in strategies if row.get("status") == "supported"]
    if supported:
        evidence.append("相对稳定的策略行为指纹：" + "、".join(supported[:4]) + "。")
    return evidence


def as_dict(result: DeepAttributionResult) -> dict[str, Any]:
    """Return a JSON-safe public representation for reports and HTTP routes."""
    return _json_safe({
        "asset_class": result.asset_class,
        "sector_exposures": result.sector_exposures,
        "strategy_fingerprints": result.strategy_fingerprints,
        "state_analysis": result.state_analysis,
        "diagnostics": result.diagnostics,
        "evidence": result.evidence,
        "warnings": result.warnings,
        "method_provenance": result.method_provenance,
        "disclaimer": result.disclaimer,
    })


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
