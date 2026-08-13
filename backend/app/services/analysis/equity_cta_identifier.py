"""Statistical market-reference inference for equity-index CTA research.

This deliberately ranks liquid index and futures *references*, not holdings.
It mirrors the commodity candidate workflow so both CTA research scopes expose
their assumptions, date alignment, sparse-regression evidence and stability.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.services.analysis._stats import safe_pearson
from app.services.analysis.variety_identifier import (
    SectorExposure,
    VarietyCandidate,
    VarietyIdentificationResult,
    _lasso_regression,
    _rolling_stability,
)
from app.services.market_data.provider import MarketDataProvider

_MIN_OBS = 20

# symbol, display name, market-reference group, colour, is_index
_EQUITY_REFERENCES = (
    ("hs300", "沪深300", "大盘指数", "#1677ff", True),
    ("zz500", "中证500", "中盘指数", "#13c2c2", True),
    ("zz1000", "中证1000", "小盘指数", "#722ed1", True),
    ("IF", "沪深300股指期货", "股指期货", "#2f54eb", False),
    ("IC", "中证500股指期货", "股指期货", "#597ef7", False),
    ("T", "十年国债期货", "利率对冲", "#52c41a", False),
)


def identify_equity_references(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    provider: MarketDataProvider,
    top_n: int = 6,
    rolling_window: int = 12,
) -> VarietyIdentificationResult:
    """Rank equity-index CTA market references using aligned return history.

    The returned shape intentionally matches ``VarietyIdentificationResult``
    so report serialization stays backwards compatible.  Its candidates are
    market references, never reconstructed contracts or positions.
    """
    if len(product_returns) < _MIN_OBS:
        return VarietyIdentificationResult([], [], 0.0, warnings=[f"样本量 {len(product_returns)} 期不足 {_MIN_OBS} 期，无法进行股指 CTA 市场线索推断"])

    product = pd.Series(product_returns, index=pd.to_datetime(product_dates), dtype=float)
    if frequency == "weekly":
        product = product.groupby(product.index.to_period("W-SUN")).last()
    elif frequency == "monthly":
        product = product.groupby(product.index.to_period("M")).last()
    else:
        product.index = product.index.normalize()

    market_series: dict[str, pd.Series] = {}
    metadata = {symbol: (name, group, color) for symbol, name, group, color, _ in _EQUITY_REFERENCES}
    for symbol, _name, _group, _color, is_index in _EQUITY_REFERENCES:
        try:
            returns = provider.get_returns(symbol, min(product_dates), max(product_dates), is_index=is_index)
        except Exception:
            continue
        if returns.empty:
            continue
        series = returns.copy()
        series.index = pd.to_datetime(series.index)
        if frequency == "weekly":
            series = (1.0 + series).resample("W-SUN").prod() - 1.0
            series.index = series.index.to_period("W-SUN")
        elif frequency == "monthly":
            series = (1.0 + series).resample("ME").prod() - 1.0
            series.index = series.index.to_period("M")
        else:
            series.index = series.index.normalize()
        market_series[symbol] = series

    if not market_series:
        return VarietyIdentificationResult([], [], 0.0, warnings=["未获取到股指或国债期货市场参考数据，已跳过股指 CTA 市场线索"])

    frame = pd.DataFrame({symbol: series.reindex(product.index) for symbol, series in market_series.items()})
    frame["__product__"] = product
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < _MIN_OBS:
        return VarietyIdentificationResult([], [], 0.0, warnings=["产品净值与股指/期货参考没有足够的实际日期交集，已停止市场线索推断"])

    symbols = [symbol for symbol in market_series if symbol in frame.columns]
    X = frame[symbols].to_numpy(dtype=float)
    y = frame["__product__"].to_numpy(dtype=float)
    correlations = {symbol: round(safe_pearson(y, X[:, index]), 4) for index, symbol in enumerate(symbols)}
    lasso = _lasso_regression(y, X, symbols)
    hit_rates, stability = _rolling_stability(y, X, symbols, rolling_window, min(3, len(symbols)))
    max_lasso = max((abs(value) for value in lasso.values()), default=0.0)

    candidates: list[VarietyCandidate] = []
    for symbol in symbols:
        name, group, _color = metadata[symbol]
        corr, coefficient, hit_rate = correlations[symbol], lasso.get(symbol, 0.0), hit_rates.get(symbol, 0.0)
        normalized_lasso = abs(coefficient) / max_lasso if max_lasso else 0.0
        score = min(100.0, 100 * (0.40 * abs(corr) + 0.35 * normalized_lasso + 0.25 * hit_rate))
        evidence = [f"与产品收益相关系数 {corr:.3f}"]
        evidence.append(f"LASSO 系数 {coefficient:.4f}" if coefficient else "LASSO 未选中")
        if hit_rate:
            evidence.append(f"在 {hit_rate:.0%} 的滚动窗口中进入 Top-3")
        candidates.append(VarietyCandidate(symbol, name, group, round(score, 1), corr, coefficient, hit_rate, evidence))
    candidates.sort(key=lambda item: item.probability_pct, reverse=True)

    grouped: dict[str, list[VarietyCandidate]] = {}
    for candidate in candidates:
        if candidate.probability_pct > 5:
            grouped.setdefault(candidate.sector, []).append(candidate)
    sectors = [
        SectorExposure(group, metadata[next(item.symbol for item in items)][2], round(sum(item.probability_pct for item in items), 1), max(items, key=lambda item: item.probability_pct).name, len(items))
        for group, items in grouped.items()
    ]
    sectors.sort(key=lambda item: item.total_probability, reverse=True)
    evidence = [f"共扫描 {len(symbols)} 个股指 CTA 市场参考", f"结论跨窗口稳定性：{stability:.0%}"]
    if candidates:
        evidence.insert(1, "排名靠前的市场参考：" + "、".join(f"{item.name}({item.probability_pct}%)" for item in candidates[:3]))
    warnings = []
    if len(y) < 60:
        warnings.append(f"样本仅 {len(y)} 期，股指 CTA 市场线索置信度受限，建议至少 60 期以上")
    if stability < 0.5:
        warnings.append(f"结论跨窗口稳定性较低（{stability:.0%}），参考排名可能随时间变化")
    return VarietyIdentificationResult(candidates[:top_n], sectors, round(stability, 3), evidence, warnings)
