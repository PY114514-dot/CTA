"""Shared utilities for the analysis pipeline endpoints.

Extracts the repeated "provider selection + periodic returns computation"
logic that was duplicated across /api/analysis/{classify,factors,varieties,report}.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.services.market_data import AKShareProvider, CsvMarketDataProvider


def select_provider(
    data_source: str,
    akshare_provider: "AKShareProvider",
    csv_provider: "CsvMarketDataProvider",
) -> "AKShareProvider | CsvMarketDataProvider":
    """Choose the market-data provider based on request preference.

    Falls back to CSV when the API is requested but unavailable.
    """
    provider = csv_provider if data_source == "upload" else akshare_provider
    if data_source == "api" and not akshare_provider.is_available():
        provider = csv_provider
    return provider


def compute_periodic_returns(
    nav_points: list,
) -> tuple[np.ndarray, list[date]]:
    """Compute simple periodic returns from a NAV point list.

    Parameters
    ----------
    nav_points : list
        Objects with ``.net_asset_value`` (float) and ``.observation_date`` (date).

    Returns
    -------
    periodic_returns : np.ndarray
        Array of length N-1 with simple returns.
    return_dates : list[date]
        Corresponding dates (the later date of each pair).
    """
    nav_values = np.array([p.net_asset_value for p in nav_points], dtype=float)
    dates = [p.observation_date for p in nav_points]
    periodic_returns = np.diff(nav_values) / nav_values[:-1]
    return_dates = dates[1:]
    return periodic_returns, return_dates


def align_product_and_market_returns(
    product_returns: np.ndarray,
    product_dates: list[date],
    market_returns: pd.Series,
    frequency: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align return series by the same disclosed calendar period.

    Never use positional truncation here: a Thursday-priced weekly NAV and a
    Sunday-labelled market week represent the same period, while two arbitrary
    "last N" observations may be weeks apart.  The function keeps only actual
    period intersections and rejects thin coverage.
    """
    if market_returns.empty or len(product_returns) != len(product_dates):
        return None
    product = pd.Series(product_returns, index=pd.to_datetime(product_dates), dtype=float)
    market = market_returns.copy().astype(float)
    market.index = pd.to_datetime(market.index)
    market = market[~market.index.duplicated(keep="last")].sort_index()

    if frequency == "weekly":
        product = product.groupby(product.index.to_period("W-SUN")).last()
        market = ((1.0 + market).resample("W-SUN").prod() - 1.0)
        market.index = market.index.to_period("W-SUN")
    elif frequency == "monthly":
        product = product.groupby(product.index.to_period("M")).last()
        market = ((1.0 + market).resample("ME").prod() - 1.0)
        market.index = market.index.to_period("M")
    else:
        product.index = product.index.normalize()
        market.index = market.index.normalize()

    common = product.index.intersection(market.index)
    if len(common) < 8 or len(common) / max(len(product), 1) < 0.70:
        return None
    paired = pd.concat([product.loc[common], market.loc[common]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 8:
        return None
    return paired.iloc[:, 0].to_numpy(dtype=float), paired.iloc[:, 1].to_numpy(dtype=float)


def assess_nav_quality(nav_points: list) -> list[str]:
    """Return transparent quality warnings before statistical attribution.

    Image tracing can create interpolated/flat daily points.  Those points are
    useful for visual review but can manufacture attractive regression results,
    so the report must show the condition before presenting any factor output.
    """
    if len(nav_points) < 2:
        return ["净值点不足，无法检查数据质量。"]

    values = np.asarray([point.net_asset_value for point in nav_points], dtype=float)
    dates = [point.observation_date for point in nav_points]
    warnings: list[str] = []
    if not np.isfinite(values).all() or (values <= 0).any():
        return ["净值存在非数值或非正值，已停止因子解释。"]

    returns = np.diff(values) / values[:-1]
    zero_ratio = float(np.mean(np.isclose(returns, 0.0, atol=1e-12)))
    unique_ratio = len(np.unique(values)) / len(values)
    max_abs_return = float(np.max(np.abs(returns)))
    max_gap_days = max((dates[index] - dates[index - 1]).days for index in range(1, len(dates)))

    if zero_ratio >= 0.35:
        warnings.append(f"净值相邻期零收益占 {zero_ratio:.1%}；可能包含图像插值/停牌填充，因子结果需复核。")
    if unique_ratio <= 0.50:
        warnings.append(f"净值仅有 {unique_ratio:.1%} 的不同取值；曲线像素量化可能降低统计有效性。")
    if max_abs_return >= 0.10:
        warnings.append(f"检测到单期 {max_abs_return:.1%} 的异常跳变；请先在原图或净值表中确认。")
    if max_gap_days > 7:
        warnings.append(f"存在最长 {max_gap_days} 天的净值日期缺口；请确认频率及缺失期处理。")
    return warnings
