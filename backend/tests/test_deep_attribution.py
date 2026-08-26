"""Tests for the evidence-first nonlinear CTA attribution module."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.services.analysis.deep_attribution import deep_attribution
from app.services.market_data.provider import MarketDataProvider


def _dates(count: int) -> list[date]:
    start = date(2024, 1, 2)
    return [start + timedelta(days=index) for index in range(count)]


def _provider(series_by_symbol: dict[str, pd.Series]) -> MarketDataProvider:
    provider = MagicMock(spec=MarketDataProvider)
    provider.get_returns.side_effect = lambda symbol, _start, _end, is_index=True: series_by_symbol.get(symbol, pd.Series(dtype=float))
    return provider


def test_deep_attribution_reports_strategy_and_sector_candidates() -> None:
    rng = np.random.default_rng(42)
    dates = _dates(140)
    base = rng.normal(0, 0.01, len(dates))
    rb = base + rng.normal(0, 0.002, len(dates))
    cu = rng.normal(0, 0.009, len(dates))
    au = rng.normal(0, 0.006, len(dates))
    product = np.sign(pd.Series(rb).rolling(5, min_periods=5).sum().shift(1).fillna(0)) * rb
    product = product.to_numpy() + rng.normal(0, 0.001, len(dates))
    market = {
        symbol: pd.Series(values, index=pd.to_datetime(dates))
        for symbol, values in {
            "rb": rb,
            "i": rb + rng.normal(0, 0.002, len(dates)),
            "cu": cu,
            "al": cu + rng.normal(0, 0.002, len(dates)),
            "au": au,
            "sc": rng.normal(0, 0.01, len(dates)),
            "ta": rng.normal(0, 0.01, len(dates)),
            "m": rng.normal(0, 0.009, len(dates)),
            "y": rng.normal(0, 0.009, len(dates)),
        }.items()
    }
    result = deep_attribution(product, dates, "daily", "commodity_cta", _provider(market), bootstrap_samples=25)

    assert result.asset_class["candidates"]
    assert result.sector_exposures
    assert result.strategy_fingerprints
    assert result.diagnostics["observation_count"] >= 24
    assert any("高分化" in item["state"] for item in result.state_analysis)
    assert all("relative_mean_return" in item for item in result.state_analysis)
    assert any(item["strategy"] == "carry_proxy" for item in result.strategy_fingerprints)
    assert "持仓" in result.disclaimer


def test_deep_attribution_stops_when_scope_is_unidentifiable() -> None:
    dates = _dates(12)
    returns = np.zeros(len(dates))
    provider = _provider({})

    result = deep_attribution(returns, dates, "daily", "commodity_cta", provider)

    assert result.diagnostics["reliability_label"] == "不可识别"
    assert result.sector_exposures == []
    assert result.warnings
