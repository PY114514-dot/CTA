"""Unit tests for the strategy-type classifier (analysis step 1).

Mocks MarketDataProvider.get_returns to control benchmark data and verify
classification outcomes without network access.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.analysis.strategy_classifier import (
    StrategyClassification,
    classify_strategy,
    _classify_disclosure_hint,
    _group_score,
    _confidence_label,
)
from app.services.market_data.provider import MarketDataProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(returns_map: dict[str, pd.Series]) -> MarketDataProvider:
    """Create a mock provider whose get_returns dispatches to *returns_map*."""
    provider = MagicMock(spec=MarketDataProvider)

    def _get_returns(symbol: str, start: date, end: date, is_index: bool = True) -> pd.Series:
        return returns_map.get(symbol, pd.Series(dtype=float))

    provider.get_returns.side_effect = _get_returns
    return provider


def _daily_dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    """Generate *n* consecutive business-day-like dates (skip weekends)."""
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_daily_series(values: np.ndarray, start: date = date(2024, 1, 2)) -> pd.Series:
    """Build a pd.Series indexed by business dates from an array of returns."""
    dates = _daily_dates(len(values), start)
    return pd.Series(values, index=pd.to_datetime(dates))


# ---------------------------------------------------------------------------
# Tests: insufficient data
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_too_few_observations(self) -> None:
        """Fewer than 8 observations → insufficient_data."""
        product_returns = np.array([0.01, 0.02, -0.01])
        product_dates = _daily_dates(3)
        provider = _make_provider({})

        result = classify_strategy(product_returns, product_dates, "daily", provider)

        assert result.strategy_type == "insufficient_data"
        assert result.confidence_pct == 0.0
        assert result.confidence_label == "低"

    def test_no_benchmark_data(self) -> None:
        """Enough product data but no benchmark → insufficient_data."""
        n = 30
        product_returns = np.random.default_rng(42).normal(0, 0.01, n)
        product_dates = _daily_dates(n)
        provider = _make_provider({})  # empty — no benchmarks

        result = classify_strategy(product_returns, product_dates, "daily", provider)

        assert result.strategy_type == "insufficient_data"
        assert "无法获取" in result.evidence[0]


# ---------------------------------------------------------------------------
# Tests: explicit strategy confirmation
# ---------------------------------------------------------------------------


class TestUserConfirmation:
    def test_user_confirmation_is_explicit_and_conditional(self) -> None:
        result = classify_strategy(
            np.array([0.01, 0.02, -0.01]),
            _daily_dates(3),
            "daily",
            _make_provider({}),
            strategy_hint="低波 CTA",
            strategy_confirmation="commodity_cta",
        )

        assert result.strategy_type == "commodity_cta"
        assert result.details["user_confirmation"]["confirmed"] is True
        assert result.details["user_confirmation"]["statistical_type"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Tests: commodity CTA classification
# ---------------------------------------------------------------------------


class TestCommodityCTA:
    def test_high_commodity_correlation(self) -> None:
        """Product perfectly correlated with commodity indices → commodity_cta."""
        rng = np.random.default_rng(123)
        n = 60
        # Shared signal drives both product and commodity benchmarks
        signal = rng.normal(0, 0.01, n)
        product_returns = signal + rng.normal(0, 0.001, n)

        dates = _daily_dates(n)
        returns_map = {
            # Commodity indices highly correlated with product
            "nh_commodity": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "nh_industrial": _make_daily_series(signal + rng.normal(0, 0.002, n)),
            "nh_agriculture": _make_daily_series(signal + rng.normal(0, 0.002, n)),
            "nh_metal": _make_daily_series(signal + rng.normal(0, 0.002, n)),
            "nh_energy": _make_daily_series(signal + rng.normal(0, 0.002, n)),
            # Equity indices uncorrelated
            "hs300": _make_daily_series(rng.normal(0, 0.01, n)),
            "zz500": _make_daily_series(rng.normal(0, 0.01, n)),
            "zz1000": _make_daily_series(rng.normal(0, 0.01, n)),
        }
        provider = _make_provider(returns_map)

        result = classify_strategy(product_returns, dates, "daily", provider)

        assert result.strategy_type == "commodity_cta"
        assert result.confidence_pct > 50
        assert len(result.correlations) > 0

    def test_correlations_dict_populated(self) -> None:
        """The correlations dict should contain entries for fetched benchmarks."""
        rng = np.random.default_rng(7)
        n = 30
        signal = rng.normal(0, 0.01, n)
        dates = _daily_dates(n)
        returns_map = {
            "nh_commodity": _make_daily_series(signal),
            "hs300": _make_daily_series(rng.normal(0, 0.01, n)),
        }
        provider = _make_provider(returns_map)

        result = classify_strategy(signal, dates, "daily", provider)

        assert "nh_commodity" in result.correlations
        assert "hs300" in result.correlations


# ---------------------------------------------------------------------------
# Tests: equity quant classification
# ---------------------------------------------------------------------------


class TestEquityQuant:
    def test_high_equity_correlation(self) -> None:
        """Product perfectly correlated with equity indices → equity_quant."""
        rng = np.random.default_rng(456)
        n = 60
        signal = rng.normal(0, 0.012, n)
        product_returns = signal + rng.normal(0, 0.001, n)

        dates = _daily_dates(n)
        returns_map = {
            # Equity indices highly correlated
            "hs300": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "zz500": _make_daily_series(signal + rng.normal(0, 0.002, n)),
            "zz1000": _make_daily_series(signal + rng.normal(0, 0.002, n)),
            # Commodity indices uncorrelated
            "nh_commodity": _make_daily_series(rng.normal(0, 0.008, n)),
            "nh_industrial": _make_daily_series(rng.normal(0, 0.008, n)),
            "nh_agriculture": _make_daily_series(rng.normal(0, 0.008, n)),
            "nh_metal": _make_daily_series(rng.normal(0, 0.008, n)),
            "nh_energy": _make_daily_series(rng.normal(0, 0.008, n)),
        }
        provider = _make_provider(returns_map)

        result = classify_strategy(product_returns, dates, "daily", provider)

        assert result.strategy_type == "equity_quant"
        assert result.confidence_pct > 50


# ---------------------------------------------------------------------------
# Tests: mixed classification
# ---------------------------------------------------------------------------


class TestMixed:
    def test_equal_correlation_both_groups(self) -> None:
        """Similar correlation to both groups → mixed."""
        rng = np.random.default_rng(789)
        n = 60
        signal = rng.normal(0, 0.01, n)
        product_returns = signal.copy()

        dates = _daily_dates(n)
        # Both groups equally correlated with product
        returns_map = {
            "hs300": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "zz500": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "zz1000": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "nh_commodity": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "nh_industrial": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "nh_agriculture": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "nh_metal": _make_daily_series(signal + rng.normal(0, 0.001, n)),
            "nh_energy": _make_daily_series(signal + rng.normal(0, 0.001, n)),
        }
        provider = _make_provider(returns_map)

        result = classify_strategy(product_returns, dates, "daily", provider)

        assert result.strategy_type == "mixed"


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_disclosure_hint_keeps_asset_class_conservative(self) -> None:
        commodity = _classify_disclosure_hint("低波商品CTA策略")
        assert commodity["detected_type"] == "commodity_cta"
        assert commodity["explicit_cta"] is True

        mixed = _classify_disclosure_hint("覆盖商品、股指和国债的多资产CTA")
        assert mixed["detected_type"] == "mixed"

        generic = _classify_disclosure_hint("低波CTA策略")
        assert generic["detected_type"] is None
        assert generic["explicit_cta"] is True

    def test_group_score_empty(self) -> None:
        assert _group_score({}) == 0.0

    def test_group_score_single(self) -> None:
        score = _group_score({"hs300": 0.8})
        # 0.7 * 0.8 + 0.3 * 0.8 = 0.8
        assert abs(score - 0.8) < 1e-10

    def test_group_score_multiple(self) -> None:
        score = _group_score({"a": 0.6, "b": 0.4})
        # max=0.6, mean=0.5 → 0.7*0.6 + 0.3*0.5 = 0.42 + 0.15 = 0.57
        assert abs(score - 0.57) < 1e-10

    def test_confidence_label_high(self) -> None:
        assert _confidence_label(75) == "高"

    def test_confidence_label_medium(self) -> None:
        assert _confidence_label(50) == "中"

    def test_confidence_label_low(self) -> None:
        assert _confidence_label(20) == "低"
