"""Wall-clock performance benchmarks for the three heaviest analysis modules.

These tests are skipped in the default suite because wall-clock benchmarks are
noisy and machine-dependent.  Run them explicitly:

    RUN_BENCHMARKS=1 pytest tests/test_performance_benchmarks.py -v -s

Design notes:
- Uses stdlib ``time.perf_counter`` (the repo has no pytest config and no
  pytest-benchmark dependency).
- Synthetic data only: seeded random walks, zero external I/O, fully offline.
- Each benchmark takes the median of 3 runs and asserts (a) structurally valid
  output and (b) a generous absolute ceiling as a regression tripwire.  The
  ceilings are deliberately 5-10x observed times, not performance targets.
- ``deep_attribution`` runs with ``bootstrap_samples=20`` (default is 80) so
  the benchmark isolates the deterministic core (ElasticNetCV + permutation
  importance); the bootstrap loop is a linear multiple of that cost.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.schemas import (
    CtaRankingMarketPoint,
    CtaRankingMarketSeries,
    CtaRankingProductInput,
    CtaRankingRequest,
    DataFrequency,
    NetAssetValuePoint,
)
from app.services.analysis.deep_attribution import deep_attribution
from app.services.analysis.factor_analyzer import analyze_factors
from app.services.codex_cta_ranking import rank_cta_products
from app.services.market_data.provider import MarketDataProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BENCHMARKS") != "1",
    reason="性能基准仅在 RUN_BENCHMARKS=1 时运行，默认套件保持快速",
)

SIZES = (100, 1000, 10000)

# Generous absolute ceilings (seconds).  These are regression tripwires for an
# order-of-magnitude slowdown (accidental O(n^2) loops etc.), not perf goals.
CEILING_SECONDS = {
    "analyze_factors": 120.0,
    "deep_attribution": 240.0,
    "rank_cta_products": 180.0,
}

_RUNS_PER_SAMPLE = 3


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _business_days(n: int, start: date = date(2023, 1, 2)) -> list[date]:
    dates: list[date] = []
    day = start
    while len(dates) < n:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return dates


def _random_walk_returns(
    rng: np.random.Generator, n: int, drift: float = 0.0004, vol: float = 0.01
) -> np.ndarray:
    """Periodic returns of a mild upward-trending random walk."""
    return rng.normal(drift, vol, n)


def _weekly_nav_points(
    rng: np.random.Generator, n: int, start: date = date(2020, 1, 3)
) -> list[NetAssetValuePoint]:
    """Positive geometric-random-walk weekly NAV series (chronological)."""
    weekly_returns = rng.normal(0.002, 0.02, n)
    nav = 1.0
    points: list[NetAssetValuePoint] = []
    day = start
    for periodic_return in weekly_returns:
        nav *= 1.0 + periodic_return
        points.append(NetAssetValuePoint(observation_date=day, net_asset_value=round(nav, 6)))
        day += timedelta(days=7)
    return points


def _make_provider(rng: np.random.Generator) -> MarketDataProvider:
    """Provider that synthesizes a deterministic random-walk series for ANY
    requested symbol over the requested calendar range (business-day index).
    """
    provider = MagicMock(spec=MarketDataProvider)

    def _get_returns(symbol: str, start: date, end: date, is_index: bool = True) -> pd.Series:
        days = max((end - start).days + 1, 1)
        index = pd.to_datetime(_business_days(days, start))
        values = _random_walk_returns(rng, len(index), vol=0.008)
        return pd.Series(values, index=index)

    provider.get_returns.side_effect = _get_returns
    return provider


def _median_seconds(call, repeats: int = _RUNS_PER_SAMPLE) -> float:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_benchmark_analyze_factors(n: int) -> None:
    """analyze_factors: daily frequency, commodity CTA, rolling OLS core."""
    rng = np.random.default_rng(42)
    returns = _random_walk_returns(rng, n)
    dates = _business_days(n)
    provider = _make_provider(rng)

    def run() -> None:
        result = analyze_factors(returns, dates, "daily", "commodity_cta", provider, rolling_window=12)
        assert result.r_squared >= 0.0
        # With 100+ aligned observations the factor set must be non-empty.
        assert result.factors, result.warnings

    seconds = _median_seconds(run)
    assert seconds < CEILING_SECONDS["analyze_factors"], f"analyze_factors n={n} took {seconds:.2f}s"
    print(f"[bench] analyze_factors n={n:>6}: {seconds * 1000:8.1f} ms")


@pytest.mark.parametrize("n", SIZES)
def test_benchmark_deep_attribution(n: int) -> None:
    """deep_attribution: daily frequency, bootstrap_samples=20 to isolate the
    deterministic ElasticNet + permutation-importance core."""
    rng = np.random.default_rng(7)
    returns = _random_walk_returns(rng, n, vol=0.009)
    dates = _business_days(n)
    provider = _make_provider(rng)

    def run() -> None:
        result = deep_attribution(
            returns, dates, "daily", "commodity_cta", provider, bootstrap_samples=20
        )
        assert result.diagnostics.get("observation_count", 0) >= 24
        assert result.asset_class["candidates"], result.warnings
        assert result.sector_exposures

    seconds = _median_seconds(run)
    assert seconds < CEILING_SECONDS["deep_attribution"], f"deep_attribution n={n} took {seconds:.2f}s"
    print(f"[bench] deep_attribution n={n:>6}: {seconds * 1000:8.1f} ms")


@pytest.mark.parametrize("n", SIZES)
def test_benchmark_rank_cta_products(n: int) -> None:
    """rank_cta_products: weekly NAV, one product, one market series, formal
    rank (>= 26 periodic returns)."""
    rng = np.random.default_rng(123)
    nav_points = _weekly_nav_points(rng, n)
    dates = [point.observation_date for point in nav_points]
    market_points = [
        CtaRankingMarketPoint(observation_date=day, return_value=float(rng.normal(0.001, 0.015)))
        for day in dates
    ]
    market_series = [
        CtaRankingMarketSeries(series_id="bench", series_name="bench-proxy", points=market_points)
    ]
    product = CtaRankingProductInput(
        product_id="bench-1",
        product_name="bench product",
        nav_points=nav_points,
        frequency=DataFrequency.WEEKLY,
        strategy="commodity_cta",
    )

    def run() -> None:
        request = CtaRankingRequest(products=[product], market_series=market_series)
        response = rank_cta_products(request)
        assert len(response.rankings) == 1
        item = response.rankings[0]
        # Single-product universe has no cross-sectional differentiation, so
        # the ranker returns status "eligible" with a neutral 50.0 score.
        assert item.status in ("eligible", "formal_rank", "insufficient_data"), item.status
        if n >= 100:
            # 99 weekly returns >= MIN_FORMAL_RETURNS(26): must be eligible.
            assert item.status == "eligible"
            assert item.score is not None

    seconds = _median_seconds(run)
    assert seconds < CEILING_SECONDS["rank_cta_products"], f"rank_cta_products n={n} took {seconds:.2f}s"
    print(f"[bench] rank_cta_products n={n:>6}: {seconds * 1000:8.1f} ms")
