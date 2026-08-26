"""Boundary and exception tests for the core statistics pipeline.

Covers the degenerate inputs called out in the code review: empty and
single-point NAV series, unsorted dates, and NaN/Inf/zero/negative NAV cells
flowing through the periodic-return computation, OLS regression and quality
gates.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.services.analysis._pipeline_utils import compute_periodic_returns
from app.services.analysis.factor_analyzer import _run_ols_regression
from app.services.nav_quality import (
    assess_machine_nav_review,
    assess_nav_quality,
)


def _nav_point(day: date, nav: float | None) -> SimpleNamespace:
    """A NAV observation with ``observation_date`` / ``nav`` attributes."""
    return SimpleNamespace(observation_date=day, nav=nav)


def _pipeline_point(day: date, nav: float | None) -> SimpleNamespace:
    """A NAV point for compute_periodic_returns (``net_asset_value``)."""
    return SimpleNamespace(observation_date=day, net_asset_value=nav)


def _day(offset: int) -> date:
    return date(2026, 1, 1) + timedelta(days=offset)


# ---------------------------------------------------------------------------
# compute_periodic_returns
# ---------------------------------------------------------------------------


class TestComputePeriodicReturns:
    def test_unsorted_input_is_sorted_by_date(self) -> None:
        points = [
            _pipeline_point(_day(2), 1.21),
            _pipeline_point(_day(0), 1.0),
            _pipeline_point(_day(1), 1.1),
        ]

        returns, dates = compute_periodic_returns(points)

        np.testing.assert_allclose(returns, [0.1, 0.1])
        assert dates == [_day(1), _day(2)]

    def test_nan_zero_negative_and_none_navs_are_dropped(self) -> None:
        points = [
            _pipeline_point(_day(0), 1.0),
            _pipeline_point(_day(1), float("nan")),
            _pipeline_point(_day(2), 1.1),
            _pipeline_point(_day(3), 0.0),
            _pipeline_point(_day(4), -0.5),
            _pipeline_point(_day(5), None),
            _pipeline_point(_day(6), 1.21),
        ]

        returns, dates = compute_periodic_returns(points)

        # Clean chain: 1.0 -> 1.1 -> 1.21 with the later date of each pair.
        np.testing.assert_allclose(returns, [0.1, 0.1])
        assert dates == [_day(2), _day(6)]

    def test_inf_nav_is_dropped(self) -> None:
        points = [
            _pipeline_point(_day(0), 1.0),
            _pipeline_point(_day(1), float("inf")),
            _pipeline_point(_day(2), 1.05),
        ]

        returns, dates = compute_periodic_returns(points)

        np.testing.assert_allclose(returns, [0.05])
        assert dates == [_day(2)]

    def test_empty_and_single_point_yield_empty(self) -> None:
        returns, dates = compute_periodic_returns([])
        assert returns.size == 0 and dates == []

        returns, dates = compute_periodic_returns([_pipeline_point(_day(0), 1.0)])
        assert returns.size == 0 and dates == []

    def test_all_invalid_points_yield_empty(self) -> None:
        points = [
            _pipeline_point(_day(0), float("nan")),
            _pipeline_point(_day(1), 0.0),
            _pipeline_point(_day(2), None),
        ]

        returns, dates = compute_periodic_returns(points)

        assert returns.size == 0 and dates == []


# ---------------------------------------------------------------------------
# _run_ols_regression
# ---------------------------------------------------------------------------


class TestOlsRegressionBoundaries:
    def test_all_nan_rows_yield_empty_result(self) -> None:
        X = pd.DataFrame({"beta": [np.nan, np.nan, np.nan]})
        y = np.array([np.nan, np.nan, np.nan])

        assert _run_ols_regression(y, X) == []

    def test_partial_nan_rows_are_dropped_not_poisoned(self) -> None:
        X = pd.DataFrame({"beta": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]})
        y = np.array([0.010, np.nan, 0.020, 0.005, float("inf"), 0.012])

        results = _run_ols_regression(y, X)

        assert len(results) == 1
        detail = results[0]
        assert np.isfinite(detail.exposure_beta)
        assert np.isfinite(detail.t_statistic)
        assert np.isfinite(detail.p_value)
        assert 0.0 <= detail.p_value <= 1.0

    def test_inf_proxy_column_is_dropped_with_row(self) -> None:
        X = pd.DataFrame({
            "beta": [0.01, float("inf"), 0.03, 0.01, -0.01],
            "momentum": [0.02, 0.01, 0.04, 0.00, 0.01],
        })
        y = np.array([0.010, 0.015, 0.020, 0.005, 0.012])

        results = _run_ols_regression(y, X)

        assert len(results) == 2
        assert all(np.isfinite(r.exposure_beta) for r in results)


# ---------------------------------------------------------------------------
# nav_quality gates
# ---------------------------------------------------------------------------


class TestNavQualityBoundaries:
    def test_empty_series_passes_without_metrics(self) -> None:
        result = assess_nav_quality([], [])

        assert result["status"] == "passed"
        assert result["blocking"] is False
        assert result["computed_cumulative_return"] is None
        assert result["computed_maximum_drawdown"] is None

    def test_single_point_passes_without_metrics(self) -> None:
        result = assess_nav_quality([_nav_point(_day(0), 1.2)], [])

        assert result["status"] == "passed"
        assert result["computed_cumulative_return"] is None

    def test_nan_negative_and_inf_navs_are_excluded_from_metrics(self) -> None:
        points = [
            _nav_point(_day(0), float("nan")),
            _nav_point(_day(1), -1.0),
            _nav_point(_day(2), 1.0),
            _nav_point(_day(3), float("inf")),
            _nav_point(_day(4), 1.05),
        ]

        result = assess_nav_quality(points, [])

        assert result["computed_cumulative_return"] == 0.05
        assert result["computed_maximum_drawdown"] == 0.0
        assert result["largest_period_change"] == 0.05
        assert np.isfinite(result["computed_cumulative_return"])

    def test_machine_review_reports_dropped_invalid_points(self) -> None:
        points = [
            _nav_point(_day(i), 1.0 + 0.001 * i) for i in range(7)
        ] + [_nav_point(_day(7), float("inf"))]

        result = assess_machine_nav_review(
            points, [], source_confidence=0.95, declared_frequency="weekly"
        )

        # The invalid cell is excluded from computation and reported openly.
        assert any("已从计算中剔除" in reason for reason in result["reasons"])
        assert result["quality"]["computed_cumulative_return"] is not None

    def test_machine_review_ignores_nan_points_for_flatness_check(self) -> None:
        points = [
            _nav_point(_day(i), 1.0 + 0.001 * i) for i in range(8)
        ] + [_nav_point(_day(8), float("nan"))]

        result = assess_machine_nav_review(
            points, [], source_confidence=0.95, declared_frequency="weekly"
        )

        # The NaN cell must not crash the flat-curve check.
        assert result["confidence"] == 0.95
