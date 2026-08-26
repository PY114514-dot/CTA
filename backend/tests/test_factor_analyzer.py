"""Unit tests for the factor exposure analyzer (analysis step 2).

Includes:
- Direct OLS verification with hand-computed coefficients.
- Full analyze_factors integration with a mocked provider.
- Edge-case handling (insufficient observations).
"""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.analysis.factor_analyzer import (
    FactorAnalysisResult,
    FactorExposureDetail,
    analyze_factors,
    _build_collinearity_diagnostics,
    _build_risk_profile,
    _principal_component_summary,
    _run_ridge_rolling_regression,
    _run_ols_regression,
    _factor_confidence,
    FACTOR_LABELS,
)
from app.services.market_data.provider import MarketDataProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(returns_map: dict[str, pd.Series]) -> MarketDataProvider:
    """Create a mock provider dispatching get_returns to *returns_map*."""
    provider = MagicMock(spec=MarketDataProvider)

    def _get_returns(symbol: str, start: date, end: date, is_index: bool = True) -> pd.Series:
        return returns_map.get(symbol, pd.Series(dtype=float))

    provider.get_returns.side_effect = _get_returns
    return provider


def _daily_dates(n: int, start: date = date(2023, 6, 1)) -> list[date]:
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_daily_series(values: np.ndarray, start: date = date(2023, 6, 1)) -> pd.Series:
    dates = _daily_dates(len(values), start)
    return pd.Series(values, index=pd.to_datetime(dates))


# ---------------------------------------------------------------------------
# Tests: _run_ols_regression with hand-verified coefficients
# ---------------------------------------------------------------------------


class TestOLSRegression:
    def test_single_factor_exact(self) -> None:
        """y = 3.0 * x + 0 (no noise) → beta ≈ 3.0, correlation ≈ 1.0."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(0, 1, n)
        y = 3.0 * x  # perfect linear, zero intercept

        X = pd.DataFrame({"beta": x})
        results = _run_ols_regression(y, X)

        assert len(results) == 1
        detail = results[0]
        assert detail.factor_name == "beta"
        assert abs(detail.exposure_beta - 3.0) < 0.01
        assert abs(detail.correlation - 1.0) < 0.01
        assert abs(detail.t_statistic) > 10  # highly significant

    def test_two_factors_known_coefficients(self) -> None:
        """y = 2.0*x1 + 0.5*x2 → verify both coefficients recovered."""
        rng = np.random.default_rng(99)
        n = 200
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2.0 * x1 + 0.5 * x2 + rng.normal(0, 0.01, n)  # tiny noise

        X = pd.DataFrame({"momentum": x1, "volatility": x2})
        results = _run_ols_regression(y, X)

        by_name = {r.factor_name: r for r in results}
        assert abs(by_name["momentum"].exposure_beta - 2.0) < 0.05
        assert abs(by_name["volatility"].exposure_beta - 0.5) < 0.05

    def test_zero_exposure(self) -> None:
        """y is pure noise, uncorrelated with x → beta ≈ 0, low confidence."""
        rng = np.random.default_rng(7)
        n = 100
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)  # independent

        X = pd.DataFrame({"carry": x})
        results = _run_ols_regression(y, X)

        detail = results[0]
        assert abs(detail.exposure_beta) < 0.3
        assert detail.confidence_label in ("低", "中")

    def test_results_sorted_by_abs_tstat(self) -> None:
        """Output factors are sorted by |t-statistic| descending."""
        rng = np.random.default_rng(11)
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        # x1 has strong effect, x2 has none
        y = 5.0 * x1 + rng.normal(0, 0.01, n)

        X = pd.DataFrame({"weak": x2, "strong": x1})
        results = _run_ols_regression(y, X)

        assert results[0].factor_name == "strong"
        assert abs(results[0].t_statistic) >= abs(results[1].t_statistic)

    def test_p_value_range(self) -> None:
        """p-values must be in [0, 1]."""
        rng = np.random.default_rng(55)
        n = 60
        x = rng.normal(0, 1, n)
        y = 1.5 * x + rng.normal(0, 0.5, n)

        X = pd.DataFrame({"beta": x})
        results = _run_ols_regression(y, X)

        for r in results:
            assert 0.0 <= r.p_value <= 1.0


class TestCollinearityDiagnostics:
    def test_highly_overlapping_proxies_trigger_ridge(self) -> None:
        """Nearly identical proxies make individual rolling OLS betas unstable."""
        rng = np.random.default_rng(123)
        base = rng.normal(0, 0.01, 80)
        X = pd.DataFrame({
            "beta": base,
            "momentum": base + rng.normal(0, 0.00001, 80),
        })

        diagnostics = _build_collinearity_diagnostics(X)

        assert diagnostics.max_abs_correlation is not None
        assert diagnostics.max_abs_correlation > 0.99
        assert diagnostics.ridge_applied is True
        assert diagnostics.ridge_alpha == 1.0
        assert diagnostics.high_correlation_pairs

        trend = _run_ridge_rolling_regression(base, X, window=20, alpha=1.0)
        assert len(trend["beta"]) > 0
        assert all(np.isfinite(value) for values in trend.values() for value in values)

    def test_independent_proxies_do_not_trigger_ridge(self) -> None:
        rng = np.random.default_rng(124)
        X = pd.DataFrame({
            "beta": rng.normal(0, 0.01, 300),
            "momentum": rng.normal(0, 0.01, 300),
        })

        diagnostics = _build_collinearity_diagnostics(X)

        assert diagnostics.max_abs_correlation is not None
        assert diagnostics.max_abs_correlation < 0.2
        assert diagnostics.ridge_applied is False


class TestFactorRiskProfile:
    def test_reports_downside_sensitivity_and_model_risk_contribution(self) -> None:
        rng = np.random.default_rng(725)
        market = rng.normal(0, 0.01, 100)
        trend = rng.normal(0, 0.01, 100)
        product = 1.4 * market + 0.2 * trend + rng.normal(0, 0.001, 100)
        X = pd.DataFrame({"beta": market, "momentum": trend})
        details = _run_ols_regression(product, X)

        profile = _build_risk_profile(product, X, details)

        assert profile.downside_betas["beta"] is not None
        assert profile.downside_betas["beta"] > 1.0
        assert profile.regime_returns["beta"]["downside_periods"] >= 6
        assert profile.variance_contributions_pct["beta"] is not None
        assert abs(sum(value for value in profile.variance_contributions_pct.values() if value is not None) - 100) < 0.2

    def test_pca_summarises_common_unlabelled_drivers(self) -> None:
        rng = np.random.default_rng(726)
        shared = rng.normal(0, 0.01, 80)
        X = pd.DataFrame({
            "beta": shared + rng.normal(0, 0.001, 80),
            "momentum": shared + rng.normal(0, 0.001, 80),
        })
        components = _principal_component_summary(shared, X)

        assert components
        assert components[0]["explained_variance_pct"] > 50
        assert {item["factor_name"] for item in components[0]["dominant_factors"]} == {"beta", "momentum"}


# ---------------------------------------------------------------------------
# Tests: full analyze_factors integration
# ---------------------------------------------------------------------------


class TestAnalyzeFactorsIntegration:
    def test_returns_valid_structure(self) -> None:
        """With sufficient mock data, analyze_factors returns a well-formed result."""
        rng = np.random.default_rng(2024)
        n = 100  # daily observations
        dates = _daily_dates(n)

        # Product returns driven by commodity index
        commodity_signal = rng.normal(0, 0.008, n + 50)  # extra for buffer
        product_returns = commodity_signal[-n:] + rng.normal(0, 0.002, n)

        # Provider returns long series for all requested symbols
        long_n = n + 50
        returns_map = {
            "nh_commodity": _make_daily_series(commodity_signal),
            "hs300": _make_daily_series(rng.normal(0, 0.01, long_n)),
            "rb": _make_daily_series(rng.normal(0, 0.009, long_n)),
            "cu": _make_daily_series(rng.normal(0, 0.009, long_n)),
            "au": _make_daily_series(rng.normal(0, 0.007, long_n)),
            "m": _make_daily_series(rng.normal(0, 0.008, long_n)),
        }
        provider = _make_provider(returns_map)

        result = analyze_factors(
            product_returns, dates, "daily", "commodity_cta", provider
        )

        assert isinstance(result, FactorAnalysisResult)
        assert len(result.factors) > 0
        assert 0.0 <= result.r_squared <= 1.0
        assert isinstance(result.evidence, list)

    def test_factor_labels_present(self) -> None:
        """Each factor detail should have a Chinese label from FACTOR_LABELS."""
        rng = np.random.default_rng(77)
        n = 80
        dates = _daily_dates(n)
        signal = rng.normal(0, 0.01, n + 40)
        product_returns = signal[-n:]

        long_n = n + 40
        returns_map = {
            "hs300": _make_daily_series(signal),
            "nh_commodity": _make_daily_series(rng.normal(0, 0.008, long_n)),
        }
        provider = _make_provider(returns_map)

        result = analyze_factors(
            product_returns, dates, "daily", "equity_quant", provider
        )

        for fd in result.factors:
            assert fd.factor_label in FACTOR_LABELS.values() or fd.factor_label == fd.factor_name


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_insufficient_observations(self) -> None:
        """Fewer than 12 observations → empty result with warning."""
        product_returns = np.array([0.01, 0.02, -0.01, 0.005])
        dates = _daily_dates(4)
        provider = _make_provider({})

        result = analyze_factors(product_returns, dates, "daily", "commodity_cta", provider)

        assert result.factors == []
        assert result.r_squared == 0.0
        assert len(result.warnings) > 0
        assert "不足" in result.warnings[0]

    def test_no_benchmark_data(self) -> None:
        """Sufficient product data but empty provider → empty factors."""
        n = 30
        product_returns = np.random.default_rng(1).normal(0, 0.01, n)
        dates = _daily_dates(n)
        provider = _make_provider({})

        result = analyze_factors(product_returns, dates, "daily", "commodity_cta", provider)

        assert result.factors == []
        assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Tests: _factor_confidence helper
# ---------------------------------------------------------------------------


class TestFactorConfidence:
    def test_high_t_and_corr(self) -> None:
        """Large t-stat and high correlation → high confidence."""
        conf = _factor_confidence(t_stat=4.0, corr=0.9, n=100)
        assert conf >= 80

    def test_low_t_and_corr(self) -> None:
        """Small t-stat and low correlation → low confidence."""
        conf = _factor_confidence(t_stat=0.5, corr=0.1, n=20)
        assert conf < 40

    def test_capped_at_100(self) -> None:
        """Confidence never exceeds 100."""
        conf = _factor_confidence(t_stat=10.0, corr=1.0, n=500)
        assert conf <= 100.0
