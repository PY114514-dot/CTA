from datetime import date, timedelta

import numpy as np

from app.services.regime_attribution import classify_observable_regimes, run_regime_attribution


def _dates(count: int) -> list[date]:
    return [date(2024, 1, 5) + timedelta(days=7 * index) for index in range(count)]


def test_regime_labels_are_causal() -> None:
    proxy = np.array([0.01, -0.005, 0.008, -0.004, 0.006, -0.003, 0.004, -0.002, 0.003, -0.001, 0.02, -0.04, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01, 0.015, -0.02])
    dates = _dates(len(proxy))
    baseline = classify_observable_regimes(proxy, dates, "weekly", vol_window=3, trend_short_window=2, trend_medium_window=4, min_history=6)
    altered = proxy.copy()
    altered[-1] = 0.80
    changed = classify_observable_regimes(altered, dates, "weekly", vol_window=3, trend_short_window=2, trend_medium_window=4, min_history=6)

    assert baseline["states"][:-1] == changed["states"][:-1]
    assert baseline["parameters"]["uses_future_data"] is False


def test_regime_attribution_reports_conditional_beta_and_crisis_alpha() -> None:
    dates = _dates(40)
    proxy = np.array([0.01] * 30 + [-0.08] + [0.02] * 9)
    factors = np.column_stack([proxy, np.array([0.01, -0.01] * 20)])
    returns = 0.02 + 0.7 * factors[:, 0] + 0.2 * factors[:, 1]
    states = classify_observable_regimes(proxy, dates, "weekly", vol_window=3, trend_short_window=2, trend_medium_window=4, min_history=6)

    result = run_regime_attribution(returns, factors, dates, ["trend", "reversal"], states, "weekly", min_regime_observations=5)

    crisis = next(item for item in result["regimes"] if item["state"] == "crisis")
    assert crisis["status"] == "available"
    assert crisis["conditional_beta"]["trend"] is not None
    assert crisis["crisis_alpha_candidate"] is not None
    assert crisis["conditional_correlation"] is not None


def test_regime_attribution_does_not_fill_insufficient_coefficients_with_zero() -> None:
    dates = _dates(20)
    proxy = np.array([0.01] * 20)
    factors = np.column_stack([proxy, proxy])
    returns = np.full(20, 0.01)
    states = classify_observable_regimes(proxy, dates, "weekly", vol_window=3, trend_short_window=2, trend_medium_window=4, min_history=6)

    result = run_regime_attribution(returns, factors, dates, ["trend", "carry"], states, "weekly", min_regime_observations=15)

    assert all(item["status"] == "insufficient" for item in result["regimes"])
    assert all(item["conditional_beta"] is None for item in result["regimes"])
    assert any("样本不足" in warning for warning in result["warnings"])


def test_regime_thresholds_use_only_prior_lagged_volatility_history() -> None:
    dates = _dates(12)
    proxy = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.50, 0.01])
    result = classify_observable_regimes(
        proxy,
        dates,
        "weekly",
        vol_window=3,
        trend_short_window=2,
        trend_medium_window=4,
        min_history=4,
    )

    # At index 6 only two earlier lagged-volatility observations exist.  The
    # current lagged volatility must not be used to form its own threshold.
    assert result["features"][6]["crisis_volatility_cutoff"] is None
