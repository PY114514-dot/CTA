from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services import factor_library
from app.services.dynamic_beta import compute_rolling_beta, filter_dynamic_beta
from app.services.dynamic_beta import prepare_factor_matrix


def _dates(count: int) -> list[date]:
    return [date(2024, 1, 5) + timedelta(days=7 * index) for index in range(count)]


def test_kalman_filter_detects_known_beta_drift() -> None:
    factors = np.tile(np.array([0.01, -0.01, 0.015, -0.015]), 20).reshape(-1, 1)
    true_beta = np.r_[np.full(40, 0.2), np.full(40, 1.2)]
    result = filter_dynamic_beta(true_beta * factors[:, 0], factors, _dates(80), ["trend"], state_variance=1e-5)

    assert result["paths"][-1]["betas"]["trend"] > result["paths"][39]["betas"]["trend"] + 0.3
    assert result["beta_range"]["trend"] > 0.3


def test_kalman_filter_does_not_create_large_drift_for_constant_beta() -> None:
    factors = np.tile(np.array([0.01, -0.01, 0.015, -0.015]), 20).reshape(-1, 1)
    result = filter_dynamic_beta(0.6 * factors[:, 0], factors, _dates(80), ["trend"], state_variance=1e-8)

    assert result["beta_range"]["trend"] < 0.15


def test_kalman_filter_is_causal() -> None:
    factors = np.tile(np.array([0.01, -0.01, 0.015, -0.015]), 20).reshape(-1, 1)
    returns = 0.4 * factors[:, 0]
    baseline = filter_dynamic_beta(returns, factors, _dates(80), ["trend"])
    altered = returns.copy()
    altered[-1] = 0.2
    changed = filter_dynamic_beta(altered, factors, _dates(80), ["trend"])

    assert baseline["paths"][-2]["betas"] == changed["paths"][-2]["betas"]
    assert baseline["paths"][-2]["beta_standard_errors"] == changed["paths"][-2]["beta_standard_errors"]
    assert baseline["paths"][-2]["long_run_style_contribution"] == changed["paths"][-2]["long_run_style_contribution"]
    assert baseline["paths"][-2]["dynamic_timing_contribution_by_factor"] == changed["paths"][-2]["dynamic_timing_contribution_by_factor"]
    assert baseline["paths"][-2]["dynamic_timing_contribution"] == changed["paths"][-2]["dynamic_timing_contribution"]


def test_prepare_factor_matrix_compounds_returns_within_one_economic_period(monkeypatch) -> None:
    dates: list[date] = []
    product_returns: list[float] = []
    factor_dates: list[date] = []
    factor_returns: list[float] = []
    start = date(2024, 1, 1)
    for week in range(20):
        for offset, product_return, factor_return in ((0, 0.01, 0.001), (1, 0.02, 0.002)):
            point = start + timedelta(days=week * 7 + offset)
            dates.append(point)
            product_returns.append(product_return)
            factor_dates.append(point)
            factor_returns.append(factor_return)

    monkeypatch.setattr(
        factor_library,
        "get_factor_series",
        lambda name, risk_profile="baseline": pd.Series(
            factor_returns, index=pd.to_datetime(factor_dates), name=name
        ),
    )

    result = prepare_factor_matrix(
        np.asarray(product_returns), dates, "weekly", ["trend"]
    )

    assert result["observation_count"] == 20
    assert result["returns"][0] == pytest.approx((1.01 * 1.02) - 1.0)
    assert result["factor_returns"][0, 0] == pytest.approx((1.001 * 1.002) - 1.0)


def test_prepare_factor_matrix_keeps_volatility_state_as_period_end_level(monkeypatch) -> None:
    dates: list[date] = []
    product_returns: list[float] = []
    factor_dates: list[date] = []
    momentum: list[float] = []
    volatility: list[float] = []
    start = date(2024, 1, 1)
    for week in range(20):
        for offset in (0, 1):
            point = start + timedelta(days=week * 7 + offset)
            dates.append(point)
            product_returns.append(0.01)
            factor_dates.append(point)
            momentum.append(0.001)
            volatility.append(0.10 + 0.01 * offset)

    monkeypatch.setattr(
        "app.services.cta_factor_bundle.get_factor_series",
        lambda name, as_of_date=None: pd.Series(
            volatility if name == "volatility_state" else momentum,
            index=pd.to_datetime(factor_dates),
            name=name,
        ),
    )
    monkeypatch.setattr(
        "app.services.cta_factor_bundle.get_cta_factor_bundle",
        lambda as_of_date=None: {"bundle_version": "cta_factor_bundle_v1", "factors": []},
    )

    result = prepare_factor_matrix(
        np.asarray(product_returns), dates, "weekly", ["short_term_trend_20", "volatility_state"]
    )

    assert result["factor_returns"][0, 1] == pytest.approx(0.11)


def test_rolling_beta_supports_configured_monthly_windows() -> None:
    factors = np.tile(np.array([0.01, -0.01, 0.015, -0.015]), 5)[:19].reshape(-1, 1)

    result = compute_rolling_beta(
        0.6 * factors[:, 0],
        factors,
        _dates(len(factors)),
        ["trend"],
        windows=[12, 24],
        minimum_window=12,
    )

    assert [report["status"] for report in result["windows"]] == ["available", "insufficient"]
    assert result["windows"][0]["n_periods"] == 8
