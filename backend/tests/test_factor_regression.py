"""Behavioral tests for the statistical CTA attribution seam."""

from datetime import date, timedelta

import numpy as np

from app.services import factor_library
from app.services.factor_library.factor_regression import run_factor_regression


def _dates(n: int) -> list[date]:
    start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _install_factor_fixture(monkeypatch, factor_returns: dict[str, np.ndarray], groups: dict[str, str]) -> None:
    metadata = [
        {
            "name": name,
            "display_name": name,
            "category": "量价",
            "factor_group": groups[name],
        }
        for name in factor_returns
    ]
    monkeypatch.setattr(factor_library, "get_all_factors", lambda: metadata)
    monkeypatch.setattr(
        factor_library,
        "get_factor_series",
        lambda name, risk_profile="baseline": __import__("pandas").Series(
            factor_returns[name], index=__import__("pandas").date_range("2020-01-01", periods=len(factor_returns[name]), freq="D")
        ),
    )


def test_hac_inference_uses_autocorrelation_robust_standard_errors(monkeypatch) -> None:
    rng = np.random.default_rng(10)
    n = 220
    factor = rng.normal(0.0, 0.01, n)
    noise = np.zeros(n)
    innovations = rng.normal(0.0, 0.004, n)
    for i in range(1, n):
        noise[i] = 0.75 * noise[i - 1] + innovations[i]
    product = 1.4 * factor + noise
    _install_factor_fixture(monkeypatch, {"trend": factor}, {"trend": "trend"})

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["trend"],
        hac_max_lags=5,
        bootstrap_reps=0,
    )

    beta = result.factors[0]
    assert result.inference_method == "OLS + HAC(Newey-West)"
    assert beta.ordinary_std_error != beta.std_error
    assert result.diagnostics["ljung_box_pvalue_6"] < 0.05
    assert result.diagnostics["residual_autocorrelation_detected"] is True


def test_stationary_bootstrap_confidence_intervals_are_reproducible(monkeypatch) -> None:
    rng = np.random.default_rng(11)
    n = 180
    factor = rng.normal(0.0, 0.01, n)
    product = 0.8 * factor + rng.normal(0.0, 0.002, n)
    _install_factor_fixture(monkeypatch, {"trend": factor}, {"trend": "trend"})

    kwargs = {
        "frequency": "daily",
        "factor_names": ["trend"],
        "hac_max_lags": 3,
        "bootstrap_reps": 160,
        "bootstrap_block_length": 12,
        "random_seed": 20260811,
    }
    first = run_factor_regression(product, _dates(n), **kwargs)
    second = run_factor_regression(product, _dates(n), **kwargs)

    first_beta = first.factors[0]
    second_beta = second.factors[0]
    assert first.bootstrap["reps"] == 160
    assert first.bootstrap["block_length"] == 12
    assert first_beta.bootstrap_ci_low == second_beta.bootstrap_ci_low
    assert first_beta.bootstrap_ci_high == second_beta.bootstrap_ci_high
    assert first_beta.bootstrap_ci_low < 0.8 < first_beta.bootstrap_ci_high


def test_factor_group_contributions_are_reported_separately(monkeypatch) -> None:
    rng = np.random.default_rng(12)
    n = 120
    trend = rng.normal(0.0, 0.01, n)
    carry = rng.normal(0.0, 0.006, n)
    product = 1.2 * trend + 0.4 * carry + rng.normal(0.0, 0.001, n)
    _install_factor_fixture(
        monkeypatch,
        {"trend": trend, "carry": carry},
        {"trend": "trend", "carry": "carry"},
    )

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["trend", "carry"],
        bootstrap_reps=0,
    )

    assert {factor.factor_group for factor in result.factors} == {"trend", "carry"}
    assert set(result.factor_group_contributions) == {"trend", "carry"}
    assert result.factor_groups == {"trend": ["trend"], "carry": ["carry"]}


def test_weekly_alignment_matches_same_economic_week(monkeypatch) -> None:
    import pandas as pd

    rng = np.random.default_rng(13)
    n = 240
    factor = np.zeros(n)
    friday_positions = [i for i, dt in enumerate(pd.date_range("2020-01-01", periods=n, freq="D")) if dt.weekday() == 4]
    factor[friday_positions] = rng.normal(0.0, 0.01, len(friday_positions))
    product_dates = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in friday_positions]
    product = 1.3 * factor[friday_positions] + rng.normal(0.0, 0.001, len(friday_positions))
    _install_factor_fixture(monkeypatch, {"trend": factor}, {"trend": "trend"})

    result = run_factor_regression(
        product,
        [dt.date() for dt in product_dates],
        frequency="weekly",
        factor_names=["trend"],
        bootstrap_reps=0,
    )

    assert result.n_observations >= 20
    assert result.factors[0].beta > 0.8


def test_expanding_walk_forward_reports_out_of_sample_fit(monkeypatch) -> None:
    rng = np.random.default_rng(14)
    n = 140
    factor = rng.normal(0.0, 0.01, n)
    product = 1.1 * factor + rng.normal(0.0, 0.0015, n)
    _install_factor_fixture(monkeypatch, {"trend": factor}, {"trend": "trend"})

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["trend"],
        bootstrap_reps=0,
        oos_train_window=60,
    )

    assert result.out_of_sample["method"] == "expanding_window"
    assert result.out_of_sample["observations"] == n - 60
    assert result.out_of_sample["r_squared"] > 0.7
    assert result.out_of_sample["tracking_error_annual"] >= 0
    assert len(result.out_of_sample["segments"]) == 3
    from app.services.factor_library.factor_regression import assess_oos_applicability
    assert assess_oos_applicability(result.out_of_sample)["status"] == "applicable"


def test_collinearity_and_joint_hac_diagnostics_are_exposed(monkeypatch) -> None:
    rng = np.random.default_rng(15)
    n = 120
    common = rng.normal(0.0, 0.01, n)
    factor_a = common + rng.normal(0.0, 0.0005, n)
    factor_b = common + rng.normal(0.0, 0.0005, n)
    product = 0.7 * factor_a + 0.3 * factor_b + rng.normal(0.0, 0.001, n)
    _install_factor_fixture(
        monkeypatch,
        {"trend": factor_a, "trend_fast": factor_b},
        {"trend": "trend", "trend_fast": "trend"},
    )

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["trend", "trend_fast"],
        bootstrap_reps=0,
    )

    assert result.joint_hac["p_value"] is not None
    assert result.collinearity["condition_number"] > 1
    assert result.collinearity["high_correlation_pairs"]


def test_empty_attribution_result_keeps_diagnostic_contract(monkeypatch) -> None:
    _install_factor_fixture(monkeypatch, {}, {})

    result = run_factor_regression(
        np.zeros(30),
        _dates(30),
        frequency="daily",
        bootstrap_reps=0,
    )

    assert result.joint_hac["p_value"] is None
    assert result.collinearity["high_correlation_pairs"] == []
    assert result.out_of_sample["r_squared"] is None


def test_exact_minimum_observations_keep_factor_exposure(monkeypatch) -> None:
    rng = np.random.default_rng(16)
    n = 20
    factor = rng.normal(0.0, 0.01, n)
    product = 0.8 * factor + rng.normal(0.0, 0.002, n)
    _install_factor_fixture(monkeypatch, {"trend": factor}, {"trend": "trend"})

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["trend"],
        bootstrap_reps=0,
    )

    assert result.n_observations == n
    assert [item.name for item in result.factors] == ["trend"]


def test_regression_accepts_a_bundle_factor_loader_without_using_legacy_loader(monkeypatch) -> None:
    import pandas as pd

    rng = np.random.default_rng(17)
    n = 40
    carry = pd.Series(
        rng.normal(0.0, 0.01, n),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    product = 0.9 * carry.to_numpy() + rng.normal(0.0, 0.001, n)

    def fail_if_legacy_loader_is_used(*_args, **_kwargs):
        raise AssertionError("CTA bundle regression must not use the legacy factor loader")

    monkeypatch.setattr(factor_library, "get_factor_series", fail_if_legacy_loader_is_used)

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["term_structure_carry"],
        factor_series_loader=lambda name: carry if name == "term_structure_carry" else None,
        bootstrap_reps=0,
    )

    assert result.n_observations == n
    assert [item.name for item in result.factors] == ["term_structure_carry"]


def test_regime_decomposition_reports_trending_and_choppy_r_squared(monkeypatch) -> None:
    rng = np.random.default_rng(42)
    n = 300
    # Trending days: volatile trend factor the product loads on.  Choppy
    # days: quiet factor and a product that decouples from it, with choppy
    # product volatility balanced to the trending variance (R² is
    # variance-weighted, so an unbalanced design would hide the story).
    # The choppy span is deliberately longer than the 60-day indicator
    # window so the median split has enough fully-choppy observations.
    trend = rng.normal(0.0, 0.01, n)
    trend[180:] = rng.normal(0.0, 0.001, 120)
    product = 1.2 * trend + rng.normal(0.0, 0.001, n)
    product[180:] = rng.normal(0.0, 0.012, 120)
    _install_factor_fixture(monkeypatch, {"trend": trend}, {"trend": "trend"})

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["trend"],
        bootstrap_reps=0,
    )

    assert result.regime is not None
    assert result.regime["regime_factor"] == "trend"
    assert result.regime["window"] == 60
    assert 0.2 < result.regime["trending_share"] < 0.8
    assert result.regime["trending"]["r_squared"] > 0.6
    assert result.regime["choppy"]["r_squared"] < 0.4
    assert result.regime["trending"]["r_squared"] > result.regime["choppy"]["r_squared"]
    assert result.regime["augmented"]["r_squared_gain"] is not None
    assert result.r_squared > 0.4


def test_attribution_quality_score_renormalizes_over_available_evidence(monkeypatch) -> None:
    rng = np.random.default_rng(43)
    n = 150
    factor = rng.normal(0.0, 0.01, n)
    # A real positive alpha on top of factor exposure: the alpha t-statistic
    # is huge, so the public-factor attribution must receive little credit
    # for unexplained-return stability.
    product = 0.004 + 1.2 * factor + rng.normal(0.0, 0.001, n)
    _install_factor_fixture(monkeypatch, {"trend": factor}, {"trend": "trend"})

    result = run_factor_regression(
        product,
        _dates(n),
        frequency="daily",
        factor_names=["trend"],
        bootstrap_reps=0,  # no bootstrap CI -> that component must be dropped
    )

    quality = result.attribution_quality
    assert quality is not None
    assert quality["method"] == "attribution_quality_v1"
    assert quality["band"] in {"high", "medium", "low"}
    assert 0.0 <= quality["score"] <= 100.0
    assert quality["components"]["alpha_significance"] < 20.0
    assert quality["components"]["alpha_bootstrap"] is None
    assert "alpha_bootstrap" not in quality["weights"]
    assert abs(sum(quality["weights"].values()) - 1.0) < 1e-9
