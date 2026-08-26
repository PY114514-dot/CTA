from datetime import date, timedelta

import numpy as np

from app.services.nonlinear_attribution import (
    evaluate_factor_interaction_increment,
    evaluate_momentum_volatility_increment,
    evaluate_nonlinear_increment,
)


def _dates(count: int) -> list[date]:
    return [date(2018, 1, 5) + timedelta(days=7 * index) for index in range(count)]


def test_phase_d_uses_contiguous_causal_oos_segments() -> None:
    rng = np.random.default_rng(20260812)
    factors = rng.normal(0.0, 0.08, size=(180, 2))
    returns = 0.35 * factors[:, 0] + 2.5 * factors[:, 1] ** 2 - 0.02 + rng.normal(0.0, 0.002, size=180)

    result = evaluate_nonlinear_increment(
        returns,
        factors,
        _dates(len(returns)),
        ["trend", "carry"],
        frequency="weekly",
        train_window=60,
        test_window=20,
        max_segments=5,
        min_segments=3,
        min_r2_uplift=0.0,
    )

    assert result["status"] == "available"
    assert len(result["segments"]) == 5
    assert result["parameters"]["uses_future_data"] is False
    assert result["parameters"]["training_scheme"] == "expanding_window"
    for segment in result["segments"]:
        assert segment["train_end_date"] < segment["test_start_date"]
        assert segment["train_observations"] >= 60
        assert segment["test_observations"] == 20
        assert "oos_r2" in segment["linear"]
        assert "oos_r2" in segment["nonlinear"]


def test_phase_d_only_reports_reliable_increment_when_uplift_is_stable() -> None:
    rng = np.random.default_rng(7)
    factors = rng.normal(0.0, 0.05, size=(180, 1))
    returns = 0.8 * factors[:, 0] + rng.normal(0.0, 0.002, size=180)

    result = evaluate_nonlinear_increment(
        returns,
        factors,
        _dates(len(returns)),
        ["trend"],
        frequency="weekly",
        train_window=60,
        test_window=20,
        max_segments=5,
        min_segments=3,
        min_r2_uplift=0.01,
    )

    assert result["status"] == "available"
    assert result["summary"]["stable_improvement"] is False
    assert result["summary"]["conclusion"] == "未发现可靠非线性增量"


def test_phase_d_reports_sensitivity_and_does_not_select_from_it() -> None:
    rng = np.random.default_rng(11)
    factors = rng.normal(0.0, 0.04, size=(120, 1))
    returns = 0.5 * factors[:, 0] + rng.normal(0.0, 0.004, size=120)

    result = evaluate_nonlinear_increment(
        returns,
        factors,
        _dates(len(returns)),
        ["trend"],
        frequency="weekly",
        train_window=40,
        test_window=15,
        max_segments=4,
        min_segments=3,
    )

    sensitivity = result["sensitivity"]
    assert sensitivity["window_sensitivity"]
    assert sensitivity["threshold_sensitivity"]
    assert sensitivity["model_parameter_sensitivity"]
    assert sensitivity["selection_policy"]["base_case_is_pre_registered"] is True
    assert sensitivity["selection_policy"]["selected_from_sensitivity"] is False
    assert sensitivity["multiple_testing"]["correction"] == "bonferroni_descriptive"
    assert sensitivity["multiple_testing"]["dsr_applicable"] is False


def test_phase_d_marks_short_series_insufficient_instead_of_filling_zero() -> None:
    result = evaluate_nonlinear_increment(
        np.array([0.01] * 20),
        np.ones((20, 1)) * 0.01,
        _dates(20),
        ["trend"],
        frequency="weekly",
        train_window=40,
        test_window=10,
    )

    assert result["status"] == "insufficient"
    assert result["segments"] == []
    assert result["summary"]["stable_improvement"] is False
    assert result["summary"]["mean_r2_delta"] is None
    assert any("样本不足" in warning for warning in result["warnings"])


def test_registered_momentum_volatility_interaction_detects_known_oos_increment() -> None:
    rng = np.random.default_rng(20260814)
    factors = rng.normal(0.0, 0.08, size=(180, 2))
    returns = (
        0.35 * factors[:, 0]
        + 0.20 * factors[:, 1]
        + 1.80 * factors[:, 0] * factors[:, 1]
        + rng.normal(0.0, 0.002, size=180)
    )

    result = evaluate_momentum_volatility_increment(
        returns,
        factors,
        _dates(len(returns)),
        frequency="weekly",
        train_window=60,
        test_window=20,
        max_segments=5,
        min_segments=3,
        min_r2_uplift=0.01,
    )

    assert result["method"] == "pre_registered_oos_momentum_x_volatility_nested_ols"
    assert result["parameters"]["standardization"] == "train_window_only"
    assert result["summary"]["stable_improvement"] is True
    assert result["summary"]["conclusion"] == "存在动量 × 波动率的稳定样本外增量"
    assert all("interaction_coefficient" in segment for segment in result["segments"])


def test_registered_interaction_rejects_no_increment() -> None:
    rng = np.random.default_rng(20260815)
    factors = rng.normal(0.0, 0.05, size=(180, 2))
    returns = 0.6 * factors[:, 0] + 0.2 * factors[:, 1] + rng.normal(0.0, 0.002, size=180)

    result = evaluate_momentum_volatility_increment(
        returns,
        factors,
        _dates(len(returns)),
        frequency="weekly",
        train_window=60,
        test_window=20,
        max_segments=5,
        min_segments=3,
        min_r2_uplift=0.01,
    )

    assert result["summary"]["stable_improvement"] is False
    assert result["summary"]["rejection_reason"] == "未发现可靠的动量 × 波动率样本外增量"
    assert result["sensitivity"]["selection_policy"]["selected_from_sensitivity"] is False


def test_named_factor_interaction_keeps_the_registered_oos_protocol() -> None:
    rng = np.random.default_rng(20260825)
    factors = rng.normal(0.0, 0.05, size=(180, 2))
    returns = 0.3 * factors[:, 0] + 0.3 * factors[:, 1] + 2.0 * factors[:, 0] * factors[:, 1]

    result = evaluate_factor_interaction_increment(
        returns,
        factors,
        _dates(len(returns)),
        factor_names=("trend", "term_structure_carry"),
        interaction_name="trend_x_term_structure_carry",
        display_name="趋势与期限结构",
        frequency="weekly",
        train_window=60,
        test_window=20,
        max_segments=5,
        min_segments=3,
        min_r2_uplift=0.01,
    )

    assert result["parameters"]["uses_future_data"] is False
    assert result["parameters"]["interaction_name"] == "trend_x_term_structure_carry"
    assert result["factor_names"] == ["trend", "term_structure_carry", "trend_x_term_structure_carry"]


def test_registered_interaction_marks_missing_observations_insufficient() -> None:
    result = evaluate_momentum_volatility_increment(
        np.ones(30) * 0.01,
        np.ones((30, 2)) * 0.01,
        _dates(30),
        frequency="weekly",
        train_window=40,
        test_window=10,
    )

    assert result["status"] == "insufficient"
    assert result["segments"] == []
    assert result["summary"]["stable_improvement"] is False
