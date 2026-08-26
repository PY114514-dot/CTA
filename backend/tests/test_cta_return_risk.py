"""收益分析与风险暴露模块的单元测试。

覆盖：收益解剖数值正确性、回撤片段统计、分年收益分组、绝对尺度评分的
单调性与门槛、统计风险（VaR/CVaR）数值、因子风险贡献与集中度、无回归
时的优雅降级。
"""

from datetime import date, timedelta

import numpy as np
import pytest

from app.services.cta_return_risk import (
    analyze_returns,
    analyze_risk_exposure,
    score_return_analysis,
    score_risk_exposure,
)
from app.services.factor_library.factor_regression import FactorBeta, RegressionResult


def _dates(count: int, start: date = date(2024, 1, 5)) -> list[date]:
    return [start + timedelta(days=7 * index) for index in range(count)]


def _nav_from_returns(returns: list[float]) -> list[float]:
    values = [1.0]
    for value in returns:
        values.append(values[-1] * (1.0 + value))
    return values


def _regression(
    *,
    r_squared: float = 0.6,
    contributions: dict[str, float] | None = None,
    factors: list[FactorBeta] | None = None,
) -> RegressionResult:
    default_factors = [
        FactorBeta(
            name="trend", display_name="趋势", beta=0.5, std_error=0.1, t_stat=5.0,
            p_value=0.001, contribution_pct=0.5, mean_return_contribution=0.001,
            significant=True, factor_group="trend",
        ),
        FactorBeta(
            name="carry", display_name="展期收益", beta=0.3, std_error=0.1, t_stat=3.0,
            p_value=0.01, contribution_pct=0.3, mean_return_contribution=0.0005,
            significant=True, factor_group="carry",
        ),
    ]
    return RegressionResult(
        intercept=0.001,
        annualized_alpha=0.052,
        alphas_t_stat=1.5,
        r_squared=r_squared,
        adj_r_squared=0.55,
        f_statistic=10.0,
        f_p_value=0.001,
        n_observations=64,
        factors=factors if factors is not None else default_factors,
        residual_annual_vol=0.08,
        residual_skew=0.0,
        residual_kurtosis=1.0,
        factor_risk_contributions=(
            contributions if contributions is not None else {"trend": 0.0004, "carry": 0.0004}
        ),
    )


# ---------------------------------------------------------------------------
# 收益分析
# ---------------------------------------------------------------------------

def test_analyze_returns_hand_computed_level_metrics() -> None:
    analysis = analyze_returns([1.0, 1.1, 1.21], _dates(3), "weekly")
    assert analysis["status"] == "available"
    metrics = analysis["metrics"]
    assert metrics["cumulative_return"] == pytest.approx(0.21)
    assert metrics["annualized_return"] == pytest.approx(1.21 ** 26 - 1.0)
    assert metrics["positive_period_ratio"] == pytest.approx(1.0)
    assert metrics["best_period_return"] == pytest.approx(0.1)
    assert metrics["maximum_drawdown"] == pytest.approx(0.0)
    assert metrics["recovery_completed"] is True
    assert metrics["drawdown_episode_count"] == 0
    assert len(analysis["drawdown_path"]) == 3
    assert analysis["warnings"]  # 2 期属于短样本，应给出提示


def test_analyze_returns_drawdown_episodes() -> None:
    analysis = analyze_returns([1.0, 1.2, 0.9, 1.0, 1.1], _dates(5), "weekly")
    metrics = analysis["metrics"]
    assert metrics["maximum_drawdown"] == pytest.approx(0.9 / 1.2 - 1.0)
    assert metrics["current_drawdown"] == pytest.approx(1.1 / 1.2 - 1.0)
    assert metrics["recovery_completed"] is False
    assert metrics["longest_drawdown_duration_periods"] == 3
    assert metrics["drawdown_episode_count"] == 1
    assert metrics["average_recovery_periods"] is None
    assert analysis["drawdown_path"][2]["drawdown"] == pytest.approx(-0.25)


def test_analyze_returns_calendar_year_grouping() -> None:
    dates = [date(2023, 12, 22), date(2023, 12, 29), date(2024, 1, 5)]
    analysis = analyze_returns([1.0, 1.1, 1.21], dates, "weekly")
    assert set(analysis["calendar_year_returns"].keys()) == {"2023", "2024"}
    assert analysis["calendar_year_returns"]["2023"] == pytest.approx(0.1)
    assert analysis["calendar_year_returns"]["2024"] == pytest.approx(0.1)


def test_analyze_returns_insufficient() -> None:
    analysis = analyze_returns([1.0], _dates(1), "weekly")
    assert analysis["status"] == "insufficient"
    assert analysis["metrics"] == {}


def test_score_return_analysis_level_monotonicity_and_weights() -> None:
    low = score_return_analysis(
        analyze_returns(_nav_from_returns([0.002] * 52), _dates(53), "weekly")
    )
    high = score_return_analysis(
        analyze_returns(_nav_from_returns([0.01] * 52), _dates(53), "weekly")
    )
    assert low["status"] == "available"
    assert high["status"] == "available"
    assert high["score"] > low["score"]
    assert low["band"] in {"high", "medium", "low"}
    assert high["band"] == "high"
    assert abs(sum(low["weights"].values()) - 1.0) < 1e-9


def test_score_return_analysis_drawdown_penalty() -> None:
    flat = score_return_analysis(
        analyze_returns(_nav_from_returns([0.005] * 52), _dates(53), "weekly")
    )
    with_drawdown = score_return_analysis(
        analyze_returns(_nav_from_returns([0.005] * 51 + [-0.30]), _dates(53), "weekly")
    )
    assert with_drawdown["score"] < flat["score"]
    assert with_drawdown["components"]["drawdown_control"] == pytest.approx(0.0)


def test_score_return_analysis_short_sample_no_score() -> None:
    result = score_return_analysis(analyze_returns([1.0, 1.1, 1.21], _dates(3), "weekly"))
    assert result["status"] == "short_sample"
    assert result["score"] is None
    assert result["band"] is None


# ---------------------------------------------------------------------------
# 风险暴露
# ---------------------------------------------------------------------------

def test_analyze_risk_exposure_statistical_values() -> None:
    returns = np.asarray([
        0.01, -0.02, 0.005, -0.01, 0.02, -0.015, 0.01, 0.008,
        -0.012, 0.004, 0.016, -0.006, 0.002, 0.012, -0.008, 0.01,
    ] * 4)
    analysis = analyze_risk_exposure(returns, None, "weekly")
    assert analysis["status"] == "available"
    assert analysis["factor_exposure_status"] == "unavailable"
    stats = analysis["statistical"]
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    assert stats["var_95"] == pytest.approx(mean + 1.6449 * std)
    assert stats["var_99"] == pytest.approx(mean + 2.3263 * std)
    assert stats["annualized_volatility"] == pytest.approx(std * np.sqrt(52))
    assert stats["worst_period_return"] == pytest.approx(float(np.min(returns)))
    assert stats["maximum_drawdown"] <= 0.0
    assert analysis["warnings"]


def test_analyze_risk_exposure_regression_concentration() -> None:
    returns = np.asarray([0.01, -0.02] * 20)
    analysis = analyze_risk_exposure(returns, _regression(), "weekly")
    assert analysis["factor_exposure_status"] == "available"
    assert len(analysis["factor_exposures"]) == 2
    assert analysis["factor_exposures"][0]["risk_contribution"] == pytest.approx(0.0004)
    assert analysis["factor_exposures"][0]["risk_contribution_pct"] == pytest.approx(50.0)
    # 两个因子风险贡献相等 → 归一化 HHI = 0（完全分散）
    assert analysis["concentration"]["hhi"] == pytest.approx(0.5)
    assert analysis["concentration"]["normalized_hhi"] == pytest.approx(0.0)
    assert analysis["systematic_variance_share"] == pytest.approx(0.6)
    assert analysis["unexplained_variance_share"] == pytest.approx(0.4)
    assert analysis["residual_annual_volatility"] == pytest.approx(0.08)
    assert set(analysis["factor_group_risk_contributions"].keys()) == {"trend", "carry"}
    assert analysis["warnings"] == []


def test_analyze_risk_exposure_single_factor_no_concentration() -> None:
    returns = np.asarray([0.01, -0.02] * 20)
    regression = _regression(contributions={"trend": 0.0009})
    analysis = analyze_risk_exposure(returns, regression, "weekly")
    assert analysis["concentration"]["normalized_hhi"] is None
    assert analysis["factor_exposures"][0]["risk_contribution_pct"] == pytest.approx(100.0)


def test_analyze_risk_exposure_insufficient_sample() -> None:
    analysis = analyze_risk_exposure(np.asarray([0.01, -0.02, 0.005]), None, "weekly")
    assert analysis["status"] == "insufficient"
    assert analysis["factor_exposure_status"] == "unavailable"
    assert analysis["statistical"] == {}


def test_score_risk_exposure_vol_penalty_and_renormalization() -> None:
    low_vol = analyze_risk_exposure(np.full(52, 0.005), None, "weekly")
    high_vol = analyze_risk_exposure(np.asarray([0.03, -0.03] * 26), None, "weekly")
    low_score = score_risk_exposure(low_vol, frequency="weekly")
    high_score = score_risk_exposure(high_vol, frequency="weekly")
    assert high_score["score"] < low_score["score"]
    # 无回归 → 集中度分量缺席，剩余权重重归一化后总和为 1
    assert "concentration_control" not in low_score["weights"]
    assert abs(sum(low_score["weights"].values()) - 1.0) < 1e-9
    assert low_score["band"] in {"high", "medium", "low"}


def test_score_risk_exposure_concentration_lowers_score() -> None:
    returns = np.asarray([0.01, -0.02] * 20)
    diversified = score_risk_exposure(
        analyze_risk_exposure(returns, _regression(contributions={"trend": 0.0004, "carry": 0.0004}), "weekly"),
        frequency="weekly",
    )
    concentrated = score_risk_exposure(
        analyze_risk_exposure(returns, _regression(contributions={"trend": 0.0009, "carry": 0.0001}), "weekly"),
        frequency="weekly",
    )
    assert concentrated["components"]["concentration_control"] < diversified["components"]["concentration_control"]
    assert concentrated["score"] < diversified["score"]


def test_score_risk_exposure_insufficient_sample_no_score() -> None:
    analysis = analyze_risk_exposure(np.asarray([0.01, -0.02]), None, "weekly")
    result = score_risk_exposure(analysis, frequency="weekly")
    assert result["status"] == "insufficient"
    assert result["score"] is None
