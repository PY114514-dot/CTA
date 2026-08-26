from datetime import date, timedelta
from types import SimpleNamespace

from app.services.cta_evaluation import build_attribution_tables, evaluate_nav_path


def test_evaluate_nav_path_reports_tail_and_recovery() -> None:
    dates = [date(2026, 1, 2) + timedelta(days=7 * index) for index in range(5)]
    result = evaluate_nav_path([1.0, 1.1, 0.9, 0.95, 1.12], dates, "weekly")

    assert result["maximum_drawdown"] < 0
    assert result["expected_shortfall_5pct"] < 0
    assert result["drawdown_duration_max_periods"] > 0
    assert result["recovery_completed"] is True
    assert len(result["drawdown_path"]) == 5


def test_attribution_tables_do_not_mix_exposure_return_or_risk() -> None:
    regression = SimpleNamespace(
        factors=[SimpleNamespace(
            name="trend", display_name="趋势", factor_group="trend", beta=0.5,
            t_stat=2.1, p_value=0.04, bootstrap_ci_low=0.1, bootstrap_ci_high=0.8,
            contribution_pct=35.0, mean_return_contribution=0.0035,
        )],
        factor_risk_contributions={"trend": 0.0123},
    )
    tables = build_attribution_tables(regression)

    assert tables["factor_exposure"][0]["beta"] == 0.5
    assert "mean_return_contribution" not in tables["factor_exposure"][0]
    assert tables["return_contribution"][0]["mean_return_contribution"] == 0.0035
    assert tables["euler_risk_contribution"][0]["component_risk_contribution"] == 0.0123
