"""Validation compares like-for-like period contributions, not beta to P&L."""

from app.services.factor_library.validation import validate_against_report


def test_validation_uses_beta_times_matched_period_factor_return() -> None:
    regression = {
        "r_squared": 0.42,
        "factors": [{"name": "trend", "beta": 1.5, "t_stat": 2.4}],
    }
    report = {
        "product_name": "示例 CTA",
        "report_date": "2026-07-24",
        "factor_contribution": {"long_term_rule": 0.03},
    }
    result = validate_against_report(regression, report, {"trend": 0.02})
    comparison = result.factor_comparisons[0]
    assert comparison.l1_predicted_contribution == 0.03
    assert comparison.direction_match is True
    assert comparison.magnitude_ratio == 1.0


def test_validation_refuses_to_compare_beta_directly_to_l4_pnl() -> None:
    regression = {"r_squared": 0.1, "factors": [{"name": "trend", "beta": -2.0, "t_stat": 1.0}]}
    report = {"factor_contribution": {"long_term_rule": 0.01}}
    result = validate_against_report(regression, report)
    comparison = result.factor_comparisons[0]
    assert comparison.l1_predicted_contribution is None
    assert comparison.direction_match is None
    assert any("不能比较" in warning for warning in result.warnings)
