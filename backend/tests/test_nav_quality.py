from dataclasses import dataclass
from datetime import date, timedelta

from app.services.nav_quality import assess_machine_nav_review, assess_nav_quality


@dataclass
class Observation:
    observation_date: date
    nav: float
    frequency: str | None = None
    review_status: str = "pending"


@dataclass
class Fact:
    field_name: str
    field_value: str


def test_quality_blocks_series_that_conflicts_with_disclosed_drawdown() -> None:
    assessment = assess_nav_quality(
        [
            Observation(date(2026, 6, 12), 1.4315),
            Observation(date(2026, 7, 17), 1.2020),
        ],
        [
            Fact("disclosed_cumulative_return", "0.202"),
            Fact("disclosed_maximum_drawdown", "-0.0171"),
        ],
    )

    assert assessment["blocking"] is True
    assert assessment["computed_maximum_drawdown"] == -0.160321
    assert any("最大回撤" in reason for reason in assessment["reasons"])


def test_quality_exposes_disclosed_sharpe_as_a_non_blocking_reference() -> None:
    assessment = assess_nav_quality(
        [Observation(date(2026, 1, 2), 1.0), Observation(date(2026, 1, 9), 1.01)],
        [Fact("disclosed_sharpe_ratio", "1.23")],
    )

    assert assessment["disclosed_sharpe_ratio"] == 1.23
    assert assessment["blocking"] is False


def test_machine_review_passes_stable_high_confidence_weekly_curve() -> None:
    points = [
        Observation(date(2026, 1, 2), 1.00),
        Observation(date(2026, 1, 9), 1.01),
        Observation(date(2026, 1, 16), 1.00),
        Observation(date(2026, 1, 23), 1.02),
        Observation(date(2026, 1, 30), 1.03),
        Observation(date(2026, 2, 6), 1.02),
        Observation(date(2026, 2, 13), 1.04),
        Observation(date(2026, 2, 20), 1.05),
    ]

    review = assess_machine_nav_review(points, [], source_confidence=0.91, declared_frequency="weekly")

    assert review["machine_reviewed"] is True
    assert review["scope"] == "可进入初步研究"


def test_machine_review_keeps_low_confidence_curve_for_human_review() -> None:
    points = [Observation(date(2026, 1, 2) + timedelta(days=index * 7), 1 + index * 0.001) for index in range(8)]

    review = assess_machine_nav_review(points, [], source_confidence=0.55, declared_frequency="weekly")

    assert review["machine_reviewed"] is False
    assert any("置信度" in reason for reason in review["reasons"])


def test_machine_review_blocks_flat_image_trace() -> None:
    points = [Observation(date(2026, 1, 2) + timedelta(days=index * 7), 1.0) for index in range(12)]

    review = assess_machine_nav_review(points, [], source_confidence=0.98, declared_frequency="weekly")

    assert review["machine_reviewed"] is False
    assert any("近似恒定" in reason for reason in review["reasons"])


def test_quality_allows_reviewed_monthly_series_with_large_real_return() -> None:
    assessment = assess_nav_quality(
        [
            Observation(date(2020, 2, 29), 2.4957, "monthly", "reviewed"),
            Observation(date(2020, 3, 31), 3.7703, "monthly", "reviewed"),
        ],
        [],
    )

    assert assessment["largest_period_change"] == 0.510718
    assert assessment["blocking"] is False
