"""Regression coverage for report parsing rules without local research data."""

from app.services.multi_product_report import _extract_product_identity, _parse_single_product_factsheet
from app.services.material_ingestion import _infer_frequency, _resample_pixel_points_weekly


def test_pixel_trace_samples_are_not_labelled_daily() -> None:
    """One sample per image column is interpolation, not a daily NAV feed."""
    from datetime import date, timedelta

    dates = [date(2025, 7, 10) + timedelta(days=index) for index in range(366)]
    assert _infer_frequency(dates, pixel_trace=True) == "weekly"
    sampled = _resample_pixel_points_weekly(
        [{"observation_date": item, "nav": 1.0 + index / 1000} for index, item in enumerate(dates)]
    )
    assert 50 <= len(sampled) <= 55
    assert sampled[0]["observation_date"] == dates[0]
    assert sampled[-1]["observation_date"] == dates[-1]


def test_chart_caption_beats_strategy_introduction_as_product_identity() -> None:
    """A strategy-page heading must not create a second pseudo-product."""
    identity = _extract_product_identity(
        "低波CTA策略产品介绍\n方普投资｜致远系列\n致远二号-周度净值与回撤 (2024.06.20-2026.07.17)",
        "",
        [],
    )
    assert identity.product_name == "致远二号"
    assert identity.manager_name == "方普投资"
    assert identity.strategy == "商品 CTA" or identity.strategy is None
    assert identity.method == "净值图标题"


def test_single_product_factsheet_keeps_disclosed_sharpe() -> None:
    metric = _parse_single_product_factsheet(
        "致远二号 1.1234 1.00% 12.00% 10.00% 3.00%\n截至2026-07-17\n夏普比率：1.23",
        "",
    )

    assert metric is not None
    assert metric.sharpe_ratio == 1.23
