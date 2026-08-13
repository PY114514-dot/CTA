"""Focused tests for the deterministic NAV metric service."""

from app.schemas import DataFrequency, NavAnalysisRequest, NetAssetValuePoint
from app.services.nav_metrics import calculate_nav_analysis


def test_calculate_nav_analysis_returns_expected_total_and_drawdown() -> None:
    """The calculator reports an 8% total return and a 10% peak-to-trough loss."""
    request = NavAnalysisRequest(
        frequency=DataFrequency.MONTHLY,
        nav_points=[
            NetAssetValuePoint(observation_date="2024-01-31", net_asset_value=1.0),
            NetAssetValuePoint(observation_date="2024-02-29", net_asset_value=1.2),
            NetAssetValuePoint(observation_date="2024-03-31", net_asset_value=1.08),
        ],
    )

    analysis = calculate_nav_analysis(request)

    assert round(analysis.metrics.cumulative_return, 6) == 0.08
    assert round(analysis.metrics.maximum_drawdown, 6) == -0.1
    assert analysis.annualization_factor == 12
