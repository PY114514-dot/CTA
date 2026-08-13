"""Focused tests for the explainable strategy-fingerprint service."""

from app.schemas import FactorReturnSeries, StrategyFingerprintRequest
from app.services.strategy_fingerprint import classify_strategy_fingerprint


def test_time_series_factor_is_classified_as_time_series_cta() -> None:
    """A perfectly aligned trend factor gives the strongest time-series signal."""
    trend_returns = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01]
    request = StrategyFingerprintRequest(
        product_periodic_returns=trend_returns,
        factor_return_series=[
            FactorReturnSeries(factor_name="time_series_momentum", periodic_returns=trend_returns),
            FactorReturnSeries(
                factor_name="cross_sectional_momentum",
                periodic_returns=[0.01, -0.02, 0.01, -0.03, 0.02, -0.02, 0.03, 0.01],
            ),
        ],
    )

    fingerprint = classify_strategy_fingerprint(request)

    assert fingerprint.strategy_style == "time_series_cta"
    assert fingerprint.factor_exposures[0].factor_name == "time_series_momentum"
    assert fingerprint.confidence == 1.0
