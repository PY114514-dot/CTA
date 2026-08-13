"""Explainable MVP classification for CTA strategy factor fingerprints.

This service reports correlation evidence rather than reconstructed holdings.
Future versions can introduce regularized rolling regressions and factor data
construction without changing the public route contract.
"""

import numpy as np

from app.schemas import FactorExposure, StrategyFingerprintRequest, StrategyFingerprintResponse
from app.services.analysis._stats import safe_pearson


TIMESERIES_FACTOR_NAMES = {"time_series_momentum", "trend", "tsm"}
CROSS_SECTIONAL_FACTOR_NAMES = {"cross_sectional_momentum", "cross_sectional", "csm"}


def classify_strategy_fingerprint(request: StrategyFingerprintRequest) -> StrategyFingerprintResponse:
    """Rank aligned factor correlations and give a conservative style label."""
    product_returns = np.asarray(request.product_periodic_returns, dtype=float)
    exposures: list[FactorExposure] = []

    for factor_series in request.factor_return_series:
        factor_returns = np.asarray(factor_series.periodic_returns, dtype=float)
        correlation = _safe_correlation(product_returns, factor_returns)
        exposures.append(
            FactorExposure(
                factor_name=factor_series.factor_name,
                correlation=correlation,
                exposure_strength=_exposure_strength(correlation),
            )
        )

    exposures.sort(key=lambda item: abs(item.correlation), reverse=True)
    strategy_style, confidence = _classify_style(exposures)
    return StrategyFingerprintResponse(
        strategy_style=strategy_style,
        confidence=confidence,
        factor_exposures=exposures,
        disclaimer=(
            "This is a statistical factor fingerprint, not a disclosure or "
            "reconstruction of actual holdings. Interpret it with sample-period and data-quality context."
        ),
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return zero when one series has no variation, avoiding undefined output."""
    return safe_pearson(left, right)


def _exposure_strength(correlation: float) -> str:
    """Convert correlation magnitude into an intentionally coarse label."""
    magnitude = abs(correlation)
    if magnitude >= 0.6:
        return "high"
    if magnitude >= 0.3:
        return "medium"
    return "low"


def _classify_style(exposures: list[FactorExposure]) -> tuple[str, float]:
    """Compare best time-series and cross-sectional evidence when available."""
    time_series_score = max(
        (abs(item.correlation) for item in exposures if item.factor_name.lower() in TIMESERIES_FACTOR_NAMES),
        default=0.0,
    )
    cross_sectional_score = max(
        (abs(item.correlation) for item in exposures if item.factor_name.lower() in CROSS_SECTIONAL_FACTOR_NAMES),
        default=0.0,
    )
    strongest_score = max(time_series_score, cross_sectional_score)

    if strongest_score < 0.3:
        return "insufficient_evidence", strongest_score
    if abs(time_series_score - cross_sectional_score) < 0.15:
        return "mixed_cta", strongest_score
    if time_series_score > cross_sectional_score:
        return "time_series_cta", time_series_score
    return "cross_sectional_cta", cross_sectional_score
