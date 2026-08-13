"""Deterministic performance-metric calculations for validated NAV histories."""

import math

import numpy as np

from app.schemas import DataFrequency, NavAnalysisRequest, NavAnalysisResponse, PerformanceMetrics


ANNUALIZATION_FACTORS: dict[DataFrequency, int] = {
    DataFrequency.DAILY: 252,
    DataFrequency.WEEKLY: 52,
    DataFrequency.MONTHLY: 12,
}


def calculate_nav_analysis(request: NavAnalysisRequest) -> NavAnalysisResponse:
    """Calculate standard metrics without filling or inventing missing NAV data.

    Returns are simple period-over-period returns. Annualization uses the
    frequency explicitly confirmed by the caller, rather than inferred dates.
    """
    nav_values = np.asarray([point.net_asset_value for point in request.nav_points], dtype=float)
    periodic_returns = nav_values[1:] / nav_values[:-1] - 1.0
    annualization_factor = ANNUALIZATION_FACTORS[request.frequency]
    elapsed_days = (request.nav_points[-1].observation_date - request.nav_points[0].observation_date).days
    if elapsed_days <= 0:
        raise ValueError("nav_points must span at least one calendar day")

    cumulative_return = float(nav_values[-1] / nav_values[0] - 1.0)
    annualized_return = float((1.0 + cumulative_return) ** (365.25 / elapsed_days) - 1.0)
    annualized_volatility = float(np.std(periodic_returns, ddof=1) * math.sqrt(annualization_factor))

    # A zero-volatility series has no meaningful risk-adjusted ratio.
    sharpe_ratio = None
    if annualized_volatility > 0:
        sharpe_ratio = float((annualized_return - request.annual_risk_free_rate) / annualized_volatility)

    running_peaks = np.maximum.accumulate(nav_values)
    maximum_drawdown = float(np.min(nav_values / running_peaks - 1.0))
    calmar_ratio = float(annualized_return / abs(maximum_drawdown)) if maximum_drawdown < 0 else None

    return NavAnalysisResponse(
        metrics=PerformanceMetrics(
            cumulative_return=cumulative_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            maximum_drawdown=maximum_drawdown,
            calmar_ratio=calmar_ratio,
        ),
        periodic_returns=[float(value) for value in periodic_returns],
        annualization_factor=annualization_factor,
    )
