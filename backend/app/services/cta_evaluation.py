"""Phase-A CTA evaluation using only confirmed, human-reviewed NAV."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np


def evaluate_nav_path(navs: list[float], dates: list[date], frequency: str) -> dict[str, Any]:
    """Compute tail and drawdown-path metrics from an already reviewed NAV series."""
    if len(navs) < 2:
        raise ValueError("至少需要两条已审核净值")
    values = np.asarray(navs, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("已审核净值必须为有限正数")
    returns = values[1:] / values[:-1] - 1.0
    annualization = {"daily": 252, "weekly": 52, "monthly": 12}.get(frequency, 52)
    wealth = values / values[0]
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    maximum_drawdown = float(drawdowns.min())
    active = drawdowns < -1e-12
    durations: list[int] = []
    start: int | None = None
    for index, in_drawdown in enumerate(active):
        if in_drawdown and start is None:
            start = index
        elif not in_drawdown and start is not None:
            durations.append(index - start)
            start = None
    current_duration = len(active) - start if start is not None else 0
    if start is not None:
        durations.append(current_duration)
    cutoff = float(np.quantile(returns, 0.05))
    expected_shortfall = float(returns[returns <= cutoff].mean())
    cagr = float((values[-1] / values[0]) ** (annualization / len(returns)) - 1.0)
    calmar = float(cagr / abs(maximum_drawdown)) if maximum_drawdown < -1e-12 else None
    return {
        "observation_count": len(values), "start_date": dates[0].isoformat(), "end_date": dates[-1].isoformat(),
        "cagr": round(cagr, 8), "maximum_drawdown": round(maximum_drawdown, 8),
        "calmar": round(calmar, 8) if calmar is not None else None,
        "expected_shortfall_5pct": round(expected_shortfall, 8),
        "drawdown_duration_max_periods": max(durations, default=0),
        "current_drawdown_duration_periods": current_duration, "recovery_completed": start is None,
        "drawdown_path": [{"date": item_date.isoformat(), "drawdown": round(float(value), 8)} for item_date, value in zip(dates, drawdowns, strict=True)],
    }


def build_attribution_tables(regression: Any) -> dict[str, Any]:
    """Return exposure, return attribution and model risk as distinct outputs."""
    exposure: list[dict[str, Any]] = []
    returns: list[dict[str, Any]] = []
    risks: list[dict[str, Any]] = []
    for factor in regression.factors:
        common = {"factor_name": factor.name, "display_name": factor.display_name, "factor_group": factor.factor_group}
        exposure.append({**common, "beta": factor.beta, "hac_t_stat": factor.t_stat, "hac_p_value": factor.p_value,
                         "bootstrap_ci_low": factor.bootstrap_ci_low, "bootstrap_ci_high": factor.bootstrap_ci_high})
        returns.append({**common, "mean_return_contribution": factor.mean_return_contribution})
        risks.append({**common, "component_risk_contribution": regression.factor_risk_contributions.get(factor.name)})
    mean_product_return = getattr(regression, "mean_product_return", None)
    mean_factor_return = getattr(regression, "mean_factor_explained_return", None)
    intercept = getattr(regression, "intercept", None)
    residual = (
        float(mean_product_return - intercept - mean_factor_return)
        if all(value is not None for value in (mean_product_return, intercept, mean_factor_return))
        else None
    )
    return {
        "factor_exposure": exposure,
        "return_contribution": returns,
        "euler_risk_contribution": risks,
        "return_reconciliation": {
            "mean_product_return": mean_product_return,
            "mean_factor_explained_return": mean_factor_return,
            "mean_intercept_return": intercept,
            "mean_residual_return": round(residual, 8) if residual is not None else None,
        },
    }
