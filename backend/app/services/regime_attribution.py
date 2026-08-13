"""Auditable observable-regime classification and conditional attribution."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np


REGIME_NAMES = ("normal", "crisis", "whipsaw")


def _annualization(frequency: str) -> int:
    return {"daily": 252, "weekly": 52, "monthly": 12}.get(frequency, 52)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return round(value, 8) if np.isfinite(value) else None


def _expected_shortfall(values: np.ndarray, quantile: float = 0.05) -> float | None:
    if values.size == 0:
        return None
    cutoff = float(np.quantile(values, quantile))
    tail = values[values <= cutoff]
    return round(float(tail.mean()), 8) if tail.size else None


def classify_observable_regimes(
    market_proxy: np.ndarray,
    dates: list[date],
    frequency: str,
    *,
    vol_window: int = 12,
    trend_short_window: int = 4,
    trend_medium_window: int = 12,
    min_history: int = 24,
) -> dict[str, Any]:
    """Assign states using only proxy observations strictly before each date.

    ``market_proxy[t]`` is deliberately excluded from the state assigned to
    ``t``. Expanding empirical thresholds are also calculated from prior
    observations only, so a future shock cannot relabel an earlier period.
    """
    proxy = np.asarray(market_proxy, dtype=float)
    if proxy.ndim != 1 or len(proxy) != len(dates):
        raise ValueError("市场代理与日期长度必须一致")
    if len(proxy) < 2 or not np.isfinite(proxy).all():
        raise ValueError("市场代理必须包含至少两个有限观测")
    if min_history < max(vol_window, trend_medium_window, 2):
        raise ValueError("min_history 必须覆盖波动率和趋势窗口")
    if vol_window < 2 or trend_short_window < 2 or trend_medium_window < trend_short_window:
        raise ValueError("状态窗口参数无效")

    states: list[str] = []
    features: list[dict[str, Any]] = []
    historical_vols: list[float] = []
    for index, point_date in enumerate(dates):
        prior = proxy[:index]
        state = "insufficient"
        lagged_vol: float | None = None
        short_trend: float | None = None
        medium_trend: float | None = None
        crisis_return_cutoff: float | None = None
        crisis_vol_cutoff: float | None = None
        normal_vol_cutoff: float | None = None
        if len(prior) >= min_history:
            lagged_slice = prior[-vol_window:]
            lagged_vol = float(np.std(lagged_slice, ddof=1)) if len(lagged_slice) >= 2 else None
            short_trend = float(prior[-trend_short_window:].sum())
            medium_trend = float(prior[-trend_medium_window:].sum())
            volatility_history = np.asarray(historical_vols, dtype=float)
            varying_returns = float(np.std(prior)) > 1e-12
            crisis_return_cutoff = float(np.quantile(prior, 0.10)) if varying_returns else None
            crisis_vol_cutoff = float(np.quantile(volatility_history, 0.90)) if len(volatility_history) >= 3 else None
            normal_vol_cutoff = float(np.quantile(volatility_history, 0.50)) if len(volatility_history) >= 3 else None
            is_crisis = (
                (crisis_return_cutoff is not None and prior[-1] <= crisis_return_cutoff)
                or (crisis_vol_cutoff is not None and lagged_vol is not None and lagged_vol >= crisis_vol_cutoff)
            )
            is_whipsaw = (
                short_trend is not None and medium_trend is not None
                and short_trend * medium_trend < 0
                and normal_vol_cutoff is not None and lagged_vol is not None and lagged_vol >= normal_vol_cutoff
            )
            state = "crisis" if is_crisis else "whipsaw" if is_whipsaw else "normal"
        states.append(state)
        features.append({
            "date": point_date.isoformat(),
            "state": state,
            "lagged_volatility": round(lagged_vol, 8) if lagged_vol is not None else None,
            "short_trend": round(short_trend, 8) if short_trend is not None else None,
            "medium_trend": round(medium_trend, 8) if medium_trend is not None else None,
            "crisis_return_cutoff": round(crisis_return_cutoff, 8) if crisis_return_cutoff is not None else None,
            "crisis_volatility_cutoff": round(crisis_vol_cutoff, 8) if crisis_vol_cutoff is not None else None,
        })
        # This date's volatility becomes historical input only for the next
        # date; otherwise a period helps define its own regime threshold.
        if lagged_vol is not None and np.isfinite(lagged_vol):
            historical_vols.append(lagged_vol)
    counts = {name: int(states.count(name)) for name in (*REGIME_NAMES, "insufficient")}
    return {
        "method": "observable_lagged_regime_rules",
        "parameters": {
            "vol_window": vol_window,
            "trend_short_window": trend_short_window,
            "trend_medium_window": trend_medium_window,
            "min_history": min_history,
            "crisis_return_quantile": 0.10,
            "crisis_volatility_quantile": 0.90,
            "uses_future_data": False,
            "hmm_used": False,
        },
        "states": states,
        "state_counts": counts,
        "features": features,
        "warnings": [
            "状态由滞后市场代理构造，不能解释为产品真实持仓状态。",
            "Normal/Crisis/Whipsaw 是预登记规则切片，不是 HMM 隐状态识别。",
        ],
    }


def run_regime_attribution(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[date],
    factor_names: list[str],
    regime_result: dict[str, Any],
    frequency: str,
    *,
    market_proxy: np.ndarray | None = None,
    min_regime_observations: int = 20,
) -> dict[str, Any]:
    """Estimate per-regime OLS outputs, preserving unavailable coefficients."""
    y = np.asarray(returns, dtype=float)
    x = np.asarray(factor_returns, dtype=float)
    states = list(regime_result.get("states", []))
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or len(y) != len(dates) or len(y) != len(states):
        raise ValueError("状态归因输入长度不一致")
    if len(factor_names) != x.shape[1]:
        raise ValueError("因子名称与因子列数不一致")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("状态归因输入必须为有限数")
    if min_regime_observations < max(4, x.shape[1] + 2):
        raise ValueError("状态最小样本量过低")
    proxy = np.asarray(market_proxy if market_proxy is not None else x[:, 0], dtype=float)
    if proxy.ndim != 1 or len(proxy) != len(y) or not np.isfinite(proxy).all():
        raise ValueError("市场代理与收益长度不一致")

    reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    ann = _annualization(frequency)
    for name in REGIME_NAMES:
        mask = np.asarray([state == name for state in states], dtype=bool)
        count = int(mask.sum())
        base: dict[str, Any] = {
            "state": name,
            "status": "insufficient" if count < min_regime_observations else "available",
            "observation_count": count,
            "conditional_return": None,
            "positive_rate": None,
            "expected_shortfall_5pct": None,
            "conditional_alpha": None,
            "conditional_alpha_annualized": None,
            "crisis_alpha_candidate": None,
            "conditional_beta": None,
            "conditional_correlation": None,
            "residual_volatility": None,
            "warnings": [],
        }
        if count < min_regime_observations:
            message = f"{name} 状态样本不足（{count}，需要至少 {min_regime_observations}），不输出条件系数。"
            base["warnings"] = [message]
            warnings.append(message)
            reports.append(base)
            continue
        selected_y = y[mask]
        selected_x = x[mask]
        design = np.column_stack([np.ones(count), selected_x])
        coefficients, _, _, _ = np.linalg.lstsq(design, selected_y, rcond=None)
        residuals = selected_y - design @ coefficients
        beta = {factor_names[index]: round(float(value), 8) for index, value in enumerate(coefficients[1:])}
        conditional_alpha = float(coefficients[0])
        base.update({
            "conditional_return": round(float(selected_y.mean()), 8),
            "positive_rate": round(float(np.mean(selected_y > 0)), 8),
            "expected_shortfall_5pct": _expected_shortfall(selected_y),
            "conditional_alpha": round(conditional_alpha, 8),
            "conditional_alpha_annualized": round(conditional_alpha * ann, 8),
            "crisis_alpha_candidate": round(conditional_alpha * ann, 8) if name == "crisis" else None,
            "conditional_beta": beta,
            "conditional_correlation": _correlation(selected_y, proxy[mask]),
            "residual_volatility": round(float(np.std(residuals, ddof=1)) if count > 1 else 0.0, 8),
        })
        reports.append(base)
    return {
        "method": "conditional_ols_by_observable_state",
        "parameters": {"min_regime_observations": min_regime_observations, "annualization": ann},
        "regimes": reports,
        "warnings": warnings,
    }
