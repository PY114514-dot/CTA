"""Causal Kalman filtering for low-dimensional return-based CTA style beta."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


DYNAMIC_FACTOR_NAMES = ("trend", "basis_carry", "short_term_trend_20", "mean_reversion_5d")
MIN_DYNAMIC_OBSERVATIONS = 19  # 20 NAV points produce 19 period returns.


def prepare_factor_matrix(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    factor_names: list[str] | tuple[str, ...] = DYNAMIC_FACTOR_NAMES,
) -> dict[str, Any]:
    """Align cached baseline factor returns to the product's economic periods.

    This function only reads the factor cache. It deliberately does not build
    factors, fetch market data, or inspect source materials.
    """
    from app.services import factor_library
    from app.services.factor_library.factor_regression import _aggregate_returns, _frequency_index

    if len(product_returns) != len(product_dates):
        raise ValueError("产品收益与日期长度必须一致")
    y_raw = pd.Series(
        np.asarray(product_returns, dtype=float),
        index=pd.to_datetime(product_dates),
        name="product",
    ).sort_index()
    if not np.isfinite(y_raw.to_numpy(dtype=float)).all():
        raise ValueError("产品收益必须为有限数")
    if (y_raw <= -1.0).any():
        raise ValueError("产品收益不能低于 -100%")
    period_index = _frequency_index(y_raw.index, frequency)
    date_periodic = pd.Series(y_raw.index, index=period_index).groupby(level=0).last()
    # Compound all observations in an economic period.  Keeping only the
    # last observation silently discards earlier sub-period returns.
    y_periodic = _aggregate_returns(y_raw, frequency)

    factor_series: dict[str, pd.Series] = {}
    warnings: list[str] = []
    for name in factor_names:
        raw = factor_library.get_factor_series(name, risk_profile="baseline")
        if raw is None or raw.empty:
            warnings.append(f"因子 '{name}' 没有已缓存的 baseline 收益，已跳过。")
            continue
        aggregated = _aggregate_returns(raw, frequency)
        if aggregated is None or aggregated.empty:
            warnings.append(f"因子 '{name}' 在当前频率没有可用收益，已跳过。")
            continue
        factor_series[name] = aggregated.astype(float)
    if not factor_series:
        raise ValueError("没有可用的已缓存 baseline 因子；请先构建因子库")

    x_raw = pd.DataFrame(factor_series)
    if frequency == "daily":
        x_raw.index = _frequency_index(x_raw.index, frequency)
    common_idx = y_periodic.index.intersection(x_raw.index).sort_values()
    if len(common_idx) < MIN_DYNAMIC_OBSERVATIONS:
        raise ValueError(
            f"产品净值与已缓存因子对齐后仅有 {len(common_idx)} 个观测，"
            f"至少需要 {MIN_DYNAMIC_OBSERVATIONS} 个"
        )
    x = x_raw.loc[common_idx]
    valid_columns = [
        column for column in x.columns
        if int(x[column].notna().sum()) >= MIN_DYNAMIC_OBSERVATIONS
    ]
    dropped = [column for column in x.columns if column not in valid_columns]
    if dropped:
        warnings.append(f"因子数据不足，已跳过：{', '.join(dropped)}")
    x = x[valid_columns]
    if x.empty:
        raise ValueError("对齐后没有达到最低数据要求的因子")
    aligned_returns = y_periodic.loc[common_idx]
    missing_rows = x.isna().any(axis=1)
    if missing_rows.any():
        warnings.append(f"因子对齐窗口内有 {int(missing_rows.sum())} 个观测缺失，已剔除这些期间。")
        x = x.loc[~missing_rows]
        aligned_returns = aligned_returns.loc[~missing_rows]
    if len(x) < MIN_DYNAMIC_OBSERVATIONS:
        raise ValueError(
            f"剔除缺失因子后仅有 {len(x)} 个观测，至少需要 {MIN_DYNAMIC_OBSERVATIONS} 个"
        )
    aligned_dates = []
    for value in date_periodic.loc[common_idx].tolist():
        timestamp = pd.Timestamp(value)
        aligned_dates.append(timestamp.date())
    return {
        "returns": aligned_returns.to_numpy(dtype=float),
        "factor_returns": x.to_numpy(dtype=float),
        "dates": aligned_dates,
        "factor_names": list(x.columns),
        "warnings": warnings,
        "observation_count": len(common_idx),
    }


def compute_rolling_beta(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[date],
    factor_names: list[str],
    windows: list[int],
    *,
    minimum_window: int = 20,
) -> dict[str, Any]:
    """Estimate trailing-window OLS beta paths and transparent drift summaries."""
    y = np.asarray(returns, dtype=float)
    x = np.asarray(factor_returns, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or len(y) != len(dates):
        raise ValueError("滚动 Beta 输入长度不一致")
    if len(factor_names) != x.shape[1]:
        raise ValueError("因子名称与因子列数不一致")
    if minimum_window < 2:
        raise ValueError("滚动窗口下限必须至少为 2 个观测")
    if not windows or any(window < minimum_window for window in windows):
        raise ValueError(f"滚动窗口必须至少为 {minimum_window} 个观测")

    reports: list[dict[str, Any]] = []
    for window in sorted(set(windows)):
        if len(y) < window:
            reports.append({"window": window, "status": "insufficient", "n_periods": 0, "paths": [], "warnings": [f"样本不足：需要 {window} 个观测"]})
            continue
        paths: list[dict[str, Any]] = []
        estimates: list[np.ndarray] = []
        for end in range(window - 1, len(y)):
            design = np.column_stack([np.ones(window), x[end - window + 1:end + 1]])
            try:
                coefficients, _, _, _ = np.linalg.lstsq(design, y[end - window + 1:end + 1], rcond=None)
            except np.linalg.LinAlgError:
                continue
            betas = coefficients[1:]
            estimates.append(betas)
            paths.append({
                "date": dates[end].isoformat(),
                "alpha": round(float(coefficients[0]), 8),
                "betas": {name: round(float(betas[index]), 8) for index, name in enumerate(factor_names)},
            })
        beta_array = np.asarray(estimates, dtype=float)
        if beta_array.size == 0:
            reports.append({"window": window, "status": "insufficient", "n_periods": 0, "paths": [], "warnings": ["滚动回归没有产生有效估计"]})
            continue
        summary: dict[str, Any] = {}
        alerts: list[dict[str, Any]] = []
        for index, name in enumerate(factor_names):
            values = beta_array[:, index]
            median = float(np.median(values))
            reference_sign = np.sign(median)
            signs = np.sign(values)
            non_zero = signs != 0
            sign_consistency = float(np.mean(signs[non_zero] == reference_sign)) if non_zero.any() else None
            sign_change_count = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
            mean = float(np.mean(values))
            coefficient_variation = float(np.std(values, ddof=1) / abs(mean)) if len(values) > 1 and abs(mean) > 1e-12 else None
            summary[name] = {
                "median_beta": round(median, 8),
                "first_beta": round(float(values[0]), 8),
                "last_beta": round(float(values[-1]), 8),
                "beta_range": round(float(np.ptp(values)), 8),
                "sign_consistency": round(sign_consistency, 6) if sign_consistency is not None else None,
                "sign_change_count": sign_change_count,
                "coefficient_variation": round(coefficient_variation, 6) if coefficient_variation is not None else None,
            }
            if sign_change_count > 0:
                alerts.append({"factor_name": name, "type": "sign_change", "severity": "warning", "message": "滚动 Beta 出现符号变化"})
            elif coefficient_variation is not None and coefficient_variation > 1.0:
                alerts.append({"factor_name": name, "type": "large_variation", "severity": "watch", "message": "滚动 Beta 相对波动较大"})
        reports.append({
            "window": window,
            "status": "available",
            "n_periods": len(paths),
            "paths": paths,
            "summary": summary,
            "alerts": alerts,
            "warnings": ["滚动 Beta 依赖窗口长度，仅用于风格漂移监控，不作为最终动态暴露估计。"],
        })
    return {"windows": reports}


def filter_dynamic_beta(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[date],
    factor_names: list[str],
    *,
    observation_variance: float | None = None,
    state_variance: float = 1e-7,
) -> dict[str, Any]:
    """Run a one-pass random-walk beta filter without any future observations.

    The state includes an intercept followed by factor betas.  Parameters are
    intentionally fixed and returned to make this monitoring model auditable.
    """
    y = np.asarray(returns, dtype=float)
    x = np.asarray(factor_returns, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or len(y) != len(dates):
        raise ValueError("收益、因子收益与日期长度必须一致")
    if len(y) < MIN_DYNAMIC_OBSERVATIONS or x.shape[1] == 0:
        raise ValueError(
            f"Kalman 动态 Beta 至少需要 {MIN_DYNAMIC_OBSERVATIONS} 个观测和一个因子"
        )
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Kalman 动态 Beta 输入必须为有限数")
    if len(factor_names) != x.shape[1]:
        raise ValueError("因子名称与因子列数必须一致")
    if state_variance <= 0:
        raise ValueError("state_variance 必须为正数")

    dimensions = x.shape[1] + 1
    # Fixed default keeps the filter causal: estimating this from the full
    # sample would allow a later return to alter earlier filtered estimates.
    measurement_variance = float(observation_variance or 1e-8)
    state = np.zeros(dimensions)
    covariance = np.eye(dimensions)
    covariance[0, 0] = 1e-8
    process_covariance = np.eye(dimensions) * state_variance
    process_covariance[0, 0] = state_variance * 0.01
    states: list[np.ndarray] = []
    standard_errors: list[np.ndarray] = []
    innovations: list[float] = []

    for target, exposures in zip(y, x, strict=True):
        # Prediction only uses the previous state; update uses this period's
        # realised return and factor value. No smoothing is applied.
        predicted_covariance = covariance + process_covariance
        design = np.r_[1.0, exposures]
        innovation = float(target - design @ state)
        innovation_variance = float(design @ predicted_covariance @ design + measurement_variance)
        gain = (predicted_covariance @ design) / innovation_variance
        state = state + gain * innovation
        covariance = (np.eye(dimensions) - np.outer(gain, design)) @ predicted_covariance
        covariance = (covariance + covariance.T) / 2.0
        states.append(state.copy())
        standard_errors.append(np.sqrt(np.maximum(np.diag(covariance), 0.0)))
        innovations.append(innovation)

    state_array = np.asarray(states)
    se_array = np.asarray(standard_errors)
    cumulative_beta = np.cumsum(state_array[:, 1:], axis=0)
    expanding_long_run_beta = cumulative_beta / np.arange(1, len(state_array) + 1, dtype=float)[:, None]
    long_run_beta = expanding_long_run_beta[-1]
    timing = (state_array[:, 1:] - expanding_long_run_beta) * x
    paths = []
    for index, point_date in enumerate(dates):
        point_long_run_beta = expanding_long_run_beta[index]
        long_run_contribution = point_long_run_beta * x[index]
        dynamic_contribution = (state_array[index, 1:] - point_long_run_beta) * x[index]
        paths.append({
            "date": point_date.isoformat(),
            "alpha": round(float(state_array[index, 0]), 8),
            "alpha_standard_error": round(float(se_array[index, 0]), 8),
            "betas": {name: round(float(state_array[index, position + 1]), 8) for position, name in enumerate(factor_names)},
            "beta_standard_errors": {name: round(float(se_array[index, position + 1]), 8) for position, name in enumerate(factor_names)},
            "causal_long_run_style_beta": {name: round(float(point_long_run_beta[position]), 8) for position, name in enumerate(factor_names)},
            "long_run_style_contribution": {name: round(float(long_run_contribution[position]), 8) for position, name in enumerate(factor_names)},
            "dynamic_timing_contribution_by_factor": {name: round(float(dynamic_contribution[position]), 8) for position, name in enumerate(factor_names)},
            "dynamic_timing_contribution": round(float(timing[index].sum()), 8),
            "innovation": round(float(innovations[index]), 8),
        })
    beta_ranges = {
        name: round(float(np.ptp(state_array[:, position + 1])), 8)
        for position, name in enumerate(factor_names)
    }
    return {
        "method": "causal_random_walk_kalman_filter",
        "factor_names": list(factor_names),
        "parameters": {
            "state_variance": state_variance,
            "observation_variance": measurement_variance,
            "smoother_used": False,
            "long_run_beta_method": "expanding_filtered_mean",
        },
        "long_run_style_beta": {name: round(float(long_run_beta[position]), 8) for position, name in enumerate(factor_names)},
        "beta_range": beta_ranges,
        "dynamic_timing_contribution_total": round(float(timing.sum()), 8),
        "innovation_root_mean_square": round(float(np.sqrt(np.mean(np.square(innovations)))), 8),
        "paths": paths,
        "warnings": [
            "动态 Beta 为过滤估计，未使用未来数据或平滑器。",
            "Dynamic Timing Contribution 是统计分解，不代表确定性的管理人择时收益。",
        ],
    }
