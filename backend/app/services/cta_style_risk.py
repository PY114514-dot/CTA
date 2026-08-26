"""Return-based CTA style risk model; it never infers holdings or positions."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np


def evaluate_style_risk(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[date],
    factor_names: list[str],
    *,
    rolling_window: int,
) -> dict[str, Any]:
    """Estimate transparent style exposure, factor risk and causal drift."""
    y = np.asarray(returns, dtype=float)
    x = np.asarray(factor_returns, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or len(y) != len(dates):
        raise ValueError("收益、因子和日期长度必须一致")
    if len(y) < 19 or x.shape[1] == 0 or len(factor_names) != x.shape[1]:
        raise ValueError("风格风险模型至少需要 19 个对齐收益和一个因子")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("风格风险模型输入必须为有限数")
    design = np.column_stack([np.ones(len(y)), x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("CTA 因子矩阵秩不足，无法区分各风格暴露")
    condition_number = float(np.linalg.cond(design[:, 1:]))
    warnings = ["该模型从净值收益反推统计风格风险，不代表真实持仓、品种权重或管理人交易意图。", "风险贡献是当前公开因子代理下的方差分解，不构成投资建议。"]
    if condition_number > 100:
        warnings.append(f"CTA 因子矩阵条件数为 {condition_number:.1f}，单因子暴露可能受共线性影响，应优先阅读总系统风险。")
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    beta = coefficients[1:]
    residual = y - design @ coefficients
    covariance = np.cov(x, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    systematic_variance = float(beta @ covariance @ beta)
    total_variance = float(np.var(y, ddof=1))
    idiosyncratic_variance = max(0.0, float(np.var(residual, ddof=1)))
    marginal = covariance @ beta
    contributions = beta * marginal
    factor_rows = [
        {"factor_name": name, "beta": round(float(beta[index]), 8),
         "risk_contribution": round(float(contributions[index] / total_variance), 8) if total_variance > 0 else None}
        for index, name in enumerate(factor_names)
    ]
    paths: list[dict[str, Any]] = []
    if rolling_window >= 2:
        for end in range(rolling_window - 1, len(y)):
            window_design = np.column_stack([np.ones(rolling_window), x[end - rolling_window + 1:end + 1]])
            rolling_coefficients, _, _, _ = np.linalg.lstsq(window_design, y[end - rolling_window + 1:end + 1], rcond=None)
            paths.append({"date": dates[end].isoformat(), "betas": {name: round(float(rolling_coefficients[index + 1]), 8) for index, name in enumerate(factor_names)}})
    drift = {
        name: round(float(np.ptp([point["betas"][name] for point in paths])), 8) if paths else None
        for name in factor_names
    }
    return {
        "method": "return_based_style_risk_v0",
        "factor_exposures": factor_rows,
        "risk": {
            "total_variance": round(total_variance, 10),
            "systematic_variance": round(systematic_variance, 10),
            "idiosyncratic_variance": round(idiosyncratic_variance, 10),
            "idiosyncratic_risk_share": round(idiosyncratic_variance / total_variance, 8) if total_variance > 0 else None,
            "factor_covariance": {name: {other: round(float(covariance[i, j]), 10) for j, other in enumerate(factor_names)} for i, name in enumerate(factor_names)},
        },
        "diagnostics": {"condition_number": round(condition_number, 4)},
        "style_drift": {"window": rolling_window, "causal": True, "beta_range": drift, "paths": paths},
        "warnings": warnings,
    }
