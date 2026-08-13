"""Phase-D OOS test for nonlinear return-based CTA attribution.

This module deliberately answers a narrow question: does a fixed nonlinear
predictive model add stable *out-of-sample* predictive information after the
registered linear factor baseline?  It does not turn model predictions into a
P&L ledger, holdings reconstruction, or a manager-skill estimate.
"""

from __future__ import annotations

from datetime import date, datetime
from math import comb
from typing import Any

import numpy as np


_MIN_TRAIN_OBSERVATIONS = 20
_MIN_TEST_OBSERVATIONS = 5
_DEFAULT_MIN_POSITIVE_FRACTION = 0.75


def _iso(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value
    return value.isoformat()


def _round(value: float | None, digits: int = 8) -> float | None:
    return round(float(value), digits) if value is not None and np.isfinite(value) else None


def _validate_inputs(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[date],
    factor_names: list[str],
    train_window: int,
    test_window: int,
    max_segments: int,
    min_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(returns, dtype=float)
    x = np.asarray(factor_returns, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or len(y) != len(dates):
        raise ValueError("收益、因子收益与日期长度不一致")
    if x.shape[1] == 0 or len(factor_names) != x.shape[1]:
        raise ValueError("因子名称与因子列数不一致")
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("收益、因子收益必须为有限数")
    if any(_iso(left) >= _iso(right) for left, right in zip(dates, dates[1:], strict=False)):
        raise ValueError("日期必须严格按时间升序排列")
    if train_window < _MIN_TRAIN_OBSERVATIONS:
        raise ValueError(f"训练窗口至少为 {_MIN_TRAIN_OBSERVATIONS} 个观测")
    if test_window < _MIN_TEST_OBSERVATIONS:
        raise ValueError(f"测试窗口至少为 {_MIN_TEST_OBSERVATIONS} 个观测")
    if max_segments < 1 or min_segments < 1 or min_segments > max_segments:
        raise ValueError("测试段数量参数无效")
    return y, x


def _metric_summary(actual: np.ndarray, predicted: np.ndarray, benchmark: float) -> dict[str, float | int | None]:
    """Calculate metrics without fitting or tuning on the test observations."""
    if len(actual) != len(predicted) or len(actual) == 0:
        return {"observations": 0, "oos_r2": None, "correlation": None, "rmse": None, "mean_residual": None}
    residuals = actual - predicted
    squared_error = float(np.sum(np.square(residuals)))
    oos_r2 = 1.0 - squared_error / benchmark if benchmark > 1e-15 else None
    actual_std = float(np.std(actual))
    predicted_std = float(np.std(predicted))
    correlation = float(np.corrcoef(actual, predicted)[0, 1]) if actual_std > 1e-15 and predicted_std > 1e-15 else None
    return {
        "observations": int(len(actual)),
        "oos_r2": _round(oos_r2, 6),
        "correlation": _round(correlation, 6),
        "rmse": _round(float(np.sqrt(np.mean(np.square(residuals)))), 8),
        "mean_residual": _round(float(np.mean(residuals)), 8),
    }


def _fit_linear_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    coefficients, _, _, _ = np.linalg.lstsq(design, train_y, rcond=None)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def _fit_nonlinear_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    # Lazy import keeps the read-only attribution service lightweight and makes
    # the dependency failure explicit at the point where Phase D is requested.
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=int(parameters["max_iter"]),
        learning_rate=float(parameters["learning_rate"]),
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        l2_regularization=float(parameters["l2_regularization"]),
        early_stopping=False,
        random_state=int(parameters["random_state"]),
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x), dtype=float)


def _default_model_parameters(train_window: int) -> dict[str, Any]:
    return {
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 7,
        "min_samples_leaf": max(5, min(12, train_window // 8)),
        "l2_regularization": 1.0,
        "random_state": 20260812,
    }


def _evaluate_segments(
    y: np.ndarray,
    x: np.ndarray,
    dates: list[date],
    *,
    train_window: int,
    test_window: int,
    max_segments: int,
    nonlinear_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Run one pre-registered expanding-window evaluation."""
    available_segments = max(0, (len(y) - train_window) // test_window)
    segment_count = min(max_segments, available_segments)
    segments: list[dict[str, Any]] = []
    failures: list[str] = []

    for segment_index in range(segment_count):
        train_end = train_window + segment_index * test_window
        test_end = train_end + test_window
        train_y = y[:train_end]
        train_x = x[:train_end]
        test_y = y[train_end:test_end]
        test_x = x[train_end:test_end]
        benchmark = float(np.sum(np.square(test_y - np.mean(train_y))))
        try:
            linear_prediction = _fit_linear_predict(train_x, train_y, test_x)
            linear = _metric_summary(test_y, linear_prediction, benchmark)
        except (np.linalg.LinAlgError, ValueError) as error:
            failures.append(f"第 {segment_index + 1} 个测试段线性模型失败：{error}")
            linear = {"observations": len(test_y), "oos_r2": None, "correlation": None, "rmse": None, "mean_residual": None}
            linear_prediction = np.full(len(test_y), np.nan)
        try:
            nonlinear_prediction = _fit_nonlinear_predict(train_x, train_y, test_x, nonlinear_parameters)
            nonlinear = _metric_summary(test_y, nonlinear_prediction, benchmark)
        except (ImportError, ValueError, RuntimeError) as error:
            failures.append(f"第 {segment_index + 1} 个测试段非线性模型失败：{error}")
            nonlinear = {"observations": len(test_y), "oos_r2": None, "correlation": None, "rmse": None, "mean_residual": None}
            nonlinear_prediction = np.full(len(test_y), np.nan)

        linear_r2 = linear["oos_r2"]
        nonlinear_r2 = nonlinear["oos_r2"]
        r2_delta = float(nonlinear_r2 - linear_r2) if nonlinear_r2 is not None and linear_r2 is not None else None
        linear_corr = linear["correlation"]
        nonlinear_corr = nonlinear["correlation"]
        correlation_delta = float(nonlinear_corr - linear_corr) if nonlinear_corr is not None and linear_corr is not None else None
        linear_rmse = linear["rmse"]
        nonlinear_rmse = nonlinear["rmse"]
        rmse_delta = float(nonlinear_rmse - linear_rmse) if nonlinear_rmse is not None and linear_rmse is not None else None
        segments.append({
            "segment": segment_index + 1,
            "train_start_date": _iso(dates[0]),
            "train_end_date": _iso(dates[train_end - 1]),
            "test_start_date": _iso(dates[train_end]),
            "test_end_date": _iso(dates[test_end - 1]),
            "train_observations": train_end,
            "test_observations": len(test_y),
            "linear": linear,
            "nonlinear": nonlinear,
            "r2_delta": _round(r2_delta, 6),
            "correlation_delta": _round(correlation_delta, 6),
            "rmse_delta": _round(rmse_delta, 8),
            "predictions_available": bool(np.isfinite(linear_prediction).all() and np.isfinite(nonlinear_prediction).all()),
        })

    return {"segments": segments, "failures": failures, "available_segments": available_segments}


def _summarize_deltas(
    segments: list[dict[str, Any]],
    *,
    min_segments: int,
    min_r2_uplift: float,
    min_positive_fraction: float,
) -> dict[str, Any]:
    deltas = np.asarray([item["r2_delta"] for item in segments if item["r2_delta"] is not None], dtype=float)
    if deltas.size == 0:
        return {
            "segment_count": len(segments),
            "evaluated_delta_count": 0,
            "mean_r2_delta": None,
            "median_r2_delta": None,
            "positive_delta_fraction": None,
            "stable_improvement": False,
            "conclusion": "未发现可靠非线性增量",
            "rule": {
                "minimum_segments": min_segments,
                "minimum_r2_uplift": min_r2_uplift,
                "minimum_positive_fraction": min_positive_fraction,
            },
        }
    positive_fraction = float(np.mean(deltas >= min_r2_uplift))
    stable = bool(
        len(deltas) >= min_segments
        and positive_fraction >= min_positive_fraction
        and float(np.mean(deltas)) >= min_r2_uplift
        and float(np.median(deltas)) >= min_r2_uplift
    )
    return {
        "segment_count": len(segments),
        "evaluated_delta_count": int(len(deltas)),
        "mean_r2_delta": _round(float(np.mean(deltas)), 6),
        "median_r2_delta": _round(float(np.median(deltas)), 6),
        "positive_delta_fraction": _round(positive_fraction, 6),
        "stable_improvement": stable,
        "conclusion": "存在多个连续测试段的非线性增量支持证据" if stable else "未发现可靠非线性增量",
        "rule": {
            "minimum_segments": min_segments,
            "minimum_r2_uplift": min_r2_uplift,
            "minimum_positive_fraction": min_positive_fraction,
            "requires_expanding_window_and_contiguous_test_segments": True,
        },
    }


def _one_sided_sign_test_p_value(deltas: list[float], threshold: float) -> float | None:
    values = [delta for delta in deltas if np.isfinite(delta)]
    if not values:
        return None
    successes = sum(delta >= threshold for delta in values)
    count = len(values)
    return float(sum(comb(count, index) for index in range(successes, count + 1)) / (2**count))


def _scenario_summary(
    y: np.ndarray,
    x: np.ndarray,
    dates: list[date],
    *,
    train_window: int,
    test_window: int,
    max_segments: int,
    min_segments: int,
    min_r2_uplift: float,
    min_positive_fraction: float,
    nonlinear_parameters: dict[str, Any],
) -> dict[str, Any]:
    evaluation = _evaluate_segments(
        y,
        x,
        dates,
        train_window=train_window,
        test_window=test_window,
        max_segments=max_segments,
        nonlinear_parameters=nonlinear_parameters,
    )
    summary = _summarize_deltas(
        evaluation["segments"],
        min_segments=min_segments,
        min_r2_uplift=min_r2_uplift,
        min_positive_fraction=min_positive_fraction,
    )
    return {
        "train_window": train_window,
        "test_window": test_window,
        "segment_count": len(evaluation["segments"]),
        "status": "available" if len(evaluation["segments"]) >= min_segments else "insufficient",
        "mean_r2_delta": summary["mean_r2_delta"],
        "median_r2_delta": summary["median_r2_delta"],
        "positive_delta_fraction": summary["positive_delta_fraction"],
        "stable_improvement": summary["stable_improvement"],
        "conclusion": summary["conclusion"],
    }


def _build_sensitivity(
    y: np.ndarray,
    x: np.ndarray,
    dates: list[date],
    *,
    train_window: int,
    test_window: int,
    max_segments: int,
    min_segments: int,
    min_r2_uplift: float,
    min_positive_fraction: float,
    nonlinear_parameters: dict[str, Any],
) -> dict[str, Any]:
    train_candidates = sorted({
        train_window,
        max(_MIN_TRAIN_OBSERVATIONS, int(round(train_window * 0.75))),
        max(_MIN_TRAIN_OBSERVATIONS, int(round(train_window * 1.25))),
    })
    test_candidates = sorted({
        test_window,
        max(_MIN_TEST_OBSERVATIONS, int(round(test_window * 0.75))),
        max(_MIN_TEST_OBSERVATIONS, int(round(test_window * 1.25))),
    })
    window_sensitivity = [
        _scenario_summary(
            y,
            x,
            dates,
            train_window=candidate_train,
            test_window=test_window,
            max_segments=max_segments,
            min_segments=min_segments,
            min_r2_uplift=min_r2_uplift,
            min_positive_fraction=min_positive_fraction,
            nonlinear_parameters=nonlinear_parameters,
        )
        for candidate_train in train_candidates
    ]
    window_sensitivity.extend(
        _scenario_summary(
            y,
            x,
            dates,
            train_window=train_window,
            test_window=candidate_test,
            max_segments=max_segments,
            min_segments=min_segments,
            min_r2_uplift=min_r2_uplift,
            min_positive_fraction=min_positive_fraction,
            nonlinear_parameters=nonlinear_parameters,
        )
        for candidate_test in test_candidates
        if candidate_test != test_window
    )

    threshold_candidates = sorted({0.0, min_r2_uplift, max(0.01, min_r2_uplift * 2), 0.05})
    # Threshold sensitivity intentionally reuses the base-case OOS deltas. It
    # changes only the pre-registered decision rule, never model selection.
    base_evaluation = _evaluate_segments(
        y,
        x,
        dates,
        train_window=train_window,
        test_window=test_window,
        max_segments=max_segments,
        nonlinear_parameters=nonlinear_parameters,
    )
    base_deltas = [item["r2_delta"] for item in base_evaluation["segments"] if item["r2_delta"] is not None]
    threshold_sensitivity = []
    for threshold in threshold_candidates:
        summary = _summarize_deltas(
            base_evaluation["segments"],
            min_segments=min_segments,
            min_r2_uplift=threshold,
            min_positive_fraction=min_positive_fraction,
        )
        threshold_sensitivity.append({
            "minimum_r2_uplift": threshold,
            "stable_improvement": summary["stable_improvement"],
            "positive_delta_fraction": summary["positive_delta_fraction"],
            "mean_r2_delta": summary["mean_r2_delta"],
        })

    conservative = {**nonlinear_parameters, "max_leaf_nodes": 3, "min_samples_leaf": max(10, int(nonlinear_parameters["min_samples_leaf"])), "l2_regularization": 2.0}
    flexible = {**nonlinear_parameters, "max_leaf_nodes": 15, "min_samples_leaf": max(3, int(nonlinear_parameters["min_samples_leaf"]) // 2), "l2_regularization": 0.1}
    model_parameter_sensitivity = [
        {"label": "conservative", **_scenario_summary(y, x, dates, train_window=train_window, test_window=test_window, max_segments=max_segments, min_segments=min_segments, min_r2_uplift=min_r2_uplift, min_positive_fraction=min_positive_fraction, nonlinear_parameters=conservative)},
        {"label": "base", **_scenario_summary(y, x, dates, train_window=train_window, test_window=test_window, max_segments=max_segments, min_segments=min_segments, min_r2_uplift=min_r2_uplift, min_positive_fraction=min_positive_fraction, nonlinear_parameters=nonlinear_parameters)},
        {"label": "flexible", **_scenario_summary(y, x, dates, train_window=train_window, test_window=test_window, max_segments=max_segments, min_segments=min_segments, min_r2_uplift=min_r2_uplift, min_positive_fraction=min_positive_fraction, nonlinear_parameters=flexible)},
    ]
    candidate_count = len(window_sensitivity) + len(model_parameter_sensitivity)
    raw_p = _one_sided_sign_test_p_value(base_deltas, min_r2_uplift)
    return {
        "window_sensitivity": window_sensitivity,
        "threshold_sensitivity": threshold_sensitivity,
        "model_parameter_sensitivity": model_parameter_sensitivity,
        "selection_policy": {
            "base_case_is_pre_registered": True,
            "selected_from_sensitivity": False,
            "sensitivity_is_diagnostic_only": True,
        },
        "multiple_testing": {
            "candidate_scenarios_count": candidate_count,
            "correction": "bonferroni_descriptive",
            "raw_one_sided_sign_test_p_value": _round(raw_p, 6),
            "bonferroni_adjusted_p_value": _round(min(1.0, raw_p * max(1, candidate_count)) if raw_p is not None else None, 6),
            "dsr_applicable": False,
            "dsr_note": "DSR 适用于风险调整收益/Sharpe 的多重选择；本层只比较预登记预测误差增量，因此不将 DSR 当作预测增量显著性检验。",
        },
    }


def evaluate_nonlinear_increment(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[date],
    factor_names: list[str],
    *,
    frequency: str,
    train_window: int,
    test_window: int,
    max_segments: int = 5,
    min_segments: int = 3,
    min_r2_uplift: float = 0.01,
    min_positive_fraction: float = _DEFAULT_MIN_POSITIVE_FRACTION,
    nonlinear_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare fixed linear and HistGradientBoosting models on causal OOS segments.

    The test segment is never used to fit, tune, scale, or select a model. The
    base case is fixed by the caller; sensitivity scenarios are reported as
    diagnostics only and cannot change the conclusion.
    """
    y, x = _validate_inputs(returns, factor_returns, dates, factor_names, train_window, test_window, max_segments, min_segments)
    if not 0.0 <= min_positive_fraction <= 1.0:
        raise ValueError("min_positive_fraction 必须在 0 和 1 之间")
    if min_r2_uplift < 0:
        raise ValueError("min_r2_uplift 不能为负数")
    model_parameters = {**_default_model_parameters(train_window), **(nonlinear_parameters or {})}
    evaluation = _evaluate_segments(
        y,
        x,
        dates,
        train_window=train_window,
        test_window=test_window,
        max_segments=max_segments,
        nonlinear_parameters=model_parameters,
    )
    summary = _summarize_deltas(
        evaluation["segments"],
        min_segments=min_segments,
        min_r2_uplift=min_r2_uplift,
        min_positive_fraction=min_positive_fraction,
    )
    status = "available" if len(evaluation["segments"]) >= min_segments else "insufficient"
    warnings = [
        "Phase D 只比较固定因子集合上的样本外预测增量，不是持仓、真实 P&L 或管理人技能归因。",
        "非线性模型的特征重要性不计入经济收益贡献。",
    ]
    if status == "insufficient":
        warnings.append(f"样本不足：需要至少 {min_segments} 个连续测试段（训练 {train_window} + 每段测试 {test_window} 个观测）。")
    warnings.extend(evaluation["failures"])
    sensitivity = _build_sensitivity(
        y,
        x,
        dates,
        train_window=train_window,
        test_window=test_window,
        max_segments=max_segments,
        min_segments=min_segments,
        min_r2_uplift=min_r2_uplift,
        min_positive_fraction=min_positive_fraction,
        nonlinear_parameters=model_parameters,
    ) if status == "available" else {
        "window_sensitivity": [],
        "threshold_sensitivity": [],
        "model_parameter_sensitivity": [],
        "selection_policy": {"base_case_is_pre_registered": True, "selected_from_sensitivity": False, "sensitivity_is_diagnostic_only": True},
        "multiple_testing": {"candidate_scenarios_count": 0, "correction": "bonferroni_descriptive", "raw_one_sided_sign_test_p_value": None, "bonferroni_adjusted_p_value": None, "dsr_applicable": False, "dsr_note": "样本不足，未进行多重场景比较。"},
    }
    return {
        "method": "causal_oos_linear_vs_hist_gradient_boosting",
        "frequency": frequency,
        "factor_names": list(factor_names),
        "status": status,
        "parameters": {
            "train_window": train_window,
            "test_window": test_window,
            "max_segments": max_segments,
            "min_segments": min_segments,
            "min_r2_uplift": min_r2_uplift,
            "min_positive_fraction": min_positive_fraction,
            "training_scheme": "expanding_window",
            "test_scheme": "contiguous_non_overlapping_segments",
            "uses_future_data": False,
            "linear_model": "OLS_with_intercept",
            "nonlinear_model": model_parameters,
        },
        "segments": evaluation["segments"],
        "summary": summary,
        "sensitivity": sensitivity,
        "warnings": warnings,
    }
