"""Pre-registered rolling out-of-sample gate for research-only factors."""

from __future__ import annotations

from datetime import date

import numpy as np


def evaluate_candidate_factor(
    returns: np.ndarray,
    factor_returns: np.ndarray,
    dates: list[date],
    factor_names: list[str],
    candidate_name: str,
    *,
    train_window: int = 52,
    test_window: int = 13,
    max_segments: int = 4,
) -> dict:
    """Compare a fixed baseline with the same baseline plus one candidate.

    The candidate is admitted only when its incremental predictive fit is
    positive in at least three of four contiguous expanding OOS segments.
    """
    candidate_index = factor_names.index(candidate_name) if candidate_name in factor_names else -1
    if candidate_index < 0 or len(returns) != len(factor_returns) or len(returns) != len(dates):
        return _insufficient(candidate_name, "候选因子未与产品收益共同覆盖")
    available_segments = min(max_segments, max(0, (len(returns) - train_window) // test_window))
    if available_segments < 3:
        return _insufficient(candidate_name, "共同覆盖样本不足，至少需要 3 个连续样本外测试段")

    base_indices = [index for index in range(len(factor_names)) if index != candidate_index]
    segments: list[dict] = []
    for segment in range(available_segments):
        train_end = train_window + segment * test_window
        test_end = train_end + test_window
        train_y, test_y = returns[:train_end], returns[train_end:test_end]
        benchmark = float(np.sum((test_y - np.mean(train_y)) ** 2))
        base_prediction = _predict(factor_returns[:train_end, base_indices], train_y, factor_returns[train_end:test_end, base_indices])
        candidate_prediction = _predict(factor_returns[:train_end], train_y, factor_returns[train_end:test_end])
        base_r2 = _oos_r2(test_y, base_prediction, benchmark)
        candidate_r2 = _oos_r2(test_y, candidate_prediction, benchmark)
        segments.append({
            "segment": segment + 1,
            "train_end_date": dates[train_end - 1].isoformat(),
            "test_start_date": dates[train_end].isoformat(),
            "test_end_date": dates[test_end - 1].isoformat(),
            "baseline_oos_r2": round(base_r2, 6),
            "candidate_oos_r2": round(candidate_r2, 6),
            "incremental_oos_r2": round(candidate_r2 - base_r2, 6),
        })
    deltas = np.asarray([segment["incremental_oos_r2"] for segment in segments], dtype=float)
    candidate_r2 = np.asarray([segment["candidate_oos_r2"] for segment in segments], dtype=float)
    positive_fraction = float(np.mean(deltas >= 0.01))
    admitted = bool(
        positive_fraction >= 0.75
        and float(np.mean(deltas)) >= 0.01
        and float(np.median(deltas)) >= 0.0
        and float(np.mean(candidate_r2)) >= 0.0
    )
    return {
        "candidate_name": candidate_name,
        "status": "admitted" if admitted else "candidate",
        "admitted": admitted,
        "segments": segments,
        "summary": {
            "evaluated_segments": len(segments),
            "mean_incremental_oos_r2": round(float(np.mean(deltas)), 6),
            "median_incremental_oos_r2": round(float(np.median(deltas)), 6),
            "positive_increment_fraction": round(positive_fraction, 6),
            "mean_candidate_oos_r2": round(float(np.mean(candidate_r2)), 6),
        },
        "rule": "至少 3 个连续样本外段；≥75% 段的增量 R² ≥ 0.01，且增量均值 ≥ 0.01、中位数 ≥ 0、候选模型平均样本外 R² ≥ 0。",
        "conclusion": "通过稳定性闸门，可进入正式归因" if admitted else "未通过稳定性闸门，仅作为研究候选，不进入正式归因",
    }


def _predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    coefficients, _, _, _ = np.linalg.lstsq(design, train_y, rcond=None)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def _oos_r2(actual: np.ndarray, predicted: np.ndarray, benchmark: float) -> float:
    return 1.0 - float(np.sum((actual - predicted) ** 2)) / benchmark if benchmark > 1e-15 else -1.0


def _insufficient(candidate_name: str, conclusion: str) -> dict:
    return {
        "candidate_name": candidate_name,
        "status": "insufficient",
        "admitted": False,
        "segments": [],
        "summary": {"evaluated_segments": 0, "mean_incremental_oos_r2": None, "median_incremental_oos_r2": None, "positive_increment_fraction": None, "mean_candidate_oos_r2": None},
        "rule": "至少 3 个连续样本外段；≥75% 段的增量 R² ≥ 0.01，且增量均值 ≥ 0.01、中位数 ≥ 0、候选模型平均样本外 R² ≥ 0。",
        "conclusion": conclusion,
    }
