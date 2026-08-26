"""数字化抽样误差审计：把 OCR 候选净值与人工核对基准逐日对齐并打分。

批次 13（精度）：图表数字化（像素追踪/OCR）产出的是候选值，不是真值。
本模块给出一条可复核的审计通道——按日期内连接两段序列，量化逐点误差、
覆盖率与容差通过率，并给出 pass/review/fail 三档判定。判定规则预注册、
可解释，绝不改写任何净值观测。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np

from app.schemas import NetAssetValuePoint

METHOD = "digitization-sampling-error-audit-v1"

# 判定阈值（预注册，2026-08 定标）：
# - 均值相对误差 ≤ 0.2% 且最大相对误差 ≤ 1% 且覆盖率 ≥ 80% → pass
# - 均值相对误差 ≤ 1% 且最大相对误差 ≤ 5% → review（需人工复核后可用）
# - 其余 → fail（数字化结果不可直接采用）
_PASS_MEAN_RELATIVE = 0.002
_PASS_MAX_RELATIVE = 0.01
_PASS_MIN_COVERAGE = 0.8
_REVIEW_MEAN_RELATIVE = 0.01
_REVIEW_MAX_RELATIVE = 0.05


def audit_digitized_nav(
    digitized_points: list[NetAssetValuePoint],
    reference_points: list[NetAssetValuePoint],
    *,
    relative_tolerance: float = 0.01,
    min_matched: int = 5,
) -> dict[str, Any]:
    """Return a sampling-error report with a pass/review/fail verdict.

    对齐规则：按 observation_date 内连接；同一段序列内重复日期不合法，
    由 Pydantic 的严格递增校验挡在入口。基准序列视为真值，数字化序列
    视为候选，误差均以基准为分母。
    """
    reference_by_date = {point.observation_date: point.net_asset_value for point in reference_points}
    digitized_by_date = {point.observation_date: point.net_asset_value for point in digitized_points}
    matched_dates = sorted(set(reference_by_date) & set(digitized_by_date))
    unmatched_digitized = len(digitized_by_date) - len(matched_dates)
    uncovered_reference = len(reference_by_date) - len(matched_dates)

    pairs: list[tuple[date, float, float, float, float]] = []
    for day in matched_dates:
        reference = float(reference_by_date[day])
        digitized = float(digitized_by_date[day])
        abs_error = abs(digitized - reference)
        relative_error = abs_error / reference if reference > 0 else float("inf")
        pairs.append((day, digitized, reference, abs_error, relative_error))

    verdict_rule = {
        "matching": "observation_date 内连接",
        "error_basis": "以基准序列为分母的相对误差",
        "minimum_matched_pairs": min_matched,
        "pass": {
            "mean_relative_error_max": _PASS_MEAN_RELATIVE,
            "max_relative_error_max": _PASS_MAX_RELATIVE,
            "coverage_min": _PASS_MIN_COVERAGE,
        },
        "review": {
            "mean_relative_error_max": _REVIEW_MEAN_RELATIVE,
            "max_relative_error_max": _REVIEW_MAX_RELATIVE,
        },
        "fail": "低于 review 门槛或匹配样本不足",
    }
    warnings: list[str] = []
    if len(digitized_by_date) != len(digitized_points):
        warnings.append("数字化序列含重复日期，已按日期去重后对齐。")
    if uncovered_reference:
        warnings.append(f"基准序列有 {uncovered_reference} 个日期无数字化点，覆盖率按 {len(matched_dates)}/{len(reference_by_date)} 计算。")

    if len(pairs) < min_matched:
        return {
            "method": METHOD,
            "matched_count": len(pairs),
            "unmatched_digitized_count": unmatched_digitized,
            "uncovered_reference_count": uncovered_reference,
            "sample_coverage_ratio": None,
            "errors": {
                "mae": None, "rmse": None, "max_abs_error": None,
                "mean_relative_error": None, "median_relative_error": None,
                "max_relative_error": None,
                "within_tolerance_count": 0, "within_tolerance_ratio": None,
            },
            "worst_points": [],
            "verdict": "fail",
            "verdict_rule": verdict_rule,
            "warnings": warnings + [f"匹配样本 {len(pairs)} 条，低于最低要求 {min_matched} 条，无法给出可复核的误差判定。"],
        }

    abs_errors = np.asarray([item[3] for item in pairs], dtype=float)
    relative_errors = np.asarray([item[4] for item in pairs], dtype=float)
    coverage = len(pairs) / len(reference_by_date)
    within = int(np.sum(relative_errors <= relative_tolerance))

    mean_relative = float(np.mean(relative_errors))
    max_relative = float(np.max(relative_errors))
    if mean_relative <= _PASS_MEAN_RELATIVE and max_relative <= _PASS_MAX_RELATIVE and coverage >= _PASS_MIN_COVERAGE:
        verdict = "pass"
    elif mean_relative <= _REVIEW_MEAN_RELATIVE and max_relative <= _REVIEW_MAX_RELATIVE:
        verdict = "review"
    else:
        verdict = "fail"

    order = np.argsort(relative_errors)[::-1][:3]
    worst_points = [
        {
            "observation_date": pairs[index][0].isoformat(),
            "digitized": pairs[index][1],
            "reference": pairs[index][2],
            "abs_error": pairs[index][3],
            "relative_error": pairs[index][4],
        }
        for index in order
    ]

    return {
        "method": METHOD,
        "matched_count": len(pairs),
        "unmatched_digitized_count": unmatched_digitized,
        "uncovered_reference_count": uncovered_reference,
        "sample_coverage_ratio": coverage,
        "errors": {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(math.sqrt(float(np.mean(abs_errors**2)))),
            "max_abs_error": float(np.max(abs_errors)),
            "mean_relative_error": mean_relative,
            "median_relative_error": float(np.median(relative_errors)),
            "max_relative_error": max_relative,
            "within_tolerance_count": within,
            "within_tolerance_ratio": within / len(pairs),
        },
        "worst_points": worst_points,
        "verdict": verdict,
        "verdict_rule": verdict_rule,
        "warnings": warnings,
    }
