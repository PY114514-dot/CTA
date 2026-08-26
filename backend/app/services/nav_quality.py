"""Quality gates for NAV series reconstructed from documents.

Image tracing is evidence, not ground truth.  When a source also discloses
headline performance statistics, compare them before allowing the series into
research.  This module intentionally returns explanations rather than silently
"fixing" a curve: only a reviewer may replace source data.
"""

from __future__ import annotations

import math
from datetime import date
from statistics import median
from typing import Any, Iterable


def _finite_positive_navs(points: Iterable[Any]) -> list[float]:
    """Extract finite, strictly positive NAV values, ignoring bad cells."""
    values: list[float] = []
    for item in points:
        try:
            nav = float(item.nav)
        except (TypeError, ValueError):
            continue
        if math.isfinite(nav) and nav > 0:
            values.append(nav)
    return values


def assess_nav_quality(observations: Iterable[Any], facts: Iterable[Any]) -> dict[str, Any]:
    points = sorted(observations, key=lambda item: item.observation_date)
    values = _finite_positive_navs(points)
    disclosed: dict[str, float] = {}
    for fact in facts:
        if getattr(fact, "field_name", "") not in {"disclosed_cumulative_return", "disclosed_annualized_return", "disclosed_maximum_drawdown", "disclosed_sharpe_ratio"}:
            continue
        try:
            disclosed[fact.field_name] = float(fact.field_value)
        except (TypeError, ValueError):
            continue

    result: dict[str, Any] = {
        "status": "passed",
        "blocking": False,
        "reasons": [],
        "computed_cumulative_return": None,
        "computed_maximum_drawdown": None,
        "largest_period_change": None,
        "disclosed_cumulative_return": disclosed.get("disclosed_cumulative_return"),
        "disclosed_annualized_return": disclosed.get("disclosed_annualized_return"),
        "disclosed_maximum_drawdown": disclosed.get("disclosed_maximum_drawdown"),
        "disclosed_sharpe_ratio": disclosed.get("disclosed_sharpe_ratio"),
    }
    if len(values) < 2:
        return result

    cumulative = values[-1] / values[0] - 1
    running_peak = values[0]
    maximum_drawdown = 0.0
    period_changes = []
    for previous, current in zip(values, values[1:]):
        running_peak = max(running_peak, current)
        maximum_drawdown = min(maximum_drawdown, current / running_peak - 1)
        period_changes.append(current / previous - 1)
    result.update({
        "computed_cumulative_return": round(cumulative, 6),
        "computed_maximum_drawdown": round(maximum_drawdown, 6),
        "largest_period_change": round(max(period_changes, key=abs), 6),
    })

    reasons: list[str] = []
    disclosed_cumulative = result["disclosed_cumulative_return"]
    if disclosed_cumulative is not None and abs(cumulative - disclosed_cumulative) > 0.03:
        reasons.append("曲线累计收益与材料披露值相差超过 3 个百分点")
    disclosed_drawdown = result["disclosed_maximum_drawdown"]
    if disclosed_drawdown is not None and abs(maximum_drawdown - disclosed_drawdown) > 0.05:
        reasons.append("曲线最大回撤与材料披露值相差超过 5 个百分点")
    # A large move in a manually reviewed monthly series can be a real CTA
    # return. The 8% check is for unreviewed or non-monthly image traces.
    reviewed_monthly = points and all(
        getattr(point, "frequency", None) == "monthly"
        and getattr(point, "review_status", None) == "reviewed"
        for point in points
    )
    if abs(result["largest_period_change"]) > 0.08 and not reviewed_monthly:
        reasons.append("存在超过 8% 的单期净值跳变，请在原图上核对该点")

    if reasons:
        result["status"] = "blocked"
        result["blocking"] = True
        result["reasons"] = reasons
    return result


def assess_machine_nav_review(
    observations: Iterable[Any],
    facts: Iterable[Any],
    *,
    source_confidence: float | None,
    declared_frequency: str | None = None,
) -> dict[str, Any]:
    """Decide whether image-derived NAV may enter preliminary research.

    This is intentionally a conservative machine review, not a claim that the
    source facts are true.  It combines CV/VLM extraction confidence with
    consistency checks that can be audited later.
    """
    points = sorted(observations, key=lambda item: item.observation_date)
    quality = assess_nav_quality(points, facts)
    reasons = list(quality["reasons"])
    if len(points) < 8:
        reasons.append("净值点不足 8 期，无法完成机器稳定性检查")
    if source_confidence is None:
        reasons.append("缺少 CV/VLM 提取置信度")
    elif source_confidence < 0.85:
        reasons.append(f"CV/VLM 提取置信度仅 {source_confidence:.0%}，低于机器复核阈值 85%")

    values = _finite_positive_navs(points)
    if len(values) != len(points):
        reasons.append(f"{len(points) - len(values)} 个净值点为非数值或非正值，已从计算中剔除")
    # A perfectly flat or nearly flat image-derived curve is usually a failed
    # colour trace (for example, a grid line or chart border), not evidence
    # that a fund had exactly zero movement for dozens of observations.
    # Do not infer a volatility threshold from a genuine short history: the
    # check only applies after the minimum eight-point review gate above.
    if len(values) >= 8:
        value_range = max(values) / min(values) - 1
        if value_range < 0.0001:
            reasons.append("提取曲线近似恒定，疑似网格线或边框，不作为有效净值序列放行")

    dates = [item.observation_date for item in points]
    if len(dates) >= 4:
        intervals = [max((right - left).days, 0) for left, right in zip(dates, dates[1:])]
        nonzero = [interval for interval in intervals if interval > 0]
        if len(nonzero) >= 3:
            typical = median(nonzero)
            irregular = sum(interval > max(typical * 2.5, typical + 7) for interval in nonzero)
            if irregular > max(1, len(nonzero) // 4):
                reasons.append("日期间隔不稳定，可能混入缺失点或频率识别错误")
            if declared_frequency == "weekly" and not 5 <= typical <= 10:
                reasons.append("声明为周频，但日期间隔与周频不一致")
            if declared_frequency == "monthly" and not 20 <= typical <= 40:
                reasons.append("声明为月频，但日期间隔与月频不一致")

    passed = not reasons and not quality["blocking"]
    return {
        "status": "machine_reviewed" if passed else "human_review_required",
        "machine_reviewed": passed,
        "confidence": round(source_confidence, 3) if source_confidence is not None else None,
        "reasons": reasons,
        "quality": quality,
        "scope": "可进入初步研究" if passed else "需人工复核",
        "method": "CV/VLM 提取置信度 + 日期频率 + 净值异常 + 披露一致性检查",
        "reviewed_at": date.today().isoformat(),
    }
