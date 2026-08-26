"""收益分析与风险暴露：CTA 产品评分主线的两个新评判标准。

质量分 / 置信分 / 归因质量分描述的是「证据可信度」；本模块补齐评判标准
缺失的另一半——产品自身的收益解剖与风险暴露：

- ``analyze_returns``          收益分析：绝对收益、分年收益、回撤路径。
- ``analyze_risk_exposure``    风险暴露：波动 / VaR / CVaR / 下行波动等统计
  风险，叠加因子回归给出的因子风险贡献与集中度。
- ``score_return_analysis``    收益分（绝对尺度 0-100，跨时点可比）。
- ``score_risk_exposure``      风险暴露分（绝对尺度 0-100，跨时点可比）。

质量分是横截面相对分（产品之间的百分位），收益分与风险暴露分是绝对分
（只看产品自身），两者互补：相对分回答「在同批产品里排第几」，绝对分
回答「这个收益 / 风险水平本身好不好」。

所有阈值都是模块顶部常量，便于单独调参。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from app.services.codex_cta_ranking import MIN_FORMAL_RETURNS

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}

RETURN_ANALYSIS_METHOD = "cta-return-analysis-v1"
RISK_EXPOSURE_METHOD = "cta-risk-exposure-v1"
RETURN_SCORE_METHOD = "cta-return-analysis-score-v2"
RISK_SCORE_METHOD = "cta-risk-exposure-score-v2"

# ---------------------------------------------------------------------------
# 绝对尺度校准常量（以周频 CTA 为基准）
# ---------------------------------------------------------------------------
# 阈值用真实宇宙（750 只可打分周频产品，2026-08 校准）的经验分布定标：
# 年化收益 12% 得 50 分（logistic 中点 = 真实中位数 11.7%），
# 24% 约 88 分，-12% 约 12 分。原 v1 用 8%，66% 的产品超中点，过于慷慨。
_ANNUALIZED_RETURN_BASIS = 0.12
# |最大回撤| 达 30% 得 0 分，无回撤得 100 分（线性衰减）。
# 真实宇宙约 20.7% 的产品 |回撤| 超过 30%，惩罚尾部有区分度，保持不变。
_MAX_DRAWDOWN_BASIS = 0.30
# 年化波动达 35% 得 0 分，无波动得 100 分（线性衰减）。
# 35% = 真实宇宙波动 p95，惩罚最差 5%；原 40% 只有 3.5% 触零，形同虚设。
_ANNUALIZED_VOLATILITY_BASIS = 0.35
# 周频每期 ES95 损失达 7% 得 0 分（其他频率按 sqrt(期数/52) 折算）。
# 7% 使约 16% 的产品触零；原 6% 使 22% 触零，过于严苛。
_CVAR_PERIODIC_BASIS_WEEKLY = 0.07

# 风险暴露统计量的最短样本；历史 CVaR 所需的最短左尾样本数。
_MIN_RISK_SAMPLE = 12
_MIN_TAIL_SAMPLE = 5

_GAIN_LOSS_CAP = 3.0  # 盈亏不对称比的上限，与排名模型保持一致


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(min(max(value, lower), upper))


def _logistic(value: float) -> float:
    """数值稳定的 logistic 函数，用于把无界的收益率映射到 0-1。"""
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exp_value = float(np.exp(value))
    return exp_value / (1.0 + exp_value)


def _band(score: float | None) -> str | None:
    if score is None:
        return None
    return "high" if score >= 75 else "medium" if score >= 50 else "low"


def _weighted_components(
    components: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float | None, dict[str, float], list[str]]:
    """按可用分量重归一化权重并加权，返回 (得分, 有效权重, 缺失提示)。"""
    available = {
        name: float(weights[name])
        for name, value in components.items()
        if value is not None and np.isfinite(value) and name in weights
    }
    if not available:
        return None, {}, [
            f"分量缺失：{name} 未纳入。"
            for name, value in components.items()
            if value is None
        ]
    total = sum(available.values())
    effective = {name: weight / total for name, weight in available.items()}
    score = float(sum(float(components[name]) * effective[name] for name in effective))
    missing = [name for name, value in components.items() if value is None]
    warnings = [f"分量缺失：{name} 未纳入，权重已重归一化。" for name in missing]
    return score, effective, warnings


# ---------------------------------------------------------------------------
# 收益分析
# ---------------------------------------------------------------------------

def analyze_returns(
    nav_values: list[float] | np.ndarray,
    dates: list[date],
    frequency: str,
) -> dict[str, Any]:
    """从已审核净值计算绝对收益解剖与完整回撤路径。

    收益用复合计算，不假设对数正态；回撤路径逐点输出，前端可画曲线。
    样本不足 26 期时仍然给出分析结果，但附短样本警告，评分函数会拒绝
    打分（与质量分的正式门槛一致）。
    """
    values = np.asarray(nav_values, dtype=float)
    if values.size < 2:
        return {
            "status": "insufficient",
            "method": RETURN_ANALYSIS_METHOD,
            "observation_count": int(values.size),
            "metrics": {},
            "calendar_year_returns": {},
            "drawdown_path": [],
            "warnings": ["净值点少于 2 个，无法生成收益分析。"],
        }
    returns = values[1:] / values[:-1] - 1.0
    n = int(returns.size)
    ppy = PERIODS_PER_YEAR.get(frequency, 52)
    warnings: list[str] = []
    if n < MIN_FORMAL_RETURNS:
        warnings.append(f"短样本：仅 {n} 个收益周期，收益分析仅供参考。")

    compound = float(np.prod(1.0 + returns))
    cumulative_return = float(compound - 1.0)
    annualized_return = float(compound ** (ppy / n) - 1.0) if compound > 0 else -1.0
    annualized_volatility = float(np.std(returns, ddof=1) * np.sqrt(ppy)) if n > 1 else None

    positive = returns[returns > 0.0]
    negative = returns[returns < 0.0]
    positive_period_ratio = float(np.mean(returns > 0.0))
    if negative.size == 0:
        gain_loss_asymmetry = float(_GAIN_LOSS_CAP)
    elif positive.size == 0:
        gain_loss_asymmetry = 0.0
    else:
        gain_loss_asymmetry = float(min(
            np.median(positive) / max(abs(float(np.median(negative))), 1e-12),
            _GAIN_LOSS_CAP,
        ))

    def _rolling(periods: int) -> float | None:
        if n < periods:
            return None
        return float(np.prod(1.0 + returns[-periods:]) - 1.0)

    # 分年收益：收益期归属其结束日所在年份，首尾年份天然是部分年份。
    calendar_year_returns: dict[str, float] = {}
    for value, end_date in zip(returns, dates[1:], strict=True):
        year = str(end_date.year)
        calendar_year_returns[year] = calendar_year_returns.get(year, 1.0) * (1.0 + float(value))
    calendar_year_returns = {
        year: round(float(accumulator - 1.0), 8)
        for year, accumulator in sorted(calendar_year_returns.items())
    }

    # 回撤：基于归一化净值的水位线路径。
    wealth = values / values[0]
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    maximum_drawdown = float(np.min(drawdowns))
    current_drawdown = float(drawdowns[-1])
    active = drawdowns < -1e-12
    episodes: list[int] = []
    start: int | None = None
    for index, in_drawdown in enumerate(active):
        if in_drawdown and start is None:
            start = index
        elif not in_drawdown and start is not None:
            episodes.append(index - start)
            start = None
    recovery_completed = start is None
    if start is not None:
        episodes.append(len(active) - start)
    longest_drawdown_duration_periods = max(episodes, default=0)
    completed = episodes[:-1] if not recovery_completed and episodes else episodes
    average_recovery_periods = (
        float(np.mean(completed)) if completed else None
    )
    drawdown_path = [
        {"date": item_date.isoformat(), "drawdown": round(float(value), 8)}
        for item_date, value in zip(dates, drawdowns, strict=True)
    ]

    return {
        "status": "available",
        "method": RETURN_ANALYSIS_METHOD,
        "observation_count": n,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "metrics": {
            "cumulative_return": round(cumulative_return, 8),
            "annualized_return": round(annualized_return, 8),
            "annualized_volatility": (
                round(annualized_volatility, 8) if annualized_volatility is not None else None
            ),
            "positive_period_ratio": round(positive_period_ratio, 8),
            "gain_loss_asymmetry": round(gain_loss_asymmetry, 8),
            "best_period_return": round(float(np.max(returns)), 8),
            "worst_period_return": round(float(np.min(returns)), 8),
            "rolling_13_period_return": round(value, 8) if (value := _rolling(13)) is not None else None,
            "rolling_26_period_return": round(value, 8) if (value := _rolling(26)) is not None else None,
            "rolling_52_period_return": round(value, 8) if (value := _rolling(52)) is not None else None,
            "maximum_drawdown": round(maximum_drawdown, 8),
            "current_drawdown": round(current_drawdown, 8),
            "longest_drawdown_duration_periods": longest_drawdown_duration_periods,
            "average_recovery_periods": (
                round(average_recovery_periods, 8) if average_recovery_periods is not None else None
            ),
            "drawdown_episode_count": len(episodes),
            "recovery_completed": recovery_completed,
        },
        "calendar_year_returns": calendar_year_returns,
        "drawdown_path": drawdown_path,
        "warnings": warnings,
    }


def score_return_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """收益分：收益水平、正收益占比、回撤控制的绝对尺度加权分。

    收益水平用 logistic 以 12% 年化为中点（真实宇宙中位数定标）；
    正收益占比与回撤控制用线性映射。任何分量缺失时剩余分量自动重归一化权重。
    """
    if analysis.get("status") != "available":
        return {
            "score": None,
            "band": None,
            "status": "insufficient",
            "components": {},
            "weights": {},
            "method": RETURN_SCORE_METHOD,
            "warnings": ["收益分析不可用，无法生成收益分。"],
        }
    metrics: dict[str, Any] = analysis["metrics"]
    if analysis.get("observation_count", 0) < MIN_FORMAL_RETURNS:
        return {
            "score": None,
            "band": None,
            "status": "short_sample",
            "components": {},
            "weights": {},
            "method": RETURN_SCORE_METHOD,
            "warnings": [
                f"收益周期仅 {analysis.get('observation_count', 0)} 个，"
                f"不足 {MIN_FORMAL_RETURNS} 期，不生成收益分。"
            ],
        }
    annualized_return = metrics.get("annualized_return")
    maximum_drawdown = metrics.get("maximum_drawdown")
    components: dict[str, float | None] = {
        "return_level": (
            100.0 * _logistic(annualized_return / _ANNUALIZED_RETURN_BASIS)
            if annualized_return is not None else None
        ),
        "consistency": (
            100.0 * float(metrics["positive_period_ratio"])
            if metrics.get("positive_period_ratio") is not None else None
        ),
        "drawdown_control": (
            100.0 * (1.0 - min(abs(float(maximum_drawdown)) / _MAX_DRAWDOWN_BASIS, 1.0))
            if maximum_drawdown is not None else None
        ),
    }
    weights = {
        "return_level": 0.40,
        "consistency": 0.25,
        "drawdown_control": 0.35,
    }
    score, effective_weights, warnings = _weighted_components(components, weights)
    return {
        "score": round(score, 4) if score is not None else None,
        "band": _band(score),
        "status": "available",
        "components": {
            name: round(value, 4) if value is not None else None
            for name, value in components.items()
        },
        "weights": {name: round(weight, 4) for name, weight in effective_weights.items()},
        "method": RETURN_SCORE_METHOD,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 风险暴露
# ---------------------------------------------------------------------------

def analyze_risk_exposure(
    returns: np.ndarray,
    regression: Any,
    frequency: str,
) -> dict[str, Any]:
    """统计风险 + 因子暴露的风险暴露画像。

    统计部分只看收益序列本身；因子暴露部分复用产品评分报告里的同一
    次因子回归（``factor_risk_contributions`` 在此前从未被评分主线消费），
    因子库未构建时优雅降级为 ``statistical_only`` 状态。
    """
    values = np.asarray(returns, dtype=float)
    n = int(values.size)
    warnings: list[str] = []
    if n < _MIN_RISK_SAMPLE:
        return {
            "status": "insufficient",
            "factor_exposure_status": "unavailable",
            "method": RISK_EXPOSURE_METHOD,
            "statistical": {},
            "factor_exposures": [],
            "factor_group_risk_contributions": {},
            "concentration": {"hhi": None, "normalized_hhi": None},
            "systematic_variance_share": None,
            "unexplained_variance_share": None,
            "residual_annual_volatility": None,
            "warnings": [f"风险暴露至少需要 {_MIN_RISK_SAMPLE} 个收益周期，当前仅 {n} 个。"],
        }
    ppy = PERIODS_PER_YEAR.get(frequency, 52)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    annualized_volatility = float(std * np.sqrt(ppy)) if std > 0 else None
    downside = np.minimum(values, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(ppy))
    var_95 = float(mean + 1.6449 * std)
    var_99 = float(mean + 2.3263 * std)
    tail_cutoff = float(np.quantile(values, 0.05))
    tail = values[values <= tail_cutoff]
    cvar_95 = float(np.mean(tail)) if tail.size >= _MIN_TAIL_SAMPLE else None
    skewness = (
        float((n / ((n - 1) * (n - 2))) * np.sum(((values - mean) / std) ** 3))
        if n > 2 and std > 0 else None
    )
    excess_kurtosis = (
        float(
            (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)))
            * np.sum(((values - mean) / std) ** 4)
            - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )
        if n > 3 and std > 0 else None
    )
    nav = np.cumprod(1.0 + values)
    maximum_drawdown = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    statistical = {
        "annualized_volatility": round(annualized_volatility, 8) if annualized_volatility is not None else None,
        "downside_deviation": round(downside_deviation, 8),
        "var_95": round(var_95, 8),
        "var_99": round(var_99, 8),
        "cvar_95": round(cvar_95, 8) if cvar_95 is not None else None,
        "skewness": round(skewness, 8) if skewness is not None else None,
        "excess_kurtosis": round(excess_kurtosis, 8) if excess_kurtosis is not None else None,
        "worst_period_return": round(float(np.min(values)), 8),
        "maximum_drawdown": round(maximum_drawdown, 8),
    }

    has_regression = regression is not None and getattr(regression, "n_observations", 0) > 0
    factor_exposures: list[dict[str, Any]] = []
    factor_group_risk_contributions: dict[str, float] = {}
    concentration: dict[str, float | None] = {"hhi": None, "normalized_hhi": None}
    systematic_variance_share: float | None = None
    unexplained_variance_share: float | None = None
    residual_annual_volatility: float | None = None
    if has_regression:
        risk_contributions = regression.factor_risk_contributions or {}
        finite_contributions = {
            name: float(value)
            for name, value in risk_contributions.items()
            if value is not None and np.isfinite(value)
        }
        total_abs = sum(abs(value) for value in finite_contributions.values())
        for factor in regression.factors:
            raw = finite_contributions.get(factor.name)
            share = (
                abs(raw) / total_abs if raw is not None and total_abs > 1e-12 else None
            )
            factor_exposures.append({
                "name": factor.name,
                "display_name": factor.display_name,
                "beta": round(float(factor.beta), 8),
                "t_stat": round(float(factor.t_stat), 8),
                "significant": bool(factor.significant),
                "risk_contribution": round(raw, 8) if raw is not None else None,
                "risk_contribution_pct": round(100.0 * share, 8) if share is not None else None,
            })
            if raw is not None:
                group = factor.factor_group or "other"
                factor_group_risk_contributions[group] = (
                    factor_group_risk_contributions.get(group, 0.0) + raw
                )
        factor_group_risk_contributions = {
            name: round(value, 8)
            for name, value in sorted(factor_group_risk_contributions.items())
        }
        if len(finite_contributions) >= 2 and total_abs > 1e-12:
            shares = np.asarray(
                [abs(value) / total_abs for value in finite_contributions.values()],
                dtype=float,
            )
            hhi = float(np.sum(shares ** 2))
            normalized_hhi = float((hhi - 1.0 / len(shares)) / (1.0 - 1.0 / len(shares)))
            concentration = {
                "hhi": round(hhi, 8),
                "normalized_hhi": round(_clip(normalized_hhi, 0.0, 1.0), 8),
            }
        r_squared = getattr(regression, "r_squared", None)
        if r_squared is not None and np.isfinite(r_squared):
            systematic_variance_share = round(float(r_squared), 8)
            unexplained_variance_share = round(float(max(1.0 - r_squared, 0.0)), 8)
        residual_annual_volatility = getattr(regression, "residual_annual_vol", None)
        if residual_annual_volatility is not None and np.isfinite(residual_annual_volatility):
            residual_annual_volatility = round(float(residual_annual_volatility), 8)
    else:
        warnings.append("CTA 因子回归不可用，因子暴露与风险集中度不纳入风险暴露分析。")

    return {
        "status": "available",
        "factor_exposure_status": "available" if has_regression else "unavailable",
        "method": RISK_EXPOSURE_METHOD,
        "statistical": statistical,
        "factor_exposures": factor_exposures,
        "factor_group_risk_contributions": factor_group_risk_contributions,
        "concentration": concentration,
        "systematic_variance_share": systematic_variance_share,
        "unexplained_variance_share": unexplained_variance_share,
        "residual_annual_volatility": residual_annual_volatility,
        "warnings": warnings,
    }


def score_risk_exposure(
    analysis: dict[str, Any],
    *,
    frequency: str,
) -> dict[str, Any]:
    """风险暴露分：波动控制、尾部控制、集中度控制的绝对尺度加权分。

    波动与 CVaR 用线性惩罚映射（基准见模块顶部常量，CVaR 基准按频率
    折算）；集中度用归一化 HHI（0=完全分散，1=完全集中）。因子回归
    缺失时集中度分量缺席，其余分量重归一化。
    """
    if analysis.get("status") != "available":
        return {
            "score": None,
            "band": None,
            "status": "insufficient",
            "components": {},
            "weights": {},
            "method": RISK_SCORE_METHOD,
            "warnings": ["风险暴露不可用，无法生成风险暴露分。"],
        }
    statistical: dict[str, Any] = analysis.get("statistical", {})
    annualized_volatility = statistical.get("annualized_volatility")
    cvar_95 = statistical.get("cvar_95")
    normalized_hhi = analysis.get("concentration", {}).get("normalized_hhi")
    ppy = PERIODS_PER_YEAR.get(frequency, 52)
    cvar_basis = _CVAR_PERIODIC_BASIS_WEEKLY * float(np.sqrt(52 / ppy))
    components: dict[str, float | None] = {
        "volatility_control": (
            100.0 * (1.0 - min(float(annualized_volatility) / _ANNUALIZED_VOLATILITY_BASIS, 1.0))
            if annualized_volatility is not None else None
        ),
        "tail_control": (
            100.0 * (1.0 - min(abs(float(cvar_95)) / cvar_basis, 1.0))
            if cvar_95 is not None else None
        ),
        "concentration_control": (
            100.0 * (1.0 - float(normalized_hhi))
            if normalized_hhi is not None else None
        ),
    }
    weights = {
        "volatility_control": 0.40,
        "tail_control": 0.30,
        "concentration_control": 0.30,
    }
    score, effective_weights, warnings = _weighted_components(components, weights)
    return {
        "score": round(score, 4) if score is not None else None,
        "band": _band(score),
        "status": "available",
        "components": {
            name: round(value, 4) if value is not None else None
            for name, value in components.items()
        },
        "weights": {name: round(weight, 4) for name, weight in effective_weights.items()},
        "method": RISK_SCORE_METHOD,
        "warnings": warnings,
    }
