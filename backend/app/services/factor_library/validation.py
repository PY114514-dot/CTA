"""P4: Validation loop, alpha measurement, and profile drift monitoring.

This module closes the loop between L1 (statistical factor regression) and
L4 (weekly report ground truth).  It answers:
1. How well does our factor library explain the product? (R² gap)
2. How much alpha does the manager generate? (residual analysis)
3. Is the product's style drifting over time? (multi-period comparison)
"""

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L1 vs L4 Validation
# ---------------------------------------------------------------------------

@dataclass
class FactorComparison:
    """Compare one factor's L1 implied period contribution to L4 disclosure."""

    factor_name: str
    display_name: str
    l1_beta: float           # from regression
    l1_t_stat: float
    l1_factor_period_return: float | None
    l1_predicted_contribution: float | None
    l4_contribution: float   # from weekly report (period return contribution)
    direction_match: bool | None
    magnitude_ratio: float | None


@dataclass
class ValidationResult:
    """Result of comparing L1 regression against L4 ground truth."""

    product_name: str
    report_date: str
    # Overall fit
    r_squared: float
    r_gap_interpretation: str  # human-readable
    # Per-factor comparison
    factor_comparisons: list[FactorComparison] = field(default_factory=list)
    # Direction agreement rate
    direction_agreement_pct: float = 0.0
    # Summary
    summary: str = ""
    warnings: list[str] = field(default_factory=list)


def validate_against_report(
    regression_result: dict,
    report_snapshot: dict,
    factor_period_returns: dict[str, float] | None = None,
) -> ValidationResult:
    """Compare L1 factor regression against L4 weekly report disclosures.

    Parameters
    ----------
    regression_result : dict from /api/factor-library/attribute
    report_snapshot : dict from /api/report/parse-weekly (ProfileSnapshot.to_dict())

    Returns
    -------
    ValidationResult with per-factor comparison and overall assessment.
    """
    product_name = report_snapshot.get("product_name", "unknown")
    report_date = report_snapshot.get("report_date", "")
    r_squared = regression_result.get("r_squared", 0.0)

    # Map public library keys to L4 disclosure fields.  A beta by itself is
    # not comparable to a weekly P&L contribution: it needs to be multiplied
    # by the public factor's realised return over the same report period.
    # L1 factor name -> L4 field in factor_contribution
    factor_mapping = {
        "trend": ("long_term_rule", "长期规则"),
        "volume_price_corr": ("volume_price_corr", "量价相关性"),
        "cross_section_mom": ("long_term_cross", "长期截面"),
        "basis_carry": ("basis", "基差"),
        "profit_margin": ("profit", "利润"),
        "warehouse_receipt": ("inventory", "库存/仓单"),
        "inventory": ("inventory", "库存"),
    }

    l4_contrib = report_snapshot.get("factor_contribution", {})
    l1_factors = regression_result.get("factors", [])

    comparisons: list[FactorComparison] = []
    direction_matches = 0
    total_compared = 0

    for l1f in l1_factors:
        name = l1f["name"]
        if name not in factor_mapping:
            continue

        l4_field, display = factor_mapping[name]
        l4_val = l4_contrib.get(l4_field)

        if l4_val is None:
            continue

        l1_beta = l1f["beta"]
        l1_t = l1f["t_stat"]

        public_factor_return = (factor_period_returns or {}).get(name)
        predicted = l1_beta * public_factor_return if public_factor_return is not None else None
        dir_match = (predicted >= 0) == (l4_val >= 0) if predicted is not None else None
        if dir_match is not None:
            total_compared += 1
            if dir_match:
                direction_matches += 1
        mag_ratio = abs(predicted) / max(abs(l4_val), 1e-8) if predicted is not None else None

        comparisons.append(FactorComparison(
            factor_name=name,
            display_name=display,
            l1_beta=round(l1_beta, 4),
            l1_t_stat=round(l1_t, 2),
            l1_factor_period_return=round(public_factor_return, 6) if public_factor_return is not None else None,
            l1_predicted_contribution=round(predicted, 6) if predicted is not None else None,
            l4_contribution=round(l4_val, 6),
            direction_match=dir_match,
            magnitude_ratio=round(mag_ratio, 2) if mag_ratio is not None else None,
        ))

    agreement_pct = (direction_matches / total_compared * 100) if total_compared > 0 else 0.0

    # R² gap interpretation
    if r_squared > 0.7:
        gap_interp = f"R²={r_squared:.2f}，因子库覆盖了大部分收益来源，残差主要为 alpha。"
    elif r_squared > 0.4:
        gap_interp = f"R²={r_squared:.2f}，因子库解释了约 {r_squared*100:.0f}% 的收益波动，存在未覆盖的来源。"
    elif r_squared > 0.15:
        gap_interp = f"R²={r_squared:.2f}，因子库覆盖有限，产品策略可能超出当前 5 因子范围。"
    else:
        gap_interp = f"R²={r_squared:.2f}，因子库几乎无解释力，需检查数据对齐或扩展因子集。"

    # Summary
    if total_compared > 0:
        summary = (
            f"在 {total_compared} 个可对比因子中，{direction_matches} 个方向一致"
            f"（{agreement_pct:.0f}%）。{gap_interp}"
        )
    else:
        summary = f"无可对比因子（周报未披露因子贡献或因子库未构建）。{gap_interp}"

    warnings = []
    if not comparisons:
        warnings.append("周报中未提取到可映射的因子贡献，无法进行 L1 vs L4 对比")
    if comparisons and total_compared == 0:
        warnings.append("缺少与周报同期间的公开因子收益；仅展示 beta，不能比较方向或量级。")
    if agreement_pct < 50 and total_compared >= 3:
        warnings.append("方向一致率低于 50%，因子库构造口径可能与管理人内部因子差异较大")

    return ValidationResult(
        product_name=product_name,
        report_date=report_date,
        r_squared=r_squared,
        r_gap_interpretation=gap_interp,
        factor_comparisons=comparisons,
        direction_agreement_pct=round(agreement_pct, 1),
        summary=summary,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Alpha Measurement
# ---------------------------------------------------------------------------

@dataclass
class AlphaMetrics:
    """Detailed alpha (residual) analysis."""

    annualized_alpha: float       # regression intercept * ann_factor
    alpha_t_stat: float
    alpha_significant: bool       # |t| > 2
    information_ratio: float      # ann_alpha / residual_vol
    residual_annual_vol: float
    residual_skew: float
    residual_kurtosis: float
    # Anomaly periods (dates where |residual| > 2.5 sigma)
    anomaly_count: int
    anomaly_dates: list[str] = field(default_factory=list)
    # Interpretation
    interpretation: str = ""


def measure_alpha(regression_result: dict) -> AlphaMetrics:
    """Compute detailed alpha metrics from regression output.

    Parameters
    ----------
    regression_result : dict from /api/factor-library/attribute

    Returns
    -------
    AlphaMetrics with information ratio, anomaly detection, interpretation.
    """
    ann_alpha = regression_result.get("annualized_alpha", 0.0)
    alpha_t = regression_result.get("alpha_t_stat", 0.0)
    residual = regression_result.get("residual", {})
    resid_vol = residual.get("annual_vol", 0.0)
    resid_skew = residual.get("skewness", 0.0)
    resid_kurt = residual.get("kurtosis", 0.0)

    # Information ratio
    ir = ann_alpha / resid_vol if resid_vol > 1e-8 else 0.0

    # Anomaly detection from rolling residuals (if available)
    # For now, use kurtosis as a proxy for tail events
    anomaly_count = 0
    anomaly_dates: list[str] = []

    # Interpretation
    significant = abs(alpha_t) > 2.0
    if significant and ann_alpha > 0:
        if ir > 1.0:
            interp = f"年化 alpha {ann_alpha*100:.2f}%（t={alpha_t:.1f}），信息比率 {ir:.2f}，管理人具备显著且稳定的超额收益能力。"
        else:
            interp = f"年化 alpha {ann_alpha*100:.2f}%（t={alpha_t:.1f}），统计显著但信息比率偏低（{ir:.2f}），alpha 波动较大。"
    elif significant and ann_alpha < 0:
        interp = f"年化 alpha {ann_alpha*100:.2f}%（t={alpha_t:.1f}），显著为负——产品收益不足以覆盖因子暴露的成本，需警惕。"
    else:
        interp = f"年化 alpha {ann_alpha*100:.2f}%（t={alpha_t:.1f}），统计上不显著，无法确认管理人具备独立超额收益。"

    if resid_kurt > 3:
        interp += f" 残差峰度 {resid_kurt:.1f}（>3），存在尾部风险集中。"

    return AlphaMetrics(
        annualized_alpha=round(ann_alpha, 6),
        alpha_t_stat=round(alpha_t, 4),
        alpha_significant=significant,
        information_ratio=round(ir, 4),
        residual_annual_vol=round(resid_vol, 6),
        residual_skew=round(resid_skew, 4),
        residual_kurtosis=round(resid_kurt, 4),
        anomaly_count=anomaly_count,
        anomaly_dates=anomaly_dates,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Profile Drift Monitoring
# ---------------------------------------------------------------------------

@dataclass
class DriftAlert:
    """A single drift detection alert."""

    factor_name: str
    display_name: str
    beta_early: float
    beta_late: float
    change: float
    change_pct: float  # relative change
    severity: str      # "low" | "medium" | "high"


@dataclass
class DriftReport:
    """Multi-period profile drift analysis."""

    n_periods: int
    date_range: str
    r_squared_trend: list[float] = field(default_factory=list)
    alerts: list[DriftAlert] = field(default_factory=list)
    style_shift_detected: bool = False
    summary: str = ""


def detect_drift(
    rolling_snapshots: list[dict],
    window_split: float = 0.5,
    threshold_pct: float = 50.0,
) -> DriftReport:
    """Detect style drift from rolling regression snapshots.

    Splits the rolling window into early/late halves and compares
    average betas.  Large relative changes trigger alerts.

    Parameters
    ----------
    rolling_snapshots : list of {date, r_squared, betas} from regression
    window_split : fraction to split at (default 0.5 = first half vs second half)
    threshold_pct : relative change threshold for "high" severity

    Returns
    -------
    DriftReport with alerts and style-shift detection.
    """
    if len(rolling_snapshots) < 20:
        return DriftReport(
            n_periods=len(rolling_snapshots),
            date_range="",
            summary="滚动窗口数据不足（<20期），无法进行漂移检测。",
        )

    split_idx = int(len(rolling_snapshots) * window_split)
    early = rolling_snapshots[:split_idx]
    late = rolling_snapshots[split_idx:]

    # Collect all factor names
    factor_names = set()
    for snap in rolling_snapshots:
        factor_names.update(snap.get("betas", {}).keys())

    # Compute average betas for each half
    early_betas: dict[str, list[float]] = {f: [] for f in factor_names}
    late_betas: dict[str, list[float]] = {f: [] for f in factor_names}

    for snap in early:
        for f, v in snap.get("betas", {}).items():
            early_betas[f].append(v)
    for snap in late:
        for f, v in snap.get("betas", {}).items():
            late_betas[f].append(v)

    # R² trend (downsample)
    step = max(1, len(rolling_snapshots) // 50)
    r2_trend = [snap["r_squared"] for snap in rolling_snapshots[::step]]

    # Compare
    alerts: list[DriftAlert] = []
    display_names = {
        "trend": "长期规则",
        "volume_price_corr": "量价相关性",
        "cross_section_mom": "长期截面",
        "basis_carry": "基差",
        "profit_margin": "利润",
    }

    for fname in sorted(factor_names):
        e_vals = early_betas.get(fname, [])
        l_vals = late_betas.get(fname, [])
        if not e_vals or not l_vals:
            continue

        e_mean = np.mean(e_vals)
        l_mean = np.mean(l_vals)
        change = l_mean - e_mean

        # Relative change (avoid division by zero)
        base = max(abs(e_mean), 0.01)
        change_pct = abs(change) / base * 100

        if change_pct > threshold_pct:
            severity = "high"
        elif change_pct > threshold_pct * 0.5:
            severity = "medium"
        else:
            severity = "low"

        if severity != "low":
            alerts.append(DriftAlert(
                factor_name=fname,
                display_name=display_names.get(fname, fname),
                beta_early=round(float(e_mean), 4),
                beta_late=round(float(l_mean), 4),
                change=round(float(change), 4),
                change_pct=round(float(change_pct), 1),
                severity=severity,
            ))

    style_shift = any(a.severity == "high" for a in alerts)

    date_range = ""
    if rolling_snapshots:
        date_range = f"{rolling_snapshots[0]['date']} ~ {rolling_snapshots[-1]['date']}"

    if style_shift:
        high_factors = [a.display_name for a in alerts if a.severity == "high"]
        summary = f"检测到风格切换：{', '.join(high_factors)} 的暴露发生显著变化。"
    elif alerts:
        summary = f"检测到 {len(alerts)} 个因子有中等程度漂移，建议持续关注。"
    else:
        summary = "因子暴露整体稳定，未检测到显著风格漂移。"

    return DriftReport(
        n_periods=len(rolling_snapshots),
        date_range=date_range,
        r_squared_trend=r2_trend,
        alerts=alerts,
        style_shift_detected=style_shift,
        summary=summary,
    )
