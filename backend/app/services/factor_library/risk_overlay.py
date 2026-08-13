"""Risk overlays for factor-return series.

The factor library deliberately keeps raw/baseline factor returns separate
from risk controls.  This module applies deterministic, ex-ante overlays to a
*cached baseline* return series for strategy evaluation.  Product attribution
continues to use the baseline series by default so a beta remains interpretable
as exposure to the underlying economic factor rather than to a particular
execution rule.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


BASELINE_PROFILE = "baseline"
VOL_TARGET_PROFILE = "vol_target"
DRAWDOWN_CONTROL_PROFILE = "drawdown_control"
SUPPORTED_PROFILES = (BASELINE_PROFILE, VOL_TARGET_PROFILE, DRAWDOWN_CONTROL_PROFILE)


@dataclass(frozen=True)
class RiskOverlayConfig:
    """Versioned default parameters for the research risk overlay."""

    target_vol: float = 0.10
    vol_lookback: int = 60
    max_leverage: float = 2.0
    drawdown_soft_limit: float = 0.08
    drawdown_hard_limit: float = 0.15
    soft_risk_scale: float = 0.70
    hard_risk_scale: float = 0.40


DEFAULT_CONFIG = RiskOverlayConfig()


def overlay_metadata(profile: str) -> dict:
    """Return displayable, reproducible metadata for an overlay profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported risk profile: {profile}")
    if profile == BASELINE_PROFILE:
        return {
            "profile": profile,
            "display_name": "基准因子",
            "description": "未叠加风控的基础因子收益；用于产品统计归因。",
            "params": {},
        }
    if profile == VOL_TARGET_PROFILE:
        return {
            "profile": profile,
            "display_name": "波动率目标",
            "description": "以前一日可得的 60 日波动率缩放仓位，目标年化波动 10%，杠杆上限 2 倍。",
            "params": {"target_vol": DEFAULT_CONFIG.target_vol, "vol_lookback": DEFAULT_CONFIG.vol_lookback,
                       "max_leverage": DEFAULT_CONFIG.max_leverage},
        }
    return {
        "profile": profile,
        "display_name": "波动率目标 + 回撤降仓",
        "description": "在波动率目标基础上，按前一日历史回撤分级降仓：8% / 15% 阈值对应 70% / 40% 风险敞口。",
        "params": {
            "target_vol": DEFAULT_CONFIG.target_vol, "vol_lookback": DEFAULT_CONFIG.vol_lookback,
            "max_leverage": DEFAULT_CONFIG.max_leverage, "drawdown_soft_limit": DEFAULT_CONFIG.drawdown_soft_limit,
            "drawdown_hard_limit": DEFAULT_CONFIG.drawdown_hard_limit, "soft_risk_scale": DEFAULT_CONFIG.soft_risk_scale,
            "hard_risk_scale": DEFAULT_CONFIG.hard_risk_scale,
        },
    }


def apply_risk_overlay(
    baseline_returns: pd.Series,
    profile: str = BASELINE_PROFILE,
    config: RiskOverlayConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Apply an ex-ante risk overlay without introducing look-ahead bias.

    Scaling at date ``t`` only uses return information through ``t-1``.  The
    first warm-up observation is retained at zero exposure rather than deleted,
    which makes overlay and baseline date alignment explicit for comparisons.
    """
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported risk profile: {profile}")
    returns = baseline_returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty or profile == BASELINE_PROFILE:
        return returns.rename(baseline_returns.name)

    trailing_vol = returns.rolling(config.vol_lookback, min_periods=config.vol_lookback).std(ddof=1)
    leverage = (config.target_vol / (trailing_vol * np.sqrt(252))).shift(1)
    leverage = leverage.clip(lower=0.0, upper=config.max_leverage).fillna(0.0)

    if profile == DRAWDOWN_CONTROL_PROFILE:
        # Drawdown is observed at the previous close, then applied to today's
        # factor return.  This prevents the current loss from deciding its own
        # position size.
        prior_nav = (1.0 + returns).cumprod().shift(1)
        prior_drawdown = prior_nav.div(prior_nav.cummax()).sub(1.0).fillna(0.0)
        risk_scale = pd.Series(1.0, index=returns.index)
        risk_scale = risk_scale.mask(prior_drawdown <= -config.drawdown_soft_limit, config.soft_risk_scale)
        risk_scale = risk_scale.mask(prior_drawdown <= -config.drawdown_hard_limit, config.hard_risk_scale)
        leverage = leverage * risk_scale

    overlaid = (returns * leverage).rename(baseline_returns.name)
    return overlaid
