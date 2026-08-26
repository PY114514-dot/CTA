"""Tests for ex-ante factor risk overlays."""

import numpy as np
import pandas as pd

from app.services.factor_library.risk_overlay import (
    BASELINE_PROFILE,
    DRAWDOWN_CONTROL_PROFILE,
    VOL_TARGET_PROFILE,
    apply_risk_overlay,
    overlay_metadata,
)


def _returns(n: int = 140) -> pd.Series:
    return pd.Series(
        np.linspace(-0.012, 0.014, n),
        index=pd.bdate_range("2024-01-02", periods=n),
        name="trend",
    )


def test_baseline_profile_preserves_returns() -> None:
    returns = _returns()
    result = apply_risk_overlay(returns, BASELINE_PROFILE)
    pd.testing.assert_series_equal(result, returns)


def test_vol_target_does_not_use_current_return_to_size_position() -> None:
    returns = _returns()
    changed = returns.copy()
    # This is after the 60-day warm-up.  It must not change the overlaid return
    # on the same day because sizing uses volatility known at t-1.
    changed.iloc[100] = 0.50

    original_overlay = apply_risk_overlay(returns, VOL_TARGET_PROFILE)
    changed_overlay = apply_risk_overlay(changed, VOL_TARGET_PROFILE)

    assert original_overlay.iloc[100] != 0
    assert changed_overlay.iloc[100] != 0
    assert np.isclose(
        changed_overlay.iloc[100] / changed.iloc[100],
        original_overlay.iloc[100] / returns.iloc[100],
    )


def test_drawdown_control_is_more_conservative_after_large_loss() -> None:
    returns = _returns()
    returns.iloc[85] = -0.30
    vol_only = apply_risk_overlay(returns, VOL_TARGET_PROFILE)
    drawdown_controlled = apply_risk_overlay(returns, DRAWDOWN_CONTROL_PROFILE)

    # The following day's risk scale reflects the prior close's drawdown.
    assert abs(drawdown_controlled.iloc[86]) < abs(vol_only.iloc[86])


def test_overlay_metadata_is_available_for_all_profiles() -> None:
    for profile in (BASELINE_PROFILE, VOL_TARGET_PROFILE, DRAWDOWN_CONTROL_PROFILE):
        metadata = overlay_metadata(profile)
        assert metadata["profile"] == profile
        assert metadata["display_name"]
