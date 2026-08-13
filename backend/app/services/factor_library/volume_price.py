"""Factor 2: Volume-Price Correlation (量价相关性).

Economic rationale
------------------
When price moves are "confirmed" by volume (positive correlation between
Δprice and Δvolume), the trend is considered more sustainable.  Varieties
with high positive volume-price correlation tend to continue trending;
those with negative correlation (price up on declining volume) may reverse.

Signal
------
For each variety, compute rolling correlation between daily price change
and daily volume change over a lookback window:
    vp_corr_t = corr(Δclose, Δvolume, window=W)

Cross-sectional interpretation:
- High vp_corr → trend confirmation → go long if recent return > 0,
  short if recent return < 0.
- We combine vp_corr with the sign of recent momentum to form a
  directional signal.

Portfolio construction
----------------------
1. Compute vp_corr per variety (rolling 40-day window).
2. Multiply by sign of 20-day momentum to get directional signal.
3. Cross-sectional z-score the signal.
4. Long top quintile, short bottom quintile, equal-weight.
5. Apply to next-day returns.
"""

import numpy as np
import pandas as pd

from app.services.factor_library.base import (
    FactorBase,
    FactorMeta,
    build_close_panel,
    build_volume_panel,
    contributions_from_weights,
    cross_sectional_zscore,
    equal_weight_long_short,
    long_short_portfolio,
)


class VolumePriceCorrFactor(FactorBase):
    """Volume-price correlation trend-confirmation factor."""

    meta = FactorMeta(
        name="volume_price_corr",
        display_name="量价相关性",
        category="量价",
        description="滚动量价相关性×动量方向: 量价配合确认趋势, 截面多空",
        params={"corr_window": 40, "momentum_window": 20, "quantile": 0.2},
        signal_rule=(
            "对每个品种计算 40 日『价格变化 vs 成交量变化』的滚动相关系数，再乘以 20 日动量的方向(±1)："
            "量价同向配合(高正相关)且近期上涨 → 强做多信号；量价配合但近期下跌 → 强做空信号。"
            "把该信号在截面上 z-score 标准化后，做多得分最高的 20% 品种、做空最低的 20%，等权、每日调仓。"
        ),
        formula=r"""
x_t = \rho_{40}(\Delta P,\,\Delta V)\;\cdot\;\operatorname{sign}(R_{20})\quad\Rightarrow\quad z(x) \to \text{LS}(20\%)
""",
        derivation=(
            "量价关系是技术分析的核心命题之一：价格上涨若伴随放量(量价正相关)，说明趋势获得资金确认、"
            "更可持续；缩量上涨则可能是强弩之末。单独的相关系数没有方向，因此乘以近期动量的符号把它转成"
            "有向信号——同样是高量价配合，涨的品种做多、跌的品种做空。截面 z-score + 五分位多空使因子市场中性，"
            "捕捉的是品种间『趋势质量』的相对差异，而非整体涨跌。"
        ),
    )

    def __init__(
        self,
        corr_window: int = 40,
        momentum_window: int = 20,
        quantile: float = 0.2,
    ):
        self.corr_window = corr_window
        self.momentum_window = momentum_window
        self.quantile = quantile

    def _weights_and_returns(
        self, close: pd.DataFrame, volume: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build long-short weights and daily returns from close/volume panels."""
        # Align to common columns and index
        common_cols = close.columns.intersection(volume.columns)
        close = close[common_cols]
        volume = volume[common_cols]

        # Daily changes
        delta_price = close.diff()
        delta_volume = volume.diff()

        # Rolling correlation per variety
        vp_corr = pd.DataFrame(index=close.index, columns=common_cols, dtype=float)
        for col in common_cols:
            vp_corr[col] = delta_price[col].rolling(
                self.corr_window, min_periods=self.corr_window
            ).corr(delta_volume[col])

        # Directional component: sign of recent momentum
        momentum = close.pct_change(self.momentum_window)
        momentum_sign = np.sign(momentum)

        # Combined signal: vp_corr * momentum_sign
        # High positive corr + positive momentum → strong long signal
        # High positive corr + negative momentum → strong short signal
        raw_signal = vp_corr * momentum_sign

        # Cross-sectional z-score
        signal = cross_sectional_zscore(raw_signal)

        # Long-short portfolio weights
        weights = equal_weight_long_short(signal, quantile=self.quantile)

        # Daily returns
        returns = close.pct_change()
        return weights, returns

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        close = build_close_panel(panels)
        volume = build_volume_panel(panels)

        if close.empty or volume.empty:
            return pd.Series(dtype=float, name=self.meta.name)

        weights, returns = self._weights_and_returns(close, volume)

        # Portfolio return (weights lagged by 1 day)
        factor_ret = long_short_portfolio(weights, returns)
        factor_ret.name = self.meta.name
        return factor_ret

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        volume = build_volume_panel(panels)
        if close.empty or volume.empty:
            return pd.DataFrame()
        weights, returns = self._weights_and_returns(close, volume)
        return contributions_from_weights(weights, returns)

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close, volume = build_close_panel(panels), build_volume_panel(panels)
        common = close.columns.intersection(volume.columns)
        price_change, volume_change = close[common].diff(), volume[common].diff()
        corr = pd.DataFrame({col: price_change[col].rolling(self.corr_window, min_periods=self.corr_window).corr(volume_change[col]) for col in common})
        return corr.mul(np.sign(close[common].pct_change(self.momentum_window)))

    def compute_weights(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        close, volume = build_close_panel(panels), build_volume_panel(panels)
        return self._weights_and_returns(close, volume)[0] if not close.empty and not volume.empty else None
