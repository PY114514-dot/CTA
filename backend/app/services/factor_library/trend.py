"""Factor 1: Long-term Rule (时序趋势 / Time-Series Momentum).

Economic rationale
------------------
Classic CTA trend-following: go long varieties trading above their long-term
moving average, go short those below.  Position size is inversely proportional
to recent realized volatility (volatility targeting), which is how most
systematic CTA managers scale exposure.

Signal
------
For each variety:
    signal_t = +1  if close_t > MA(fast)  AND  close_t > MA(slow)
             = -1  if close_t < MA(fast)  AND  close_t < MA(slow)
             =  0  otherwise (mixed / transitional)

An alternative continuous signal uses the normalized distance from MA:
    signal_t = (close_t - MA(slow)) / (ATR * slow_window)

We implement the discrete version (more robust, matches practitioner style)
with vol-targeted position sizing.

Portfolio construction
----------------------
1. Per-variety signal in {-1, 0, +1}.
2. Per-variety return: signal_{t-1} * r_t (no look-ahead).
3. Weight by inverse vol (20d realized vol, annualized).
4. Normalize to target portfolio vol of 10% annualized.
5. Factor return = weighted sum of per-variety signal returns.
"""

import numpy as np
import pandas as pd

from app.services.factor_library.base import (
    FactorBase,
    FactorMeta,
    build_close_panel,
    realized_vol,
)


class TrendFactor(FactorBase):
    """Long-term trend-following (dual moving average)."""

    meta = FactorMeta(
        name="trend",
        display_name="长期规则(趋势)",
        category="量价",
        description="双均线时序动量: 价格在长期均线之上做多、之下做空, 波动率目标化",
        params={"fast_window": 120, "slow_window": 250, "vol_window": 20, "target_vol": 0.10},
        signal_rule=(
            "对每个品种：收盘价同时站上快线(120日)与慢线(250日)均线时做多(signal=+1)，"
            "同时跌破两条均线时做空(signal=-1)，介于两者之间空仓(signal=0)。"
            "仓位按 20 日实现波动率的倒数缩放(低波动品种权重更高)，"
            "组合整体再加杠杆把年化波动目标控制在 10%(杠杆上限 3 倍)。信号滞后一天应用以避免未来函数。"
        ),
        formula=r"""
s_t = \begin{cases}+1 & P_t > MA_{120} \;\land\; P_t > MA_{250} \\-1 & P_t < MA_{120} \;\land\; P_t < MA_{250} \\\;\;0 & \text{else}\end{cases}\quad\Rightarrow\quad w \propto s_{t-1}/\sigma_{20},\;\; \text{VT}(10\%)
""",
        derivation=(
            "经典 CTA 趋势跟踪：假设价格趋势具有持续性，站上长期均线视为上升趋势、跌破视为下降趋势。"
            "采用离散信号(±1/0)而非连续信号更稳健、更贴近实务。波动率倒数加权让各品种贡献相近的风险，"
            "避免高波动品种主导组合；波动率目标化(lev = target/realized vol)使因子收益序列在不同市场环境下"
            "风险敞口稳定，便于横向比较与回归。所有仓位 shift(1) 后再乘当日收益，确保 t 日信号只用 t-1 及之前的信息。"
        ),
    )

    def __init__(
        self,
        fast_window: int = 120,
        slow_window: int = 250,
        vol_window: int = 20,
        target_vol: float = 0.10,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.vol_window = vol_window
        self.target_vol = target_vol

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        if close.empty:
            return pd.DataFrame()

        # Daily returns per variety
        returns = close.pct_change()

        # Moving averages
        ma_fast = close.rolling(self.fast_window, min_periods=self.fast_window).mean()
        ma_slow = close.rolling(self.slow_window, min_periods=self.slow_window).mean()

        # Discrete signal: +1 / -1 / 0
        signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signal[(close > ma_fast) & (close > ma_slow)] = 1.0
        signal[(close < ma_fast) & (close < ma_slow)] = -1.0

        # Per-variety vol for inverse-vol weighting
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        # Position sizing for t can only use volatility observed through t-1.
        inv_vol = 1.0 / vol.shift(1).replace(0, np.nan)

        # Position = signal * inverse_vol (signal applied with 1-day lag)
        raw_position = signal.shift(1) * inv_vol

        # Per-variety PnL: position * next-day return
        variety_pnl = raw_position * returns

        # Cross-sectional sum (equal risk contribution before vol targeting)
        # Normalize by number of active varieties to avoid concentration
        active_count = signal.shift(1).abs().sum(axis=1).replace(0, np.nan)
        portfolio_ret = variety_pnl.sum(axis=1) / active_count

        # Volatility targeting: scale the whole portfolio
        port_vol = portfolio_ret.rolling(60, min_periods=20).std() * np.sqrt(252)
        leverage = (self.target_vol / port_vol.shift(1).replace(0, np.nan)).clip(upper=3.0)

        # Per-variety contribution that row-sums to the vol-targeted factor
        # return: (variety_pnl / active_count) * leverage, broadcast per day.
        return variety_pnl.div(active_count, axis=0).mul(leverage, axis=0)

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        fast = close.rolling(self.fast_window, min_periods=self.fast_window).mean()
        slow = close.rolling(self.slow_window, min_periods=self.slow_window).mean()
        signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signal[(close > fast) & (close > slow)] = 1.0
        signal[(close < fast) & (close < slow)] = -1.0
        return signal

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        contrib = self.compute_contributions(panels)
        if contrib.empty:
            return pd.Series(dtype=float, name=self.meta.name)

        # Sum per-variety contributions and drop warm-up NaN
        factor_ret = contrib.sum(axis=1).dropna()
        factor_ret.name = self.meta.name
        return factor_ret
