"""Factor 3: Long-term Cross-section (截面动量 / Cross-Sectional Momentum).

Economic rationale
------------------
Relative-value across varieties: go long the best performers over the past
6-12 months and short the worst.  Unlike time-series momentum (which has a
market-direction component), cross-sectional momentum is market-neutral by
construction — it captures dispersion in commodity returns.

Signal
------
For each variety at time t:
    momentum_t = close_t / close_{t - lookback} - 1

Cross-sectional ranking of momentum → long top quintile, short bottom quintile.

We use a 126-day (≈6 month) lookback as the default, which balances signal
strength against turnover.  A skip of the most recent 5 days avoids
short-term reversal contamination.

Portfolio construction
----------------------
1. Compute 126-day return per variety (skip last 5 days).
2. Cross-sectional z-score.
3. Long top 20%, short bottom 20%, equal-weight within each leg.
4. Gross exposure normalized to 1.0 (0.5 long + 0.5 short).
5. Rebalance daily (weights shift slowly due to long lookback).
6. Apply weights to next-day returns.
"""

import numpy as np
import pandas as pd

from app.services.factor_library.base import (
    FactorBase,
    FactorMeta,
    build_close_panel,
    contributions_from_weights,
    cross_sectional_zscore,
    equal_weight_long_short,
    long_short_portfolio,
)


class CrossSectionMomentumFactor(FactorBase):
    """Long-term cross-sectional momentum (relative value across varieties)."""

    meta = FactorMeta(
        name="cross_section_mom",
        display_name="长期截面(截面动量)",
        category="量价",
        description="6个月截面动量: 做多过去强势品种、做空弱势品种, 市场中性",
        params={"lookback": 126, "skip": 5, "quantile": 0.2},
        signal_rule=(
            "对每个品种计算过去约 6 个月(126 日)的累计收益，但跳过最近 5 天以规避短期反转噪声。"
            "把所有品种的动量在截面上 z-score 标准化，做多相对最强的 20%、做空相对最弱的 20%，"
            "等权、毛敞口归一为 1(0.5 多 + 0.5 空)。组合天然市场中性，每日再平衡(因长回看窗口，权重变化缓慢)。"
        ),
        formula=r"""
\text{mom}_t = P_{t-5}\,/\,P_{t-126} - 1\quad\Rightarrow\quad z(\text{mom}) \to \text{LS}(20\%)
""",
        derivation=(
            "截面动量是商品期货里最稳健的异象之一：过去 6-12 个月相对强势的品种倾向于继续强势。"
            "与时序动量(含市场方向敞口)不同，截面动量通过『多强空弱』对冲掉整体涨跌，只赚品种间分化(离散度)的钱，"
            "因此是市场中性因子。跳过最近 5 天是为了避开短期反转(过去一周涨太多的品种往往小幅回调)。"
            "毛敞口归一为 1 使不同因子的收益序列量纲一致，方便后续做因子回归与归因。"
        ),
    )

    def __init__(
        self,
        lookback: int = 126,
        skip: int = 5,
        quantile: float = 0.2,
    ):
        self.lookback = lookback
        self.skip = skip
        self.quantile = quantile

    def _weights_and_returns(self, close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build long-short weights and daily returns from a close panel."""
        # Momentum: return from t-lookback to t-skip
        # close.shift(skip) is the price `skip` days ago
        # close.shift(lookback) is the price `lookback` days ago
        past_price = close.shift(self.lookback)
        recent_price = close.shift(self.skip)
        momentum = recent_price / past_price - 1.0

        # Cross-sectional z-score
        signal = cross_sectional_zscore(momentum)

        # Long-short weights
        weights = equal_weight_long_short(signal, quantile=self.quantile)

        # Daily returns
        returns = close.pct_change()
        return weights, returns

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Persist pre-ranking momentum, not the later long-short PnL."""
        close = build_close_panel(panels)
        return close.shift(self.skip).div(close.shift(self.lookback)).sub(1.0)

    def compute_weights(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        close = build_close_panel(panels)
        return self._weights_and_returns(close)[0] if not close.empty else None

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        close = build_close_panel(panels)
        if close.empty:
            return pd.Series(dtype=float, name=self.meta.name)

        weights, returns = self._weights_and_returns(close)

        # Portfolio return
        factor_ret = long_short_portfolio(weights, returns)
        factor_ret.name = self.meta.name
        return factor_ret

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        if close.empty:
            return pd.DataFrame()
        weights, returns = self._weights_and_returns(close)
        return contributions_from_weights(weights, returns)
