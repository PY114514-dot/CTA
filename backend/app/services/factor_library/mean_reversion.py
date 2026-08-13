"""Factor 10: Short-term Mean Reversion (短期反转).

Economic rationale
------------------
Over short horizons (1-2 weeks), commodity futures exhibit mean reversion:
variety that rose sharply tends to pull back, and vice versa.  This is the
natural complement to the momentum/trend factors already in the library —
those earn from persistence over weeks-to-months, while reversal earns from
the snap-back over days.

The effect is typically attributed to temporary liquidity shocks and
overreaction: a burst of buying pushes price above fundamental value, and
the subsequent unwinding produces a predictable reversion.

Signal
------
For each variety at time t:
    signal_t = -R_5 = -(P_t / P_{t-5} - 1)

Negative sign: past losers (R_5 < 0) get a positive signal (go long),
past winners (R_5 > 0) get a negative signal (go short).

Portfolio construction
----------------------
1. Compute 5-day return per variety.
2. Negate to get reversal signal.
3. Cross-sectional z-score.
4. Long top 20% (biggest recent losers), short bottom 20% (biggest winners).
5. Equal-weight, gross exposure = 1.
6. Apply weights to next-day returns.
"""

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


class MeanReversionFactor(FactorBase):
    """Short-term (5-day) cross-sectional mean reversion."""

    meta = FactorMeta(
        name="mean_reversion_5d",
        display_name="短期反转(5日)",
        category="量价",
        description="5日收益反转: 做多近期超跌品种、做空近期超涨品种, 截面多空",
        params={"lookback": 5, "quantile": 0.2},
        signal_rule=(
            "对每个品种计算过去 5 个交易日的累计收益并取反作为反转信号："
            "近期跌幅最大的品种获得最强做多信号(超跌反弹)，近期涨幅最大的品种获得最强做空信号(获利回吐)。"
            "截面 z-score 标准化后，做多信号最高的 20%、做空最低的 20%，等权、毛敞口归一为 1，每日调仓。"
        ),
        formula=r"""
s_t = -\,R_{5} = -\!\left(\frac{P_t}{P_{t-5}} - 1\right)
\quad\Rightarrow\quad
z(s) \to \text{LS}(20\%)
""",
        derivation=(
            "短期反转是商品期货里与动量互补的经典异象：过去 1-2 周的极端涨跌倾向于在随后几天回吐。"
            "成因通常归结为流动性冲击与过度反应——集中买/卖把价格暂时推离均衡，随后均值回归。"
            "与库中已有的趋势/动量因子(捕捉周-月级持续性)形成对冲：趋势因子在动量崩溃期亏损时，"
            "反转因子往往盈利，两者组合可降低整体因子收益的尾部风险。"
            "信号取 5 日收益的相反数，截面 z-score + 五分位多空确保市场中性，只赚品种间过度反应的截面差异。"
        ),
    )

    def __init__(
        self,
        lookback: int = 5,
        quantile: float = 0.2,
    ):
        self.lookback = lookback
        self.quantile = quantile

    def _weights_and_returns(self, close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build long-short weights and daily returns from a close panel."""
        # Short-term return
        ret = close.pct_change(self.lookback, fill_method=None)

        # Reversal signal: negate (past losers → long, past winners → short)
        signal = cross_sectional_zscore(-ret)

        # Long-short weights
        weights = equal_weight_long_short(signal, quantile=self.quantile)

        # Daily returns
        returns = close.pct_change(fill_method=None)
        return weights, returns

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        close = build_close_panel(panels)
        if close.empty:
            return pd.Series(dtype=float, name=self.meta.name)

        weights, returns = self._weights_and_returns(close)

        factor_ret = long_short_portfolio(weights, returns)
        factor_ret.name = self.meta.name
        return factor_ret

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        if close.empty:
            return pd.DataFrame()
        weights, returns = self._weights_and_returns(close)
        return contributions_from_weights(weights, returns)

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -build_close_panel(panels).pct_change(self.lookback, fill_method=None)

    def compute_weights(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        close = build_close_panel(panels)
        return self._weights_and_returns(close)[0] if not close.empty else None
