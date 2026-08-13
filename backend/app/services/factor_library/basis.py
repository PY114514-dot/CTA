"""Factor 4: Basis / Carry (基差 / 期限结构).

Economic rationale
------------------
In commodity futures, the term structure slope (near vs far month spread)
reflects carry: backwardation = positive carry, contango = negative carry.
Varieties with high positive carry tend to deliver excess returns as the
futures price "rolls up" the curve.

Data constraint
---------------
True term structure data (near/far month prices) requires multi-contract
fetching which is unreliable via free APIs.  We use a well-established
academic proxy (Koijen, Moskowitz, Pedersen 2018): the difference between
short-horizon and long-horizon returns captures the roll-yield component.

Signal
------
For each variety at time t:
    carry_proxy = ret_1m(t) - ret_6m(t)

where ret_Nm is the N-month (≈21/126 trading day) cumulative return.

Interpretation:
- Positive carry_proxy → recent momentum exceeds long-term trend →
  consistent with backwardation / positive roll yield → go long.
- Negative carry_proxy → recent underperformance vs trend →
  consistent with contango / negative carry → go short.

Portfolio construction
----------------------
1. Compute carry_proxy per variety (21d return minus 126d return).
2. Cross-sectional z-score.
3. Long top quintile (high carry), short bottom quintile (low carry).
4. Equal-weight within each leg, gross exposure = 1.
5. Apply weights to next-day returns.
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


class BasisCarryFactor(FactorBase):
    """Term structure carry proxy via multi-horizon return differential."""

    meta = FactorMeta(
        name="basis_carry",
        display_name="基差(Carry)",
        category="基本面",
        description="多期限收益差代理期限结构: 短期超额收益反映正carry, 截面多空",
        params={"short_window": 21, "long_window": 126, "quantile": 0.2},
        signal_rule=(
            "对每个品种计算『短期收益(21日) 减 长期收益(126日)』作为 carry 代理："
            "短期跑赢长期趋势(代理值为正)对应现货紧张/贴水(backwardation)、正展期收益，做多；"
            "短期跑输(代理值为负)对应升水(contango)、负 carry，做空。截面 z-score 后做多最高 20%、做空最低 20%，等权。"
        ),
        formula=r"""
\text{carry}_t = R_{21} - R_{126}\quad\Rightarrow\quad z(\text{carry}) \to \text{LS}(20\%)
""",
        derivation=(
            "商品期货的期限结构斜率反映持有收益(carry)：贴水(backwardation)意味着展期获利、升水(contango)意味着展期亏损，"
            "高 carry 品种长期倾向于提供超额收益。真实的近远月价差需要多合约数据，免费接口难以稳定获取，"
            "因此采用学术上常用的代理(Koijen, Moskowitz, Pedersen 2018)：短周期收益减长周期收益近似剥离出展期分量。"
            "该代理虽含噪声，但保留了 carry 的截面排序信息，足以构造市场中性多空因子。"
        ),
    )

    def __init__(
        self,
        short_window: int = 21,
        long_window: int = 126,
        quantile: float = 0.2,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.quantile = quantile

    def _weights_and_returns(self, close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build long-short weights and daily returns from a close panel."""
        # Multi-horizon cumulative returns
        ret_short = close.pct_change(self.short_window)
        ret_long = close.pct_change(self.long_window)

        # Carry proxy: short-horizon excess over long-horizon
        carry_proxy = ret_short - ret_long

        # Cross-sectional z-score
        signal = cross_sectional_zscore(carry_proxy)

        # Long-short weights
        weights = equal_weight_long_short(signal, quantile=self.quantile)

        # Daily returns
        returns = close.pct_change()
        return weights, returns

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

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        return close.pct_change(self.short_window).sub(close.pct_change(self.long_window))

    def compute_weights(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        close = build_close_panel(panels)
        return self._weights_and_returns(close)[0] if not close.empty else None
