"""Short-horizon (20 day) time-series momentum factor.

Uses the same risk-parity position sizing and volatility targeting as the
long-term trend factor so that the two are directly comparable: inverse
annualized-vol weighting per variety, then portfolio-level vol targeting.
"""
import numpy as np
import pandas as pd
from app.services.factor_library.base import FactorBase, FactorMeta, build_close_panel


class ShortTermTrendFactor(FactorBase):
    meta = FactorMeta(
        name="short_term_trend_20",
        display_name="短期时序动量(20日)",
        category="量价",
        description="20日价格动量的时序多空，按品种波动率缩放并波动率目标化。",
        params={"lookback": 20, "vol_window": 20, "target_vol": 0.10},
        signal_rule=(
            "对每个品种：过去 20 日收益为正做多、为负做空(signal=±1)。"
            "仓位按 20 日年化实现波动率的倒数缩放(低波动品种权重更高)，"
            "组合整体再加杠杆把年化波动目标控制在 10%(杠杆上限 3 倍)。信号滞后一天应用以避免未来函数。"
        ),
        formula=r"""
s_t = \operatorname{sign}(R_{20})\quad\Rightarrow\quadw \propto s_{t-1}/\sigma_{20},\;\; \text{VT}(10\%)
""",
        derivation=(
            "短期时序动量捕捉 20 日级别的价格惯性：近期上涨的品种倾向延续上涨。"
            "采用与长期趋势因子相同的风险平价口径——波动率倒数加权让各品种贡献相近的风险，"
            "避免高波动品种主导组合；波动率目标化(lev = target/realized vol，上限 3 倍)使因子收益序列"
            "在不同市场环境下风险敞口稳定在 10% 年化，便于与其它因子横向比较与回归。"
            "所有仓位 shift(1) 后再乘当日收益，确保 t 日信号只用 t-1 及之前的信息。"
        ),
    )

    def __init__(
        self,
        lookback: int = 20,
        vol_window: int = 20,
        target_vol: float = 0.10,
    ):
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        if close.empty:
            return pd.DataFrame()

        returns = close.pct_change()

        # Directional signal: sign of trailing *lookback*-day return, lagged 1 day
        signal = np.sign(close.pct_change(self.lookback)).shift(1)

        # Per-variety ANNUALIZED vol for inverse-vol weighting (matches trend.py)
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        inv_vol = 1.0 / vol.replace(0, np.nan)

        # Per-variety PnL: signal * inverse_vol * return
        pnl = signal * inv_vol * returns

        # Normalize by number of active varieties
        active = signal.abs().sum(axis=1).replace(0, np.nan)
        portfolio_ret = pnl.sum(axis=1) / active

        # Volatility targeting: scale the whole portfolio to target_vol
        port_vol = portfolio_ret.rolling(60, min_periods=20).std() * np.sqrt(252)
        leverage = (self.target_vol / port_vol.replace(0, np.nan)).clip(upper=3.0)

        # Per-variety contribution that row-sums to the vol-targeted factor return
        return pnl.div(active, axis=0).mul(leverage, axis=0)

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        return close.pct_change(self.lookback)

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        contrib = self.compute_contributions(panels)
        if contrib.empty:
            return pd.Series(dtype=float, name=self.meta.name)
        result = contrib.sum(axis=1).dropna()
        result.name = self.meta.name
        return result
