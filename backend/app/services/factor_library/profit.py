"""Factor 5: Profit / Processing Margin (利润 / 盘面利润).

Economic rationale
------------------
Industrial processing margins (e.g., steel profit = rebar - iron ore - coke)
reflect supply-demand balance.  Extreme high margins attract new supply
(bearish), while extreme low margins force capacity cuts (bullish).  This
creates a mean-reversion dynamic exploitable cross-sectionally.

Data approach
-------------
True spot processing margins require industrial data (Wind/Mysteel).  We use
"exchange-traded margins" (盘面利润) synthesized from futures prices of
products and raw materials.  This is data-feasible and captures the same
economic signal with some noise.

Recipes (approximate stoichiometric ratios):
- Steel margin: rb - 1.6*i - 0.5*j  (1t steel ≈ 1.6t ore + 0.5t coke)
- PP margin: pp - 1.2*sc  (polypropylene from naphtha/crude)
- Palm crush: p - 0.9*m  (palm oil premium over soybean meal)
- Coke margin: j - 1.3*jm  (coke from coking coal, if jm available)

Signal
------
1. For each definable margin, compute the raw margin level.
2. Rolling z-score over 250 days (≈1 year lookback).
3. Signal = -z_score (mean reversion: high margin → short, low → long).
4. Map margin signals back to the "product" variety (rb, pp, p, j).
5. Varieties without a margin recipe get signal = 0 (excluded from portfolio).
6. Cross-sectional long-short on the available signals.

Portfolio construction
----------------------
Long varieties with low margin z-score (margin compression → bullish),
short varieties with high margin z-score (margin expansion → bearish).
Equal-weight, quintile-based.
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


# Processing margin recipes: product_symbol -> (formula description, components)
# Each component: (symbol, coefficient).  Margin = sum(coef * price).
MARGIN_RECIPES: dict[str, list[tuple[str, float]]] = {
    # Steel: rebar - 1.6*iron_ore - 0.5*coke
    "rb": [("rb", 1.0), ("i", -1.6), ("j", -0.5)],
    # Hot coil: similar to rebar
    "hc": [("hc", 1.0), ("i", -1.6), ("j", -0.5)],
    # Polypropylene: PP - 1.2*crude_oil
    "pp": [("pp", 1.0), ("sc", -1.2)],
    # Palm oil crush: palm_oil - 0.9*soybean_meal
    "p": [("p", 1.0), ("m", -0.9)],
    # Coke: coke - 1.3*coking_coal (if jm available)
    "j": [("j", 1.0), ("jm", -1.3)],
    # PTA: PTA - 0.65*crude (PX from naphtha)
    "ta": [("ta", 1.0), ("sc", -0.65)],
    # Methanol: MA - 1.5*coal_proxy(use j as coal proxy)
    "ma": [("ma", 1.0), ("j", -0.8)],
}


class ProfitMarginFactor(FactorBase):
    """Processing margin mean-reversion factor (exchange-traded margins)."""

    meta = FactorMeta(
        name="profit_margin",
        display_name="利润(盘面利润)",
        category="基本面",
        description="盘面利润均值回归: 高利润偏空(供给扩张)、低利润偏多(产能收缩), 截面多空",
        params={"zscore_window": 250, "quantile": 0.25},
        signal_rule=(
            "用期货价格合成各品种的『盘面利润』(如螺纹利润 = rb - 1.6×铁矿i - 0.5×焦炭j)，"
            "除以产品价格归一成利润率，再做 250 日滚动 z-score。信号取 z-score 的相反数(均值回归)："
            "利润处于历史高位 → 偏空(高利润吸引供给扩张)，利润处于历史低位 → 偏多(亏损倒逼产能收缩)。"
            "在有利润配方的品种间做截面 z-score，多低利润、空高利润，等权五分位(品种较少时用 25% 分位)。"
        ),
        formula=r"""
\text{margin}_t = \textstyle\sum_k c_k\,P_k,\quad\text{rate}_t = \text{margin}_t\,/\,P_{\text{prod}}\\[4pt]s_t = -\,z_{250}(\text{rate})\quad\Rightarrow\quad z(s) \to \text{LS}(25\%)
""",
        derivation=(
            "产业加工利润是供需平衡的温度计：利润极高会吸引新增产能/复产(供给扩张、利空)，"
            "利润极低甚至亏损会倒逼检修减产(供给收缩、利多)，由此形成均值回归。真实现货利润需 Wind/Mysteel 产业数据，"
            "这里用『盘面利润』——由产品与原料期货价按近似化学计量比合成(如 1 吨螺纹≈1.6 吨铁矿+0.5 吨焦炭)，"
            "数据可得且捕捉同一经济信号。对利润率取滚动 z-score 再反号，即在高利润时做空、低利润时做多。"
            "只有具备完整原料配方的品种(螺纹/PP/棕榈油/焦炭等)才进入组合，其余品种信号置零剔除。"
        ),
    )

    def __init__(
        self,
        zscore_window: int = 250,
        quantile: float = 0.25,
    ):
        self.zscore_window = zscore_window
        self.quantile = quantile

    def _weights_and_returns(
        self, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Build long-short weights and daily returns from a close panel.

        Returns None when fewer than 3 varieties have a complete margin
        recipe (not enough breadth for a cross-sectional signal).
        """
        available_symbols = set(close.columns)

        # Compute margin level for each recipe where all ingredients exist
        margin_signals = pd.DataFrame(index=close.index)

        for product, components in MARGIN_RECIPES.items():
            # Check all component symbols are available
            needed = {sym for sym, _ in components}
            if not needed.issubset(available_symbols):
                continue
            if product not in available_symbols:
                continue

            # Compute raw margin = sum(coef * price)
            margin = pd.Series(0.0, index=close.index)
            for sym, coef in components:
                margin = margin + coef * close[sym]

            # Normalize by product price to get a "margin rate"
            product_price = close[product].replace(0, np.nan)
            margin_rate = margin / product_price

            # Rolling z-score
            roll_mean = margin_rate.rolling(self.zscore_window, min_periods=60).mean()
            roll_std = margin_rate.rolling(self.zscore_window, min_periods=60).std()
            zscore = (margin_rate - roll_mean) / roll_std.replace(0, np.nan)

            # Mean reversion: invert sign (high margin → short signal)
            margin_signals[product] = -zscore

        if margin_signals.empty or margin_signals.shape[1] < 3:
            # Not enough varieties with margin data for cross-sectional signal
            return None

        # Cross-sectional z-score across available margin signals
        signal = cross_sectional_zscore(margin_signals)

        # Long-short weights (use wider quantile since fewer varieties)
        weights = equal_weight_long_short(signal, quantile=self.quantile)

        # Daily returns for the product varieties
        returns = close.pct_change()
        return weights, returns

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        close = build_close_panel(panels)
        if close.empty:
            return pd.Series(dtype=float, name=self.meta.name)

        built = self._weights_and_returns(close)
        if built is None:
            return pd.Series(dtype=float, name=self.meta.name)
        weights, returns = built

        # Portfolio return
        factor_ret = long_short_portfolio(weights, returns)
        factor_ret.name = self.meta.name
        return factor_ret

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        if close.empty:
            return pd.DataFrame()
        built = self._weights_and_returns(close)
        if built is None:
            return pd.DataFrame()
        weights, returns = built
        return contributions_from_weights(weights, returns)
