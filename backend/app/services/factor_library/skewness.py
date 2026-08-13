"""Cross-sectional realised-return skewness factor."""
import pandas as pd
from app.services.factor_library.base import FactorBase, FactorMeta, build_close_panel, contributions_from_weights, cross_sectional_zscore, equal_weight_long_short, long_short_portfolio


class SkewnessFactor(FactorBase):
    meta = FactorMeta(
        name="skewness", display_name="偏度", category="量价",
        description="60日收益偏度的截面多空因子。", params={"window": 60, "quantile": 0.2},
        signal_rule="计算每个品种过去 60 日日收益偏度，截面做多高偏度、做空低偏度品种。",
        formula=r"\text{skew}_t = \operatorname{skew}(r_{t-59:t})\quad\Rightarrow\quad z(\text{skew})\to\text{LS}(20\%)",
        derivation="用于复现外部周报的偏度因子类别；方向与参数应以持续的外部样本回测校验。",
    )

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        close = build_close_panel(panels)
        returns = close.pct_change()
        signal = cross_sectional_zscore(returns.rolling(60, min_periods=40).skew())
        return long_short_portfolio(equal_weight_long_short(signal), returns).rename(self.meta.name)

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = build_close_panel(panels)
        returns = close.pct_change()
        signal = cross_sectional_zscore(returns.rolling(60, min_periods=40).skew())
        return contributions_from_weights(equal_weight_long_short(signal), returns)

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return build_close_panel(panels).pct_change().rolling(60, min_periods=40).skew()
