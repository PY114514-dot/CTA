"""Fundamental CTA factors backed by auditable user-supplied CSV inputs.

CSV schema: date,symbol,value.  Values are warehouse receipts or inventories;
the library deliberately returns no series until real data is provided.
"""
import numpy as np
import pandas as pd
from app.config import DATA_DIRECTORY
from app.services.factor_library.base import FactorBase, FactorMeta, build_close_panel, contributions_from_weights, cross_sectional_zscore, equal_weight_long_short, long_short_portfolio


class _FundamentalLevelFactor(FactorBase):
    filename = ""
    direction = -1.0

    def input_note(self) -> str | None:
        path = DATA_DIRECTORY / "factor_inputs" / self.filename
        if not path.exists():
            return f"需要上传输入数据 {path}（格式: date,symbol,value）"
        return None

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        path = DATA_DIRECTORY / "factor_inputs" / self.filename
        if not path.exists():
            return pd.Series(dtype=float, name=self.meta.name)
        raw = pd.read_csv(path, parse_dates=["date"])
        if not {"date", "symbol", "value"}.issubset(raw.columns):
            return pd.Series(dtype=float, name=self.meta.name)
        levels = raw.pivot_table(index="date", columns="symbol", values="value", aggfunc="last").sort_index()
        z = (levels - levels.rolling(252, min_periods=60).mean()) / levels.rolling(252, min_periods=60).std().replace(0, np.nan)
        close = build_close_panel(panels)
        common = close.columns.intersection(z.columns)
        if len(common) < 4:
            return pd.Series(dtype=float, name=self.meta.name)
        signal = cross_sectional_zscore(self.direction * z.reindex(close.index)[common])
        return long_short_portfolio(equal_weight_long_short(signal), close[common].pct_change()).rename(self.meta.name)

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        path = DATA_DIRECTORY / "factor_inputs" / self.filename
        if not path.exists():
            return pd.DataFrame()
        raw = pd.read_csv(path, parse_dates=["date"])
        if not {"date", "symbol", "value"}.issubset(raw.columns):
            return pd.DataFrame()
        levels = raw.pivot_table(index="date", columns="symbol", values="value", aggfunc="last").sort_index()
        z = (levels - levels.rolling(252, min_periods=60).mean()) / levels.rolling(252, min_periods=60).std().replace(0, np.nan)
        close = build_close_panel(panels)
        common = close.columns.intersection(z.columns)
        if len(common) < 4:
            return pd.DataFrame()
        signal = cross_sectional_zscore(self.direction * z.reindex(close.index)[common])
        return contributions_from_weights(equal_weight_long_short(signal), close[common].pct_change())


class WarehouseReceiptFactor(_FundamentalLevelFactor):
    filename = "warehouse_receipts.csv"
    meta = FactorMeta(name="warehouse_receipt", display_name="仓单", category="基本面", description="交易所仓单水平的截面均值回归。", params={"zscore_window": 252}, signal_rule="高仓单通常代表可交割供给充裕，信号偏空；低仓单偏多。输入为 date,symbol,value CSV。", formula=r"s_t = -\,z_{252}(\text{receipt})\quad\Rightarrow\quad z(s)\to\text{LS}(20\%)", derivation="只使用可追溯的交易所仓单数据；未上传输入数据时不会生成代理收益。")


class InventoryFactor(_FundamentalLevelFactor):
    filename = "inventory.csv"
    meta = FactorMeta(name="inventory", display_name="库存", category="基本面", description="库存水平的截面均值回归。", params={"zscore_window": 252}, signal_rule="高库存偏空、低库存偏多；输入为 date,symbol,value CSV。", formula=r"s_t = -\,z_{252}(\text{inventory})\quad\Rightarrow\quad z(s)\to\text{LS}(20\%)", derivation="库存可来自交易所、社会库存或经核验的产业数据，必须记录来源与口径。")
