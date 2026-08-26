"""南华商品指数的本地周频市场因子。"""

from datetime import date
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT
from app.services.factor_library.base import FactorBase, FactorMeta


_NANHUA_COMMODITY_CODE = "142162"
_FUTURES_DB_GLOB = "fof919_futures_weekly_nav_*.sqlite"


class NanhuaCommodityIndexFactor(FactorBase):
    """Use the complete-week returns of the Nanhua Commodity Index."""

    meta = FactorMeta(
        name="nanhua_commodity",
        display_name="南华商品指数",
        category="市场",
        description="期货 SQLite 中南华商品指数（代码 142162）的完整周收益，用作商品市场基准。",
        params={"code": 142162},
        signal_rule="每个完整自然周使用南华商品指数的收盘净值涨跌幅；尚未结束的当周不纳入。",
        formula=r"r_t = \frac{\mathrm{NHCI}_t}{\mathrm{NHCI}_{t-1}} - 1",
        derivation="该指数反映国内商品期货市场的广义方向性风险，适合解释周频或月频产品对商品市场的 Beta，不代表 CTA 的交易信号或策略收益。",
        frequency="weekly",
    )

    def __init__(self, database_path: Path | None = None) -> None:
        self._database_path = database_path

    def requires_market_panels(self) -> bool:
        return False

    def input_note(self) -> str | None:
        if self._resolve_database() is None:
            return "未找到期货周度 SQLite（期望文件：fof919_futures_weekly_nav_*.sqlite）。"
        return None

    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        """Read valid weekly returns from the local read-only SQLite source."""
        database = self._resolve_database()
        if database is None:
            return pd.Series(dtype=float, name=self.meta.name)

        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
            frame = pd.read_sql_query(
                """
                SELECT week_key, week_end, pc, fetched_at
                FROM futures_index_weekly
                WHERE code = ?
                ORDER BY week_end
                """,
                connection,
                params=(_NANHUA_COMMODITY_CODE,),
            )

        if frame.empty:
            return pd.Series(dtype=float, name=self.meta.name)

        frame["week_end"] = pd.to_datetime(frame["week_end"], errors="coerce")
        frame["pc"] = pd.to_numeric(frame["pc"], errors="coerce")
        frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], errors="coerce")
        frame = frame.dropna(subset=["week_end", "pc"])
        frame = frame[np.isfinite(frame["pc"])]
        frame = frame[frame.apply(_is_complete_week, axis=1)]
        frame = frame.drop_duplicates("week_end", keep="last").sort_values("week_end")

        returns = frame.set_index("week_end")["pc"].astype(float)
        returns.name = self.meta.name
        return returns

    def _resolve_database(self) -> Path | None:
        if self._database_path is not None:
            return self._database_path if self._database_path.is_file() else None
        candidates = sorted(PROJECT_ROOT.glob(_FUTURES_DB_GLOB))
        return candidates[-1] if candidates else None


def _is_complete_week(row: pd.Series) -> bool:
    """Exclude the source snapshot when it belongs to its still-open ISO week."""
    fetched_at = row["fetched_at"]
    if pd.isna(fetched_at):
        return True
    try:
        year_text, week_text = str(row["week_key"]).split("-W", maxsplit=1)
        week_end = date.fromisocalendar(int(year_text), int(week_text), 7)
        return week_end <= pd.Timestamp(fetched_at).date()
    except (TypeError, ValueError):
        return True
