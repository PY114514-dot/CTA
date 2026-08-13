"""CSV-based market data provider for user-uploaded price data.

Accepts CSV files with columns: date, open, high, low, close, volume.
Used as a fallback when the online API is unavailable or when the user
prefers to supply their own data source.
"""

import io
import logging
from datetime import date

import pandas as pd

from app.services.market_data.provider import MarketDataProvider

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class CsvMarketDataProvider(MarketDataProvider):
    """Serve market data from in-memory CSV uploads keyed by symbol."""

    def __init__(self) -> None:
        # symbol -> DataFrame (standard OHLCV format)
        self._store: dict[str, pd.DataFrame] = {}

    def is_available(self) -> bool:
        return True  # always available once data is loaded

    def load_csv(self, symbol: str, csv_content: bytes | str) -> int:
        """Parse and store a CSV upload for the given symbol.

        Returns the number of rows loaded.
        Raises ValueError if the CSV cannot be parsed or lacks required columns.
        """
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8-sig")

        try:
            df = pd.read_csv(io.StringIO(csv_content))
        except Exception as exc:
            raise ValueError(f"CSV 解析失败: {exc}") from exc

        df = self._normalize(df)
        if df.empty:
            raise ValueError("CSV 中无有效数据行（需包含 date 和 close 列）")

        self._store[symbol] = df
        return len(df)

    def get_index_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self._query(symbol, start, end)

    def get_futures_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self._query(symbol, start, end)

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._store

    def loaded_symbols(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()

    # ------------------------------------------------------------------

    def _query(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        df = self._store.get(symbol)
        if df is None:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[mask].reset_index(drop=True)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize uploaded CSV to standard OHLCV format."""
        df = df.copy()

        # Handle Chinese column names
        col_map = {
            "日期": "date", "时间": "date",
            "开盘": "open", "开盘价": "open",
            "最高": "high", "最高价": "high",
            "最低": "low", "最低价": "low",
            "收盘": "close", "收盘价": "close",
            "成交量": "volume", "持仓量": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Require at least date and close
        if "date" not in df.columns:
            # Try first column as date
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: "date"})
            else:
                return pd.DataFrame(columns=_OHLCV_COLUMNS)

        if "close" not in df.columns:
            # If only date + one value column, treat it as close
            non_date_cols = [c for c in df.columns if c != "date"]
            if len(non_date_cols) == 1:
                df = df.rename(columns={non_date_cols[0]: "close"})
            else:
                return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0

        # Fill missing OHLC with close if only close is provided
        if (df["open"] == 0).all():
            df["open"] = df["close"]
        if (df["high"] == 0).all():
            df["high"] = df["close"]
        if (df["low"] == 0).all():
            df["low"] = df["close"]

        df = df.sort_values("date").reset_index(drop=True)
        return df[_OHLCV_COLUMNS]
