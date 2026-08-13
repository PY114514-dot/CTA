"""Abstract market-data provider interface.

All concrete providers (AKShare, CSV upload, future paid APIs) implement this
contract so the analysis pipeline is agnostic to the data source.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class MarketDataProvider(ABC):
    """Uniform interface for fetching index and futures price data."""

    @abstractmethod
    def get_index_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return daily OHLCV for a benchmark index.

        Columns: date, open, high, low, close, volume
        Sorted by date ascending.  Empty DataFrame if symbol unavailable.
        """

    @abstractmethod
    def get_futures_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return daily OHLCV for a futures main-continuous contract.

        Columns: date, open, high, low, close, volume
        Sorted by date ascending.  Empty DataFrame if symbol unavailable.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Quick connectivity / dependency check."""

    def get_returns(self, symbol: str, start: date, end: date, is_index: bool = True) -> pd.Series:
        """Convenience: compute simple daily returns from close prices.

        Returns a Series indexed by date with the first row dropped.
        """
        df = self.get_index_daily(symbol, start, end) if is_index else self.get_futures_daily(symbol, start, end)
        if df.empty or len(df) < 2:
            return pd.Series(dtype=float)
        close = df.set_index("date")["close"].sort_index()
        return close.pct_change().dropna()
