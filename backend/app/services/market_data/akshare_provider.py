"""AKShare-based market data provider (free, no API key required).

AKShare is a community-maintained Python library that scrapes public financial
data sources.  API signatures change between versions, so each fetch method is
wrapped defensively with fallback logic.
"""

import logging
from datetime import date, datetime

import pandas as pd

from app.services.market_data.provider import MarketDataProvider
from app.services.market_data.index_registry import INDEX_BENCHMARKS, FUTURES_VARIETIES

logger = logging.getLogger(__name__)

# Standard output columns
_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class AKShareProvider(MarketDataProvider):
    """Fetch daily index and futures data via AKShare."""

    def __init__(self) -> None:
        self._ak = None
        self._available: bool | None = None

    def _ensure_akshare(self):
        """Lazy-import akshare to avoid hard dependency at module load."""
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                logger.warning("akshare not installed; market data API unavailable")
                self._available = False
        return self._ak

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        ak = self._ensure_akshare()
        self._available = ak is not None
        return self._available

    # ------------------------------------------------------------------
    # Index data
    # ------------------------------------------------------------------

    def get_index_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch daily data for a benchmark index.

        Supports:
        - Stock indices (hs300, zz500, zz1000) via stock_zh_index_daily
        - Nanhua commodity indices via index_nh_daily (if available)
        """
        ak = self._ensure_akshare()
        if ak is None:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        info = INDEX_BENCHMARKS.get(symbol)
        if info is None:
            logger.warning("Unknown index symbol: %s", symbol)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        try:
            if info.asset_class == "equity":
                return self._fetch_stock_index(ak, info.code, start, end)
            elif info.asset_class == "commodity":
                return self._fetch_nanhua_index(ak, info.code, start, end)
        except Exception as exc:
            logger.warning("Failed to fetch index %s: %s", symbol, exc)
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    def _fetch_stock_index(self, ak, code: str, start: date, end: date) -> pd.DataFrame:
        """Fetch A-share index daily via ak.stock_zh_index_daily."""
        df = ak.stock_zh_index_daily(symbol=code)
        return self._normalize_stock_index(df, start, end)

    def _fetch_nanhua_index(self, ak, code: str, start: date, end: date) -> pd.DataFrame:
        """Fetch Nanhua commodity index with multiple fallbacks.

        Tries: index_nh_daily → stock_zh_index_daily → synthetic basket.
        """
        # Try the dedicated nanhua index endpoint
        try:
            df = ak.index_nh_daily(symbol=code)
            return self._normalize_nanhua(df, start, end)
        except (AttributeError, TypeError):
            pass

        # Fallback: try generic index endpoint with nanhua code
        try:
            df = ak.stock_zh_index_daily(symbol=code)
            return self._normalize_stock_index(df, start, end)
        except Exception:
            pass

        # A generic basket is only an acceptable fallback for the broad
        # commodity index.  Reusing it for every sector makes five different
        # benchmarks identical and corrupts strategy classification.
        if code == "NHCI":
            logger.info("Building synthetic broad commodity index from futures basket")
            return self._build_synthetic_commodity_index(ak, start, end)
        logger.warning("No independent source available for commodity sub-index %s", code)
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    def _build_synthetic_commodity_index(self, ak, start: date, end: date) -> pd.DataFrame:
        """Build a synthetic commodity index by averaging major futures returns.

        Uses a basket of liquid, sector-diverse futures as a proxy when
        dedicated commodity index endpoints are unavailable.
        """
        basket = ["rb", "cu", "au", "m", "sc"]  # 黑色+有色+贵金属+农产品+能化
        all_closes: dict[str, pd.Series] = {}

        for symbol in basket:
            df = self.get_futures_daily(symbol, start, end)
            if not df.empty and len(df) > 10:
                series = df.set_index("date")["close"]
                all_closes[symbol] = series

        if not all_closes:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        # Align all series on common dates, compute equal-weight return index
        combined = pd.DataFrame(all_closes)
        combined = combined.dropna()
        if combined.empty or len(combined) < 2:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        # Equal-weight cumulative return → synthetic index (base 1000)
        returns = combined.pct_change().dropna()
        index_returns = returns.mean(axis=1)  # equal-weight average
        index_level = (1 + index_returns).cumprod() * 1000

        result = pd.DataFrame({
            "date": index_level.index,
            "open": index_level.values,
            "high": index_level.values,
            "low": index_level.values,
            "close": index_level.values,
            "volume": 0.0,
        })
        return result[_OHLCV_COLUMNS]

    # ------------------------------------------------------------------
    # Futures data
    # ------------------------------------------------------------------

    # Mapping from our symbol codes to East Money Chinese names for futures_hist_em
    _EM_NAMES: dict[str, str] = {
        "rb": "螺纹钢主力", "hc": "热卷主力", "i": "铁矿石主力",
        "j": "焦炭主力", "jm": "焦煤主力", "ss": "不锈钢主力",
        "cu": "沪铜主力", "al": "沪铝主力", "zn": "沪锌主力",
        "ni": "沪镍主力", "sn": "沪锡主力",
        "au": "沪金主力", "ag": "沪银主力",
        "sc": "原油主力", "fu": "燃料油主力", "ta": "PTA主力",
        "ma": "甲醇主力", "pp": "聚丙烯主力", "eg": "乙二醇主力",
        "bu": "沥青主力",
        "m": "豆粕主力", "y": "豆油主力", "p": "棕榈油主力",
        "oi": "菜油主力", "cf": "棉花主力", "sr": "白糖主力", "rm": "菜粕主力",
        "IF": "沪深300主力", "IC": "中证500主力", "T": "十年国债主力",
    }

    def get_futures_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch daily data for a futures main-continuous contract.

        Tries multiple data sources: Sina → East Money (futures_hist_em).
        """
        ak = self._ensure_akshare()
        if ak is None:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        info = FUTURES_VARIETIES.get(symbol)
        if info is None:
            logger.warning("Unknown futures symbol: %s", symbol)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        # Attempt 1: Sina main continuous
        try:
            sina_symbol = f"{symbol}0"
            df = ak.futures_main_sina(symbol=sina_symbol, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
            result = self._normalize_futures(df, start, end)
            if not result.empty:
                return result
        except Exception as exc:
            logger.debug("Sina fetch failed for %s: %s", symbol, exc)

        # Attempt 2: East Money (futures_hist_em)
        em_name = self._EM_NAMES.get(symbol)
        if em_name:
            try:
                df = ak.futures_hist_em(
                    symbol=em_name,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                result = self._normalize_futures_em(df, start, end)
                if not result.empty:
                    return result
            except Exception as exc:
                logger.debug("East Money fetch failed for %s: %s", symbol, exc)

        # Attempt 3: Sina daily (no date filter)
        try:
            sina_symbol = f"{symbol}0"
            df = ak.futures_zh_daily_sina(symbol=sina_symbol)
            return self._normalize_futures(df, start, end)
        except Exception as exc:
            logger.debug("Sina daily fallback failed for %s: %s", symbol, exc)

        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    def get_shfe_contracts(self, on_date: date) -> pd.DataFrame:
        """Return SHFE contracts with their exchange-published expiry date."""
        ak = self._ensure_akshare()
        if ak is None:
            return pd.DataFrame(columns=["contract", "symbol", "expiry"])
        try:
            raw = ak.futures_contract_info_shfe(date=on_date.strftime("%Y%m%d"))
        except Exception as exc:
            logger.debug("SHFE contract list fetch failed for %s: %s", on_date, exc)
            return pd.DataFrame(columns=["contract", "symbol", "expiry"])
        if raw is None or raw.empty or not {"合约代码", "到期日"}.issubset(raw.columns):
            return pd.DataFrame(columns=["contract", "symbol", "expiry"])
        result = pd.DataFrame({
            "contract": raw["合约代码"].astype(str).str.lower(),
            "expiry": pd.to_datetime(raw["到期日"], errors="coerce").dt.date,
        }).dropna(subset=["expiry"])
        result["symbol"] = result["contract"].str.extract(r"^([a-zA-Z]+)", expand=False).str.lower()
        return result.dropna(subset=["symbol"]).sort_values(["symbol", "expiry"]).reset_index(drop=True)

    def get_shfe_daily_prices(self, on_date: date) -> pd.DataFrame:
        """Return exchange-published settlement prices for all SHFE contracts."""
        ak = self._ensure_akshare()
        if ak is None:
            return pd.DataFrame(columns=["contract", "close"])
        try:
            raw = ak.get_futures_daily(
                start_date=on_date.strftime("%Y%m%d"),
                end_date=on_date.strftime("%Y%m%d"),
                market="SHFE",
            )
        except Exception as exc:
            logger.debug("SHFE daily price fetch failed for %s: %s", on_date, exc)
            return pd.DataFrame(columns=["contract", "close"])
        if raw is None or raw.empty or not {"symbol", "settle"}.issubset(raw.columns):
            return pd.DataFrame(columns=["contract", "close"])
        result = pd.DataFrame({
            "contract": raw["symbol"].astype(str).str.lower(),
            "close": pd.to_numeric(raw["settle"], errors="coerce"),
        })
        return result.dropna(subset=["close"])[lambda frame: frame["close"] > 0].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_stock_index(self, df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Normalize stock index DataFrame to standard OHLCV format."""
        if df is None or df.empty:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        # AKShare stock_zh_index_daily typically returns: date, open, high, low, close, volume
        df = df.copy()

        # Ensure date column exists and is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        elif "日期" in df.columns:
            df["date"] = pd.to_datetime(df["日期"]).dt.date
            df = df.rename(columns={"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
        else:
            # Assume first column is date
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.date

        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0

        # Filter date range
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.sort_values("date").reset_index(drop=True)

        return df[_OHLCV_COLUMNS]

    def _normalize_nanhua(self, df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Normalize Nanhua index DataFrame."""
        if df is None or df.empty:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df = df.copy()

        # Nanhua index typically has: date, open, high, low, close, volume (or Chinese)
        col_map = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        else:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0

        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.sort_values("date").reset_index(drop=True)

        return df[_OHLCV_COLUMNS]

    def _normalize_futures(self, df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Normalize futures DataFrame to standard OHLCV format."""
        if df is None or df.empty:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df = df.copy()

        # Handle Chinese column names from Sina
        col_map = {"日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low", "收盘价": "close", "成交量": "volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Also handle English variants
        alt_map = {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        df = df.rename(columns={k: v for k, v in alt_map.items() if k in df.columns and k != v})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        elif df.index.name == "date":
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"]).dt.date
        else:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0

        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.sort_values("date").reset_index(drop=True)

        return df[_OHLCV_COLUMNS]

    def _normalize_futures_em(self, df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Normalize East Money futures_hist_em DataFrame.

        Typical columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df = df.copy()

        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "date" not in df.columns:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df["date"] = pd.to_datetime(df["date"]).dt.date

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0

        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.sort_values("date").reset_index(drop=True)

        return df[_OHLCV_COLUMNS]
