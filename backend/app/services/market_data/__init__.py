"""Market data sub-package: providers, registry, and convenience accessors."""

from app.services.market_data.provider import MarketDataProvider
from app.services.market_data.akshare_provider import AKShareProvider
from app.services.market_data.csv_provider import CsvMarketDataProvider
from app.services.market_data.index_registry import (
    INDEX_BENCHMARKS,
    FUTURES_VARIETIES,
    SECTOR_COLORS,
    get_all_symbols,
    get_varieties_by_sector,
    get_all_sectors,
)

__all__ = [
    "MarketDataProvider",
    "AKShareProvider",
    "CsvMarketDataProvider",
    "INDEX_BENCHMARKS",
    "FUTURES_VARIETIES",
    "SECTOR_COLORS",
    "get_all_symbols",
    "get_varieties_by_sector",
    "get_all_sectors",
]
