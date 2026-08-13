"""Batch data fetcher for the factor library.

Orchestrates fetching daily OHLCV data for all core varieties using the
existing market_data provider abstraction (AKShare online + CSV fallback).

The fetcher adds:
- Batch orchestration with progress tracking
- Minimum data-length filtering (skip varieties with too few rows)
- A unified panel dict ready for factor computation
"""

import logging
from datetime import date, timedelta

import pandas as pd

from app.services.market_data.provider import MarketDataProvider
from app.services.factor_library.registry import get_core_symbols

logger = logging.getLogger(__name__)

# Minimum trading days required for a variety to participate in factor
# construction.  Factors with long lookbacks (250d MA) need at least ~300
# rows to produce meaningful output.
MIN_ROWS = 300


def fetch_panels(
    providers: list[MarketDataProvider],
    start: date,
    end: date,
    symbols: list[str] | None = None,
    min_rows: int = MIN_ROWS,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV panels for all core varieties.

    Tries each provider in order for each symbol; uses the first that
    returns sufficient data.

    Parameters
    ----------
    providers : list[MarketDataProvider]
        Ordered by priority (e.g., [csv_provider, akshare_provider]).
    start, end : date
        Date range for data fetching.
    symbols : list[str] | None
        Override the default core variety list.
    min_rows : int
        Minimum rows to accept from a provider.

    Returns
    -------
    dict[symbol, DataFrame]
        Only varieties with >= min_rows of valid data.
    """
    if symbols is None:
        symbols = get_core_symbols()

    panels: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for symbol in symbols:
        df = _fetch_single(symbol, providers, start, end, min_rows)
        if df is not None:
            panels[symbol] = df
        else:
            failed.append(symbol)

    if failed:
        logger.warning(
            "Factor data fetch: %d/%d varieties failed: %s",
            len(failed), len(symbols), ", ".join(failed),
        )

    logger.info(
        "Factor data fetch complete: %d/%d varieties, range %s to %s",
        len(panels), len(symbols), start, end,
    )
    return panels


def _fetch_single(
    symbol: str,
    providers: list[MarketDataProvider],
    start: date,
    end: date,
    min_rows: int,
) -> pd.DataFrame | None:
    """Try each provider in order until one returns sufficient data."""
    for provider in providers:
        if not provider.is_available():
            continue
        try:
            df = provider.get_futures_daily(symbol, start, end)
            if df is not None and len(df) >= min_rows:
                # Basic quality: drop rows with zero close
                df = df[df["close"] > 0].reset_index(drop=True)
                if len(df) >= min_rows:
                    return df
        except Exception as exc:
            logger.debug("Provider %s failed for %s: %s", type(provider).__name__, symbol, exc)
    return None


def get_data_summary(panels: dict[str, pd.DataFrame]) -> dict:
    """Produce a summary of the fetched data for reporting to the user."""
    if not panels:
        return {"varieties": 0, "start": None, "end": None, "symbols": []}

    all_dates = []
    for df in panels.values():
        if not df.empty:
            all_dates.extend(df["date"].tolist())

    return {
        "varieties": len(panels),
        "start": min(all_dates).isoformat() if all_dates else None,
        "end": max(all_dates).isoformat() if all_dates else None,
        "symbols": sorted(panels.keys()),
        "rows_per_symbol": {sym: len(df) for sym, df in panels.items()},
    }
