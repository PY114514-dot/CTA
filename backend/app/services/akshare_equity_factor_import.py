"""Build the AKShare-supported subset of the equity factor contract."""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.services.market_data.provider import MarketDataProvider


def build_equity_factor_rows(provider: MarketDataProvider, start: date, end: date) -> list[dict]:
    """Build only reproducible index-based proxies from public daily closes.

    Value and quality have no defensible free source in the current stack, so
    they deliberately remain uncovered instead of being proxied by prices.
    """
    series: dict[str, pd.Series] = {}
    for symbol in ("hs300", "zz1000"):
        frame = provider.get_index_daily(symbol, start, end)
        if frame.empty or "date" not in frame or "close" not in frame:
            raise ValueError(f"AKShare 未返回 {symbol} 的有效指数收盘价")
        values = pd.to_numeric(frame.set_index("date")["close"], errors="coerce").dropna()
        values.index = pd.to_datetime(values.index)
        series[symbol] = values.pct_change().dropna()
    aligned = pd.concat(series, axis=1).dropna()
    if len(aligned) < 21:
        raise ValueError("AKShare 股票指数共同历史不足 21 个交易日")
    market = aligned["hs300"]
    momentum_signal = (1.0 + market).rolling(20, min_periods=20).apply(lambda values: values.prod() - 1.0, raw=True).shift(1)
    volatility = market.rolling(20, min_periods=20).std().mul(252 ** 0.5).shift(1)
    factors = pd.DataFrame({
        "equity_market": market,
        "equity_size_spread": aligned["zz1000"] - market,
        "equity_momentum": market * momentum_signal.map(lambda value: 1.0 if value >= 0 else -1.0),
        "equity_volatility": volatility.diff(),
    }).dropna()
    return [
        {"observation_date": item_date.date(), "available_date": item_date.date(), "factor_name": factor_name, "value": float(value)}
        for item_date, row in factors.iterrows()
        for factor_name, value in row.items()
    ]
