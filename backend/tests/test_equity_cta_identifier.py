"""Tests for the equity-index CTA market-reference inference layer."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.services.analysis.equity_cta_identifier import identify_equity_references
from app.services.market_data.provider import MarketDataProvider


def _dates(count: int) -> list[date]:
    current = date(2023, 1, 2)
    result: list[date] = []
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def test_equity_cta_references_rank_and_label_market_proxies() -> None:
    """The equity path must return indices/futures as references, not commodities."""
    rng = np.random.default_rng(72)
    dates = _dates(80)
    hs300 = rng.normal(0, 0.01, len(dates))
    provider = MagicMock(spec=MarketDataProvider)
    reference_series = {
        "hs300": hs300,
        "zz500": rng.normal(0, 0.01, len(dates)),
        "zz1000": rng.normal(0, 0.01, len(dates)),
        "IF": hs300 + rng.normal(0, 0.001, len(dates)),
        "IC": rng.normal(0, 0.01, len(dates)),
        "T": rng.normal(0, 0.003, len(dates)),
    }

    def returns(symbol: str, _start: date, _end: date, is_index: bool = True) -> pd.Series:
        assert is_index == (symbol in {"hs300", "zz500", "zz1000"})
        return pd.Series(reference_series[symbol], index=pd.to_datetime(dates))

    provider.get_returns.side_effect = returns
    product = 1.1 * hs300 + rng.normal(0, 0.001, len(dates))
    result = identify_equity_references(product, dates, "daily", provider)

    assert result.top_varieties
    assert result.top_varieties[0].symbol in {"hs300", "IF"}
    assert {candidate.sector for candidate in result.top_varieties} <= {"大盘指数", "中盘指数", "小盘指数", "股指期货", "利率对冲"}
    assert any("股指 CTA" in line for line in result.evidence)
    assert all("商品" not in candidate.name for candidate in result.top_varieties)
