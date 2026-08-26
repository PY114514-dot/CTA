from datetime import date, timedelta

import pandas as pd

from app.services.akshare_equity_factor_import import build_equity_factor_rows


class FakeProvider:
    def get_index_daily(self, symbol, start, end):
        dates = [start + timedelta(days=index) for index in range(35)]
        base = 100.0 if symbol == "hs300" else 120.0
        slope = 1.002 if symbol == "hs300" else 1.003
        closes = [base * slope ** index for index in range(35)]
        return pd.DataFrame({"date": dates, "close": closes})


def test_builds_only_supported_equity_proxies() -> None:
    rows = build_equity_factor_rows(FakeProvider(), date(2024, 1, 1), date(2024, 2, 4))
    names = {row["factor_name"] for row in rows}
    assert names == {"equity_market", "equity_size_spread", "equity_momentum", "equity_volatility"}
    assert all(row["available_date"] == row["observation_date"] for row in rows)
