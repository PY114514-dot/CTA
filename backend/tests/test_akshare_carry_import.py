from datetime import date, timedelta

import pandas as pd

import pytest

from app.services.akshare_carry_import import assert_carry_coverage, import_shfe_carry


class FakeProvider:
    def get_shfe_contracts(self, on_date):
        rows = []
        for symbol in ("rb", "cu", "al", "zn"):
            rows.extend([
                {"contract": f"{symbol}2609", "symbol": symbol, "expiry": date(2026, 9, 15)},
                {"contract": f"{symbol}2612", "symbol": symbol, "expiry": date(2026, 12, 15)},
            ])
        return pd.DataFrame(rows)

    def get_shfe_daily_prices(self, on_date):
        if on_date not in {date(2026, 8, 7), date(2026, 8, 14)}:
            return pd.DataFrame(columns=["contract", "close"])
        base = 100 if on_date == date(2026, 8, 7) else 101
        return pd.DataFrame([
            {"contract": f"{symbol}{month}", "close": base}
            for symbol in ("rb", "cu", "al", "zn")
            for month in ("2609", "2612")
        ])


def test_import_shfe_carry_uses_named_contracts_and_published_expiry() -> None:
    rows = import_shfe_carry(
        date(2026, 8, 1),
        date(2026, 8, 14),
        provider=FakeProvider(),
        symbols=("rb", "cu", "al", "zn"),
    )

    assert len(rows) == 4
    assert {row["symbol"] for row in rows} == {"rb", "cu", "al", "zn"}
    assert all(row["near_contract"] != row["far_contract"] for row in rows)
    assert all(row["near_next_price"] > 0 and row["far_next_price"] > 0 for row in rows)
    assert all(row["observation_date"] - date(2026, 8, 7) >= timedelta(days=7) for row in rows)
    assert all(row["far_days_to_expiry"] > row["near_days_to_expiry"] for row in rows)
    assert all(row["available_date"] == row["observation_date"] for row in rows)


def test_carry_coverage_rejects_partial_download() -> None:
    with pytest.raises(ValueError, match="Carry"):
        assert_carry_coverage([], date(2026, 8, 1), date(2026, 8, 14))
