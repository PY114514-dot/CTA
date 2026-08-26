from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.services import shfe_xls_carry_import as importer


def test_delivery_month_midpoint_is_derived_from_standard_contract_code() -> None:
    assert importer._delivery_month_midpoint("rb2405") == date(2024, 5, 15)
    assert importer._delivery_month_midpoint("invalid") is None


def test_build_shfe_carry_rows_holds_the_same_named_contracts(monkeypatch) -> None:
    rows = []
    for day, near_price, far_price in (("2024-01-05", 100.0, 110.0), ("2024-01-12", 105.0, 115.0)):
        for symbol in ("rb", "cu", "al", "zn"):
            rows.extend([
                {"contract": f"{symbol}2405", "trade_date": pd.Timestamp(day), "settle": near_price, "volume": 10.0, "open_interest": 10.0, "symbol": symbol, "delivery_date": date(2024, 5, 15)},
                {"contract": f"{symbol}2410", "trade_date": pd.Timestamp(day), "settle": far_price, "volume": 10.0, "open_interest": 10.0, "symbol": symbol, "delivery_date": date(2024, 10, 15)},
            ])
    monkeypatch.setattr(importer, "load_shfe_reports", lambda _directory: pd.DataFrame(rows))

    result = importer.build_shfe_carry_rows(
        directory=Path("ignored"),
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
    )

    assert len(result) == 4
    assert result[0]["near_contract"].endswith("2405")
    assert result[0]["far_contract"].endswith("2410")
    assert result[0]["holding_return"] == pytest.approx(0.05)
