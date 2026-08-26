from datetime import date, timedelta

import pytest

from app.services import multi_asset_factor_contracts as contracts


def test_equity_contract_requires_all_factor_series_and_preserves_version(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contracts, "ATTRIBUTION_FACTOR_CONTRACT_DIRECTORY", tmp_path)
    start = date(2024, 1, 5)
    rows = [
        {
            "observation_date": start + timedelta(days=7 * index),
            "available_date": start + timedelta(days=7 * index),
            "factor_name": name,
            "value": index / 1000,
        }
        for name, _ in contracts.CONTRACTS["equity_quant"]["factors"]
        for index in range(20)
    ]

    saved = contracts.save_factor_rows("equity_quant", rows, source="人工审核 CSV", data_version="2024Q2-v1")

    assert saved["status"] == "available"
    assert saved["contract_version"] == "equity_quant_v1"
    series = contracts.load_factor_series("equity_quant", "equity_market", as_of_date=start + timedelta(days=7 * 10))
    assert series is not None
    assert len(series) == 11


def test_contract_rejects_future_available_date(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contracts, "ATTRIBUTION_FACTOR_CONTRACT_DIRECTORY", tmp_path)
    with pytest.raises(contracts.MultiAssetFactorContractError, match="available_date"):
        contracts.save_factor_rows("options_volatility", [{
            "observation_date": "2024-01-05", "available_date": "2024-01-06",
            "factor_name": "option_underlying_market", "value": 0.01,
        }], source="审核 CSV", data_version="v1")
