from datetime import date, timedelta

import pytest

from app.services import cta_factor_bundle as bundle


def _carry_rows(*, first_day: date = date(2024, 1, 1), holding_returns: list[float] | None = None) -> list[dict]:
    returns = holding_returns or [0.01, 0.02, 0.00, 0.00]
    rows = []
    for index, (near, far, holding_return) in enumerate(
        zip((110.0, 100.0, 100.0, 100.0), (100.0, 110.0, 100.0, 100.0), returns, strict=True)
    ):
        observation_date = first_day
        rows.append({
            "observation_date": observation_date,
            "symbol": f"S{index}",
            "near_contract": f"S{index}2401",
            "far_contract": f"S{index}2405",
            "near_price": near,
            "far_price": far,
            "near_next_price": near * (1 + holding_return),
            "far_next_price": far * (1 + holding_return / 2),
            "near_days_to_expiry": 30,
            "far_days_to_expiry": 120,
            "holding_return": holding_return,
            "available_date": observation_date,
        })
    return rows


def test_real_carry_uses_term_structure_sort_and_holding_return() -> None:
    result = bundle.compute_real_carry_returns(
        _carry_rows(holding_returns=[0.12, -0.08, 0.01, 0.02]),
        quantile=0.25,
        minimum_symbols=4,
    )

    assert result.name == "term_structure_carry"
    assert result.iloc[0] == pytest.approx(0.20)


def test_real_carry_does_not_use_price_return_proxy() -> None:
    base = _carry_rows(holding_returns=[0.00, 0.00, 0.00, 0.00])
    changed = _carry_rows(holding_returns=[0.20, -0.10, 0.00, 0.00])

    base_result = bundle.compute_real_carry_returns(base, quantile=0.25, minimum_symbols=4)
    changed_result = bundle.compute_real_carry_returns(changed, quantile=0.25, minimum_symbols=4)

    assert base_result.iloc[0] == pytest.approx(0.0)
    assert changed_result.iloc[0] == pytest.approx(0.30)


def test_calendar_spread_momentum_uses_same_contract_spread_returns() -> None:
    rows = []
    for offset in range(5):
        for index, symbol in enumerate(("S0", "S1", "S2", "S3")):
            near, far = 100.0, 100.0 + (index + offset) * 2.0
            rows.append({
                "observation_date": date(2024, 1, 1) + timedelta(days=offset), "symbol": symbol,
                "near_contract": f"{symbol}2401", "far_contract": f"{symbol}2405",
                "near_price": near, "far_price": far, "near_next_price": near,
                "far_next_price": far * (1.1 if index == 3 else 1.0),
                "near_days_to_expiry": 30, "far_days_to_expiry": 120,
                "holding_return": 0.0, "available_date": date(2024, 1, 1) + timedelta(days=offset),
            })
    result = bundle.compute_calendar_spread_momentum(rows, lookback_periods=1, quantile=0.25)

    assert result.name == "calendar_spread_momentum"
    assert result.iloc[-1] == pytest.approx(-0.1)


def test_carry_rows_reject_duplicate_observation_and_invalid_expiry() -> None:
    duplicate = _carry_rows() + [_carry_rows()[0]]
    with pytest.raises(bundle.CtaFactorContractError, match="同一日期和品种"):
        bundle.validate_carry_rows(duplicate)

    invalid = _carry_rows()
    invalid[0]["far_days_to_expiry"] = invalid[0]["near_days_to_expiry"]
    with pytest.raises(bundle.CtaFactorContractError, match="far_days_to_expiry"):
        bundle.validate_carry_rows(invalid)


def test_carry_loader_obeys_as_of_date(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    carry_path = tmp_path / "term_structure_carry.csv"
    manifest_path = tmp_path / "term_structure_carry_manifest.json"
    monkeypatch.setattr(bundle, "REAL_CARRY_PATH", carry_path)
    monkeypatch.setattr(bundle, "REAL_CARRY_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(bundle, "CTA_FACTOR_DIRECTORY", tmp_path)

    rows = []
    for index in range(4):
        rows.extend(_carry_rows(first_day=date(2024, 1, 1) + timedelta(days=index)))
    bundle.save_real_carry_rows(
        rows,
        source="test",
        data_version="test-v1",
        continuous_contract_rule="declared-near-far",
    )

    assert len(bundle.load_real_carry_rows(as_of_date=date(2024, 1, 1))) == 4
    assert len(bundle.load_real_carry_rows(as_of_date=date(2024, 1, 3))) == 12


def test_bundle_reports_carry_not_covered_without_real_input(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(bundle, "REAL_CARRY_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(bundle, "REAL_CARRY_MANIFEST_PATH", tmp_path / "missing.json")

    manifest = bundle.get_cta_factor_bundle(as_of_date=date(2024, 1, 1))
    carry = next(item for item in manifest["factors"] if item["name"] == "term_structure_carry")

    assert carry["status"] == "not_covered"
    assert bundle.get_factor_series("term_structure_carry", as_of_date=date(2024, 1, 1)) is None
