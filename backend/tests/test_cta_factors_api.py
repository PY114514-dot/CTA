from datetime import date

import pytest
from fastapi import HTTPException

from app.routers.cta_factors import CarryBundleUploadRequest, get_factor_bundle, upload_real_carry
from app.services import cta_factor_bundle as bundle


def _rows() -> list[dict]:
    return [
        {
            "observation_date": date(2024, 1, 1),
            "symbol": f"S{index}",
            "near_contract": f"S{index}2401",
            "far_contract": f"S{index}2405",
            "near_price": 100.0 - index,
            "far_price": 101.0 + index,
            "near_next_price": 100.5 - index,
            "far_next_price": 101.5 + index,
            "near_days_to_expiry": 30,
            "far_days_to_expiry": 120,
            "holding_return": 0.01 * index,
            "available_date": date(2024, 1, 1),
        }
        for index in range(4)
    ]


def test_factor_bundle_endpoint_reports_contract_and_not_covered(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(bundle, "REAL_CARRY_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(bundle, "REAL_CARRY_MANIFEST_PATH", tmp_path / "missing.json")

    response = get_factor_bundle(as_of_date=date(2024, 1, 1))
    carry = next(item for item in response["factors"] if item["name"] == "term_structure_carry")

    assert response["bundle_version"] == "cta_factor_bundle_v1"
    assert carry["status"] == "not_covered"


def test_factor_bundle_upload_persists_real_carry(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(bundle, "CTA_FACTOR_DIRECTORY", tmp_path)
    monkeypatch.setattr(bundle, "REAL_CARRY_PATH", tmp_path / "carry.csv")
    monkeypatch.setattr(bundle, "REAL_CARRY_MANIFEST_PATH", tmp_path / "manifest.json")
    request = CarryBundleUploadRequest(
        source="reviewed-test",
        data_version="test-v1",
        continuous_contract_rule="declared-near-far",
        rows=_rows(),
    )

    response = upload_real_carry(request)

    assert response["factor_name"] == "term_structure_carry"
    assert response["bundle_version"] == "cta_factor_bundle_v1"
    assert response["version_id"].startswith("cta_factor_bundle_v1:test-v1:")


def test_factor_bundle_upload_maps_contract_error_to_422(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(bundle, "CTA_FACTOR_DIRECTORY", tmp_path)
    monkeypatch.setattr(bundle, "REAL_CARRY_PATH", tmp_path / "carry.csv")
    monkeypatch.setattr(bundle, "REAL_CARRY_MANIFEST_PATH", tmp_path / "manifest.json")
    rows = _rows()
    rows.append(dict(rows[0]))
    request = CarryBundleUploadRequest(
        source="reviewed-test",
        data_version="duplicate-v1",
        continuous_contract_rule="declared-near-far",
        rows=rows,
    )

    with pytest.raises(HTTPException) as error:
        upload_real_carry(request)

    assert error.value.status_code == 422
