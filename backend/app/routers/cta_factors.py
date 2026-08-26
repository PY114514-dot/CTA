"""Versioned CTA factor-contract and reviewed Carry data endpoints."""

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.cta_factor_bundle import (
    CtaFactorContractError,
    get_cta_factor_bundle,
    save_real_carry_rows,
)
from app.services.akshare_carry_import import assert_carry_coverage, import_shfe_carry

router = APIRouter(prefix="/api/cta-factors", tags=["CTA 因子合同"])


class RealCarryInput(BaseModel):
    observation_date: date
    symbol: str = Field(min_length=1, max_length=80)
    near_contract: str = Field(min_length=1, max_length=80)
    far_contract: str = Field(min_length=1, max_length=80)
    near_price: float = Field(gt=0)
    far_price: float = Field(gt=0)
    near_next_price: float = Field(gt=0)
    far_next_price: float = Field(gt=0)
    near_days_to_expiry: float = Field(gt=0)
    far_days_to_expiry: float = Field(gt=0)
    holding_return: float = Field(ge=-1)
    available_date: date


class CarryBundleUploadRequest(BaseModel):
    source: str = Field(min_length=1, max_length=160)
    data_version: str = Field(min_length=1, max_length=160)
    continuous_contract_rule: str = Field(min_length=1, max_length=160)
    rows: list[RealCarryInput] = Field(min_length=1, max_length=200000)


class AkShareCarryImportRequest(BaseModel):
    start: date
    end: date


@router.get("/bundle")
def get_factor_bundle(as_of_date: date | None = None) -> dict:
    """Return the active factor contract and point-in-time coverage."""
    return get_cta_factor_bundle(as_of_date=as_of_date)


@router.post("/carry")
def upload_real_carry(request: CarryBundleUploadRequest) -> dict:
    """Replace the reviewed Carry input with one declared data version."""
    try:
        return save_real_carry_rows(
            request.rows,
            source=request.source,
            data_version=request.data_version,
            continuous_contract_rule=request.continuous_contract_rule,
        )
    except CtaFactorContractError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/carry/akshare")
def import_real_carry_from_akshare(request: AkShareCarryImportRequest) -> dict:
    """Replace Carry data with exchange-verifiable SHFE observations."""
    try:
        rows = import_shfe_carry(request.start, request.end)
        assert_carry_coverage(rows, request.start, request.end)
        return save_real_carry_rows(
            rows,
            source="AKShare / SHFE 合约信息 + Sina 指定合约日线",
            data_version=f"akshare-shfe-weekly-{request.start.isoformat()}-{request.end.isoformat()}",
            continuous_contract_rule="SHFE published expiry; named near/far contracts; no main-continuous carry proxy",
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except CtaFactorContractError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
