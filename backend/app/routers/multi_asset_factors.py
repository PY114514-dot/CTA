"""Reviewed uploads for non-CTA attribution factor contracts."""

from datetime import date

import csv
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.multi_asset_factor_contracts import (
    CONTRACTS,
    MultiAssetFactorContractError,
    contract_metadata,
    load_factor_rows,
    save_factor_rows,
)
from app.services.akshare_equity_factor_import import build_equity_factor_rows
from app.services.market_data.akshare_provider import AKShareProvider

router = APIRouter(prefix="/api/attribution-factors", tags=["多资产因子合同"])


class FactorObservation(BaseModel):
    observation_date: date
    available_date: date
    factor_name: str = Field(min_length=1, max_length=80)
    value: float


class FactorContractUpload(BaseModel):
    source: str = Field(min_length=1, max_length=160)
    data_version: str = Field(min_length=1, max_length=160)
    rows: list[FactorObservation] = Field(min_length=1, max_length=500000)


class AkShareEquityImportRequest(BaseModel):
    start: date
    end: date


@router.get("/{pool}")
def get_contract(pool: str, as_of_date: date | None = None) -> dict:
    if pool not in CONTRACTS:
        raise HTTPException(404, "未知因子合同")
    return contract_metadata(pool, as_of_date=as_of_date)


@router.post("/{pool}")
def upload_contract(pool: str, request: FactorContractUpload) -> dict:
    try:
        return save_factor_rows(pool, request.rows, source=request.source, data_version=request.data_version)
    except MultiAssetFactorContractError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/{pool}/csv")
async def upload_contract_csv(
    pool: str,
    file: UploadFile = File(...),
    source: str = Form(...),
    data_version: str = Form(...),
) -> dict:
    """Upload a reviewed factor CSV instead of manually assembling JSON."""
    if pool not in CONTRACTS:
        raise HTTPException(404, "未知因子合同")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "仅支持 CSV 文件")
    try:
        text = (await file.read()).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        required = {"observation_date", "available_date", "factor_name", "value"}
        if not rows or not required.issubset(rows[0]):
            raise MultiAssetFactorContractError("CSV 必须包含 observation_date、available_date、factor_name、value 四列")
        return save_factor_rows(pool, rows, source=source, data_version=data_version)
    except UnicodeDecodeError as error:
        raise HTTPException(422, "CSV 必须使用 UTF-8 编码") from error
    except MultiAssetFactorContractError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/equity_quant/akshare")
def import_equity_subset_from_akshare(request: AkShareEquityImportRequest) -> dict:
    """Fetch the supported equity market proxies; never invent value/quality."""
    try:
        if load_factor_rows("equity_quant"):
            raise MultiAssetFactorContractError("已有股票因子数据；为避免覆盖已审核版本，请通过 CSV 明确替换")
        rows = build_equity_factor_rows(AKShareProvider(), request.start, request.end)
        return save_factor_rows(
            "equity_quant", rows,
            source="AKShare：沪深300与中证1000日线",
            data_version=f"akshare-equity-{request.start.isoformat()}-{request.end.isoformat()}",
        )
    except (ValueError, MultiAssetFactorContractError) as error:
        raise HTTPException(422, str(error)) from error
