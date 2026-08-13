"""Market data endpoints: symbol listing, provider status, CSV upload."""

from datetime import date

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.dependencies import akshare_provider, csv_provider
from app.schemas import (
    SymbolListResponse,
    SymbolInfo,
    MarketDataStatusResponse,
    CsvUploadResponse,
)
from app.services.market_data import get_all_symbols, get_all_sectors, SECTOR_COLORS

router = APIRouter(tags=["行情数据"])


@router.get("/api/market-data/symbols", response_model=SymbolListResponse)
def list_market_symbols() -> SymbolListResponse:
    """Return all available indices and futures varieties for analysis."""
    raw_symbols = get_all_symbols()
    symbols = [SymbolInfo(**item) for item in raw_symbols]
    return SymbolListResponse(
        symbols=symbols,
        sectors=get_all_sectors(),
        sector_colors=SECTOR_COLORS,
    )


@router.get("/api/market-data/status", response_model=MarketDataStatusResponse)
def market_data_status() -> MarketDataStatusResponse:
    """Check whether the online API is reachable and what CSV data is loaded."""
    return MarketDataStatusResponse(
        api_available=akshare_provider.is_available(),
        api_provider="akshare",
        uploaded_symbols=csv_provider.loaded_symbols(),
    )


@router.post("/api/market-data/upload", response_model=CsvUploadResponse)
async def upload_market_csv(
    symbol: str = Form(...),
    file: UploadFile = File(...),
) -> CsvUploadResponse:
    """Upload a CSV of daily OHLCV data for a specific symbol."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="仅支持 CSV 文件")
    try:
        content = await file.read()
        rows = csv_provider.load_csv(symbol, content)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    df = csv_provider.get_futures_daily(symbol, date.min, date.max)
    date_range = None
    if not df.empty:
        date_range = (df["date"].iloc[0], df["date"].iloc[-1])

    return CsvUploadResponse(
        symbol=symbol,
        rows_loaded=rows,
        date_range=date_range,
        warnings=[] if rows > 0 else ["未解析到有效数据行"],
    )
