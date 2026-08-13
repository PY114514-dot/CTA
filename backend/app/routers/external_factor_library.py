"""External CTA factor library — Guotai Junan weekly report benchmark."""

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.services.external_factor_library import (
    ExternalFactorLibraryError,
    content_disposition,
    export_latest_csv,
    export_latest_json,
    export_latest_xlsx,
    extract_pdf_text,
    ingest_external_document,
    ingest_gtja_document,
    latest_observations,
    latest_snapshot_date,
    verify_snapshot,
)
from app.services.paddleocr_service import PaddleOcrServiceError, extract_document

router = APIRouter(tags=["外部因子库"])


@router.post("/api/external-factor-library/guotai-junan/import")
async def import_guotai_junan_factor_report(file: UploadFile = File(...)) -> dict:
    """Import one weekly PDF/image; preserves source and creates a revision."""
    filename = file.filename or "source"
    content = await file.read()
    try:
        if filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
            text = await run_in_threadpool(extract_pdf_text, content)
            method = "pdf_text"
        else:
            parsed = await run_in_threadpool(extract_document, content, filename)
            text = "\n".join(page.markdown for page in parsed.pages)
            method = "paddleocr"
        return await run_in_threadpool(ingest_gtja_document, content, filename, text, method)
    except (ExternalFactorLibraryError, PaddleOcrServiceError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/external-factor-library/import")
async def import_external_factor_report(
    file: UploadFile = File(...),
    organization_name: str | None = Form(None),
) -> dict:
    """Import an external CTA report, auto-detecting issuer unless overridden."""
    filename, content = file.filename or "source", await file.read()
    try:
        if filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
            text, method = await run_in_threadpool(extract_pdf_text, content), "pdf_text"
        else:
            parsed = await run_in_threadpool(extract_document, content, filename)
            text, method = "\n".join(page.markdown for page in parsed.pages), "paddleocr"
        return await run_in_threadpool(ingest_external_document, content, filename, text, method, organization_name)
    except (ExternalFactorLibraryError, PaddleOcrServiceError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/api/external-factor-library/guotai-junan/observations")
def get_guotai_junan_factor_observations() -> dict:
    """Return latest (non-superseded) observations; raw revisions remain stored."""
    return {
        "source": "guotai_junan_cta",
        "latest_as_of_date": latest_snapshot_date(),
        "observations": latest_observations(),
    }


@router.post("/api/external-factor-library/guotai-junan/verify")
def verify_guotai_junan_snapshot(request: dict) -> dict:
    """Explicitly confirm the selected external report against its original source."""
    as_of_date = request.get("as_of_date")
    if not isinstance(as_of_date, str):
        raise HTTPException(status_code=422, detail="需要提供 as_of_date")
    try:
        changed = verify_snapshot(as_of_date, source=request.get("source"), verified_by=str(request.get("verified_by", "user")))
        return {"as_of_date": as_of_date, "verified_rows": changed}
    except ExternalFactorLibraryError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/api/external-factor-library/guotai-junan/export/{format_name}")
def export_guotai_junan_factor_library(format_name: Literal["csv", "json", "xlsx"]) -> Response:
    """Download latest non-superseded factor observations."""
    if format_name == "csv":
        as_of_date, content = export_latest_csv()
        return Response(content, media_type="text/csv", headers={"Content-Disposition": content_disposition("csv", as_of_date)})
    if format_name == "xlsx":
        as_of_date, content = export_latest_xlsx()
        return Response(content, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": content_disposition("xlsx", as_of_date)})
    as_of_date, content = export_latest_json()
    return Response(content, media_type="application/json", headers={"Content-Disposition": content_disposition("json", as_of_date)})
