"""NAV analysis, image digitization, OCR, product recognition, and document parsing."""

from datetime import date
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas import (
    NavAnalysisRequest,
    NavAnalysisResponse,
    NavImageDigitizationResponse,
    DataFrequency,
    DigitizationAuditReport,
    DigitizationAuditRequest,
    OcrRegionResponse,
    PaddleOcrDocumentResponse,
    ProductImageRecognitionResponse,
    ProductStrategyProfileRequest,
    ProductStrategyProfileResponse,
    MultiProductReportResponse,
    StrategyFingerprintRequest,
    StrategyFingerprintResponse,
)
from app.services.nav_metrics import calculate_nav_analysis
from app.services.nav_image_digitizer import digitize_nav_image
from app.services.digitization_audit import audit_digitized_nav
from app.services.multi_product_report import extract_multi_product_report
from app.services.product_strategy_profile import build_strategy_profile, recognize_product_name
from app.services.image_ocr import ocr_image_region
from app.services.paddleocr_service import PaddleOcrServiceError, extract_document
from app.services.strategy_fingerprint import classify_strategy_fingerprint

router = APIRouter(tags=["净值与图片"])

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/api/nav/analyze", response_model=NavAnalysisResponse)
def analyze_nav(request: NavAnalysisRequest) -> NavAnalysisResponse:
    """Calculate performance metrics from a reviewed NAV history."""
    try:
        return calculate_nav_analysis(request)
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/nav/digitize-audit", response_model=DigitizationAuditReport)
def audit_nav_digitization(request: DigitizationAuditRequest) -> DigitizationAuditReport:
    """把数字化候选净值与人工核对基准逐日对齐，给出误差审计报告。

    数字化序列是候选值，基准序列视为真值；判定只用于提示是否可采纳，
    绝不改写任何净值观测。匹配样本不足时直接判 fail。
    """
    report = audit_digitized_nav(
        request.digitized_points,
        request.reference_points,
        relative_tolerance=request.relative_tolerance,
        min_matched=request.min_matched,
    )
    return DigitizationAuditReport.model_validate(report)


@router.post("/api/nav/digitize-image", response_model=NavImageDigitizationResponse)
async def digitize_image_nav_chart(
    image: UploadFile = File(...),
    start_date: date = Form(...),
    end_date: date = Form(...),
    nav_min: float = Form(...),
    nav_max: float = Form(...),
    value_mode: Literal["nav", "cumulative_return"] = Form(...),
    # ``auto`` is accepted for chart images only.  It is deliberately kept
    # separate from DataFrequency: analysis itself must always use a concrete
    # disclosure frequency.
    frequency: Literal["auto", "daily", "weekly", "monthly"] = Form("auto"),
    start_x_ratio: float | None = Form(None),
    end_x_ratio: float | None = Form(None),
    line_kind: Literal["product", "benchmark"] = Form("product"),
    line_color: str | None = Form(None),
    top_y_ratio: float | None = Form(None),
    bottom_y_ratio: float | None = Form(None),
) -> NavImageDigitizationResponse:
    """Extract reviewable NAV candidates from an uploaded chart image."""
    if image.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 PNG、JPG/JPEG 或 WebP 图片")
    try:
        return digitize_nav_image(
            await image.read(), start_date, end_date, nav_min, nav_max,
            value_mode, frequency, start_x_ratio, end_x_ratio, line_kind,
            top_y_ratio, bottom_y_ratio, line_color,
        )
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/product/recognize-image", response_model=ProductImageRecognitionResponse)
async def recognize_product_from_image(image: UploadFile = File(...)) -> ProductImageRecognitionResponse:
    """Extract a reviewable product-name candidate from an uploaded image."""
    if image.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 PNG、JPG/JPEG 或 WebP 图片")
    try:
        return recognize_product_name(await image.read())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/report/extract-multiple", response_model=MultiProductReportResponse)
async def extract_multiple_products_from_report(image: UploadFile = File(...)) -> MultiProductReportResponse:
    """Locate product curves and disclosed performance rows in a weekly report."""
    if image.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 PNG、JPG/JPEG 或 WebP 图片")
    try:
        return extract_multi_product_report(await image.read())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/image/ocr-region", response_model=OcrRegionResponse)
async def read_image_region(
    image: UploadFile = File(...),
    left_ratio: float = Form(...),
    top_ratio: float = Form(...),
    right_ratio: float = Form(...),
    bottom_ratio: float = Form(...),
) -> OcrRegionResponse:
    """OCR a normalized rectangular region of an uploaded image."""
    if image.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 PNG、JPG/JPEG 或 WebP 图片")
    try:
        text = ocr_image_region(await image.read(), left_ratio, top_ratio, right_ratio, bottom_ratio)
        return OcrRegionResponse(text=text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/document/paddleocr", response_model=PaddleOcrDocumentResponse)
async def parse_document_with_paddleocr(document: UploadFile = File(...)) -> PaddleOcrDocumentResponse:
    """Extract reviewable Markdown from a PDF or image with PaddleOCR-VL."""
    allowed_types = {"application/pdf", "image/jpeg", "image/png", "image/webp"}
    if document.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="仅支持 PDF、PNG、JPG/JPEG 或 WebP 文件")
    try:
        return await run_in_threadpool(
            extract_document, await document.read(), document.filename or "document"
        )
    except PaddleOcrServiceError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


# ---------------------------------------------------------------------------
# Product strategy profile & fingerprint
# ---------------------------------------------------------------------------


@router.post("/api/product/strategy-profile", response_model=ProductStrategyProfileResponse)
def create_product_strategy_profile(request: ProductStrategyProfileRequest) -> ProductStrategyProfileResponse:
    """Build a transparent keyword-evidence strategy profile."""
    return build_strategy_profile(request)


@router.post("/api/strategy/fingerprint", response_model=StrategyFingerprintResponse)
def create_strategy_fingerprint(request: StrategyFingerprintRequest) -> StrategyFingerprintResponse:
    """Create an explainable initial CTA factor fingerprint."""
    try:
        return classify_strategy_fingerprint(request)
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
