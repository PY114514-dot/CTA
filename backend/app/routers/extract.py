"""Chart extraction router — PDF/image → structured NAV data.

Two modes:
1. AUTO: upload PDF/image, VLM reads chart structure, CV traces pixels
2. MANUAL: user specifies curve colors (hex or click anchors) + axis anchors,
   no VLM needed (zero cost)

Endpoints:
  POST /api/extract/pdf             Upload PDF, start pipeline
  POST /api/extract/single-image    Extract from one chart image (auto or manual)
  POST /api/extract/sample-color    Sample color at a clicked pixel
  GET  /api/extract/status/{id}     Poll job progress
  GET  /api/extract/preview/{id}    Get results for review
  GET  /api/extract/jobs            List jobs
  GET  /api/config/vlm              Get VLM config
  PUT  /api/config/vlm              Update VLM config
"""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import persist_local_environment
from app.services.chart_extractor import (
    extract_single_image,
    get_job,
    list_jobs,
    run_extraction,
    sample_color_at_pixel,
)
from app.services.chart_extractor.vlm_extractor import DashScopeProvider, OpenAICompatibleProvider

logger = logging.getLogger(__name__)

router = APIRouter(tags=["图表提取"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class VLMConfig(BaseModel):
    provider: str = "dashscope"
    model: str = "qwen3-vl-flash"
    api_key_set: bool = False
    base_url: str = ""


class VLMConfigUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


class ExtractResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    total_pages: int
    message: str


class CurveSpecInput(BaseModel):
    """User-provided curve specification."""

    name: str = ""
    color_hex: str = ""
    color_name: str = ""
    is_benchmark: bool = False
    anchor_px: list[int] | None = None  # [x, y] clicked point on the curve


class AxisAnchorInput(BaseModel):
    """User-provided axis anchor (pixel ↔ value)."""

    px: int
    value: float | None = None
    label: str = ""


class SingleImageRequest(BaseModel):
    """Manual extraction parameters for single-image endpoint."""

    curves: list[CurveSpecInput] | None = None
    y_anchors: list[AxisAnchorInput] | None = None
    x_anchors: list[AxisAnchorInput] | None = None
    x_labels: list[str] | None = None
    use_vlm: bool = True


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


@router.post("/api/extract/pdf", response_model=ExtractResponse)
async def extract_pdf(
    file: UploadFile = File(...),
    max_pages: int | None = None,
    skip_detection: bool = False,
    use_vlm: bool = True,
    curve_color_hex: str | None = None,
    curve_color_name: str | None = None,
) -> ExtractResponse:
    """Upload a PDF and start the chart extraction pipeline.

    If curve_color_hex/name is provided, VLM is skipped (manual mode).
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    if content[:5] != b"%PDF-":
        raise HTTPException(400, "Not a PDF. Use /api/extract/single-image for images.")

    # Build manual curve specs if color provided
    curve_specs = None
    if curve_color_hex or curve_color_name:
        curve_specs = [{
            "name": "产品净值",
            "color_hex": curve_color_hex or "",
            "color_name": curve_color_name or "",
            "is_benchmark": False,
        }]
        use_vlm = False
    elif use_vlm:
        # Check VLM availability
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key and os.getenv("VLM_PROVIDER", "dashscope") == "dashscope":
            raise HTTPException(
                503,
                "VLM not configured. Set DASHSCOPE_API_KEY, or provide "
                "curve_color_hex for manual (VLM-free) extraction.",
            )

    job = await run_extraction(
        pdf_bytes=content,
        filename=file.filename,
        max_pages=max_pages,
        skip_detection=skip_detection,
        use_vlm=use_vlm,
        curve_specs=curve_specs,
    )

    n_curves = sum(len(r.curves) for r in job.results)
    return ExtractResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status.value,
        total_pages=job.total_pages,
        message=f"{'Completed' if job.status.value == 'completed' else 'Failed'}. "
        f"Charts on {len(job.pages_with_charts)} pages, {n_curves} curves traced.",
    )


# ---------------------------------------------------------------------------
# Single image extraction (auto or manual)
# ---------------------------------------------------------------------------


@router.post("/api/extract/single-image")
async def extract_single(
    file: UploadFile = File(...),
    curves: str | None = Form(None),  # JSON string of CurveSpecInput list
    y_anchors: str | None = Form(None),  # JSON string
    x_anchors: str | None = Form(None),  # JSON string
    x_labels: str | None = Form(None),  # JSON string (comma-separated dates)
    y_ticks: str | None = Form(None),  # JSON array or comma-separated tick labels (bottom→top)
    use_vlm: bool = Form(True),
) -> dict[str, Any]:
    """Extract chart data from a single image.

    Manual mode: pass curves/y_anchors/x_anchors as JSON form fields.
    Auto mode: leave them empty and set use_vlm=true.
    """
    import json

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    if content[:5] == b"%PDF-":
        raise HTTPException(400, "This is a PDF. Use /api/extract/pdf.")

    # Parse optional JSON fields
    curve_specs = None
    if curves:
        try:
            curve_specs = json.loads(curves)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid 'curves' JSON")

    y_anchor_list = None
    if y_anchors:
        try:
            y_anchor_list = json.loads(y_anchors)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid 'y_anchors' JSON")

    x_anchor_list = None
    if x_anchors:
        try:
            x_anchor_list = json.loads(x_anchors)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid 'x_anchors' JSON")

    x_label_list = None
    if x_labels:
        try:
            x_label_list = json.loads(x_labels)
        except json.JSONDecodeError:
            # Treat as comma-separated
            x_label_list = [s.strip() for s in x_labels.split(",") if s.strip()]

    y_tick_labels = None
    if y_ticks:
        try:
            parsed_ticks = json.loads(y_ticks)
            if isinstance(parsed_ticks, list):
                y_tick_labels = [str(t).strip() for t in parsed_ticks if str(t).strip()]
            else:
                y_tick_labels = [s.strip() for s in str(parsed_ticks).split(",") if s.strip()]
        except json.JSONDecodeError:
            y_tick_labels = [s.strip() for s in y_ticks.split(",") if s.strip()]

    # If manual curves provided, skip VLM
    if curve_specs:
        use_vlm = False
    elif use_vlm:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key and os.getenv("VLM_PROVIDER", "dashscope") == "dashscope":
            raise HTTPException(
                503,
                "VLM not configured. Provide 'curves' for manual extraction, "
                "or set DASHSCOPE_API_KEY.",
            )

    try:
        result = await extract_single_image(
            content,
            curve_specs=curve_specs,
            y_anchors=y_anchor_list,
            x_anchors=x_anchor_list,
            x_labels=x_label_list,
            y_tick_labels=y_tick_labels,
            use_vlm=use_vlm,
        )
        return {
            "status": "completed" if not result.error else "partial",
            "result": result.model_dump(),
        }
    except Exception as e:
        logger.exception("Single image extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")


# ---------------------------------------------------------------------------
# Color sampling (click-to-pick)
# ---------------------------------------------------------------------------


@router.post("/api/extract/sample-color")
async def sample_color(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    radius: int = Form(3),
) -> dict[str, Any]:
    """Sample the dominant color at a clicked pixel position.

    Frontend workflow: user clicks on a curve in the preview image,
    this endpoint returns the hex color to use for tracing.
    """
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")

    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise HTTPException(400, f"Click position ({x},{y}) out of image bounds ({w}x{h})")

    color_hex = sample_color_at_pixel(img, x, y, radius)
    return {"x": x, "y": y, "color_hex": color_hex}


# ---------------------------------------------------------------------------
# Job status & preview
# ---------------------------------------------------------------------------


@router.get("/api/extract/status/{job_id}")
async def extract_status(job_id: str) -> dict[str, Any]:
    """Poll extraction job progress."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")

    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "total_pages": job.total_pages,
        "pages_with_charts": job.pages_with_charts,
        "num_results": len(job.results),
        "error": job.error,
    }


@router.get("/api/extract/preview/{job_id}")
async def extract_preview(job_id: str) -> dict[str, Any]:
    """Get full extraction results for user review and confirmation."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")

    if job.status.value not in ("completed", "failed"):
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": job.progress,
            "message": "Job still in progress",
        }

    return {
        "job_id": job_id,
        "filename": job.filename,
        "status": job.status.value,
        "total_pages": job.total_pages,
        "pages_with_charts": job.pages_with_charts,
        "regions": [r.model_dump() for r in job.regions],
        "results": [r.model_dump() for r in job.results],
        "error": job.error,
    }


@router.get("/api/extract/preview/{job_id}/image/{page_index}/{region_index}")
async def extract_region_image(job_id: str, page_index: int, region_index: int):
    """Serve a saved crop image for frontend display."""
    from app.services.chart_extractor.pipeline import EXTRACT_WORK_DIR

    crop_path = EXTRACT_WORK_DIR / job_id / f"p{page_index}_r{region_index}.png"
    if not crop_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(crop_path), media_type="image/png")


@router.get("/api/extract/jobs")
async def list_extract_jobs() -> list[dict[str, Any]]:
    """List all extraction jobs (most recent first)."""
    jobs = list_jobs()
    return [
        {
            "job_id": j.job_id,
            "filename": j.filename,
            "status": j.status.value,
            "total_pages": j.total_pages,
            "num_results": len(j.results),
            "progress": j.progress,
        }
        for j in jobs[:20]
    ]


# ---------------------------------------------------------------------------
# VLM Configuration
# ---------------------------------------------------------------------------


@router.get("/api/config/vlm", response_model=VLMConfig)
async def get_vlm_config() -> VLMConfig:
    """Get current VLM configuration."""
    return VLMConfig(
        provider=os.getenv("VLM_PROVIDER", "dashscope"),
        model=os.getenv("VLM_MODEL", "qwen3-vl-flash"),
        api_key_set=bool(os.getenv("DASHSCOPE_API_KEY", "")),
        base_url=os.getenv("VLM_BASE_URL", ""),
    )


@router.put("/api/config/vlm")
async def update_vlm_config(config: VLMConfigUpdate) -> dict[str, str]:
    """Update VLM configuration for the running process."""
    if config.provider is not None:
        os.environ["VLM_PROVIDER"] = config.provider
    if config.model is not None:
        os.environ["VLM_MODEL"] = config.model
    if config.api_key is not None:
        os.environ["DASHSCOPE_API_KEY"] = config.api_key
    if config.base_url is not None:
        os.environ["VLM_BASE_URL"] = config.base_url
    persist_local_environment({
        key: value
        for key, value in {
            "VLM_PROVIDER": config.provider,
            "VLM_MODEL": config.model,
            "DASHSCOPE_API_KEY": config.api_key,
            "VLM_BASE_URL": config.base_url,
        }.items()
        if value is not None
    })

    return {"status": "ok", "message": "VLM configuration updated"}


@router.post("/api/config/vlm/test")
async def test_vlm_config() -> dict[str, object]:
    """Verify the configured VLM with a real, tiny image request."""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    provider_name = os.getenv("VLM_PROVIDER", "dashscope")
    model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
    base_url = os.getenv("VLM_BASE_URL", "")
    if not api_key and provider_name == "dashscope":
        raise HTTPException(400, "请先保存 DashScope API Key。")
    if provider_name != "dashscope" and not base_url:
        raise HTTPException(400, "自定义 VLM 请先填写 Base URL。")

    # Use a real 32×32 PNG. Qwen rejects images whose width or height is
    # smaller than 11 px, so a 1×1 connectivity probe always returns 400.
    # A successful response proves the endpoint, credential and image-input
    # path—not the accuracy of chart recognition.
    probe = np.full((32, 32, 3), 255, dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", probe)
    if not encoded_ok:
        raise HTTPException(500, "无法生成视觉模型连通性测试图片。")
    image = encoded.tobytes()
    provider = DashScopeProvider(api_key=api_key, model=model) if provider_name == "dashscope" else OpenAICompatibleProvider(base_url=base_url, api_key=api_key or "not-needed", model=model)
    started = perf_counter()
    try:
        response = await provider.extract_structure(image, "请只回复 OK。")
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:300] or error.response.reason_phrase
        raise HTTPException(error.response.status_code, f"视觉模型返回错误：{detail}") from error
    except Exception as error:
        raise HTTPException(502, f"无法连接视觉模型：{error}") from error
    return {
        "ok": True,
        "model": model,
        "latency_ms": round((perf_counter() - started) * 1000),
        "detail": f"图像请求已由模型响应：{str(response).strip()[:80] or '（空响应）'}",
    }
