"""Chart extraction pipeline orchestrator.

Two modes:
1. AUTO: PDF render → chart detect → VLM structure → pixel trace → calibrate
2. MANUAL: user provides curve colors (hex or click anchors) + axis anchors,
   skips VLM entirely (zero cost, fully deterministic)

VLM is optional — it only reads chart STRUCTURE (colors, tick labels,
plot area).  All numerical extraction is done by pixel_tracer.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import DATA_DIRECTORY

from .chart_detector import ChartRegion, crop_region, detect_chart_regions
from .models import (
    AxisAnchor,
    ChartExtractResult,
    ChartStructure,
    ExtractJob,
    ExtractJobStatus,
    PageChartRegion,
    PlotArea,
    TracedCurve,
)
from .pdf_renderer import render_pdf_pages
from .pixel_tracer import (
    TraceConfig,
    detect_plot_area,
    refine_colors_kmeans,
    sample_color_at_pixel,
    trace_all_curves,
)
from .preprocessor import preprocess_chart_image
from .tick_locator import locate_x_ticks, locate_y_ticks
from .vlm_extractor import (
    VLMProvider,
    Y_AXIS_PROMPT,
    create_provider,
    parse_structure_response,
    parse_y_axis_response,
)

logger = logging.getLogger(__name__)

# Job store (in-memory, same pattern as v1 BuildJobStore)
_jobs: dict[str, ExtractJob] = {}

EXTRACT_WORK_DIR = DATA_DIRECTORY / "extract_jobs"


def get_job(job_id: str) -> ExtractJob | None:
    return _jobs.get(job_id)


def list_jobs() -> list[ExtractJob]:
    return sorted(_jobs.values(), key=lambda j: j.job_id, reverse=True)


# ---------------------------------------------------------------------------
# Core: extract one chart image
# ---------------------------------------------------------------------------


async def extract_chart(
    image_bytes: bytes,
    curve_specs: list[dict] | None = None,
    y_anchors: list[dict] | None = None,
    x_anchors: list[dict] | None = None,
    x_labels: list[str] | None = None,
    y_tick_labels: list[str] | None = None,
    provider: VLMProvider | None = None,
    use_vlm: bool = True,
    config: TraceConfig | None = None,
    structure_override: ChartStructure | dict | None = None,
) -> ChartExtractResult:
    """Extract data from a single chart image.

    Parameters
    ----------
    image_bytes : PNG/JPEG of the chart (cropped region or full page)
    curve_specs : manual curve specifications. Each dict:
        {"name": "...", "color_hex": "#FF0000", "color_name": "red",
         "is_benchmark": false, "anchor_px": [x, y]}
        If anchor_px given without color_hex, color is sampled at that pixel.
    y_anchors : manual Y axis anchors: [{"px": 100, "value": 1.5, "label": "1.5"}]
    x_anchors : manual X axis tick pixel positions: [{"px": 50, "label": "2024-01"}]
    x_labels : X axis date labels (paired with x_anchors by order)
    provider : VLM provider (created from config if None)
    use_vlm : whether to use VLM for structure when curve_specs is incomplete
    config : tracing configuration
    structure_override : previously captured VLM structure to replay.  This
        skips the VLM call and makes CV strategy comparisons deterministic.

    Returns
    -------
    ChartExtractResult with traced + calibrated curves.
    """
    # Decode image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ChartExtractResult(error="Cannot decode image")

    structure: ChartStructure | None = None
    vlm_attempted = False
    vlm_succeeded = False
    vlm_provider = None
    vlm_model = None
    vlm_error = None
    if structure_override is not None:
        structure = (
            structure_override
            if isinstance(structure_override, ChartStructure)
            else ChartStructure.model_validate(structure_override)
        )
        curve_specs = curve_specs or structure.curves
        x_labels = x_labels or structure.x_ticks
        logger.info("Replaying captured VLM structure: %d curves, %d x_ticks", len(curve_specs), len(structure.x_ticks))

    # --- Step 1: Get chart structure ---
    if structure is None and use_vlm:
        # VLM reads the axes/plot frame even when a trusted colour hint came
        # from the previous layout stage.  Earlier code made these paths
        # mutually exclusive, so keeping a good colour silently disabled axis
        # calibration and asking for axes silently discarded that colour.
        trusted_curve_specs = list(curve_specs or [])
        if use_vlm:
            vlm_attempted = True
            vlm_provider = os.getenv("VLM_PROVIDER", "dashscope")
            vlm_model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
            try:
                if provider is None:
                    provider = _default_provider()
                enhanced = preprocess_chart_image(image_bytes)
                raw = await provider.extract_structure(enhanced)
                structure = parse_structure_response(raw)
                vlm_succeeded = True
                curve_specs = trusted_curve_specs or structure.curves
                x_labels = x_labels or structure.x_ticks
                logger.info("VLM structure: %d curves, %d x_ticks, %d y_ticks",
                            len(curve_specs), len(structure.x_ticks), len(structure.y_ticks))
            except Exception as e:
                vlm_error = str(e)[:300]
                logger.warning("VLM structure extraction failed: %s", e)
                fallback_area = detect_plot_area(img)
                local_y_anchors = _read_y_axis_ocr(img, fallback_area)
                fallback_structure = _structure_with_y_anchors(None, local_y_anchors)
                return ChartExtractResult(
                    structure=fallback_structure,
                    plot_area=fallback_area,
                    plot_area_source="cv",
                    warnings=[
                        "VLM 调用失败，已保留 CV 候选图框，并用本地 OCR 读取纵轴；请人工取色与校准日期首尾。"
                        if local_y_anchors
                        else "VLM 调用失败，已保留 CV 候选图框供人工取色与日期首尾校准；纵轴未识别时需在高级校准中填写。"
                    ],
                    needs_color_pick=True,
                    error=None,
                    vlm_attempted=vlm_attempted,
                    vlm_succeeded=vlm_succeeded,
                    vlm_provider=vlm_provider,
                    vlm_model=vlm_model,
                    vlm_error=vlm_error,
                )
    elif structure is None and not curve_specs:
        return ChartExtractResult(
            error="No curve specs provided and VLM disabled. "
            "Please specify curve colors manually.",
        )

    if not curve_specs:
        # Keep a successful VLM location available to the browser.  The
        # recovery UI can then ask for only a colour sample and the two date
        # endpoints instead of making the user redraw the whole curve.  Some
        # VLMs identify the chart semantically but omit the numeric bbox; in
        # that case expose the CV frame as a reviewable fallback rather than
        # returning no frame at all.
        fallback_area, fallback_source = _choose_plot_area(img, detect_plot_area(img), structure)
        y_anchors_for_review: list[AxisAnchor] = []
        if structure is not None and len(structure.y_range) != 2:
            if provider is not None:
                values = await _read_y_axis_focused(img, fallback_area, provider)
                if len(values) >= 2:
                    y_anchors_for_review = _anchors_from_values(values, fallback_area, axis="y")
            if not y_anchors_for_review:
                y_anchors_for_review = _read_y_axis_ocr(img, fallback_area)
            structure = _structure_with_y_anchors(structure, y_anchors_for_review)
        return ChartExtractResult(
            structure=structure,
            plot_area=fallback_area,
            plot_area_source=fallback_source,
            warnings=[
                "VLM 已定位图框，但未可靠识别产品线颜色；请直接点击产品曲线取色。"
                if fallback_source == "vlm"
                else "VLM 未提供可靠图框，已保留 CV 候选图框；请直接点击产品曲线取色。"
            ],
            needs_color_pick=True,
            vlm_attempted=vlm_attempted,
            vlm_succeeded=vlm_succeeded,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            vlm_error=vlm_error,
        )

    # --- Step 2: Resolve anchor pixel colors ---
    for spec in curve_specs:
        anchor = spec.get("anchor_px")
        if anchor and not spec.get("color_hex"):
            ax, ay = int(anchor[0]), int(anchor[1])
            spec["color_hex"] = sample_color_at_pixel(img, ax, ay)
            logger.info("Sampled color %s at (%d,%d)", spec["color_hex"], ax, ay)

    # --- Step 3: Locate the drawable plot area ---
    # CV remains the precise default.  VLM supplies a semantic rectangle that
    # prevents titles, legends and performance tables from being mistaken for
    # the NAV curve when it agrees with the visual detector.
    cv_plot_area = detect_plot_area(img)
    plot_area, plot_area_source = _choose_plot_area(img, cv_plot_area, structure)

    # --- Step 3b: Refine VLM colors via K-means pixel clustering ---
    if structure is not None and curve_specs:
        curve_specs = refine_colors_kmeans(img, plot_area, curve_specs)

    # --- Step 4: Build axis anchors (manual override OR auto-locate) ---
    y_axis_anchors = _build_axis_anchors(y_anchors, "y")
    x_axis_anchors = _build_axis_anchors(x_anchors, "x")

    # Auto-locate tick pixel positions from VLM labels when no manual anchors
    if not y_axis_anchors and structure and structure.y_ticks:
        y_axis_anchors = locate_y_ticks(img, plot_area, structure.y_ticks)
        logger.info("Auto-located %d Y tick anchors", len(y_axis_anchors))

    # Zero-cost manual path: user typed the Y tick labels (bottom→top).
    # Reuse locate_y_ticks to find their pixel positions via grid lines /
    # even spacing — no VLM required.
    if not y_axis_anchors and y_tick_labels:
        y_axis_anchors = locate_y_ticks(img, plot_area, y_tick_labels)
        logger.info("Y ticks from %d user-provided labels", len(y_axis_anchors))

    # Fallback: if Y labels were unparseable (e.g. VLM read "%"), do a
    # focused re-read on a zoomed crop of the Y-axis strip.
    if not y_axis_anchors and use_vlm and provider is not None:
        y_values = await _read_y_axis_focused(img, plot_area, provider)
        if y_values:
            y_axis_anchors = _anchors_from_values(y_values, plot_area, axis="y")
            logger.info("Y-axis focused re-read: %d anchors", len(y_axis_anchors))

    # Last resort: use VLM-reported y_range [min, max] for 2-point calibration
    if not y_axis_anchors and structure and len(structure.y_range) == 2:
        y_min, y_max = structure.y_range
        if y_max > y_min:
            y_axis_anchors = [
                AxisAnchor(axis="y", px=plot_area.bottom, value=y_min, label=str(y_min)),
                AxisAnchor(axis="y", px=plot_area.top, value=y_max, label=str(y_max)),
            ]
            logger.info("Y calibration from y_range: [%.4f, %.4f]", y_min, y_max)

    if not x_axis_anchors and x_labels:
        x_axis_anchors = locate_x_ticks(img, plot_area, x_labels)
        logger.info("Auto-located %d X tick anchors", len(x_axis_anchors))

    # Dense Chinese weekly-report labels are frequently omitted by a full-page
    # VLM pass.  Re-read only the X-axis strip with OCR, which supplies both
    # the date text and its actual pixel position.
    if not x_axis_anchors:
        x_axis_anchors = _read_x_axis_ocr(img, plot_area)
        if x_axis_anchors:
            x_labels = [anchor.label for anchor in x_axis_anchors]
            logger.info("X-axis focused OCR read: %d dated anchors", len(x_axis_anchors))

    # --- Step 5: Trace curves ---
    traced = trace_all_curves(
        img,
        curve_specs,
        plot_area=plot_area,
        y_anchors=y_axis_anchors,
        x_anchors=x_axis_anchors,
        x_labels=x_labels,
        config=config,
    )

    if not traced:
        return ChartExtractResult(
            structure=structure,
            plot_area=plot_area,
            plot_area_source=plot_area_source,
            warnings=["未按当前颜色追踪到可靠产品曲线；请直接点击产品曲线取色。"],
            needs_color_pick=True,
            error=None,
            vlm_attempted=vlm_attempted,
            vlm_succeeded=vlm_succeeded,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            vlm_error=vlm_error,
        )

    # --- Step 6: Assemble result ---
    frequency = structure.frequency if structure else "unknown"
    confidence = _estimate_confidence(traced, y_axis_anchors)
    unreliable = [curve for curve in traced if not curve.is_benchmark and not curve.quality.get("is_reliable", False)]
    if unreliable:
        names = ", ".join(curve.name or "产品曲线" for curve in unreliable)
        return ChartExtractResult(
            structure=structure,
            plot_area=plot_area,
            plot_area_source=plot_area_source,
            curves=traced,
            frequency=frequency,
            confidence=min(confidence, 0.45),
            warnings=["CV 曲线未通过覆盖率或连续性质量门槛，已转入取色/日期首尾兜底。"],
            needs_color_pick=True,
            error=None,
            vlm_attempted=vlm_attempted,
            vlm_succeeded=vlm_succeeded,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            vlm_error=vlm_error,
        )

    return ChartExtractResult(
        structure=structure,
        plot_area=plot_area,
        plot_area_source=plot_area_source,
        curves=traced,
        frequency=frequency,
        confidence=confidence,
        vlm_attempted=vlm_attempted,
        vlm_succeeded=vlm_succeeded,
        vlm_provider=vlm_provider,
        vlm_model=vlm_model,
        vlm_error=vlm_error,
    )


def _choose_plot_area(
    img_bgr: np.ndarray,
    cv_area: PlotArea,
    structure: ChartStructure | None,
) -> tuple[PlotArea, str]:
    """Constrain CV to the semantic VLM chart frame whenever available.

    For single-product weekly reports the important failure mode is CV finding
    a same-colour header, legend or table outside the NAV chart.  Once VLM has
    provided a validated drawable rectangle, it is therefore authoritative;
    CV is only used as a fallback when VLM cannot locate a frame at all.
    """
    vlm_area = _plot_area_from_structure(img_bgr, structure)
    if vlm_area is None:
        return cv_area, "cv"

    logger.info("Using VLM plot area as CV tracing constraint: %s", vlm_area)
    return vlm_area, "vlm"


def _plot_area_from_structure(
    img_bgr: np.ndarray,
    structure: ChartStructure | None,
) -> PlotArea | None:
    """Convert a validated normalized VLM rectangle to source pixels."""
    if structure is None or len(structure.plot_bbox_1000) != 4:
        return None

    height, width = img_bgr.shape[:2]
    left, top, right, bottom = structure.plot_bbox_1000
    vlm_area = PlotArea(
        left=round(left * width / 1000),
        top=round(top * height / 1000),
        right=round(right * width / 1000),
        bottom=round(bottom * height / 1000),
    )
    if vlm_area.right - vlm_area.left < width * 0.12 or vlm_area.bottom - vlm_area.top < height * 0.12:
        return None
    return vlm_area


def _build_axis_anchors(raw: list[dict] | None, axis: str) -> list[AxisAnchor]:
    """Convert raw dicts to AxisAnchor objects."""
    if not raw:
        return []
    return [
        AxisAnchor(
            axis=axis,
            px=int(a.get("px", 0)),
            value=float(a.get("value", 0.0)),
            label=str(a.get("label", "")),
        )
        for a in raw
    ]


async def _read_y_axis_focused(
    img_bgr: np.ndarray,
    plot_area: PlotArea,
    provider: VLMProvider,
    upscale: int = 3,
) -> list[float]:
    """Re-read Y-axis tick numbers from a zoomed crop of the axis strip.

    Small axis labels are often unreadable in the full-chart VLM pass.
    Cropping just the Y-axis region and upscaling dramatically improves
    digit recognition.

    Returns parsed numbers ordered bottom→top, or [] on failure.
    """
    h, w = img_bgr.shape[:2]

    # Crop the left strip where Y labels live (left of the plot area)
    strip_w = max(plot_area.left, int(w * 0.08))
    strip = img_bgr[plot_area.top:plot_area.bottom, 0:strip_w]
    if strip.size == 0:
        return []


    # Upscale for better digit recognition
    sh, sw = strip.shape[:2]
    zoomed = cv2.resize(strip, (sw * upscale, sh * upscale), interpolation=cv2.INTER_CUBIC)

    ok, buf = cv2.imencode(".png", zoomed)
    if not ok:
        return []

    try:
        raw = await provider.extract_structure(buf.tobytes(), Y_AXIS_PROMPT)
        values = parse_y_axis_response(raw)
        # VLM often returns values top-to-bottom (visual order).
        # _anchors_from_values expects bottom-to-top. Auto-detect and fix.
        if len(values) >= 2 and values[0] > values[-1]:
            values = values[::-1]
        logger.info("Y-axis focused read: %s -> %s", raw[:80], values)
        return values
    except Exception as e:
        logger.warning("Y-axis focused read failed: %s", e)
        return []


def _read_x_axis_ocr(img_bgr: np.ndarray, plot_area: PlotArea, upscale: int = 4) -> list[AxisAnchor]:
    """Read complete date labels from a zoomed X-axis strip with OCR boxes."""
    try:
        import pytesseract
    except ImportError:
        return []
    h, _ = img_bgr.shape[:2]
    height = plot_area.bottom - plot_area.top
    top = max(0, plot_area.bottom - max(4, int(height * 0.03)))
    bottom = min(h, plot_area.bottom + max(24, int(height * 0.22)))
    crop = img_bgr[top:bottom, plot_area.left:plot_area.right]
    if crop.size == 0:
        return []
    zoomed = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    try:
        data = pytesseract.image_to_data(zoomed, lang="chi_sim+eng", config="--psm 6", output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    pattern = re.compile(r"(20\d{2})[./年\-\s]*(\d{1,2})[./月\-\s]*(\d{1,2})")
    found: list[AxisAnchor] = []
    for text, left, width, confidence in zip(data["text"], data["left"], data["width"], data["conf"], strict=False):
        match = pattern.search(str(text).replace(" ", ""))
        if not match or float(confidence) < 25:
            continue
        try:
            label = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).date().isoformat()
        except ValueError:
            continue
        px = plot_area.left + int((int(left) + int(width) / 2) / upscale)
        if plot_area.left <= px <= plot_area.right:
            found.append(AxisAnchor(axis="x", px=px, value=float(len(found)), label=label))
    anchors = sorted({anchor.label: anchor for anchor in found}.values(), key=lambda anchor: anchor.px)
    return anchors if len(anchors) >= 2 and [anchor.label for anchor in anchors] == sorted(anchor.label for anchor in anchors) else []


def _read_y_axis_ocr(img_bgr: np.ndarray, plot_area: PlotArea, upscale: int = 4) -> list[AxisAnchor]:
    """Read numeric Y-axis labels locally, retaining their actual row positions."""
    try:
        import pytesseract
    except ImportError:
        return []
    height, width = img_bgr.shape[:2]
    plot_width = max(1, plot_area.right - plot_area.left)
    left = max(0, plot_area.left - max(int(width * 0.22), int(plot_width * 0.18)))
    right = min(width, plot_area.left + max(8, int(width * 0.025)))
    top = max(0, plot_area.top - 8)
    bottom = min(height, plot_area.bottom + 8)
    crop = img_bgr[top:bottom, left:right]
    if crop.size == 0:
        return []
    zoomed = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    try:
        data = pytesseract.image_to_data(
            zoomed,
            lang="eng",
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []
    found: list[AxisAnchor] = []
    numeric = re.compile(r"^\s*[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*[%％]?\s*$")
    for label, box_top, box_height, confidence in zip(
        data["text"], data["top"], data["height"], data["conf"], strict=False,
    ):
        text = str(label).strip().replace("，", ",").replace("．", ".")
        if not numeric.match(text) or float(confidence) < 25:
            continue
        is_percent = "%" in text or "％" in text
        try:
            value = float(text.replace("%", "").replace("％", "").replace(",", ""))
        except ValueError:
            continue
        if is_percent:
            value /= 100.0
        px = top + int((int(box_top) + int(box_height) / 2) / upscale)
        if plot_area.top - 5 <= px <= plot_area.bottom + 5:
            found.append(AxisAnchor(axis="y", px=px, value=value, label=text))
    ordered = sorted(found, key=lambda anchor: anchor.px, reverse=True)
    deduped: list[AxisAnchor] = []
    for anchor in ordered:
        if deduped and abs(anchor.px - deduped[-1].px) < 4:
            continue
        deduped.append(anchor)
    values = [anchor.value for anchor in deduped]
    if len(values) < 2 or not all(values[index] < values[index + 1] for index in range(len(values) - 1)):
        return []
    return deduped


def _structure_with_y_anchors(
    structure: ChartStructure | None,
    anchors: list[AxisAnchor],
) -> ChartStructure | None:
    """Expose a validated OCR range to the browser's lightweight recovery flow."""
    if len(anchors) < 2:
        return structure
    result = structure.model_copy(deep=True) if structure is not None else ChartStructure(source="ocr")
    result.y_ticks = [anchor.label for anchor in anchors]
    result.y_range = [anchors[0].value, anchors[-1].value]
    return result


def _anchors_from_values(
    values: list[float],
    plot_area: PlotArea,
    axis: str = "y",
) -> list[AxisAnchor]:
    """Build evenly-spaced AxisAnchors from parsed tick values.

    values are ordered bottom→top for the Y axis (as the VLM reports).
    Pixel positions are assigned by even spacing across the plot area.
    """
    n = len(values)
    if n == 0:
        return []

    anchors = []
    if axis == "y":
        # bottom→top values map to large→small pixel y
        for i, val in enumerate(values):
            if n == 1:
                px = (plot_area.top + plot_area.bottom) // 2
            else:
                frac = i / (n - 1)  # 0 = bottom, 1 = top
                px = int(plot_area.bottom - frac * (plot_area.bottom - plot_area.top))
            anchors.append(AxisAnchor(axis="y", px=px, value=val, label=str(val)))
    else:
        for i, val in enumerate(values):
            if n == 1:
                px = (plot_area.left + plot_area.right) // 2
            else:
                px = int(plot_area.left + (i / (n - 1)) * (plot_area.right - plot_area.left))
            anchors.append(AxisAnchor(axis="x", px=px, value=val, label=str(val)))

    return anchors


def _estimate_confidence(traced: list[TracedCurve], y_anchors: list[AxisAnchor]) -> float:
    """Heuristic confidence score for the extraction."""
    if not traced:
        return 0.0

    score = 0.5  # base: curves were traced

    # More points = more confident
    avg_points = sum(c.num_pixels_traced for c in traced) / len(traced)
    if avg_points > 100:
        score += 0.2
    elif avg_points > 30:
        score += 0.1

    # Calibrated values = more confident
    if y_anchors:
        score += 0.3

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Full PDF pipeline
# ---------------------------------------------------------------------------


async def run_extraction(
    pdf_bytes: bytes,
    filename: str,
    provider: VLMProvider | None = None,
    max_pages: int | None = None,
    skip_detection: bool = False,
    use_vlm: bool = True,
    curve_specs: list[dict] | None = None,
) -> ExtractJob:
    """Run the extraction pipeline on a PDF.

    If curve_specs is provided, VLM is skipped (manual mode).
    """
    job_id = uuid.uuid4().hex[:12]
    job = ExtractJob(job_id=job_id, filename=filename)
    _jobs[job_id] = job

    job_dir = EXTRACT_WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Manual mode: no VLM needed
    if curve_specs:
        use_vlm = False

    if use_vlm and provider is None:
        provider = _default_provider()

    try:
        # --- Phase 1: Render pages ---
        job.status = ExtractJobStatus.RENDERING
        render_result = render_pdf_pages(pdf_bytes, max_pages=max_pages)
        job.total_pages = render_result.total_pages

        # --- Phase 2: Detect chart regions ---
        job.status = ExtractJobStatus.DETECTING
        page_regions: dict[int, list[ChartRegion]] = {}

        for page in render_result.pages:
            if skip_detection:
                page_regions[page.page_index] = [ChartRegion(
                    x=0, y=0, w=page.width_px, h=page.height_px, confidence=1.0,
                )]
            else:
                regions = detect_chart_regions(page.image_bytes)
                if regions:
                    page_regions[page.page_index] = regions
                    job.pages_with_charts.append(page.page_index)
            job.progress = (page.page_index + 1) / len(render_result.pages) * 0.2

        if not page_regions:
            logger.info("Job %s: no chart regions detected, using full pages", job_id)
            for page in render_result.pages:
                page_regions[page.page_index] = [ChartRegion(
                    x=0, y=0, w=page.width_px, h=page.height_px, confidence=0.5,
                )]

        # --- Phase 3: Structure + Trace ---
        job.status = ExtractJobStatus.STRUCTURE if use_vlm else ExtractJobStatus.TRACING
        total_regions = sum(len(r) for r in page_regions.values())
        processed = 0

        for page in render_result.pages:
            for region_idx, region in enumerate(page_regions.get(page.page_index, [])):
                crop_bytes = crop_region(page.image_bytes, region)
                if not crop_bytes:
                    processed += 1
                    continue

                # Save crop for preview
                crop_path = job_dir / f"p{page.page_index}_r{region_idx}.png"
                crop_path.write_bytes(crop_bytes)

                job.regions.append(PageChartRegion(
                    page_index=page.page_index,
                    region_index=region_idx,
                    x=region.x, y=region.y, w=region.w, h=region.h,
                    confidence=region.confidence,
                    image_path=str(crop_path),
                ))

                # Extract this chart
                result = await extract_chart(
                    crop_bytes,
                    curve_specs=curve_specs,
                    provider=provider,
                    use_vlm=use_vlm,
                )
                job.results.append(result)

                processed += 1
                job.progress = 0.2 + (processed / max(total_regions, 1)) * 0.8

        job.status = ExtractJobStatus.COMPLETED
        job.progress = 1.0
        logger.info("Job %s: completed with %d results", job_id, len(job.results))

    except Exception as e:
        job.status = ExtractJobStatus.FAILED
        job.error = str(e)
        logger.exception("Job %s failed", job_id)

    return job


def _select_auto_crop_region(
    regions: list[ChartRegion],
    page_width: int,
    page_height: int,
) -> ChartRegion | None:
    """Choose a likely NAV line chart and reject report headers/tables.

    A report header often contains one very wide horizontal rule.  It can be
    wider than the NAV chart but is not a drawable chart region.  The previous
    implementation chose the *widest* line-like candidate, which selected
    that header and made the VLM correctly report that it saw no curve.
    """
    page_area = page_width * page_height
    interior = [region for region in regions if 0.05 < (region.w * region.h) / page_area < 0.8]
    if not interior:
        return None

    # NAV plots are normally wide but not banner-shaped.  Keep a lower bound
    # to exclude tables and a conservative upper bound to exclude title bands.
    line_like = [
        region for region in interior
        if region.w >= page_width * 0.35 and 1.5 <= region.w / max(region.h, 1) <= 4.5
    ]
    if line_like:
        # Detector confidence distinguishes a true coloured plot from a
        # low-confidence axis/header rule; aspect is a tie-breaker only.
        return max(
            line_like,
            key=lambda region: (
                region.confidence,
                region.w * region.h,
                -abs(region.w / max(region.h, 1) - 2.8),
            ),
        )

    fallback = max(interior, key=lambda region: (region.confidence, region.w * region.h))
    return fallback if fallback.confidence >= 0.5 else None


def _auto_crop_chart_region(image_bytes: bytes) -> tuple[bytes, int, int] | None:
    """If the image is a full page/report containing a chart, crop to the chart.

    Users often upload an entire report slide rather than a pre-cropped chart.
    Tracing the whole slide picks up same-colored table text and titles, and the
    plot-area/calibration logic assumes a single chart. When a confident chart
    region clearly smaller than the page is detected, crop to it.

    Returns ``(cropped_bytes, offset_x, offset_y)`` or ``None`` if no crop is
    warranted (already a clean chart, or nothing confidently detected).
    """
    from .chart_detector import crop_region, detect_chart_regions

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]

    regions = detect_chart_regions(image_bytes)
    best = _select_auto_crop_region(regions, w, h)
    if best is None:
        return None

    cropped = crop_region(image_bytes, best)
    if not cropped:
        return None
    # crop_region adds left padding (15% of region width) to capture Y labels.
    # Cast to plain int: best.x/best.y can be numpy int32 (from HoughLinesP),
    # which pydantic cannot JSON-serialize.
    offset_x = int(max(0, best.x - max(5, int(best.w * 0.15))))
    offset_y = int(max(0, best.y - 5))
    logger.info(
        "Auto-cropped chart region x=%d y=%d w=%d h=%d (conf=%.2f) from %dx%d page",
        best.x, best.y, best.w, best.h, best.confidence, w, h,
    )
    return cropped, offset_x, offset_y


async def extract_single_image(
    image_bytes: bytes,
    curve_specs: list[dict] | None = None,
    y_anchors: list[dict] | None = None,
    x_anchors: list[dict] | None = None,
    x_labels: list[str] | None = None,
    y_tick_labels: list[str] | None = None,
    use_vlm: bool = True,
    structure_override: ChartStructure | dict | None = None,
) -> ChartExtractResult:
    """Extract from a single chart image (no PDF, no detection)."""
    crop_offset = [0, 0]
    # Only auto-crop when the caller gave no pixel coordinates (which would be
    # invalidated by cropping). A full report slide is cropped to its chart.
    has_manual_coords = bool(y_anchors) or bool(x_anchors) or any(
        spec.get("anchor_px") for spec in (curve_specs or [])
    )
    if not has_manual_coords:
        cropped = _auto_crop_chart_region(image_bytes)
        if cropped is not None:
            image_bytes, offset_x, offset_y = cropped
            crop_offset = [offset_x, offset_y]

    result = await extract_chart(
        image_bytes,
        curve_specs=curve_specs,
        y_anchors=y_anchors,
        x_anchors=x_anchors,
        x_labels=x_labels,
        y_tick_labels=y_tick_labels,
        use_vlm=use_vlm,
        structure_override=structure_override,
    )
    result.crop_offset = crop_offset
    return result


def _default_provider() -> VLMProvider:
    """Create provider from environment configuration."""
    import os

    provider_type = os.getenv("VLM_PROVIDER", "dashscope")
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
    base_url = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")

    return create_provider(
        provider_type=provider_type,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
