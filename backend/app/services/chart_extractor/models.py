"""Pydantic models for the chart extraction pipeline."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExtractSource(str, Enum):
    PDF_VLM = "pdf_vlm"
    PDF_CV = "pdf_cv"
    EXCEL = "excel"
    PASTE = "paste"
    OCR = "ocr"


# ---------------------------------------------------------------------------
# Chart structure (from VLM or manual user input)
# ---------------------------------------------------------------------------


class CurveSpec(BaseModel):
    """Specification of one curve to trace — from VLM or user."""

    name: str = ""
    color_hex: str = ""  # e.g. "#E03030"
    color_name: str = ""  # e.g. "red"
    is_benchmark: bool = False
    # Pixel anchor points (from user clicks on the image)
    anchor_px: tuple[int, int] | None = None  # a clicked point on this curve


class ChartStructure(BaseModel):
    """Structural understanding of a chart (no data values).

    Can come from VLM analysis OR manual user input.
    """

    curves: list[dict] = Field(default_factory=list)  # CurveSpec-like dicts
    x_ticks: list[str] = Field(default_factory=list)  # raw tick labels, left→right
    y_ticks: list[str] = Field(default_factory=list)  # raw tick labels, bottom→top
    y_range: list[float] = Field(default_factory=list)  # [min, max] from VLM
    y_axis_label: str = ""
    chart_title: str = ""
    frequency: str = "unknown"
    has_grid_lines: bool = False
    # Bounding box of the drawable plot area in a 0–1000 normalized image
    # coordinate system: [left, top, right, bottom].  It is intentionally
    # kept separate from ``PlotArea`` because VLM coordinates are approximate.
    plot_bbox_1000: list[int] = Field(default_factory=list)
    raw_response: str = ""
    source: str = "vlm"  # "vlm" | "manual"


# ---------------------------------------------------------------------------
# Axis calibration (pixel ↔ value mapping)
# ---------------------------------------------------------------------------


class AxisAnchor(BaseModel):
    """A known pixel↔value correspondence on an axis.

    From user clicks or VLM-inferred tick positions.
    """

    axis: str = "y"  # "x" | "y"
    px: int = 0  # pixel coordinate (x for x-axis, y for y-axis)
    value: float = 0.0  # data value
    label: str = ""  # original tick label text


class PlotArea(BaseModel):
    """The chart's plot region in pixel coordinates."""

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0


# ---------------------------------------------------------------------------
# Tracing results
# ---------------------------------------------------------------------------


class TracedPoint(BaseModel):
    """A single traced data point (pixel → value)."""

    x_px: int = 0
    y_px: int = 0
    date: str = ""  # interpolated date string
    value: float | None = None


class TracedCurve(BaseModel):
    """One fully traced curve with calibrated values."""

    name: str = ""
    color_hex: str = ""
    is_benchmark: bool = False
    points: list[TracedPoint] = Field(default_factory=list)
    num_pixels_traced: int = 0
    # Deterministic tracing diagnostics used to decide whether the browser
    # may auto-apply this curve or must switch to the lightweight fallback.
    quality: dict[str, float | bool] = Field(default_factory=dict)


class ChartExtractResult(BaseModel):
    """Complete extraction result for one chart image."""

    structure: ChartStructure | None = None
    plot_area: PlotArea | None = None
    # Auditable provenance for the rectangle that constrained CV.  A parsed
    # VLM response is not proof that it supplied a usable plot rectangle.
    plot_area_source: str | None = None  # "vlm" | "cv"
    curves: list[TracedCurve] = Field(default_factory=list)
    frequency: str = "unknown"
    confidence: float = 0.0
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    # VLM provenance is part of the result so callers can distinguish a real
    # model call from a configured-but-unused fallback.
    vlm_attempted: bool = False
    vlm_succeeded: bool = False
    vlm_provider: str | None = None
    vlm_model: str | None = None
    vlm_error: str | None = None
    # A successful VLM frame with no reliable line colour is a recoverable
    # review state, not a failed extraction.  The browser uses this to enter
    # one-click colour-pick mode while preserving the located chart frame.
    needs_color_pick: bool = False
    # One auditable gate for every automatic result.  A trace can be visually
    # sound yet still require axis calibration before it is safe NAV data.
    review_required: bool = True
    review_reasons: list[str] = Field(default_factory=list)
    # If the chart was auto-cropped from a larger page, the (x, y) offset of the
    # crop's top-left corner in the original image. Traced point coordinates are
    # relative to the crop; add this offset to place them on the original image.
    crop_offset: list[int] = Field(default_factory=lambda: [0, 0])


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------


class ExtractJobStatus(str, Enum):
    PENDING = "pending"
    RENDERING = "rendering"
    DETECTING = "detecting"
    STRUCTURE = "structure"  # VLM structure analysis
    TRACING = "tracing"  # pixel tracing
    COMPLETED = "completed"
    FAILED = "failed"


class PageChartRegion(BaseModel):
    """A detected chart region within a rendered page."""

    page_index: int
    region_index: int
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0
    image_path: str | None = None


class ExtractJob(BaseModel):
    """Tracks the state of a PDF extraction job."""

    job_id: str
    filename: str
    status: ExtractJobStatus = ExtractJobStatus.PENDING
    total_pages: int = 0
    pages_with_charts: list[int] = Field(default_factory=list)
    results: list[ChartExtractResult] = Field(default_factory=list)
    regions: list[PageChartRegion] = Field(default_factory=list)
    error: str | None = None
    progress: float = 0.0


# ---------------------------------------------------------------------------
# NAV series (universal output)
# ---------------------------------------------------------------------------


class NavPoint(BaseModel):
    """A confirmed NAV data point."""

    date: date
    nav: float
    acc_nav: float | None = None


class NavSeries(BaseModel):
    """Standard NAV series — the universal output of all import channels."""

    product_name: str
    manager_name: str | None = None
    series: list[NavPoint] = Field(default_factory=list)
    frequency: str = "unknown"
    source: ExtractSource = ExtractSource.PDF_CV
    confidence: float = 0.0
    raw_image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
