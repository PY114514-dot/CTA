"""PDF / long-image slicer for weekly report parsing.

CTA weekly reports typically arrive as single-page ultra-long PDFs
(e.g., 1725x17368pt) or equivalently tall PNG images.  This module renders
them into manageable horizontal slices suitable for OCR / visual recognition.

Uses PyMuPDF (import pymupdf) for PDF rendering.  Note: the old
`import fitz` path is deprecated in PyMuPDF >= 1.24.
"""

import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default slice height in PDF points (≈ 1/8 of a typical A4-ish width)
DEFAULT_SLICE_HEIGHT_PT = 2000
# Overlap between slices to avoid cutting through table rows
OVERLAP_PT = 100
# DPI for rendering (150 is a good balance of quality vs memory)
RENDER_DPI = 150


@dataclass
class SliceInfo:
    """Metadata for one rendered slice."""

    index: int
    y_start_pt: float
    y_end_pt: float
    image_bytes: bytes  # PNG
    width_px: int
    height_px: int


def detect_input_type(data: bytes) -> str:
    """Detect whether input is PDF or image based on magic bytes."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    return "unknown"


def slice_pdf(
    pdf_bytes: bytes,
    slice_height_pt: float = DEFAULT_SLICE_HEIGHT_PT,
    overlap_pt: float = OVERLAP_PT,
    dpi: int = RENDER_DPI,
    page_index: int = 0,
) -> list[SliceInfo]:
    """Render a PDF page into horizontal slices.

    Parameters
    ----------
    pdf_bytes : raw PDF file content
    slice_height_pt : height of each slice in PDF points
    overlap_pt : vertical overlap between consecutive slices
    dpi : rendering resolution
    page_index : which page to slice (default 0, first page)

    Returns
    -------
    List of SliceInfo with rendered PNG bytes.
    """
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    if page_index >= len(doc):
        raise ValueError(f"PDF only has {len(doc)} pages, requested page {page_index}")

    page = doc[page_index]
    page_rect = page.rect
    total_height = page_rect.height
    page_width = page_rect.width

    logger.info(
        "Slicing PDF page %d: %.0f x %.0f pt (%.1f x %.1f inches)",
        page_index, page_width, total_height,
        page_width / 72, total_height / 72,
    )

    slices: list[SliceInfo] = []
    y = 0.0
    idx = 0
    zoom = dpi / 72.0

    while y < total_height:
        y_end = min(y + slice_height_pt, total_height)

        # Define clip rectangle
        clip = pymupdf.Rect(0, y, page_width, y_end)

        # Render to pixmap
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        png_bytes = pix.tobytes("png")

        slices.append(SliceInfo(
            index=idx,
            y_start_pt=y,
            y_end_pt=y_end,
            image_bytes=png_bytes,
            width_px=pix.width,
            height_px=pix.height,
        ))

        idx += 1
        y = y_end - overlap_pt if y_end < total_height else total_height

    doc.close()
    logger.info("Produced %d slices from PDF page", len(slices))
    return slices


def slice_image(
    image_bytes: bytes,
    slice_height_px: int = 3000,
    overlap_px: int = 150,
) -> list[SliceInfo]:
    """Slice a tall image (PNG/JPEG) into horizontal strips.

    Uses OpenCV for image manipulation.
    """
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")

    h, w = img.shape[:2]
    logger.info("Slicing image: %d x %d px", w, h)

    slices: list[SliceInfo] = []
    y = 0
    idx = 0

    while y < h:
        y_end = min(y + slice_height_px, h)
        strip = img[y:y_end, :]

        # Encode to PNG
        ok, buf = cv2.imencode(".png", strip)
        if not ok:
            raise ValueError("Failed to encode slice")

        slices.append(SliceInfo(
            index=idx,
            y_start_pt=float(y),  # pixel coords for images
            y_end_pt=float(y_end),
            image_bytes=buf.tobytes(),
            width_px=w,
            height_px=y_end - y,
        ))

        idx += 1
        y = y_end - overlap_px if y_end < h else h

    logger.info("Produced %d slices from image", len(slices))
    return slices


def slice_report(
    data: bytes,
    slice_height: int | None = None,
    dpi: int = RENDER_DPI,
) -> list[SliceInfo]:
    """Auto-detect input type and slice accordingly.

    This is the main entry point for the pipeline.
    """
    input_type = detect_input_type(data)

    if input_type == "pdf":
        height = slice_height or DEFAULT_SLICE_HEIGHT_PT
        return slice_pdf(data, slice_height_pt=height, dpi=dpi)
    elif input_type in ("png", "jpeg"):
        height = slice_height or 3000
        return slice_image(data, slice_height_px=height)
    else:
        raise ValueError(
            f"Unsupported input format (magic bytes: {data[:8]!r}). "
            "Expected PDF, PNG, or JPEG."
        )
