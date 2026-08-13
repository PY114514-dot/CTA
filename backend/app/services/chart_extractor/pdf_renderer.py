"""PDF page rendering for chart extraction.

Renders each page of a multi-page PDF at high DPI for downstream
chart detection and VLM reading.  Extends the v1 pdf_slicer concept
to handle normal multi-page documents (not just ultra-long single pages).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RENDER_DPI = 300  # higher than v1's 150 for better VLM readability


@dataclass
class RenderedPage:
    """A single rendered PDF page."""

    page_index: int
    image_bytes: bytes  # PNG
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float


@dataclass
class RenderResult:
    """All rendered pages from a PDF."""

    pages: list[RenderedPage] = field(default_factory=list)
    total_pages: int = 0
    filename: str = ""


def render_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = RENDER_DPI,
    max_pages: int | None = None,
) -> RenderResult:
    """Render PDF pages to PNG images.

    Parameters
    ----------
    pdf_bytes : raw PDF content
    dpi : rendering resolution (300 for VLM readability)
    max_pages : optional limit on pages to render

    Returns
    -------
    RenderResult with rendered pages.
    """
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    pages_to_render = min(total, max_pages) if max_pages else total

    logger.info(
        "Rendering PDF: %d pages total, rendering %d at %d DPI",
        total, pages_to_render, dpi,
    )

    result = RenderResult(total_pages=total)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    for i in range(pages_to_render):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")

        result.pages.append(RenderedPage(
            page_index=i,
            image_bytes=png_bytes,
            width_px=pix.width,
            height_px=pix.height,
            width_pt=page.rect.width,
            height_pt=page.rect.height,
        ))

    doc.close()
    logger.info("Rendered %d pages", len(result.pages))
    return result


def render_single_page(
    pdf_bytes: bytes,
    page_index: int = 0,
    dpi: int = RENDER_DPI,
) -> RenderedPage:
    """Render a single PDF page."""
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    if page_index >= len(doc):
        doc.close()
        raise ValueError(f"PDF has {len(doc)} pages, requested {page_index}")

    page = doc[page_index]
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")

    rendered = RenderedPage(
        page_index=page_index,
        image_bytes=png_bytes,
        width_px=pix.width,
        height_px=pix.height,
        width_pt=page.rect.width,
        height_pt=page.rect.height,
    )
    doc.close()
    return rendered
