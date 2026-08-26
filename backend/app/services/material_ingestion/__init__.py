"""Automatic ingestion for materials uploaded from the FOF workbench.

The ingestion result is deliberately reviewable: product entities and NAV
observations created here remain pending until a user confirms or reviews them.

Parsing strategy per file type:
- Image (weekly report): table OCR for disclosed metrics, then chart_extractor
  pixel tracing (VLM when configured, else auto-color CV) for NAV curves.
- PDF: text-layer extraction for context + chart pipeline (render → detect →
  trace) for NAV curves.
- XLSX/CSV: long format (date + nav column) or wide format (date + one
  column per product).

Layout:
- ``entry``: top-level dispatcher and public API.
- ``helpers``: date/NAV normalisation, audit merging, name heuristics.
- ``image``: report-image ingestion and disclosed-table recovery.
- ``curves``: pixel curve tracing via chart_extractor.
- ``table`` / ``office`` / ``pdf``: per-file-type ingestion.
"""

from .curves import (
    _auto_curve_colors,
    _cross_validate_with_table,
    _extract_points_cv_only,
    _extract_points_from_image,
    _resample_pixel_points_weekly,
    _run_async,
    _trace_image_curves,
    _trace_whole_image,
)
from .entry import (
    _ingest_context_material,
    ingest_uploaded_material,
    ingestion_queue_snapshot,
    recover_pending_disclosed_nav,
)
from .helpers import (
    _empty_extraction_audit,
    _find_column,
    _find_or_create_product,
    _infer_frequency,
    _is_multi_product_material,
    _looks_like_product_name,
    _looks_like_strategy_label,
    _manager_name_from_filename,
    _merge_extraction_audit,
    _normalise_manifest_product_name,
    _normalise_nav_points,
    _normalise_observation_date,
    _product_name_from_filename,
    _valid_nav_points,
    _visual_method,
)
from .image import (
    _disclosed_nav_image_variants,
    _ingest_report_image,
    _parse_disclosed_nav_payload,
    _parse_monthly_return_payload,
    _parse_monthly_return_rows,
    _recover_disclosed_nav_table,
    _recover_monthly_nav_from_return_table,
)
from .office import _extract_office_parts, _ingest_office_document
from .pdf import (
    _auto_curve_colors_from_pdf,
    _ingest_pdf,
    _looks_like_pdf_product_curve,
    _recover_monthly_nav_from_pdf_text,
)
from .table import _ingest_nav_frame, _ingest_nav_table

__all__ = [
    "ingest_uploaded_material",
    "ingestion_queue_snapshot",
    "recover_pending_disclosed_nav",
]
