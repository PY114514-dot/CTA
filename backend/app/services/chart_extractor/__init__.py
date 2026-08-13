"""Chart extractor service — public API.

Usage:
    from app.services.chart_extractor import (
        run_extraction, extract_single_image, extract_chart,
        sample_color_at_pixel,
    )
"""

from .models import (
    AxisAnchor,
    ChartExtractResult,
    ChartStructure,
    ExtractJob,
    ExtractJobStatus,
    NavPoint,
    NavSeries,
    PageChartRegion,
    PlotArea,
    TracedCurve,
    TracedPoint,
)
from .pipeline import (
    extract_chart,
    extract_single_image,
    get_job,
    list_jobs,
    run_extraction,
)
from .pixel_tracer import (
    calibrate_from_disclosed,
    calibrate_from_reference,
    refine_colors_kmeans,
    sample_color_at_pixel,
    sanitize_nav_points,
)

__all__ = [
    "AxisAnchor",
    "ChartExtractResult",
    "ChartStructure",
    "ExtractJob",
    "ExtractJobStatus",
    "NavPoint",
    "NavSeries",
    "PageChartRegion",
    "PlotArea",
    "TracedCurve",
    "TracedPoint",
    "calibrate_from_disclosed",
    "calibrate_from_reference",
    "extract_chart",
    "extract_single_image",
    "get_job",
    "list_jobs",
    "refine_colors_kmeans",
    "run_extraction",
    "sample_color_at_pixel",
    "sanitize_nav_points",
]
