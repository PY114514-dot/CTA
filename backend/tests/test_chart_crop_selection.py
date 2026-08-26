from app.services.chart_extractor.chart_detector import ChartRegion
from app.services.chart_extractor.pipeline import _select_auto_crop_region


def test_auto_crop_prefers_nav_plot_over_wide_report_header():
    """A banner-shaped report header must not replace the actual NAV chart."""
    nav_plot = ChartRegion(x=372, y=3432, w=2202, h=775, confidence=0.6)
    side_table = ChartRegion(x=2881, y=3297, w=1110, h=1301, confidence=0.6)
    header_rule = ChartRegion(x=0, y=0, w=4252, h=764, confidence=0.3)

    selected = _select_auto_crop_region(
        [nav_plot, side_table, header_rule],
        page_width=4252,
        page_height=6236,
    )

    assert selected is nav_plot


def test_auto_crop_excludes_a_page_border_mistaken_for_axes():
    page_border = ChartRegion(x=68, y=98, w=1156, h=439, confidence=0.9)
    nav_plot = ChartRegion(x=187, y=691, w=915, h=299, confidence=0.8)

    selected = _select_auto_crop_region(
        [page_border, nav_plot],
        page_width=1280,
        page_height=1810,
    )

    assert selected is nav_plot
