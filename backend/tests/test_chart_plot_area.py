import numpy as np

from app.services.chart_extractor.models import ChartStructure, PlotArea
from app.services.chart_extractor.pipeline import _endpoint_x_axis_anchors, _plot_area_from_structure


def test_vlm_plot_area_keeps_a_safety_margin_for_edge_observations() -> None:
    image = np.zeros((900, 1400, 3), dtype=np.uint8)
    area = _plot_area_from_structure(
        image,
        ChartStructure(plot_bbox_1000=[100, 100, 980, 880]),
    )

    assert area is not None
    assert area.left == 84
    assert area.right == 1400
    assert area.top == 54
    assert area.bottom == 828


def test_vlm_endpoint_dates_supply_reviewable_x_axis_anchors() -> None:
    anchors = _endpoint_x_axis_anchors(
        PlotArea(left=80, top=50, right=1320, bottom=800),
        ["2024-09-03", "2024-09-17", "2026-06-12"],
    )

    assert [(anchor.px, anchor.label) for anchor in anchors] == [
        (80, "2024-09-03"),
        (1320, "2026-06-12"),
    ]


def test_vlm_endpoint_dates_reject_a_short_partial_read() -> None:
    assert _endpoint_x_axis_anchors(
        PlotArea(left=80, top=50, right=1320, bottom=800),
        ["2024-09-03", "2024-09-17"],
    ) == []
