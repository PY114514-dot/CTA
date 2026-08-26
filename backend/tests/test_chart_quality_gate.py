import asyncio

import cv2
import numpy as np

from app.services.chart_extractor.models import ChartStructure
from app.services.chart_extractor.pipeline import extract_chart


def _chart() -> bytes:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    points = np.array([(x, int(50 + 12 * np.sin(x / 25))) for x in range(10, 190)], dtype=np.int32)
    cv2.polylines(image, [points], False, (20, 20, 220), 2)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_reliable_trace_without_axis_evidence_requires_review() -> None:
    result = asyncio.run(extract_chart(
        _chart(),
        curve_specs=[{"name": "产品", "color_hex": "#DC1414"}],
        use_vlm=False,
    ))

    assert result.curves
    assert result.review_required is True
    assert result.needs_color_pick is False
    assert result.review_reasons == ["y_axis_unverified", "x_axis_unverified"]
    assert result.confidence <= 0.45


def test_reliable_trace_with_both_axes_can_be_accepted() -> None:
    result = asyncio.run(extract_chart(
        _chart(),
        curve_specs=[{"name": "产品", "color_hex": "#DC1414"}],
        y_anchors=[{"px": 80, "value": 0.8}, {"px": 20, "value": 1.2}],
        x_anchors=[{"px": 10, "label": "2024-01-01"}, {"px": 189, "label": "2024-12-31"}],
        x_labels=["2024-01-01", "2024-12-31"],
        structure_override=ChartStructure(curves=[], frequency="weekly"),
        use_vlm=False,
    ))

    assert result.curves
    assert result.review_required is False
    assert result.review_reasons == []


def test_axis_labels_without_pixel_evidence_still_require_review() -> None:
    result = asyncio.run(extract_chart(
        _chart(),
        curve_specs=[{"name": "产品", "color_hex": "#DC1414"}],
        structure_override=ChartStructure(
            curves=[],
            x_ticks=["2024-01-01", "2024-12-31"],
            y_ticks=["0.8", "1.2"],
            y_range=[0.8, 1.2],
            plot_bbox_1000=[50, 0, 950, 1000],
            frequency="weekly",
        ),
        use_vlm=False,
    ))

    assert result.curves
    assert result.review_required is True
    assert result.review_reasons == ["y_axis_unverified", "x_axis_interpolated"]
