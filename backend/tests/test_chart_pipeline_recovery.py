"""Recovery-state coverage for the lightweight colour/date fallback."""

import asyncio

import cv2
import numpy as np

from app.services.chart_extractor.pipeline import extract_chart
from app.services.chart_extractor.models import ChartStructure
from app.services.chart_extractor.vlm_extractor import VLMProvider


class _NoColorProvider(VLMProvider):
    async def extract_structure(self, _image_bytes: bytes, _prompt: str = "") -> str:
        return """{
          "curves": [],
          "x_ticks": ["2024-01-01", "2024-12-31"],
          "y_ticks": ["1.0", "1.2"],
          "y_range": [1.0, 1.2],
          "plot_area": {"bbox_1000": [100, 100, 900, 800]},
          "frequency": "weekly"
        }"""


class _BrokenProvider(VLMProvider):
    async def extract_structure(self, _image_bytes: bytes, _prompt: str = "") -> str:
        raise RuntimeError("provider unavailable")


class _MalformedProvider(VLMProvider):
    async def extract_structure(self, _image_bytes: bytes, _prompt: str = "") -> str:
        return "I cannot locate a chart"


def test_vlm_without_colour_returns_recoverable_pick_state() -> None:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = asyncio.run(extract_chart(encoded.tobytes(), provider=_NoColorProvider(), use_vlm=True))

    assert result.needs_color_pick is True
    assert result.error is None
    assert result.plot_area is not None
    assert result.plot_area_source == "vlm"
    assert result.structure is not None
    assert "点击产品曲线取色" in result.warnings[0]


def test_captured_structure_replays_without_calling_vlm() -> None:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = asyncio.run(extract_chart(
        encoded.tobytes(),
        use_vlm=False,
        structure_override={
            "curves": [],
            "x_ticks": ["2024-01-01", "2024-12-31"],
            "y_range": [1.0, 1.2],
            "plot_bbox_1000": [100, 100, 900, 800],
        },
    ))

    assert result.needs_color_pick is True
    assert result.error is None
    assert result.structure is not None
    assert result.structure.plot_bbox_1000 == [100, 100, 900, 800]


def test_vlm_failure_keeps_cv_frame_for_manual_recovery() -> None:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = asyncio.run(extract_chart(encoded.tobytes(), provider=_BrokenProvider(), use_vlm=True))

    assert result.vlm_attempted is True
    assert result.vlm_succeeded is False
    assert result.vlm_error == "provider unavailable"
    assert result.plot_area is not None
    assert result.plot_area_source == "cv"
    assert result.needs_color_pick is True
    assert result.error is None


def test_malformed_vlm_response_is_not_reported_as_success() -> None:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = asyncio.run(extract_chart(encoded.tobytes(), provider=_MalformedProvider(), use_vlm=True))

    assert result.vlm_attempted is True
    assert result.vlm_succeeded is False
    assert "JSON" in (result.vlm_error or "")
    assert result.plot_area_source == "cv"


def test_unreliable_colour_trace_enters_pick_state_without_hard_error() -> None:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    cv2.line(image, (30, 80), (65, 80), (0, 0, 255), 2)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = asyncio.run(extract_chart(
        encoded.tobytes(),
        use_vlm=False,
        curve_specs=[{"name": "产品", "color_hex": "#FF0000", "is_benchmark": False}],
        structure_override=ChartStructure(
            curves=[{"name": "产品", "color_hex": "#FF0000", "is_benchmark": False}],
            x_ticks=["2024-01-01", "2024-12-31"],
            y_range=[0.8, 1.2],
            plot_bbox_1000=[100, 100, 900, 900],
        ),
    ))

    assert result.needs_color_pick is True
    assert result.error is None
