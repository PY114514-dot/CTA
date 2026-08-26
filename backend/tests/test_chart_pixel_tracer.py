import cv2
import numpy as np

from app.services.chart_extractor.models import PlotArea
from app.services.chart_extractor.pixel_tracer import TraceConfig, TracedPoint, assess_trace_quality, refine_colors_kmeans, sanitize_nav_points, trace_all_curves, trace_curve


def test_trace_curve_does_not_jump_to_same_colour_bottom_legend() -> None:
    """A gap in a red NAV line must not reconnect through a red legend swatch."""
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    red = (0, 0, 220)
    # Main curve, deliberately interrupted near the legend's X range.
    cv2.line(image, (10, 42), (68, 42), red, 2)
    cv2.line(image, (96, 45), (165, 45), red, 2)
    # Same-colour legend sample below the drawable region.
    cv2.line(image, (65, 78), (92, 78), red, 3)

    points = trace_curve(
        image,
        PlotArea(left=10, top=10, right=170, bottom=90),
        color_hex="#DC0000",
        config=TraceConfig(max_slope_px=8),
    )

    assert len(points) > 60
    assert max(point.y_px for point in points) < 60
    assert assess_trace_quality(points, PlotArea(left=10, top=10, right=170, bottom=90))["is_reliable"] is True


def test_trace_quality_rejects_a_short_legend_like_segment() -> None:
    points = [
        # A short same-colour swatch is not eligible for automatic NAV output.
        *[type("Point", (), {"x_px": x, "y_px": 75})() for x in range(60, 85)],
    ]
    quality = assess_trace_quality(points, PlotArea(left=10, top=10, right=170, bottom=90))
    assert quality["coverage_ratio"] < 0.40
    assert quality["is_reliable"] is False


def test_trace_curve_keeps_an_early_low_return_segment() -> None:
    """A real early curve segment can be short and close to the X axis."""
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    blue = (196, 114, 68)
    cv2.line(image, (10, 78), (48, 72), blue, 2)
    cv2.line(image, (60, 42), (170, 42), blue, 2)

    points = trace_curve(
        image,
        PlotArea(left=10, top=10, right=170, bottom=90),
        color_hex="#4472C4",
        config=TraceConfig(max_slope_px=8),
    )

    assert min(point.x_px for point in points) == 10


def test_default_trace_config_keeps_a_sharp_data_turn() -> None:
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    blue = (196, 114, 68)
    cv2.line(image, (10, 70), (48, 70), blue, 2)
    cv2.line(image, (49, 48), (170, 48), blue, 2)

    points = trace_curve(image, PlotArea(left=10, top=10, right=170, bottom=90), color_hex="#4472C4")

    assert min(point.x_px for point in points) == 10


def test_colour_refinement_never_reassigns_a_distant_series() -> None:
    """A red product hint must not become a dominant blue label/axis cluster."""
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 50), (170, 50), (100, 70, 40), 4)

    refined = refine_colors_kmeans(
        image,
        PlotArea(left=5, top=5, right=175, bottom=95),
        [{"name": "产品", "color_hex": "#FF0000", "is_benchmark": False}],
    )

    assert refined[0]["color_hex"] == "#FF0000"


def test_trace_all_curves_keeps_uncalibrated_pixels_for_review() -> None:
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 60), (170, 40), (196, 114, 68), 2)

    curves = trace_all_curves(
        image,
        [{"name": "产品", "color_hex": "#4472C4"}],
        plot_area=PlotArea(left=10, top=10, right=170, bottom=90),
    )

    assert len(curves) == 1
    assert len(curves[0].points) > 100
    assert all(point.value is None for point in curves[0].points)


def test_sanitize_keeps_cumulative_returns_near_zero() -> None:
    points = [TracedPoint(x_px=index, y_px=0, date=f"2024-01-{index + 1:02d}", value=index / 1000)
              for index in range(1, 20)]

    assert len(sanitize_nav_points(points)) == len(points)
