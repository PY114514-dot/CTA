"""Unit checks for the image-curve tracing primitive."""

from datetime import date

import cv2
import numpy as np

from app.services.nav_image_digitizer import _trace_curve, digitize_nav_image


def test_trace_curve_follows_a_continuous_colored_line() -> None:
    mask = np.zeros((40, 80), dtype=bool)
    for x in range(mask.shape[1]):
        mask[20 + x // 20, x] = True

    points = _trace_curve(mask)

    assert len(points) == 80
    assert points[0] == (0, 20)
    assert points[-1] == (79, 23)


def test_manual_chart_bounds_constrain_curve_search_and_overlay() -> None:
    """A user-framed chart must exclude a same-colour header decoration."""
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    red = (50, 50, 200)  # BGR, approximately #C83232.
    # A wide decorative line above the chart used to win automatic detection.
    cv2.line(image, (8, 26), (292, 26), red, 3)
    # The actual product curve lies wholly inside the user-marked chart box.
    for x in range(32, 270):
        y = 158 - (x - 32) // 10
        image[y:y + 3, x] = red
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok

    result = digitize_nav_image(
        encoded.tobytes(),
        date(2020, 1, 1), date(2021, 1, 1), 1.0, 2.0, "nav", "monthly",
        start_x_ratio=0.10, end_x_ratio=0.90,
        top_y_ratio=0.50, bottom_y_ratio=0.95,
        line_color="#C83232",
    )

    ys = [point.y_ratio for point in result.candidate_curve]
    assert min(ys) >= 0.50
    assert max(ys) <= 0.95
