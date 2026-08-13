"""Unit checks for the image-curve tracing primitive."""

import numpy as np

from app.services.nav_image_digitizer import _trace_curve


def test_trace_curve_follows_a_continuous_colored_line() -> None:
    mask = np.zeros((40, 80), dtype=bool)
    for x in range(mask.shape[1]):
        mask[20 + x // 20, x] = True

    points = _trace_curve(mask)

    assert len(points) == 80
    assert points[0] == (0, 20)
    assert points[-1] == (79, 23)
