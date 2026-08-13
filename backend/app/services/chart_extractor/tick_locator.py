"""Tick localization — finding where axis ticks sit in pixel space.

The VLM reads tick LABELS (dates, values) but not their pixel positions.
This module locates tick pixel positions so pixel_tracer can calibrate:

- X axis (dates): detect vertical grid lines.
- Y axis (values): detect horizontal grid lines and match to labels.

No OCR involved — pure geometry + the labels the VLM already read.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .models import AxisAnchor, PlotArea

logger = logging.getLogger(__name__)


def detect_grid_lines(
    img_bgr: np.ndarray,
    plot_area: PlotArea,
) -> tuple[list[int], list[int]]:
    """Detect grid line positions within the plot area.

    Returns
    -------
    (vertical_x_positions, horizontal_y_positions) sorted.
    Grid lines are light gray, so we look for thin light lines.
    """
    crop = img_bgr[plot_area.top:plot_area.bottom, plot_area.left:plot_area.right]
    if crop.size == 0:
        return [], []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    vertical_xs: list[int] = []
    horizontal_ys: list[int] = []

    # Grid lines are usually light gray (darker than white bg, lighter than curves)
    # Detect columns that are consistently light-gray across most of the height
    for x in range(w):
        col = gray[:, x]
        # Count pixels in the "light gray" band
        grid_px = np.sum((col > 180) & (col < 245))
        if grid_px > h * 0.6:
            vertical_xs.append(x)

    for y in range(h):
        row = gray[y, :]
        grid_px = np.sum((row > 180) & (row < 245))
        if grid_px > w * 0.6:
            horizontal_ys.append(y)

    # Cluster adjacent pixels into single line positions
    vertical_xs = _cluster_positions(vertical_xs)
    horizontal_ys = _cluster_positions(horizontal_ys)

    # Convert back to full-image coordinates
    vertical_xs = [x + plot_area.left for x in vertical_xs]
    horizontal_ys = [y + plot_area.top for y in horizontal_ys]

    return vertical_xs, horizontal_ys


def _cluster_positions(positions: list[int], min_gap: int = 5) -> list[int]:
    """Cluster adjacent pixel positions into single line centers."""
    if not positions:
        return []

    clusters: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return [int(np.mean(c)) for c in clusters]


def locate_x_ticks(
    img_bgr: np.ndarray,
    plot_area: PlotArea,
    x_labels: list[str],
) -> list[AxisAnchor]:
    """Locate X axis tick pixel positions for the given date labels.

    Labels without image positions are not enough for calibration.  Return no
    anchors when the chart does not contain a usable set of vertical grid
    lines; the caller will route the result to manual review.
    """
    n = len(x_labels)
    if n == 0:
        return []

    v_lines, _ = detect_grid_lines(img_bgr, plot_area)

    # Filter grid lines that span a reasonable range
    usable = [x for x in v_lines if plot_area.left < x < plot_area.right]

    if len(usable) != n:
        logger.info("X tick pixel evidence insufficient: %d grid lines for %d labels", len(usable), n)
        return []

    # Every returned anchor must correspond to a detected grid line.
    positions = [float(position) for position in sorted(usable)]
    logger.info("X ticks from %d grid lines -> %d positions", len(usable), n)

    return [
        AxisAnchor(axis="x", px=int(pos), value=float(i), label=x_labels[i])
        for i, pos in enumerate(positions)
    ]


def locate_y_ticks(
    img_bgr: np.ndarray,
    plot_area: PlotArea,
    y_labels: list[str],
) -> list[AxisAnchor]:
    """Locate Y axis tick pixel positions and parse their values.

    y_labels are ordered bottom→top (as the VLM reports them).
    Returns anchors with parsed numeric values; labels that can't be
    parsed to a number are skipped.
    """
    # Parse labels to values (handle %, commas, Chinese)
    parsed: list[tuple[str, float]] = []
    for label in y_labels:
        val = _parse_tick_value(label)
        if val is not None:
            parsed.append((label, val))

    if not parsed:
        logger.warning("No Y tick labels parsed to numbers: %s", y_labels)
        return []

    n = len(parsed)

    _, h_lines = detect_grid_lines(img_bgr, plot_area)
    usable = [y for y in h_lines if plot_area.top < y < plot_area.bottom]
    if not usable or len(usable) != n:
        logger.info("Y tick pixel evidence insufficient: %d grid lines for %d labels", len(usable), n)
        return []

    positions = [float(position) for position in sorted(usable)]
    # Detected lines are ascending pixel-y (top→bottom), but parsed labels are
    # bottom→top. Reverse to align.
    positions = positions[::-1]
    logger.info("Y ticks from %d grid lines -> %d positions", len(usable), n)

    anchors = []
    for i, (label, val) in enumerate(parsed):
        anchors.append(AxisAnchor(axis="y", px=int(positions[i]), value=val, label=label))

    return anchors


def _parse_tick_value(label: str) -> float | None:
    """Parse a Y axis tick label into a float.

    Handles: "1.05", "105%", "1,000", "20%", "-0.5", "１.０" (fullwidth).
    """
    if not label:
        return None

    s = label.strip()
    # Normalize fullwidth digits/punctuation to ASCII
    s = _normalize_fullwidth(s)
    # Remove thousand separators and percent signs
    s = s.replace(",", "").replace("，", "")

    is_percent = "%" in s or "％" in s
    s = s.replace("%", "").replace("％", "").strip()

    try:
        val = float(s)
    except ValueError:
        return None

    if is_percent:
        val = val / 100.0

    return val


def _normalize_fullwidth(s: str) -> str:
    """Convert fullwidth digits and punctuation to ASCII equivalents."""
    result = []
    for ch in s:
        code = ord(ch)
        # Fullwidth digits ０-９ -> 0-9
        if 0xFF10 <= code <= 0xFF19:
            result.append(chr(code - 0xFF10 + ord("0")))
        # Fullwidth period ．
        elif ch == "．":
            result.append(".")
        # Fullwidth minus －
        elif ch == "－":
            result.append("-")
        else:
            result.append(ch)
    return "".join(result)
