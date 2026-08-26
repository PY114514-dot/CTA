"""Deterministic pixel-based curve tracing and coordinate calibration.

This is the numerical core of the extraction pipeline.  Given:
- a chart image (cropped region)
- curve color specifications (from VLM or user clicks)
- axis anchors (pixel ↔ value correspondences)

It traces each curve's pixel path and maps pixels to data values via
linear interpolation.  No ML involved — pure geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import AxisAnchor, PlotArea, TracedCurve, TracedPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

# Common color name → HSV range mapping (hue in OpenCV's 0-180 scale)
COLOR_NAME_HSV: dict[str, list[tuple[int, int, int, int, int, int]]] = {
    # (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) — multiple ranges for wrap-around
    "red": [(0, 10, 80, 255, 80, 255), (170, 180, 80, 255, 80, 255)],
    "blue": [(95, 135, 80, 255, 60, 255)],
    "dark blue": [(95, 135, 80, 255, 40, 180)],
    "light blue": [(85, 130, 40, 200, 120, 255)],
    "green": [(35, 85, 60, 255, 60, 255)],
    "orange": [(10, 25, 120, 255, 120, 255)],
    "yellow": [(20, 35, 120, 255, 150, 255)],
    "purple": [(130, 170, 60, 255, 60, 255)],
    "pink": [(140, 170, 40, 200, 150, 255)],
    "gray": [(0, 180, 0, 60, 60, 200)],
    "grey": [(0, 180, 0, 60, 60, 200)],
    "black": [(0, 180, 0, 255, 0, 80)],
    "brown": [(10, 20, 80, 200, 60, 160)],
    "cyan": [(80, 100, 80, 255, 120, 255)],
    "gold": [(20, 30, 150, 255, 150, 255)],
}


def hex_to_hsv_ranges(color_hex: str) -> list[tuple[int, int, int, int, int, int]]:
    """Convert a hex color to HSV tolerance ranges for masking.

    Uses a generous tolerance to account for anti-aliasing and JPEG artifacts.
    """
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        return []

    try:
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
    except ValueError:
        return []

    # Convert to HSV (OpenCV scale: H 0-180, S 0-255, V 0-255)
    pixel = np.uint8([[[b, g, r]]])  # BGR order for OpenCV
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Tolerance scales with saturation (gray-ish colors need wider hue tolerance)
    h_tol = 25 if s < 80 else 12
    s_tol = 70
    v_tol = 70

    ranges = []
    h_lo, h_hi = h - h_tol, h + h_tol

    if h_lo < 0:
        # Wrap around red
        ranges.append((h_lo + 180, 180, max(0, s - s_tol), min(255, s + s_tol), max(0, v - v_tol), min(255, v + v_tol)))
        ranges.append((0, h_hi, max(0, s - s_tol), min(255, s + s_tol), max(0, v - v_tol), min(255, v + v_tol)))
    elif h_hi > 180:
        ranges.append((h_lo, 180, max(0, s - s_tol), min(255, s + s_tol), max(0, v - v_tol), min(255, v + v_tol)))
        ranges.append((0, h_hi - 180, max(0, s - s_tol), min(255, s + s_tol), max(0, v - v_tol), min(255, v + v_tol)))
    else:
        ranges.append((h_lo, h_hi, max(0, s - s_tol), min(255, s + s_tol), max(0, v - v_tol), min(255, v + v_tol)))

    return ranges


def color_name_to_hsv_ranges(name: str) -> list[tuple[int, int, int, int, int, int]]:
    """Map a color name to HSV ranges."""
    return COLOR_NAME_HSV.get(name.lower(), [])


def build_color_mask(
    hsv_img: np.ndarray,
    color_hex: str = "",
    color_name: str = "",
) -> np.ndarray:
    """Build a binary mask for pixels matching the given color.

    Prefers hex (more precise), falls back to color name.
    For low-saturation (gray-ish) colors, near-white background pixels
    are excluded to avoid matching the chart background.
    """
    ranges = []
    if color_hex:
        ranges = hex_to_hsv_ranges(color_hex)
    if not ranges and color_name:
        ranges = color_name_to_hsv_ranges(color_name)
    if not ranges:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in ranges:
        lower = np.array([h_lo, s_lo, v_lo])
        upper = np.array([h_hi, s_hi, v_hi])
        mask |= cv2.inRange(hsv_img, lower, upper)

    # For gray-ish colors (low saturation), exclude near-white background.
    # White background: S≈0, V≈255. Real gray curves sit below V≈250.
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]
    is_grayish = s_channel < 60
    near_white = v_channel > 250
    mask[is_grayish & near_white] = 0

    return mask


def refine_colors_kmeans(
    img_bgr: np.ndarray,
    plot_area: "PlotArea",
    curve_specs: list[dict],
    max_clusters: int = 6,
) -> list[dict]:
    """Refine VLM-provided curve colors using K-means on actual plot pixels.

    The VLM often returns approximate hex colors (e.g. "#FF0000" when the
    actual curve is "#E63946"). This function:
    1. Extracts all saturated (non-background) pixels in the plot area
    2. Clusters them via K-means to find the true dominant colors
    3. Matches each VLM curve to the nearest cluster center
    4. Returns updated curve_specs with refined color_hex values

    This dramatically improves tracing accuracy when VLM colors are imprecise.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    roi_hsv = hsv[plot_area.top:plot_area.bottom, plot_area.left:plot_area.right]
    roi_bgr = img_bgr[plot_area.top:plot_area.bottom, plot_area.left:plot_area.right]

    # Filter: only saturated, non-white, non-black pixels (actual curve ink)
    s_ch = roi_hsv[:, :, 1].ravel().astype(np.float32)
    v_ch = roi_hsv[:, :, 2].ravel().astype(np.float32)
    valid = (s_ch > 50) & (v_ch > 40) & (v_ch < 250)

    pixels_bgr = roi_bgr.reshape(-1, 3).astype(np.float32)
    curve_pixels = pixels_bgr[valid]

    if len(curve_pixels) < 20:
        logger.warning("Too few saturated pixels (%d) for K-means refinement", len(curve_pixels))
        return curve_specs

    # Determine K: number of non-benchmark curves, clamped
    n_curves = max(1, sum(1 for s in curve_specs if not s.get("is_benchmark", False)))
    k = min(max_clusters, max(n_curves + 1, 3))  # +1 for potential grid/axis remnants

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(curve_pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Sort clusters by size (largest = most likely a real curve)
    cluster_sizes = np.bincount(labels.ravel(), minlength=k)
    # Filter out tiny clusters (noise, <2% of pixels)
    min_size = len(curve_pixels) * 0.02
    valid_centers = []
    for i in np.argsort(cluster_sizes)[::-1]:
        if cluster_sizes[i] >= min_size:
            b, g, r = int(centers[i][0]), int(centers[i][1]), int(centers[i][2])
            valid_centers.append(f"#{r:02X}{g:02X}{b:02X}")

    if not valid_centers:
        return curve_specs

    logger.info("K-means found %d dominant colors: %s", len(valid_centers), valid_centers)

    # Match each curve spec to the nearest cluster center (by RGB distance)
    refined = []
    used_centers: set[int] = set()

    for spec in curve_specs:
        spec = dict(spec)  # copy
        vlm_hex = spec.get("color_hex", "")
        if not vlm_hex or spec.get("is_benchmark", False):
            refined.append(spec)
            continue

        vlm_rgb = _hex_to_rgb(vlm_hex)
        if vlm_rgb is None:
            refined.append(spec)
            continue

        # Find nearest unused cluster center
        best_idx = -1
        best_dist = float("inf")
        for idx, center_hex in enumerate(valid_centers):
            if idx in used_centers:
                continue
            c_rgb = _hex_to_rgb(center_hex)
            if c_rgb is None:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(vlm_rgb, c_rgb)) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        # Only refine if the match is reasonable (within distance 150)
        if best_idx >= 0 and best_dist < 150:
            old_hex = spec.get("color_hex", "")
            spec["color_hex"] = valid_centers[best_idx]
            used_centers.add(best_idx)
            if old_hex.upper() != spec["color_hex"].upper():
                logger.info("Refined color for '%s': %s → %s (dist=%.0f)",
                            spec.get("name", ""), old_hex, spec["color_hex"], best_dist)
        elif best_idx >= 0:
            # A distant cluster is more likely an axis, label or another
            # series than a refined version of this curve.  Keep the semantic
            # VLM colour so CV cannot silently switch product ownership.
            logger.warning(
                "VLM color %s is far from every plot cluster (nearest=%s, dist=%.0f); preserving it",
                vlm_hex,
                valid_centers[best_idx],
                best_dist,
            )

        refined.append(spec)

    return refined


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int] | None:
    """Parse #RRGGBB to (R, G, B) tuple."""
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        return None
    try:
        return int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
    except ValueError:
        return None


def _build_grid_exclusion_mask(
    img_bgr: np.ndarray,
    plot_area: "PlotArea",
    line_width: int = 2,
) -> np.ndarray:
    """Build a mask of grid-line pixels to exclude from curve tracing.

    Grid lines are light gray lines spanning the plot. Excluding them
    prevents gray curves from picking up the grid, and stops other
    curves from counting anti-aliased grid pixels.
    """
    from .tick_locator import detect_grid_lines

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    v_lines, h_lines = detect_grid_lines(img_bgr, plot_area)
    for x in v_lines:
        x0 = max(0, x - line_width)
        x1 = min(w, x + line_width + 1)
        mask[plot_area.top:plot_area.bottom, x0:x1] = 255
    for y in h_lines:
        y0 = max(0, y - line_width)
        y1 = min(h, y + line_width + 1)
        mask[y0:y1, plot_area.left:plot_area.right] = 255

    return mask


def sample_color_at_pixel(img_bgr: np.ndarray, x: int, y: int, radius: int = 3) -> str:
    """Sample the dominant color around a pixel, return hex string.

    Used for the "click on curve to pick color" frontend interaction.
    """
    h, w = img_bgr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)

    patch = img_bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    # Exclude near-white/near-black pixels (likely background/axes)
    brightness = patch.mean(axis=1)
    valid = (brightness > 30) & (brightness < 240)
    if valid.sum() == 0:
        valid = np.ones(len(patch), dtype=bool)

    mean_bgr = patch[valid].mean(axis=0)
    b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# Plot area detection
# ---------------------------------------------------------------------------


def detect_plot_area(
    img_bgr: np.ndarray,
    margin_pct: float = 0.01,
) -> PlotArea:
    """Detect the plot area (the region enclosed by axes).

    Strategy:
    1. Find bounding box of non-background content.
    2. Refine using long axis lines (filtered by span).
    3. Always validate the result; fall back to margin-based estimate
       if detection is degenerate (left >= right, etc.).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Fallback: percentage-based estimate (always valid) ---
    # Financial charts: Y labels on left (~6%), X labels at bottom (~10%),
    # title/legend on top (~8%), small right margin.
    fallback = PlotArea(
        left=int(w * 0.07),
        top=int(h * 0.08),
        right=int(w * 0.98),
        bottom=int(h * 0.88),
    )

    # Content mask: anything that isn't near-white background
    content = gray < 240
    if not content.any():
        return fallback

    ys, xs = np.where(content)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())

    # --- Refine with axis lines (only trust long, well-placed lines) ---
    edges = cv2.Canny(gray, 50, 150)
    content_h = bottom - top
    content_w = right - left
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=min(w, h) // 6,
        minLineLength=min(w, h) // 4,
        maxLineGap=10,
    )

    if lines is not None and content_h > 50 and content_w > 50:
        v_xs = []  # candidate Y-axis x positions (vertical lines)
        h_ys = []  # candidate X-axis y positions (horizontal lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            span_y = abs(y2 - y1)
            span_x = abs(x2 - x1)
            # Vertical line spanning most of the content height = Y axis
            if 80 < angle < 100 and abs(x1 - x2) < 5 and span_y > content_h * 0.5:
                v_xs.append((x1 + x2) // 2)
            # Horizontal line spanning most of the content width = X axis
            elif (angle < 10 or angle > 170) and abs(y1 - y2) < 5 and span_x > content_w * 0.5:
                h_ys.append((y1 + y2) // 2)

        # Y axis = leftmost long vertical line within content bounds
        for axis_x in sorted(v_xs):
            if left <= axis_x < right and axis_x < left + content_w * 0.3:
                left = axis_x
                break
        # X axis = bottommost long horizontal line within content bounds
        for axis_y in sorted(h_ys, reverse=True):
            if top < axis_y <= bottom and axis_y > top + content_h * 0.5:
                bottom = axis_y
                break

    # Apply small margin
    mx = int(w * margin_pct)
    my = int(h * margin_pct)
    candidate = PlotArea(
        left=max(0, left + mx),
        top=max(0, top + my),
        right=min(w, right - mx),
        bottom=min(h, bottom - my),
    )

    # --- Validate: fall back if degenerate or too small ---
    cand_w = candidate.right - candidate.left
    cand_h = candidate.bottom - candidate.top
    if cand_w < w * 0.3 or cand_h < h * 0.3 or cand_w <= 0 or cand_h <= 0:
        logger.warning(
            "Plot area detection degenerate (l=%d t=%d r=%d b=%d), using fallback",
            candidate.left, candidate.top, candidate.right, candidate.bottom,
        )
        return fallback

    return candidate


# ---------------------------------------------------------------------------
# Curve tracing
# ---------------------------------------------------------------------------


@dataclass
class TraceConfig:
    """Configuration for curve tracing."""

    vertical_merge_px: int = 3  # merge Y values within this distance in a column
    min_points: int = 5  # minimum traced points to consider a curve valid
    dilate_px: int = 1  # dilate mask to connect anti-aliased pixels
    max_slope_px: int = 25  # max Y-pixel movement per column (continuity constraint)


def trace_curve(
    img_bgr: np.ndarray,
    plot_area: PlotArea,
    color_hex: str = "",
    color_name: str = "",
    config: TraceConfig | None = None,
) -> list[TracedPoint]:
    """Trace a single curve within the plot area.

    Uses a full-width continuity-constrained path search.  Competing Y
    clusters remain alive until the scan completes, so a same-colour legend or
    annotation cannot win merely because it appears first.

    Returns raw pixel points with sub-pixel Y accuracy (calibration to
    values happens separately).
    """
    if config is None:
        config = TraceConfig()

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv, color_hex, color_name)

    if mask.sum() == 0:
        logger.warning("Color mask is empty for hex=%s name=%s", color_hex, color_name)
        return []

    # Exclude grid-line pixels (prevents gray curves matching the grid)
    grid_mask = _build_grid_exclusion_mask(img_bgr, plot_area)
    mask[grid_mask > 0] = 0

    # Dilate to connect anti-aliased / broken curve segments
    if config.dilate_px > 0:
        kernel = np.ones((config.dilate_px * 2 + 1, config.dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Restrict to plot area
    region_mask = np.zeros_like(mask)
    region_mask[plot_area.top:plot_area.bottom, plot_area.left:plot_area.right] = 255
    mask = mask & region_mask

    # Keep competing trajectories until the whole image has been scanned.
    # Choosing a single cluster in the first column is brittle: a same-colour
    # annotation can sit nearer the plot centre than the real curve.  The
    # longest smooth path is identifiable only with full-width evidence.
    frontier: list[tuple[int, float, int, float, tuple | None]] = []
    for x in range(plot_area.left, plot_area.right):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) == 0:
            continue
        clusters = _cluster_values(ys, config.vertical_merge_px)
        next_frontier = []
        for y_center in clusters:
            predecessors = [
                node for node in frontier
                if config.max_slope_px <= 0
                or abs(node[1] - y_center) <= config.max_slope_px * max(1, x - node[0])
            ]
            predecessor = max(
                predecessors,
                key=lambda node: (node[2], -(node[3] + abs(node[1] - y_center) ** 2)),
                default=None,
            )
            length = predecessor[2] + 1 if predecessor else 1
            delta = abs(predecessor[1] - y_center) if predecessor else 0.0
            roughness = predecessor[3] + delta * delta if predecessor else 0.0
            next_frontier.append((x, y_center, length, roughness, predecessor))
        frontier = next_frontier

    if not frontier:
        return []

    node = max(frontier, key=lambda item: (item[2], -item[3]))
    path: list[tuple[int, float]] = []
    while node is not None:
        path.append((node[0], node[1]))
        node = node[4]
    path.reverse()

    points: list[TracedPoint] = []
    for x, y_center in path:
        col = mask[:, x]
        y_int = int(round(y_center))
        y_lo = max(0, y_int - config.vertical_merge_px)
        y_hi = min(mask.shape[0], y_int + config.vertical_merge_px + 1)
        local_ys = np.arange(y_lo, y_hi)
        local_weights = col[y_lo:y_hi].astype(np.float64)
        if local_weights.sum() > 0:
            y_subpixel = float(np.average(local_ys, weights=local_weights))
        else:
            y_subpixel = y_center
        points.append(TracedPoint(x_px=x, y_px=int(round(y_subpixel))))

    return points


def _cluster_values(values: np.ndarray, max_gap: int) -> list[float]:
    """Cluster sorted integer values, return cluster centers."""
    if len(values) == 0:
        return []

    clusters: list[list[int]] = [[int(values[0])]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= max_gap:
            clusters[-1].append(int(v))
        else:
            clusters.append([int(v)])

    return [float(np.mean(c)) for c in clusters]


# ---------------------------------------------------------------------------
# Coordinate calibration
# ---------------------------------------------------------------------------


def calibrate_y(
    points: list[TracedPoint],
    y_anchors: list[AxisAnchor],
    plot_area: PlotArea,
) -> list[TracedPoint]:
    """Map Y pixels to data values using axis anchors.

    Uses linear interpolation between anchors.  Note: image Y axis is
    inverted (top=0), so higher pixel Y = lower data value.
    """
    if not y_anchors:
        return points

    # Sort anchors by pixel position (top to bottom in image = high to low value)
    anchors = sorted(y_anchors, key=lambda a: a.px)

    for pt in points:
        pt.value = _interpolate(pt.y_px, [(a.px, a.value) for a in anchors])

    return points


def calibrate_x_dates(
    points: list[TracedPoint],
    x_anchors: list[AxisAnchor],
    x_labels: list[str],
    plot_area: PlotArea,
) -> list[TracedPoint]:
    """Map X pixels to date strings using axis anchors + tick labels.

    x_anchors: pixel positions of X axis ticks (from user clicks or detection)
    x_labels: the date labels at those ticks (from VLM or user input)
    """
    if not x_anchors or not x_labels:
        return points

    # Pair anchors with labels (assume same order, left→right)
    anchors = sorted(x_anchors, key=lambda a: a.px)
    n = min(len(anchors), len(x_labels))
    pairs = [(anchors[i].px, x_labels[i]) for i in range(n)]

    for pt in points:
        pt.date = _interpolate_label(pt.x_px, pairs)

    return points


def _interpolate(px: int, pairs: list[tuple[int, float]]) -> float | None:
    """Linear interpolation/extrapolation from (pixel, value) pairs."""
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0][1]

    pairs = sorted(pairs, key=lambda p: p[0])

    # Clamp to range (slight extrapolation allowed)
    if px <= pairs[0][0]:
        p0, p1 = pairs[0], pairs[1]
    elif px >= pairs[-1][0]:
        p0, p1 = pairs[-2], pairs[-1]
    else:
        for i in range(len(pairs) - 1):
            if pairs[i][0] <= px <= pairs[i + 1][0]:
                p0, p1 = pairs[i], pairs[i + 1]
                break
        else:
            return None

    if p1[0] == p0[0]:
        return p0[1]

    t = (px - p0[0]) / (p1[0] - p0[0])
    return p0[1] + t * (p1[1] - p0[1])


def _interpolate_label(px: int, pairs: list[tuple[int, str]]) -> str:
    """Interpolate date label for a pixel position using linear date interpolation.

    Parses date labels to ordinal numbers, interpolates linearly, then
    converts back to ISO format. Falls back to nearest-neighbor if parsing fails.
    """
    if not pairs:
        return ""

    # Try to parse labels as dates for linear interpolation
    from datetime import date, datetime

    parsed: list[tuple[int, float]] = []  # (pixel, ordinal)
    for tick_px, label in pairs:
        ordinal = _parse_date_to_ordinal(label)
        if ordinal is not None:
            parsed.append((tick_px, ordinal))

    if len(parsed) >= 2:
        # Linear interpolation on ordinal values
        ordinal = _interpolate(px, parsed)
        if ordinal is not None:
            try:
                d = date.fromordinal(int(round(ordinal)))
                return d.isoformat()
            except (ValueError, OverflowError):
                pass

    # Fallback: nearest-neighbor
    best_label = pairs[0][1]
    best_dist = abs(px - pairs[0][0])
    for tick_px, label in pairs[1:]:
        dist = abs(px - tick_px)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label


def _parse_date_to_ordinal(label: str) -> float | None:
    """Parse a date label string to a Python ordinal (days since 0001-01-01).

    Supports formats: 2024-01, 2024-01-15, 2024/01, 20240115, 2024年1月, etc.
    """
    from datetime import date, datetime
    import re as _re

    label = label.strip()
    if not label:
        return None

    # Try common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d", "%Y-%m", "%Y/%m", "%Y.%m"):
        try:
            dt = datetime.strptime(label, fmt)
            return float(dt.date().toordinal())
        except ValueError:
            continue

    # Chinese format: 2024年1月15日 or 2024年1月
    m = _re.match(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?", label)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 1
        try:
            return float(date(y, mo, d).toordinal())
        except ValueError:
            return None

    # Partial date like "2024-01" → first of month
    m = _re.match(r"(\d{4})-(\d{1,2})$", label)
    if m:
        try:
            return float(date(int(m.group(1)), int(m.group(2)), 1).toordinal())
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Reference calibration (table cross-validation)
# ---------------------------------------------------------------------------


def calibrate_from_reference(
    points: list[TracedPoint],
    reference: list[tuple[str, float]],
    min_matches: int = 2,
) -> list[TracedPoint]:
    """Recalibrate traced values using known reference points (e.g. from a NAV table).

    When a report contains both a chart and a data table for the same product,
    the table provides ground-truth (date, NAV) pairs. This function:
    1. Finds traced points whose dates match reference dates
    2. Fits a linear model: true_value = scale * traced_value + offset
    3. Applies the correction to all traced points

    This corrects systematic Y-axis calibration errors (wrong tick reading,
    imprecise plot area bounds, etc.) using the table as anchor.

    Parameters
    ----------
    points : traced points with .date and .value set
    reference : list of (date_str, nav_value) from table extraction
    min_matches : minimum matching points required to apply correction

    Returns
    -------
    The same points list with corrected .value fields.
    """
    if not reference or not points:
        return points

    # Build lookup: date → reference value
    ref_map: dict[str, float] = {}
    for date_str, val in reference:
        if date_str and val > 0:
            ref_map[date_str] = val

    if len(ref_map) < min_matches:
        return points

    # Find matching points
    matched_traced: list[float] = []
    matched_ref: list[float] = []
    for p in points:
        if p.date and p.value and p.date in ref_map:
            matched_traced.append(p.value)
            matched_ref.append(ref_map[p.date])

    if len(matched_traced) < min_matches:
        logger.debug("Reference calibration: only %d date matches (need %d), skipping",
                     len(matched_traced), min_matches)
        return points

    # Least-squares linear fit: ref = scale * traced + offset
    n = len(matched_traced)
    sum_x = sum(matched_traced)
    sum_y = sum(matched_ref)
    sum_xy = sum(x * y for x, y in zip(matched_traced, matched_ref))
    sum_x2 = sum(x * x for x in matched_traced)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        # All traced values are identical — just apply offset
        offset = sum_y / n - sum_x / n
        scale = 1.0
    else:
        scale = (n * sum_xy - sum_x * sum_y) / denom
        offset = (sum_y - scale * sum_x) / n

    # Sanity: scale should be close to 1.0 (within 0.5x to 2x)
    if scale < 0.5 or scale > 2.0:
        logger.warning("Reference calibration scale=%.3f out of range, skipping", scale)
        return points

    # Apply correction
    corrected_count = 0
    for p in points:
        if p.value is not None:
            new_val = scale * p.value + offset
            if new_val > 0:
                p.value = round(new_val, 6)
                corrected_count += 1

    logger.info("Reference calibration: %d matches, scale=%.4f, offset=%.6f, corrected %d points",
                n, scale, offset, corrected_count)
    return points


def calibrate_from_disclosed(
    points: list[TracedPoint],
    start_date: str,
    end_date: str,
    cumulative_return: float,
    start_nav: float = 1.0,
    edge_window: int = 5,
) -> list[TracedPoint]:
    """Calibrate a raw pixel curve using disclosed start/end metrics.

    Used when axis labels are unavailable (a tight crop around the curve
    excludes them). The curve endpoints are anchored to the disclosed values:

    - leftmost traced position  -> (start_date, start_nav)
    - rightmost traced position -> (end_date, start_nav * (1 + cumulative_return))

    This assumes the chart spans the full disclosed period — the standard
    layout for weekly-report NAV charts. Endpoint Y is taken as the median
    over a small window to reduce single-pixel noise.

    Returns the same points with .value and .date populated.
    """
    if len(points) < 2 or not start_date or not end_date:
        return points

    # Normalize to ISO strings (inputs may be datetime.date or str).
    start_date = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)
    end_date = end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date)

    end_nav = start_nav * (1.0 + cumulative_return)

    # Robust endpoint positions: median over a small window at each end.
    window = min(edge_window, len(points))
    left_y = float(np.median([p.y_px for p in points[:window]]))
    right_y = float(np.median([p.y_px for p in points[-window:]]))
    left_x = points[0].x_px
    right_x = points[-1].x_px

    if right_x <= left_x:
        return points

    # Y mapping (pixel y -> NAV). Guard the degenerate flat-curve case.
    if abs(right_y - left_y) < 1.0:
        for p in points:
            p.value = round(start_nav, 6)
    else:
        y_anchors = [
            AxisAnchor(axis="y", px=int(round(left_y)), value=start_nav, label=f"{start_nav:.4f}"),
            AxisAnchor(axis="y", px=int(round(right_y)), value=end_nav, label=f"{end_nav:.4f}"),
        ]
        calibrate_y(points, y_anchors, None)

    # X mapping (pixel x -> date), linear between the two endpoints.
    x_anchors = [
        AxisAnchor(axis="x", px=left_x, value=0.0, label=start_date),
        AxisAnchor(axis="x", px=right_x, value=0.0, label=end_date),
    ]
    calibrate_x_dates(points, x_anchors, [start_date, end_date], None)

    return points


# ---------------------------------------------------------------------------
# High-level: trace all curves in a chart
# ---------------------------------------------------------------------------


def trace_all_curves(
    img_bgr: np.ndarray,
    curve_specs: list[dict],
    plot_area: PlotArea | None = None,
    y_anchors: list[AxisAnchor] | None = None,
    x_anchors: list[AxisAnchor] | None = None,
    x_labels: list[str] | None = None,
    config: TraceConfig | None = None,
) -> list[TracedCurve]:
    """Trace all specified curves in a chart image.

    Parameters
    ----------
    img_bgr : chart image (BGR)
    curve_specs : list of dicts with name/color_hex/color_name/is_benchmark
    plot_area : plot region (auto-detected if None)
    y_anchors : Y axis pixel↔value anchors
    x_anchors : X axis tick pixel positions
    x_labels : X axis tick date labels
    config : tracing configuration

    Returns
    -------
    List of TracedCurve with calibrated points.
    """
    if config is None:
        config = TraceConfig()

    if plot_area is None:
        plot_area = detect_plot_area(img_bgr)

    results: list[TracedCurve] = []

    for spec in curve_specs:
        color_hex = spec.get("color_hex", "")
        color_name = spec.get("color_name", "")
        name = spec.get("name", "")
        is_benchmark = spec.get("is_benchmark", False)

        # If user provided an anchor pixel, sample the actual color there
        anchor_px = spec.get("anchor_px")
        if anchor_px and not color_hex:
            ax, ay = anchor_px
            color_hex = sample_color_at_pixel(img_bgr, ax, ay)
            logger.info("Sampled color %s at (%d,%d) for curve '%s'", color_hex, ax, ay, name)

        points = trace_curve(img_bgr, plot_area, color_hex, color_name, config)

        if len(points) < config.min_points:
            logger.warning(
                "Curve '%s' (hex=%s, name=%s): only %d points traced, skipping",
                name, color_hex, color_name, len(points),
            )
            continue

        # Quality belongs to the raw pixel path.  Calibration may merge many
        # pixel columns into one disclosed date, which must not make a good
        # visual trace appear to have poor horizontal coverage.
        raw_quality = assess_trace_quality(points, plot_area, config)

        # Calibrate
        if y_anchors:
            points = calibrate_y(points, y_anchors, plot_area)
        if x_anchors and x_labels:
            points = calibrate_x_dates(points, x_anchors, x_labels, plot_area)

        # Raw pixels are useful evidence in the calibration preview.  They
        # have neither dates nor values yet, so sanitising them here would
        # erase a valid trace before the user can calibrate it.
        if y_anchors or (x_anchors and x_labels):
            points = sanitize_nav_points(points)

        results.append(TracedCurve(
            name=name,
            color_hex=color_hex,
            is_benchmark=is_benchmark,
            points=points,
            num_pixels_traced=len(points),
            quality=raw_quality,
        ))

    return results


def assess_trace_quality(
    points: list[TracedPoint],
    plot_area: PlotArea,
    config: TraceConfig | None = None,
) -> dict[str, float | bool]:
    """Return deterministic evidence that a trace is safe to auto-apply.

    The checks intentionally operate on the path returned by CV rather than
    on the NAV values, so they work before any potentially unreliable axis
    calibration.  A long same-colour legend can fail both coverage and slope
    checks; valid sparse/dashed lines retain their trace but are routed to
    the click-to-calibrate review path instead of silently becoming NAV data.
    """
    if config is None:
        config = TraceConfig()
    width = max(1, plot_area.right - plot_area.left)
    if len(points) < 2:
        return {
            "coverage_ratio": len(points) / width,
            "max_y_jump_per_x": 0.0,
            "is_reliable": False,
        }

    coverage = len({point.x_px for point in points}) / width
    jumps = [
        abs(current.y_px - previous.y_px) / max(1, current.x_px - previous.x_px)
        for previous, current in zip(points, points[1:])
        if current.x_px > previous.x_px
    ]
    max_jump = max(jumps, default=0.0)
    # A real curve may have a sharp disclosed NAV move, but should not exceed
    # the same continuity bound that the tracker used to select it.
    reliable = coverage >= 0.40 and max_jump <= config.max_slope_px * 1.25
    return {
        "coverage_ratio": round(coverage, 4),
        "max_y_jump_per_x": round(max_jump, 4),
        "is_reliable": reliable,
    }


def sanitize_nav_points(
    points: list[TracedPoint],
    max_jump_pct: float = 0.30,
) -> list[TracedPoint]:
    """Post-process traced points: deduplicate dates and remove outlier jumps.

    Steps:
    1. Keep only points with calibrated values (value is not None)
    2. Group by date, take median value for duplicates
    3. Sort by date
    4. Remove points that jump > max_jump_pct from their neighbors (likely
       tracing errors where the mask picked up a different curve)
    """
    # Filter to calibrated points only
    calibrated = [p for p in points if p.value is not None and p.value > 0]
    if len(calibrated) < 3:
        return calibrated

    # Deduplicate by date (keep median)
    date_groups: dict[str, list[TracedPoint]] = {}
    no_date: list[TracedPoint] = []
    for p in calibrated:
        if p.date:
            date_groups.setdefault(p.date, []).append(p)
        else:
            no_date.append(p)

    deduped: list[TracedPoint] = []
    for date_str, group in date_groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Keep the point with median value
            values = sorted(g.value for g in group)
            median_val = values[len(values) // 2]
            best = min(group, key=lambda g: abs(g.value - median_val))
            deduped.append(best)

    # If no dates, just use pixel order (already sorted by x)
    if not deduped and no_date:
        deduped = no_date

    # Sort by date
    deduped.sort(key=lambda p: p.date or "")

    # Cumulative-return charts legitimately start close to zero.  Applying a
    # percentage jump rule there mistakes ordinary basis-point moves for NAV
    # outliers; callers convert these review candidates to NAV afterwards.
    if deduped and max(float(point.value or 0.0) for point in deduped) < 0.5:
        return deduped

    # Remove outlier jumps
    if len(deduped) < 3:
        return deduped

    cleaned: list[TracedPoint] = [deduped[0]]
    for i in range(1, len(deduped)):
        prev_val = cleaned[-1].value
        curr_val = deduped[i].value
        if prev_val and prev_val > 0:
            jump = abs(curr_val - prev_val) / prev_val
            if jump > max_jump_pct:
                # Check if it's a spike (next point returns to normal)
                if i + 1 < len(deduped):
                    next_val = deduped[i + 1].value
                    if next_val and abs(next_val - prev_val) / prev_val < max_jump_pct * 0.5:
                        # Spike: skip this point
                        logger.debug("Removing spike at %s: %.4f (jump %.1f%%)",
                                     deduped[i].date, curr_val, jump * 100)
                        continue
                # Large jump but might be real — keep if within 2x max_jump
                if jump > max_jump_pct * 2:
                    logger.debug("Removing outlier at %s: %.4f (jump %.1f%%)",
                                 deduped[i].date, curr_val, jump * 100)
                    continue
        cleaned.append(deduped[i])

    removed = len(deduped) - len(cleaned)
    if removed > 0:
        logger.info("Sanitize: removed %d outlier/duplicate points (%d → %d)",
                    removed, len(deduped), len(cleaned))

    return cleaned
