"""Chart region detection using OpenCV.

Detects line-chart regions in rendered PDF pages by finding:
- Long straight lines (axes) via Hough transform
- Dense colored regions (curves) via color saturation analysis
- Rectangular contours that enclose chart-like content

This is the zero-cost preprocessing step before VLM extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChartRegion:
    """A detected chart region in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0
    score_breakdown: dict = field(default_factory=dict)


def detect_chart_regions(
    image_bytes: bytes,
    min_area_ratio: float = 0.05,
    max_regions: int = 5,
) -> list[ChartRegion]:
    """Detect chart-like regions in a rendered page image.

    Strategy:
    1. Find long horizontal/vertical lines (axes candidates)
    2. Find regions with high color saturation (curve candidates)
    3. Combine into bounding boxes and score them

    Parameters
    ----------
    image_bytes : PNG image bytes
    min_area_ratio : minimum region area as fraction of page area
    max_regions : maximum number of regions to return

    Returns
    -------
    List of ChartRegion sorted by confidence (descending).
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Failed to decode image for chart detection")
        return []

    h, w = img.shape[:2]
    page_area = h * w
    min_area = page_area * min_area_ratio

    regions: list[ChartRegion] = []

    # --- Strategy 1: Axis line detection ---
    axis_regions = _detect_by_axis_lines(img, w, h, min_area)
    regions.extend(axis_regions)

    # --- Strategy 2: Color saturation clustering ---
    color_regions = _detect_by_color_saturation(img, w, h, min_area)
    regions.extend(color_regions)

    # --- Merge overlapping regions ---
    merged = _merge_overlapping(regions, iou_threshold=0.3)

    # A region spanning nearly the whole page is the page itself, not a chart.
    merged = [r for r in merged if (r.w * r.h) / page_area < 0.85]

    # --- Score and sort ---
    for region in merged:
        region.confidence = _score_region(img, region)

    merged.sort(key=lambda r: r.confidence, reverse=True)
    return merged[:max_regions]


def _detect_by_axis_lines(
    img: np.ndarray, page_w: int, page_h: int, min_area: float
) -> list[ChartRegion]:
    """Find chart regions by detecting axis-like long lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines via probabilistic Hough
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=min(page_w, page_h) // 8,
        minLineLength=min(page_w, page_h) // 6,
        maxLineGap=20,
    )

    if lines is None or len(lines) < 2:
        return []

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:
            h_lines.append((x1, y1, x2, y2))
        elif 80 < angle < 100:
            v_lines.append((x1, y1, x2, y2))

    if not h_lines or not v_lines:
        return []

    # Find clusters of H+V lines that form an L-shape (axis)
    regions = []
    for v_line in v_lines:
        vx, vy1, _, vy2 = v_line
        v_top = min(vy1, vy2)
        v_bot = max(vy1, vy2)

        for h_line in h_lines:
            hx1, hy, hx2, _ = h_line
            h_left = min(hx1, hx2)
            h_right = max(hx1, hx2)

            # Check if they form an axis corner (vertical meets horizontal)
            if abs(vx - h_left) < 30 or abs(vx - h_right) < 30:
                if abs(v_bot - hy) < 30 or abs(v_top - hy) < 30:
                    # Bounding box of the chart area
                    x = min(vx, h_left)
                    y = min(v_top, hy)
                    rw = abs(h_right - h_left)
                    rh = abs(v_bot - v_top)

                    if rw * rh >= min_area and rw > 100 and rh > 80:
                        regions.append(ChartRegion(
                            x=max(0, x - 10),
                            y=max(0, y - 10),
                            w=min(page_w - x, rw + 20),
                            h=min(page_h - y, rh + 20),
                            score_breakdown={"method": "axis_lines"},
                        ))

    return regions


def _detect_by_color_saturation(
    img: np.ndarray, page_w: int, page_h: int, min_area: float
) -> list[ChartRegion]:
    """Find regions with concentrated color (chart curves)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # High saturation pixels (colored curves vs gray text)
    sat_mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)

    # Horizontally-biased closing: connect curve segments left-to-right while
    # NOT merging vertically-stacked elements (a report page's title, tables and
    # logo would otherwise collapse into one whole-page blob with the old
    # isotropic 30x30 close + heavy dilation).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 8))
    closed = cv2.morphologyEx(sat_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)

    # Modest dilation to thicken the curve blob (kept light to avoid re-merging)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area >= min_area and rw > 150 and rh > 100:
            # Check aspect ratio (charts are usually wider than tall)
            aspect = rw / max(rh, 1)
            if 0.5 < aspect < 5.0:
                regions.append(ChartRegion(
                    x=x, y=y, w=rw, h=rh,
                    score_breakdown={"method": "color_saturation"},
                ))

    return regions


def _merge_overlapping(
    regions: list[ChartRegion], iou_threshold: float = 0.3
) -> list[ChartRegion]:
    """Merge overlapping regions using non-maximum suppression style."""
    if not regions:
        return []

    # Sort by area descending
    regions.sort(key=lambda r: r.w * r.h, reverse=True)
    merged: list[ChartRegion] = []

    for region in regions:
        should_add = True
        for existing in merged:
            if _iou(region, existing) > iou_threshold:
                # Merge: take the union bounding box
                x1 = min(region.x, existing.x)
                y1 = min(region.y, existing.y)
                x2 = max(region.x + region.w, existing.x + existing.w)
                y2 = max(region.y + region.h, existing.y + existing.h)
                existing.x, existing.y = x1, y1
                existing.w, existing.h = x2 - x1, y2 - y1
                should_add = False
                break
        if should_add:
            merged.append(region)

    return merged


def _iou(a: ChartRegion, b: ChartRegion) -> float:
    """Compute intersection-over-union of two regions."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter
    return inter / max(union, 1)


def _score_region(img: np.ndarray, region: ChartRegion) -> float:
    """Score a region based on chart-likeness heuristics."""
    score = 0.0

    # Crop region
    crop = img[region.y:region.y + region.h, region.x:region.x + region.w]
    if crop.size == 0:
        return 0.0

    # 1. Color diversity (charts have multiple colored lines)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    colored_mask = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 40)
    color_ratio = colored_mask.sum() / colored_mask.size
    if 0.01 < color_ratio < 0.6:
        score += 0.3  # some color but not overwhelming

    # 2. Edge density (axes + grid lines + curves create edges)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.sum() / (edges.size * 255)
    if 0.02 < edge_ratio < 0.3:
        score += 0.3

    # 3. Aspect ratio bonus (charts tend to be ~16:9 to 4:3)
    aspect = region.w / max(region.h, 1)
    if 1.0 < aspect < 2.5:
        score += 0.2

    # 4. Size bonus (larger regions more likely to be main charts)
    page_h, page_w = img.shape[:2]
    area_ratio = (region.w * region.h) / (page_w * page_h)
    if area_ratio > 0.1:
        score += 0.2

    # 5. Method bonus
    if region.score_breakdown.get("method") == "axis_lines":
        score += 0.1  # axis detection is more reliable

    return min(score, 1.0)


def crop_region(
    image_bytes: bytes,
    region: ChartRegion,
    padding: int = 5,
    left_padding: int | None = None,
) -> bytes:
    """Crop a chart region from the full page image, with padding.

    left_padding defaults to 15% of region width to capture Y-axis labels
    that sit outside the detected plot area.

    Returns PNG bytes of the cropped region.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return b""

    h, w = img.shape[:2]
    lp = left_padding if left_padding is not None else max(padding, int(region.w * 0.15))
    x1 = max(0, region.x - lp)
    y1 = max(0, region.y - padding)
    x2 = min(w, region.x + region.w + padding)
    y2 = min(h, region.y + region.h + padding)

    crop = img[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else b""
