"""Conservative chart-image digitization with optional local OCR.

The extracted values are candidates only.  Axis calibration remains explicit
because chart screenshots rarely provide enough reliable information to infer
units or dates safely.
"""

from __future__ import annotations

import re
import warnings
from datetime import date, timedelta
from typing import Literal

import numpy as np

# Optional dependency: imported once at module level instead of lazily inside
# each function (issue #5).  Functions that need it check for None.
try:
    import cv2
except ImportError:  # pragma: no cover - environment without opencv
    cv2 = None  # type: ignore[assignment]

from app.schemas import ChartCurvePoint, DataFrequency, NavImageDigitizationResponse, NetAssetValuePoint


def _require_cv2():
    """Return the cv2 module or raise a user-facing error (issue #5)."""
    if cv2 is None:
        raise ValueError("图像识别依赖未安装。请运行 pip install -r requirements.txt")
    return cv2


def digitize_nav_image(
    image_bytes: bytes,
    start_date: date,
    end_date: date,
    nav_min: float,
    nav_max: float,
    value_mode: Literal["nav", "cumulative_return"],
    frequency: Literal["auto", "daily", "weekly", "monthly"],
    start_x_ratio: float | None = None,
    end_x_ratio: float | None = None,
    line_kind: Literal["product", "benchmark"] = "product",
    top_y_ratio: float | None = None,
    bottom_y_ratio: float | None = None,
    line_color: str | None = None,
) -> NavImageDigitizationResponse:
    """Extract a saturated line and map it with explicitly supplied axes."""
    if not image_bytes:
        raise ValueError("image file is empty")
    if end_date <= start_date:
        raise ValueError("end_date must be later than start_date")
    if nav_max <= nav_min:
        raise ValueError("nav_max must be greater than nav_min")
    if (start_x_ratio is None) != (end_x_ratio is None):
        raise ValueError("start_x_ratio and end_x_ratio must be supplied together")
    if start_x_ratio is not None and (not 0 <= start_x_ratio < end_x_ratio <= 1):
        raise ValueError("date anchor positions must satisfy 0 <= start < end <= 1")
    if (top_y_ratio is None) != (bottom_y_ratio is None):
        raise ValueError("top_y_ratio and bottom_y_ratio must be supplied together")
    if top_y_ratio is not None and (not 0 <= top_y_ratio < bottom_y_ratio <= 1):
        raise ValueError("value anchor positions must satisfy 0 <= top < bottom <= 1")

    cv2 = _require_cv2()

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片；请上传 PNG、JPG 或 JPEG 文件")

    height, width = image.shape[:2]
    if width < 80 or height < 80:
        raise ValueError("图片分辨率过低，无法可靠识别曲线")

    # In weekly reports, a blue table often spans wider than the chart.  The
    # red product curve is a more reliable chart locator than "any colour".
    selected_mask = _colour_line_mask(image, line_color) if line_color else (
        _red_line_mask(image) if line_kind == "product" else _grey_line_mask(image)
    )
    detected_left, detected_top, detected_right, detected_bottom = _chart_bounds_from_curve(selected_mask)
    # Manual anchors define the search area as well as the numerical axis.
    # Previously they only calibrated values after detection, so a red header
    # or decorative block above the selected chart could win curve detection
    # and then be drawn as a candidate outside the user's framed region.
    left = round(start_x_ratio * width) if start_x_ratio is not None else detected_left
    right = round(end_x_ratio * width) if end_x_ratio is not None else detected_right
    top = round(top_y_ratio * height) if top_y_ratio is not None else detected_top
    bottom = round(bottom_y_ratio * height) if bottom_y_ratio is not None else detected_bottom
    if right - left < 8 or bottom - top < 8:
        raise ValueError("手工框选区域过小，请重新标记横轴和纵轴范围")
    chart = image[top:bottom, left:right]
    mask = _colour_line_mask(chart, line_color) if line_color else (
        _red_line_mask(chart) if line_kind == "product" else _grey_line_mask(chart)
    )
    mask = _keep_widest_component(mask)
    points = _trace_curve(mask)
    if len(points) < 12:
        line_label = f"{line_color} 曲线" if line_color else ("红色产品" if line_kind == "product" else "灰色基准")
        raise ValueError(f"未找到足够连续的{line_label}曲线。请上传清晰图表或改选另一条曲线")

    span_days = (end_date - start_date).days
    chart_height, chart_width = mask.shape
    curve_start_x, curve_end_x = points[0][0], points[-1][0]
    curve_width = max(curve_end_x - curve_start_x, 1)
    anchor_start_x = (start_x_ratio * width - left) if start_x_ratio is not None else None
    anchor_end_x = (end_x_ratio * width - left) if end_x_ratio is not None else None
    # A curve has one coordinate for nearly every pixel column, not one genuine
    # NAV observation per calendar day.  Resolve the disclosure frequency
    # before sampling so that a weekly report can never be silently expanded
    # into noisy daily returns.
    ocr_text = _try_ocr(image)
    resolved_frequency, inference_note = _resolve_extraction_frequency(
        frequency, ocr_text, (end_date - start_date).days
    )
    selected = _resample_curve(
        points, start_date, end_date, resolved_frequency, anchor_start_x, anchor_end_x
    )

    # Prefer reading the printed Y-axis labels: this recovers the true scale even
    # when the curve does not span the full axis range, which the bounding-box
    # fallback assumes.  It removes the need for pixel-perfect manual calibration
    # on full-page report images.
    axis_mapping = None
    if top_y_ratio is None:
        strip_right = min(width, max(left + round((right - left) * 0.12), round(width * 0.10)))
        chart_span = bottom - top
        references = _y_axis_references(
            image, strip_right, top - round(chart_span * 0.10), bottom + round(chart_span * 0.30)
        )
        axis_mapping = _fit_axis_mapping(references)

    # Second priority: detect horizontal grid lines to infer the Y scale.
    # Expand the search area vertically: grid lines span the full axis range,
    # which is often larger than the curve's bounding box.
    grid_mapping = None
    if axis_mapping is None and top_y_ratio is None:
        chart_span = bottom - top
        grid_top = max(0, top - round(chart_span * 0.35))
        grid_bottom = min(height, bottom + round(chart_span * 0.35))
        grid_mapping = _grid_line_mapping(image, left, grid_top, right, grid_bottom, nav_min, nav_max)

    if axis_mapping is not None:
        slope, intercept = axis_mapping
        axis_values = [intercept + slope * (top + y) for _, _, y in selected]
    elif grid_mapping is not None:
        g_slope, g_intercept = grid_mapping
        axis_values = [g_intercept + g_slope * (top + y) for _, _, y in selected]
    elif top_y_ratio is None:
        axis_values = [nav_max - y / max(chart_height - 1, 1) * (nav_max - nav_min) for _, _, y in selected]
    else:
        top_y, bottom_y = top_y_ratio * height, bottom_y_ratio * height
        axis_values = [nav_max - ((top + y) - top_y) / (bottom_y - top_y) * (nav_max - nav_min) for _, _, y in selected]
    if value_mode == "cumulative_return":
        # A 80% cumulative return corresponds to a normalised NAV of 1.80.
        axis_values = [1.0 + axis_value / 100.0 for axis_value in axis_values]
    nav_points = [
        NetAssetValuePoint(
            observation_date=observation_date,
            net_asset_value=axis_value,
        )
        for (observation_date, _, _), axis_value in zip(selected, axis_values, strict=True)
    ]
    nav_points = _deduplicate_dates(nav_points)
    coverage = curve_width / max(chart_width - 1, 1)
    warnings = [
        "识别结果为候选净值点；请在计算前核对日期、坐标范围、复权和分红口径。",
        "当前版本按图片横轴位置在所填日期范围内等距映射，未将 OCR 文字直接作为数值依据。",
        inference_note,
        f"已按{_frequency_label(resolved_frequency)}提取 {len(selected)} 个候选点；不会按像素列补齐为日频。",
    ]
    if value_mode == "cumulative_return":
        warnings.append("已将纵轴累计收益率转换为标准化净值：净值 = 1 + 累计收益率。")
    if start_x_ratio is not None:
        warnings.append("已使用人工点击的首尾日期锚线校准横轴。")
    if top_y_ratio is not None or start_x_ratio is not None:
        warnings.append("曲线搜索已限制在人工框选的图表区域内。")
    if axis_mapping is not None:
        warnings.append("已自动识别纵轴刻度标签并据此换算数值（优先于自动图框估计）。")
    elif grid_mapping is not None:
        warnings.append("已通过检测图表水平网格线推断纵轴比例（精度优于图框估计，建议仍核对首尾净值）。")
    elif top_y_ratio is not None:
        warnings.append("已使用人工点击的最高/最低数值锚线校准纵轴。")
    else:
        warnings.append("未能识别纵轴刻度标签，纵轴按曲线外接框估算，误差可能较大；建议点击[标记纵轴最大/最小值]校准。")
    if ocr_text is None:
        warnings.append("未检测到可用的本地 Tesseract OCR；已使用手工填写的坐标范围。")
    confidence = float(min(0.9, max(0.15, coverage * min(1.0, len(points) / chart_width))))
    return NavImageDigitizationResponse(
        nav_points=nav_points,
        confidence=confidence,
        image_width=width,
        image_height=height,
        ocr_text=ocr_text,
        warnings=warnings,
        detected_line=line_kind,
        extraction_frequency=resolved_frequency,
        candidate_curve=[
            ChartCurvePoint(x_ratio=(left + x) / width, y_ratio=(top + y) / height)
            for _, x, y in selected
        ],
    )


def _red_line_mask(image: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 80)).astype(np.uint8)


def _grey_line_mask(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0].astype(int), image[:, :, 1].astype(int), image[:, :, 2].astype(int)
    # The benchmark is a medium-grey stroke; light grid lines are deliberately
    # excluded by the upper brightness bound.
    return ((np.abs(blue - green) <= 18) & (np.abs(green - red) <= 18) & (red >= 55) & (red <= 185)).astype(np.uint8)


def _colour_line_mask(image: np.ndarray, color_hex: str) -> np.ndarray:
    """Return pixels close to a user-selected line colour, including dark strokes."""
    token = color_hex.strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", token):
        raise ValueError("曲线颜色应为 #RRGGBB 格式")
    target = np.array([int(token[4:6], 16), int(token[2:4], 16), int(token[:2], 16)], dtype=np.float32)
    target_hsv = cv2.cvtColor(target.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # For a saturated chosen colour, compare hue rather than raw RGB alone.
    # Charts often draw a dark burgundy curve while the UI swatch is bright
    # red; they are visually the same colour but differ by more than 62 RGB
    # units.  Hue matching remains safely bounded by the user-framed ROI.
    if int(target_hsv[1]) >= 50:
        hue_delta = np.abs(image_hsv[:, :, 0].astype(np.int16) - int(target_hsv[0]))
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        minimum_saturation = max(45, int(target_hsv[1]) // 4)
        return ((hue_delta <= 10) & (image_hsv[:, :, 1] >= minimum_saturation) & (image_hsv[:, :, 2] >= 45)).astype(np.uint8)
    # Squaring int16 channel differences can overflow (e.g. 255²), producing
    # NaN distances and an empty match for an otherwise valid colour choice.
    pixels = image.astype(np.float32)
    distance = np.sqrt(np.sum((pixels - target) ** 2, axis=2))
    # Anti-aliasing makes a plotted line span nearby shades.  62 keeps the
    # user-selected gold/red/blue line while rejecting black text and grids.
    return (distance <= 62).astype(np.uint8)


_AXIS_LABEL_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%?$")


def _y_axis_references(image: np.ndarray, strip_right: int, row_top: int, row_bottom: int) -> list[tuple[int, float]]:
    """OCR the left axis-label column and return (absolute_row, value) pairs.

    Only tokens that parse as a bare number or percentage are kept, so chart
    titles, dates and table text are ignored.  Returns an empty list when no
    usable labels are found or OCR is unavailable.
    """
    if cv2 is None:
        return []
    try:
        import pytesseract
    except Exception:
        return []
    row_top = max(0, row_top)
    row_bottom = min(image.shape[0], row_bottom)
    strip_right = max(1, min(image.shape[1], strip_right))
    if row_bottom <= row_top or strip_right < 10:
        return []
    strip = image[row_top:row_bottom, 0:strip_right]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        data = pytesseract.image_to_data(upscaled, lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    references: list[tuple[int, float]] = []
    for text, top, token_height in zip(data["text"], data["top"], data["height"]):
        token = text.strip().replace(",", "")
        if not token:
            continue
        match = _AXIS_LABEL_PATTERN.fullmatch(token)
        if not match:
            continue
        # Undo the 2x upscale and take the token's vertical centre.
        absolute_row = row_top + int((top + token_height / 2) / 2)
        references.append((absolute_row, float(match.group(1))))
    return references


def _fit_axis_mapping(references: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Fit ``value = intercept + slope * absolute_row`` from axis labels.

    A least-squares line is refined by dropping outliers so that stray OCR
    tokens (table numbers, dates) do not distort the scale.  Returns
    ``(slope, intercept)`` or ``None`` when no consistent mapping exists.
    """
    if len(references) < 2:
        return None
    rows = np.asarray([row for row, _ in references], dtype=float)
    values = np.asarray([value for _, value in references], dtype=float)
    # Suppress harmless polyfit conditioning warnings locally (issue #4) rather
    # than polluting the whole process via a module-level filterwarnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        slope, intercept = np.polyfit(rows, values, 1)
        if slope >= 0:  # axis values must decrease as pixel row increases
            return None
        residuals = np.abs(values - (intercept + slope * rows))
        tolerance = max(0.06 * float(values.max() - values.min()), 0.5)
        inliers = residuals <= tolerance
        if int(inliers.sum()) < 2:
            return None
        slope, intercept = np.polyfit(rows[inliers], values[inliers], 1)
    if slope >= 0:
        return None
    return float(slope), float(intercept)


def _grid_line_mapping(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    nav_min: float,
    nav_max: float,
) -> tuple[float, float] | None:
    """Detect horizontal grid lines and infer a linear Y-axis mapping.

    Most financial charts draw light-grey horizontal lines at each Y-axis tick.
    By finding these rows and assuming they are evenly spaced between nav_min
    and nav_max, we recover the true scale without OCR.

    Returns ``(slope, intercept)`` mapping absolute pixel row → value, or None.
    """
    cv2 = _require_cv2()

    # Work on the chart region only.
    chart_region = image[top:bottom, left:right]
    if chart_region.size == 0:
        return None
    gray = cv2.cvtColor(chart_region, cv2.COLOR_BGR2GRAY)
    region_height, region_width = gray.shape

    # Grid lines are rows where many pixels share a similar light-grey value
    # (typically 200-240 for grid lines on white background, or 160-200 on
    # coloured backgrounds).  We look for rows with low variance and moderate
    # brightness that span a significant width.
    row_means = gray.mean(axis=1)
    row_stds = gray.std(axis=1)

    # A grid line row has: relatively uniform brightness (low std) and is
    # distinguishable from the background.  We detect rows where a large
    # fraction of pixels fall in a narrow brightness band.
    grid_rows: list[int] = []
    min_line_pixels = region_width * 0.35  # at least 35% of width

    for row_idx in range(region_height):
        row = gray[row_idx]
        # Check if this row looks like a horizontal line: many pixels in a
        # narrow brightness range that differs from pure white background.
        mean_val = row_means[row_idx]
        # Grid lines are typically grey (150-230) on white (240-255) background
        if 140 <= mean_val <= 235 and row_stds[row_idx] < 30:
            # Count pixels that are close to the row mean (part of the line)
            in_line = np.sum(np.abs(row.astype(float) - mean_val) < 20)
            if in_line >= min_line_pixels:
                grid_rows.append(row_idx)

    if len(grid_rows) < 3:
        return None

    # Cluster adjacent rows into single grid lines (a line may be 1-3px thick).
    clusters: list[float] = []
    cluster_start = grid_rows[0]
    cluster_end = grid_rows[0]
    for row_idx in grid_rows[1:]:
        if row_idx - cluster_end <= 3:
            cluster_end = row_idx
        else:
            clusters.append((cluster_start + cluster_end) / 2.0)
            cluster_start = row_idx
            cluster_end = row_idx
    clusters.append((cluster_start + cluster_end) / 2.0)

    if len(clusters) < 3:
        return None

    # Verify roughly equal spacing (tolerance 25% of median gap).
    gaps = [clusters[i + 1] - clusters[i] for i in range(len(clusters) - 1)]
    median_gap = float(np.median(gaps))
    if median_gap < 5:  # too close together to be real grid lines
        return None
    consistent = [g for g in gaps if abs(g - median_gap) <= median_gap * 0.25]
    if len(consistent) < len(gaps) * 0.6:
        return None

    # The grid lines span from nav_max (top) to nav_min (bottom) with equal
    # intervals.  Use the first and last detected grid line as anchors.
    top_row_abs = top + clusters[0]
    bottom_row_abs = top + clusters[-1]
    # Assume the topmost grid line = nav_max, bottommost = nav_min.
    # This is the standard chart convention.
    n_intervals = len(clusters) - 1
    value_per_interval = (nav_max - nav_min) / n_intervals

    # Fit a linear mapping: value = intercept + slope * absolute_row
    grid_values = [nav_max - i * value_per_interval for i in range(len(clusters))]
    grid_abs_rows = [top + c for c in clusters]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        slope, intercept = np.polyfit(grid_abs_rows, grid_values, 1)
    if slope >= 0:
        return None
    return float(slope), float(intercept)


def _keep_widest_component(mask: np.ndarray) -> np.ndarray:
    """Discard labels/legend samples and retain the longest continuous stroke."""
    cv2 = _require_cv2()

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    component = max(range(1, count), key=lambda index: (stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_AREA]))
    return (labels == component).astype(np.uint8)


def _chart_bounds_from_curve(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the largest long coloured component, ignoring side-panel icons."""
    cv2 = _require_cv2()

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [
        stat
        for stat in stats[1:component_count]
        if stat[cv2.CC_STAT_WIDTH] >= mask.shape[1] * 0.12
    ]
    if not candidates:
        return (max(1, round(mask.shape[1] * 0.08)), max(1, round(mask.shape[0] * 0.08)), min(mask.shape[1] - 1, round(mask.shape[1] * 0.96)), min(mask.shape[0] - 1, round(mask.shape[0] * 0.90)))
    # The curve usually spans the greatest horizontal distance; labels and
    # coloured decorative blocks are short even when their filled area is big.
    curve = max(candidates, key=lambda stat: (stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_AREA]))
    x, y, component_width, component_height, _ = curve
    horizontal_padding = max(16, round(component_width * 0.05))
    # A cumulative-return curve often starts at the 0% baseline.  Preserve that
    # lower bound instead of adding a large blank margin, which would compress
    # a visible 80% return back toward a 1.1 NAV.
    top_padding = max(8, round(component_height * 0.08))
    bottom_padding = 0
    return (
        max(0, x - horizontal_padding),
        max(0, y - top_padding),
        min(mask.shape[1], x + component_width + horizontal_padding),
        min(mask.shape[0], y + component_height + bottom_padding),
    )


def _trace_curve(mask: np.ndarray) -> list[tuple[int, int]]:
    """Follow the centre of the coloured stroke column-by-column for one curve.

    Tracing the centre (rather than the edge pixel nearest the previous point)
    removes a systematic bias of up to one stroke-width, which matters on
    high-resolution charts where the line is several pixels thick.
    """
    columns = [np.flatnonzero(mask[:, x]) for x in range(mask.shape[1])]
    valid = [index for index, values in enumerate(columns) if values.size]
    if not valid:
        return []
    previous = int(np.median(columns[valid[0]]))
    traced: list[tuple[int, int]] = [(valid[0], previous)]
    max_gap = max(8, round(mask.shape[0] * 0.08))
    for x in range(valid[0] + 1, mask.shape[1]):
        candidates = columns[x]
        if not candidates.size:
            continue
        center = _nearest_run_center(candidates, previous)
        if abs(center - previous) > max_gap:
            continue
        previous = center
        traced.append((x, center))
    return traced


def _nearest_run_center(candidates: np.ndarray, previous: int) -> int:
    """Return the centre of the contiguous candidate run closest to ``previous``.

    A thick stroke yields a vertical run of pixels; taking its centre avoids
    hugging one edge.  If a column holds two separate runs (e.g. a crossing
    line), the run nearest the previous point is chosen to keep continuity.
    """
    runs: list[tuple[int, int]] = []
    run_start = int(candidates[0])
    run_prev = int(candidates[0])
    for value in candidates[1:]:
        value = int(value)
        if value - run_prev > 2:  # tolerate tiny anti-aliasing gaps
            runs.append((run_start, run_prev))
            run_start = value
        run_prev = value
    runs.append((run_start, run_prev))
    best = min(runs, key=lambda run: abs((run[0] + run[1]) / 2 - previous))
    return round((best[0] + best[1]) / 2)


def _resample_curve(
    curve: list[tuple[int, int]],
    start_date: date,
    end_date: date,
    frequency: DataFrequency,
    start_x: float | None,
    end_x: float | None,
) -> list[tuple[date, int, int]]:
    """Sample pixel coordinates at the user-confirmed analysis frequency."""
    interval_days = {
        DataFrequency.DAILY: 1,
        DataFrequency.WEEKLY: 7,
        DataFrequency.MONTHLY: 30,
    }[frequency]
    span_days = (end_date - start_date).days
    offsets = list(range(0, span_days, interval_days)) + [span_days]
    curve_x = np.asarray([point[0] for point in curve], dtype=float)
    curve_y = np.asarray([point[1] for point in curve], dtype=float)
    x_start = curve_x[0] if start_x is None else start_x
    x_end = curve_x[-1] if end_x is None else end_x
    sampled: list[tuple[date, int, int]] = []
    for offset in offsets:
        x = x_start + offset / max(span_days, 1) * (x_end - x_start)
        sampled.append((start_date + timedelta(days=offset), round(x), round(float(np.interp(x, curve_x, curve_y)))))
    return sampled


def _resolve_extraction_frequency(
    requested: Literal["auto", "daily", "weekly", "monthly"],
    ocr_text: str | None,
    span_days: int,
) -> tuple[DataFrequency, str]:
    """Choose a safe chart sampling frequency without inventing daily data.

    OCR is evidence only: a user-selected frequency always wins.  In automatic
    mode the conservative fallback for a long chart is weekly, because CV
    traces represent pixels rather than daily disclosed NAVs.
    """
    explicit = {
        "daily": DataFrequency.DAILY,
        "weekly": DataFrequency.WEEKLY,
        "monthly": DataFrequency.MONTHLY,
    }
    if requested in explicit:
        chosen = explicit[requested]
        return chosen, f"已按用户指定的{_frequency_label(chosen)}提取。"

    text = (ocr_text or "").lower().replace(" ", "")
    if re.search(r"周度|周报|周频|weekly|week", text):
        return DataFrequency.WEEKLY, "自动识别到“周度/周报”披露信息，已按周频提取。"
    if re.search(r"月度|月报|月频|monthly|month", text):
        return DataFrequency.MONTHLY, "自动识别到“月度/月报”披露信息，已按月频提取。"
    if re.search(r"日度|日报|日频|daily|day", text):
        return DataFrequency.DAILY, "自动识别到“日度/日报”披露信息，已按日频提取。"
    if span_days >= 60:
        return DataFrequency.WEEKLY, "未能可靠读出披露频率；为避免将 CV 像素噪声误作日收益，已保守按周频提取。"
    return DataFrequency.DAILY, "未能可靠读出披露频率，且区间较短，暂按日频提取；请在复核前确认。"


def _frequency_label(frequency: DataFrequency) -> str:
    return {
        DataFrequency.DAILY: "日频",
        DataFrequency.WEEKLY: "周频",
        DataFrequency.MONTHLY: "月频",
    }[frequency]


def _deduplicate_dates(points: list[NetAssetValuePoint]) -> list[NetAssetValuePoint]:
    """Keep the last pixel position for dates that round to the same day."""
    by_date = {point.observation_date: point for point in points}
    return list(by_date.values())


def _try_ocr(image: np.ndarray) -> str | None:
    """Return diagnostic OCR text when Tesseract is installed locally."""
    try:
        import pytesseract

        text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
        return text or None
    except Exception:  # Tesseract is optional and must never block digitization.
        return None
