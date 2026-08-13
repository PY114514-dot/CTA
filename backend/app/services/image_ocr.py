"""OCR helpers for arbitrary image regions, used for axis-label auto-fill."""

from __future__ import annotations

import numpy as np


REGION_LANGUAGES = ("chi_sim+eng", "eng")

# Minimum mean word confidence to accept a vertical-text (90 degree)
# rotation result.  Below this the crop was probably not vertical text.
_VERTICAL_MIN_CONF = 50.0


def _rotate(binary: "np.ndarray", angle: float) -> "np.ndarray":
    """Rotate a binary image counter-clockwise by ``angle`` degrees."""
    import cv2

    h, w = binary.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _ink_points(binary: "np.ndarray") -> "np.ndarray | None":
    """Return the merged foreground points of all significant contours."""
    import cv2

    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = binary.shape[:2]
    min_area = (h * w) * 0.001
    points = [c for c in contours if cv2.contourArea(c) > min_area]
    if not points:
        points = [max(contours, key=cv2.contourArea)]
    combined = np.vstack(points)
    if combined.shape[0] < 5:
        return None
    return combined


def _deskew(binary: "np.ndarray") -> "np.ndarray":
    """Rotate a binarized image so that its dominant text line becomes horizontal.

    Handles moderate skew (up to ~45 degrees), which is common for slanted
    X-axis date labels.  Fully vertical text (90 degrees) is detected and
    handled separately by the caller because the min-area-rectangle angle
    normalizes to zero for axis-aligned tall blobs.
    """
    import cv2

    combined = _ink_points(binary)
    if combined is None:
        return binary

    rect = cv2.minAreaRect(combined)
    angle = rect[2]  # in (-90, 0]
    rect_w, rect_h = rect[1]
    # Determine the actual skew: if the bounding box is taller than wide the
    # text baseline is steep and we shift the angle by 90 degrees.
    if rect_h > rect_w:
        angle += 90.0
    # Normalize into [-45, 45] — we never want to flip text upside-down.
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    # Skip negligible skew to avoid interpolation artifacts.
    if abs(angle) < 2.0:
        return binary
    return _rotate(binary, angle)


def _looks_vertical(binary: "np.ndarray") -> bool:
    """Heuristic: is the dominant ink blob taller than wide?

    A single axis label rendered horizontally is always wider than tall, so
    a tall blob means the text was rotated by roughly 90 degrees (vertical
    labels appear on some chart Y axes and densely-packed X axes).  At exactly
    45 degrees the bounding box is square (ratio 1.0), so any ratio above ~1
    implies the text is closer to vertical than to horizontal.
    """
    import cv2

    combined = _ink_points(binary)
    if combined is None:
        return False
    _, _, bw, bh = cv2.boundingRect(combined)
    return bh > bw * 1.05


def _ocr_with_confidence(
    binary: "np.ndarray",
    languages: tuple[str, ...] = REGION_LANGUAGES,
) -> tuple[str | None, float]:
    """OCR returning (text, mean word confidence) using per-word scores.

    ``languages`` controls which Tesseract language models are tried (each
    scored independently, best mean confidence wins).  Vertical axis labels
    are pure ASCII (dates / numbers / percent signs), so the vertical path
    passes ``("eng",)`` — chi_sim inflates confidence on upside-down strokes
    (reading them as '一'), which would let the wrong orientation win.
    """
    import pytesseract

    best_text: str | None = None
    best_conf = -1.0
    for language in languages:
        try:
            data = pytesseract.image_to_data(
                binary, lang=language, config="--psm 7",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        words: list[str] = []
        confs: list[float] = []
        for token, conf in zip(data["text"], data["conf"]):
            token = str(token).strip()
            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                continue
            if token and conf_value >= 0:
                words.append(token)
                confs.append(conf_value)
        if not words:
            continue
        mean_conf = sum(confs) / len(confs)
        if mean_conf > best_conf:
            best_text = " ".join(words)
            best_conf = mean_conf
    return best_text, best_conf


def _ocr_first_match(binary: "np.ndarray") -> str | None:
    """OCR returning the first non-empty text across language fallbacks."""
    import pytesseract

    for language in REGION_LANGUAGES:
        try:
            text = pytesseract.image_to_string(binary, lang=language, config="--psm 7").strip()
            if text:
                return text
        except Exception:
            continue
    return None


def ocr_image_region(
    image_bytes: bytes,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> str | None:
    """Return raw OCR text from a normalized crop of the uploaded image.

    The crop is upscaled and binarized, then automatically corrected for
    rotated labels before Tesseract recognition:

    * slanted labels (up to ~45 degrees) are deskewed via the min-area
      rectangle of the ink region;
    * vertical labels (~90 degrees, detected by a taller-than-wide ink
      blob) are rotated both clockwise and counter-clockwise and the
      higher-confidence result is kept (scored with ``eng`` only, since
      vertical axis labels are ASCII dates/numbers).

    Horizontal labels try ``chi_sim+eng`` first, falling back to ``eng``.
    """
    if not image_bytes:
        raise ValueError("image file is empty")
    try:
        import cv2
        import pytesseract  # noqa: F401  (fail fast if dependency missing)
    except ImportError as error:
        raise ValueError("图像识别依赖未安装。请运行 pip install -r requirements.txt") from error

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片")

    height, width = image.shape[:2]
    x1 = max(0, int(left_ratio * width))
    y1 = max(0, int(top_ratio * height))
    x2 = min(width, int(right_ratio * width))
    y2 = min(height, int(bottom_ratio * height))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Upscale small axis labels before thresholding.
    scale = 2.0
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Vertical (90 degree rotated) labels: try both rotation directions,
    # deskew any residual tilt, and keep the higher-confidence reading so
    # that upside-down attempts lose automatically.  np.rot90 is used (not
    # warpAffine) because a 90 degree turn swaps width/height — an in-place
    # affine rotation would clip the now-horizontal text inside the original
    # tall/narrow frame.
    if _looks_vertical(binary):
        best_text: str | None = None
        best_conf = 0.0
        for k in (1, 3):  # 90 deg counter-clockwise / clockwise
            candidate = _deskew(np.rot90(binary, k=k))
            # eng-only: vertical axis labels are ASCII; chi_sim would score
            # upside-down strokes as '一' and let the wrong orientation win.
            text, conf = _ocr_with_confidence(candidate, languages=("eng",))
            if text and conf > best_conf:
                best_text, best_conf = text, conf
        if best_text and best_conf >= _VERTICAL_MIN_CONF:
            return best_text
        # Otherwise fall through — the tall blob may not be vertical text.

    # Horizontal / moderately slanted labels.
    return _ocr_first_match(_deskew(binary))
