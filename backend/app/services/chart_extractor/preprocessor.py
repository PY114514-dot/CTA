"""Image preprocessing for chart extraction.

Applies CV enhancements to chart crops before VLM reading:
- Bilateral filter (denoise while preserving edges)
- CLAHE (contrast enhancement for faded charts)
- Optional upscaling for low-resolution crops
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_chart_image(
    image_bytes: bytes,
    enhance_contrast: bool = True,
    denoise: bool = True,
    upscale_if_small: bool = True,
    min_width: int = 800,
) -> bytes:
    """Apply preprocessing to a chart crop for better VLM readability.

    Parameters
    ----------
    image_bytes : PNG bytes of the chart crop
    enhance_contrast : apply CLAHE
    denoise : apply bilateral filter
    upscale_if_small : upscale if width < min_width
    min_width : minimum width threshold for upscaling

    Returns
    -------
    Enhanced PNG bytes.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Cannot decode image for preprocessing")
        return image_bytes

    h, w = img.shape[:2]

    # Upscale small images (VLMs read larger images better)
    if upscale_if_small and w < min_width:
        scale = min_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug("Upscaled %dx%d -> %dx%d", w, h, new_w, new_h)

    # Denoise (preserve edges — important for thin curve lines)
    if denoise:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # CLAHE contrast enhancement (helps with faded/low-contrast charts)
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return image_bytes
    return buf.tobytes()


def crop_axis_labels(
    image_bytes: bytes,
    axis: str = "x",
    margin_ratio: float = 0.12,
) -> bytes:
    """Crop just the axis label region from a chart image.

    Useful for separate OCR of axis labels when VLM struggles.

    Parameters
    ----------
    image_bytes : full chart crop PNG
    axis : "x" for bottom labels, "y" for left labels
    margin_ratio : fraction of image to crop from the edge

    Returns
    -------
    PNG bytes of the axis label strip.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    h, w = img.shape[:2]

    if axis == "x":
        # Bottom strip
        strip_h = int(h * margin_ratio)
        crop = img[h - strip_h:h, int(w * 0.08):int(w * 0.95)]
    else:
        # Left strip
        strip_w = int(w * margin_ratio)
        crop = img[int(h * 0.05):int(h * 0.9), 0:strip_w]

    if crop.size == 0:
        return image_bytes

    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else image_bytes
