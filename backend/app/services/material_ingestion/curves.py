"""Curve tracing: pixel-to-NAV conversion via chart_extractor (VLM + CV)."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date
from typing import Any

import numpy as np

from app.services import product_store as store

from .helpers import (
    _find_or_create_product,
    _infer_frequency,
    _normalise_nav_points,
    _valid_nav_points,
    _visual_method,
)

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Bridge async chart_extractor calls from the sync background task."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _trace_image_curves(
    session: Any,
    file_id: str,
    content: bytes,
    candidates: list,
    ordered_products: list,
    filename: str,
    metrics: list | None = None,
) -> list[dict[str, Any]]:
    """Trace each colour-identified curve into NAV observations."""
    stem = Path(filename).stem if filename else "未命名产品"

    metrics_by_id = {m.product_id: m for m in (metrics or [])}
    products_by_source_id = {
        metric.product_id: ordered_products[index]
        for index, metric in enumerate(metrics or [])
        if index < len(ordered_products) and ordered_products[index] is not None
    }
    single_product = ordered_products[0] if len(ordered_products) == 1 else None
    audits: list[dict[str, Any]] = []
    primary_curves = [
        candidate for candidate in candidates
        if str(getattr(candidate, "legend_label", "")).strip() in {"累计收益率", "累计收益"}
    ]
    if single_product is not None and primary_curves:
        candidates = primary_curves

    for idx, candidate in enumerate(candidates):
        # A VLM-provided source ID is authoritative. Positional matching is
        # only safe for a one-product report; for multiple products create a
        # pending candidate instead of attaching a curve to the wrong fund.
        candidate_product_id = getattr(candidate, "product_id", None)
        product = products_by_source_id.get(candidate_product_id) if candidate_product_id else None
        if product is None:
            product = single_product
        is_unmatched = product is None
        fallback_name = product.standard_name if product else f"{stem} / 待绑定曲线 {idx + 1}"
        # Keep the curve as unbound evidence rather than fabricating a
        # “待绑定曲线” product.  The user can bind it to a real product from
        # the source-file queue, and the formal product library stays clean.

        metric = metrics_by_id.get(candidate.product_id or "")
        disclosed = None
        if metric is not None:
            disclosed = {
                "start_date": metric.start_date,
                "end_date": metric.end_date,
                "cumulative_return": metric.cumulative_return,
            }

        points, frequency_hint, trace_confidence, audit = _extract_points_from_image(
            content, fallback_name, disclosed, color_hint=getattr(candidate, "color_hex", None)
        )
        audits.append(audit)
        points = _valid_nav_points(points)
        if len(points) >= 3:
            if product is None:
                continue
            target = product

            # Cross-validate against existing table-sourced NAV data
            points = _cross_validate_with_table(session, target.id, points)

            binding_status = "unmatched" if is_unmatched else "matched"
            fragment = store.add_fragment(
                session,
                file_id=file_id,
                fragment_type="chart_traced",
                content_text=f"曲线追踪 {len(points)} 个数据点",
                content_data={
                    "curve_index": candidate.curve_index,
                    "num_points": len(points),
                    "method": _visual_method(),
                    "confidence": trace_confidence,
                    "review_status": "pending",
                    "binding_status": binding_status,
                    "binding_reason": None if binding_status == "matched" else "多产品曲线未返回可验证的产品 ID",
                    "binding_confidence": candidate.binding_confidence,
                    "binding_evidence": candidate.binding_evidence,
                    "legend_label": candidate.legend_label,
                    "layout_product_name": candidate.layout_product_name,
                    "color_hex": candidate.color_hex,
                },
                ocr_confidence=trace_confidence,
                product_id=target.id,
            )
            frequency = frequency_hint or _infer_frequency([
                    date.fromisoformat(point["observation_date"])
                    for point in points
                ], pixel_trace=True)
            candidate_version = store.create_nav_candidate_version(
                session, target.id, file_id, points,
                source_fragment_id=fragment.id, frequency=frequency, confidence=trace_confidence,
            )
            fragment.content_data = {**(fragment.content_data or {}), "candidate_version_id": candidate_version.id}
            session.commit()
            logger.info("Created candidate NAV version %s with %d points for %s from %s", candidate_version.id, len(points), target.standard_name, filename)
        else:
            # Never hide a failed trace behind a generic “completed” status.
            # Preserve the chart bounds and binding evidence so the browser
            # can open the exact source region for colour picking/calibration.
            store.add_fragment(
                session,
                file_id=file_id,
                fragment_type="chart_traced",
                content_text=f"曲线追踪未生成足够净值点（{len(points)} 个）",
                content_data={
                    "curve_index": candidate.curve_index,
                    "num_points": 0,
                    "method": _visual_method(),
                    "review_status": "pending",
                    "binding_status": "unmatched" if is_unmatched else "matched",
                    "binding_reason": "多产品曲线未返回可验证的产品 ID" if is_unmatched else "曲线像素追踪不足 3 点；需要重新取色或校准坐标轴",
                    "binding_confidence": candidate.binding_confidence,
                    "binding_evidence": candidate.binding_evidence,
                    "legend_label": candidate.legend_label,
                    "layout_product_name": candidate.layout_product_name,
                    "color_hex": candidate.color_hex,
                },
                ocr_confidence=0.2,
                product_id=product.id if product else None,
            )

    return audits


def _trace_whole_image(session: Any, file_id: str, content: bytes, product_name: str) -> dict[str, Any]:
    """Trace the entire image as a single NAV chart (no table/candidates found)."""
    points, frequency_hint, trace_confidence, audit = _extract_points_from_image(content, product_name)
    points = _valid_nav_points(points)
    if len(points) < 3:
        return audit
    product = _find_or_create_product(session, product_name, None, None, None)
    fragment = store.add_fragment(
        session,
        file_id=file_id,
        fragment_type="chart_traced",
        content_text=f"整图曲线追踪 {len(points)} 个数据点",
        content_data={"num_points": len(points), "method": _visual_method(), "confidence": trace_confidence, "review_status": "pending"},
        ocr_confidence=trace_confidence,
        product_id=product.id,
    )
    frequency = frequency_hint or _infer_frequency([
            date.fromisoformat(point["observation_date"])
            for point in points
        ], pixel_trace=True)
    candidate_version = store.create_nav_candidate_version(
        session, product.id, file_id, points,
        source_fragment_id=fragment.id, frequency=frequency, confidence=trace_confidence,
    )
    fragment.content_data = {**(fragment.content_data or {}), "candidate_version_id": candidate_version.id}
    session.commit()
    return audit


def _cross_validate_with_table(
    session: Any,
    product_id: str,
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Correct chart-traced NAV values using existing table-sourced observations.

    If the product already has NAV data from a table (higher confidence),
    find matching dates and apply a linear correction (scale + offset) to
    fix systematic Y-axis calibration errors in the chart tracing.
    """
    from sqlalchemy import select as sa_select

    from app.models import NavObservation

    # Get existing observations for this product (table-sourced = ground truth)
    existing = session.execute(
        sa_select(NavObservation)
        .where(NavObservation.product_id == product_id)
        .order_by(NavObservation.observation_date)
    ).scalars().all()

    if len(existing) < 2:
        return points

    # Build reference map: date_str → nav
    ref_map: dict[str, float] = {}
    for obs in existing:
        ref_map[obs.observation_date.isoformat()] = obs.nav

    # Find matching points
    matched_traced: list[float] = []
    matched_ref: list[float] = []
    for p in points:
        d = p.get("observation_date", "")
        v = p.get("nav")
        if d and v and d in ref_map:
            matched_traced.append(float(v))
            matched_ref.append(ref_map[d])

    if len(matched_traced) < 2:
        return points

    # Least-squares linear fit: ref = scale * traced + offset
    n = len(matched_traced)
    sum_x = sum(matched_traced)
    sum_y = sum(matched_ref)
    sum_xy = sum(x * y for x, y in zip(matched_traced, matched_ref))
    sum_x2 = sum(x * x for x in matched_traced)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        offset = sum_y / n - sum_x / n
        scale = 1.0
    else:
        scale = (n * sum_xy - sum_x * sum_y) / denom
        offset = (sum_y - scale * sum_x) / n

    # Sanity check
    if scale < 0.5 or scale > 2.0:
        logger.warning("Cross-validation scale=%.3f out of range, skipping", scale)
        return points

    # Apply correction
    for p in points:
        v = p.get("nav")
        if v:
            corrected = scale * float(v) + offset
            if corrected > 0:
                p["nav"] = round(corrected, 6)

    logger.info("Table cross-validation: %d matches, scale=%.4f, offset=%.6f", n, scale, offset)
    return points


def _extract_points_from_image(
    image_bytes: bytes,
    curve_name: str,
    disclosed: dict[str, Any] | None = None,
    color_hint: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, float, dict[str, Any]]:
    """Run chart extraction on one image; return NAV point dicts.

    When a VLM is configured it reads the chart structure (and axis anchors)
    and the standard pipeline calibrates the trace. In CV-only mode there are
    no axis anchors, so the curve is traced as raw pixels and calibrated
    against the disclosed table metrics (start/end dates + cumulative return)
    via endpoint anchoring.
    """
    # DashScope needs an API key; local / OpenAI-compatible vLLM only needs a
    # reachable base URL.  Treat both as a configured visual fallback.
    from app.config import load_local_environment

    load_local_environment()
    provider_type = os.getenv("VLM_PROVIDER", "dashscope")
    use_vlm = bool(os.getenv("DASHSCOPE_API_KEY", "")) if provider_type == "dashscope" else bool(os.getenv("VLM_BASE_URL", ""))

    if not use_vlm:
        points = _extract_points_cv_only(image_bytes, disclosed)
        return points, None, 0.55 if len(points) >= 3 else 0.2, {
            "configured": False,
            "attempted": False,
            "succeeded": False,
            "methods": ["CV 像素追踪（无 VLM）"],
        }

    from app.services.chart_extractor import extract_single_image

    def points_from_result(result: Any) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for curve in result.curves:
            if curve.is_benchmark:
                continue
            raw = [p for p in curve.points if p.value is not None and p.value > 0]
            # A product-name hint can replace the VLM curve label.  Values
            # below 0.5 are not valid NAV levels here and identify a return
            # axis even when that label is no longer available.
            is_cumulative_return = "收益" in curve.name or bool(raw and max(float(p.value) for p in raw) < 0.5)
            for p in curve.points:
                if p.date and p.value is not None and p.value > 0:
                    points.append({
                        "observation_date": p.date,
                        "nav": 1.0 + p.value if is_cumulative_return else p.value,
                    })
            if points:
                break  # one product curve per crop
            # A VLM often reads the frame, colour and Y scale correctly but
            # omits dense Chinese X-axis labels.  When the same factsheet
            # discloses a complete start/end period, map the already traced
            # pixel sequence across that interval and retain weekly samples.
            # This is explicitly interpolation evidence, never OCR dates.
            if len(raw) >= 3 and disclosed and disclosed.get("start_date") and disclosed.get("end_date"):
                try:
                    start = date.fromisoformat(str(disclosed["start_date"]))
                    end = date.fromisoformat(str(disclosed["end_date"]))
                    span = max((end - start).days, 1)
                    left, right = raw[0].x_px, raw[-1].x_px
                    width = max(right - left, 1)
                    points = [
                        {
                            "observation_date": (start.fromordinal(start.toordinal() + round((point.x_px - left) / width * span))).isoformat(),
                            "nav": 1.0 + float(point.value) if is_cumulative_return else float(point.value),
                        }
                        for point in raw
                    ]
                except (TypeError, ValueError):
                    points = []
            if points:
                break
        return points

    # Do not pass an auto-detected colour when VLM is enabled: a non-empty
    # curve_specs list intentionally switches the chart pipeline into manual
    # mode and silently skips VLM.  Let the model identify the line first, then
    # use CV colour tracing for the numerical pixels.  A manual CV pass remains
    # the deterministic fallback when the model is unavailable or uncertain.
    if use_vlm:
        audit = {
            "configured": True,
            "attempted": False,
            "succeeded": False,
            "provider": provider_type,
            "model": os.getenv("VLM_MODEL", "qwen3-vl-flash"),
            "calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "errors": [],
            "methods": ["VLM 结构识别", "CV 像素追踪"],
        }
        try:
            curve_specs = None
            if color_hint:
                curve_specs = [{
                    "name": curve_name,
                    "color_hex": color_hint,
                    "color_name": "layout_hint",
                    "is_benchmark": False,
                }]
            result = _run_async(extract_single_image(image_bytes, curve_specs=curve_specs, use_vlm=True))
            audit.update({
                "attempted": bool(result.vlm_attempted),
                "succeeded": bool(result.vlm_succeeded),
                "provider": result.vlm_provider or provider_type,
                "model": result.vlm_model or os.getenv("VLM_MODEL", "qwen3-vl-flash"),
                "calls": 1 if result.vlm_attempted else 0,
                "successful_calls": 1 if result.vlm_succeeded else 0,
                "failed_calls": 1 if result.vlm_attempted and not result.vlm_succeeded else 0,
                "errors": [result.vlm_error] if result.vlm_error else [],
            })
            points = points_from_result(result)
            # A frequency is authoritative only when the VLM also returned
            # dated observations.  On a pixel-only fallback, a model may say
            # "daily" simply because the chart is dense; those are still
            # interpolation samples and are handled as weekly below.
            frequency_hint = result.frequency if result.frequency in {"daily", "weekly", "monthly", "quarterly"} and points else None
            if len(points) >= 3:
                return _resample_pixel_points_weekly(points), frequency_hint, max(0.3, min(0.98, float(result.confidence))), audit
            logger.info("VLM chart structure returned no usable points for %s; falling back to CV", curve_name)
        except Exception as exc:
            logger.warning("VLM chart extraction failed for %s: %s", curve_name, exc)
            frequency_hint = None
            audit.update({"attempted": True, "calls": 1, "failed_calls": 1, "errors": [str(exc)[:300]]})
    else:
        frequency_hint = None

    # VLM often identifies the line and Y-axis but cannot read every X-axis
    # date on a dense report.  In that case its points have empty dates and
    # cannot enter the NAV table.  Use the deterministic CV tracer with the
    # disclosed endpoints to generate dated values instead of sending an
    # uncalibrated manual extraction through the VLM pipeline again.
    cv_points = _extract_points_cv_only(image_bytes, disclosed)
    if len(cv_points) >= 3:
        return cv_points, frequency_hint, 0.55, audit

    # Last resort for a chart with no disclosed period: a manual colour trace
    # may still produce reviewable pixel evidence, even though it has no date
    # calibration and will be kept out of the research-ready NAV series.
    colors = _auto_curve_colors(image_bytes)
    if not colors:
        return [], frequency_hint, 0.2, audit
    curve_specs = [{"name": curve_name, "color_hex": colors[0], "color_name": "", "is_benchmark": False}]
    try:
        result = _run_async(extract_single_image(image_bytes, curve_specs=curve_specs, use_vlm=False))
    except Exception as exc:
        logger.warning("CV chart extraction failed: %s", exc)
        audit.update({"failed_calls": int(audit.get("failed_calls", 0)) + 1, "errors": [*audit.get("errors", []), str(exc)[:300]]})
        return [], frequency_hint, 0.2, audit
    points = points_from_result(result)
    return points, frequency_hint, max(0.3, min(0.8, float(getattr(result, "confidence", 0.55)))), audit


def _extract_points_cv_only(
    image_bytes: bytes,
    disclosed: dict[str, Any] | None,
    color_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Trace a curve with pure CV and calibrate it to real NAV values.

    No VLM / axis labels are used here. The curve is traced as raw pixels and
    then anchored to the disclosed start/end NAV values (endpoint calibration):
    leftmost traced pixel -> (start_date, 1.0), rightmost -> (end_date,
    1 + cumulative_return). Dates are interpolated linearly between the two
    endpoints. Axis-label OCR was evaluated and rejected as too unreliable.
    """
    import cv2

    from app.services.chart_extractor import (
        calibrate_from_disclosed,
        sanitize_nav_points,
    )
    from app.services.chart_extractor.pixel_tracer import detect_plot_area, trace_curve

    colors = [color_hint] if color_hint else _auto_curve_colors(image_bytes)
    if not colors:
        return []

    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []

    plot_area = detect_plot_area(img)
    points = trace_curve(img, plot_area, color_hex=colors[0])
    if len(points) < 3:
        return []

    if disclosed:
        points = calibrate_from_disclosed(
            points,
            start_date=disclosed.get("start_date", ""),
            end_date=disclosed.get("end_date", ""),
            cumulative_return=float(disclosed.get("cumulative_return", 0.0)),
            start_nav=float(disclosed.get("start_nav", 1.0)),
        )

    points = sanitize_nav_points(points)
    raw_points = [
        {"observation_date": p.date, "nav": p.value}
        for p in points
        if p.date and p.value is not None and p.value > 0
    ]
    return _resample_pixel_points_weekly(raw_points)


def _resample_pixel_points_weekly(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn dense per-pixel samples into one observation every seven days.

    The CV tracer emits one point for almost every image column.  Those are
    interpolation samples, not observations published by the manager.  Taking
    the last available pixel at each seven-day target preserves the curve
    shape and endpoints while preventing anti-aliasing noise from becoming
    artificial daily returns.
    """
    if len(points) < 3:
        return points
    points = _normalise_nav_points(points)
    ordered = sorted(
        (
            item
            for item in points
            if item.get("observation_date") and item.get("nav") is not None
        ),
        key=lambda item: str(item["observation_date"]),
    )
    if len(ordered) < 3:
        return points
    dates = [date.fromisoformat(item["observation_date"]) for item in ordered]
    if (dates[-1] - dates[0]).days < 30:
        return ordered

    sampled: list[dict[str, Any]] = []
    cursor = dates[0]
    index = 0
    while cursor <= dates[-1]:
        while index + 1 < len(ordered) and dates[index + 1] <= cursor:
            index += 1
        source = ordered[index]
        sampled.append({"observation_date": cursor, "nav": source["nav"]})
        cursor = cursor.fromordinal(cursor.toordinal() + 7)

    # Always retain the disclosed end point; otherwise a one-week endpoint
    # mismatch can make the table's cumulative return appear inconsistent.
    if sampled[-1]["observation_date"] != dates[-1]:
        sampled.append({"observation_date": dates[-1], "nav": ordered[-1]["nav"]})
    return sampled


def _auto_curve_colors(image_bytes: bytes, max_colors: int = 2) -> list[str]:
    """Detect dominant curve colors via HSV scan (red/green/blue families)."""
    import cv2

    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    h, w = img.shape[:2]
    roi = img[int(h * 0.1):int(h * 0.85), int(w * 0.08):int(w * 0.95)]
    if roi.size == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    saturated = (sat >= 60) & (val >= 80)

    families: list[tuple[int, np.ndarray]] = []
    red_mask = saturated & ((hue <= 10) | (hue >= 170))
    green_mask = saturated & (hue >= 35) & (hue <= 90)
    blue_mask = saturated & (hue >= 100) & (hue <= 135)
    for mask in (red_mask, green_mask, blue_mask):
        count = int(mask.sum())
        if count > (roi.shape[0] * roi.shape[1]) * 0.001:
            families.append((count, mask))

    families.sort(key=lambda item: item[0], reverse=True)
    colors: list[str] = []
    for _, mask in families[:max_colors]:
        # Average only the masked (curve) pixels; averaging the bitwise_and
        # output would dilute the color with the zeroed background.
        mean_bgr = cv2.mean(roi, mask=mask.astype(np.uint8))
        b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


