"""Offline evaluation for chart-structure annotations and VLM predictions.

This module intentionally evaluates only recorded JSON annotations.  It never
uploads source material or calls a VLM, which keeps private chart review data
under the analyst's control.
"""

from __future__ import annotations

from typing import Any


def _iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
             + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def evaluate_structure_prediction(expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    """Score one structural prediction against an analyst-reviewed manifest.

    Coordinates use the same 0–1000 convention as ``ChartStructure``.  A
    sample marked ``recognizable=False`` is correct only when prediction also
    rejects automatic extraction.
    """
    if not expected.get("recognizable", True):
        rejected = bool(predicted.get("rejected") or predicted.get("review_required"))
        return {"status": "passed" if rejected else "failed", "rejection_correct": rejected, "metrics": {}}

    expected_bbox = expected.get("plot_bbox_1000", [])
    predicted_bbox = predicted.get("plot_bbox_1000", [])
    bbox_iou = round(_iou(expected_bbox, predicted_bbox), 6)
    expected_curves = {str(item.get("name", "")).strip().lower() for item in expected.get("curves", []) if item.get("name")}
    predicted_curves = {str(item.get("name", "")).strip().lower() for item in predicted.get("curves", []) if item.get("name")}
    curve_recall = len(expected_curves & predicted_curves) / len(expected_curves) if expected_curves else 1.0
    expected_ticks = list(expected.get("y_ticks", []))
    predicted_ticks = list(predicted.get("y_ticks", []))
    tick_recall = len(set(expected_ticks) & set(predicted_ticks)) / len(set(expected_ticks)) if expected_ticks else 1.0
    passed = bbox_iou >= 0.7 and curve_recall >= 0.8 and tick_recall >= 0.8
    return {
        "status": "passed" if passed else "failed",
        "rejection_correct": None,
        "metrics": {"plot_bbox_iou": bbox_iou, "curve_name_recall": round(curve_recall, 6), "y_tick_recall": round(tick_recall, 6)},
    }


def evaluate_manifest(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a local JSONL/JSON manifest already paired with predictions."""
    results = [evaluate_structure_prediction(item["expected"], item["predicted"]) for item in manifest]
    passed = sum(result["status"] == "passed" for result in results)
    return {"sample_count": len(results), "passed_count": passed, "pass_rate": round(passed / len(results), 6) if results else None, "results": results}
