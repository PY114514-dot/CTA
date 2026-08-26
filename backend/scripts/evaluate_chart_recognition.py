"""Deterministic quality benchmark for the production curve tracer.

The cases are generated from geometry, not copied from customer reports.  This
keeps the benchmark reproducible and makes improvements compete on aggregate
recognition quality instead of one document layout.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chart_extractor.models import PlotArea  # noqa: E402
from app.services.chart_extractor.pixel_tracer import (  # noqa: E402
    TraceConfig,
    assess_trace_quality,
    trace_curve,
)


PLOT = PlotArea(left=10, top=10, right=190, bottom=90)
RED = (20, 20, 220)


@dataclass(frozen=True)
class Case:
    name: str
    image: np.ndarray
    expected_curve: dict[int, int] | None


def _canvas() -> np.ndarray:
    return np.full((100, 200, 3), 255, dtype=np.uint8)


def _curve(image: np.ndarray, ys: dict[int, int], *, gaps: set[int] | None = None) -> None:
    gaps = gaps or set()
    points = [(x, y) for x, y in ys.items() if x not in gaps]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x1 == x0 + 1:
            cv2.line(image, (x0, y0), (x1, y1), RED, 2)


def build_cases() -> list[Case]:
    xs = range(PLOT.left, PLOT.right)
    smooth = {x: int(round(48 + 14 * np.sin((x - 10) / 28))) for x in xs}
    upper = {x: int(round(28 + 8 * np.sin((x - 10) / 24))) for x in xs}

    smooth_image = _canvas()
    _curve(smooth_image, smooth)

    broken_image = _canvas()
    _curve(broken_image, smooth, gaps=set(range(72, 80)) | set(range(132, 139)))
    cv2.line(broken_image, (65, 80), (92, 80), RED, 3)

    competing_image = _canvas()
    _curve(competing_image, upper)
    cv2.line(competing_image, (10, 50), (64, 50), RED, 3)

    swatch_image = _canvas()
    cv2.line(swatch_image, (70, 78), (100, 78), RED, 3)

    return [
        Case("smooth_curve", smooth_image, smooth),
        Case("broken_curve_with_legend", broken_image, smooth),
        Case("competing_same_colour_annotation", competing_image, upper),
        Case("short_legend_only", swatch_image, None),
        Case("empty_chart", _canvas(), None),
    ]


def evaluate() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    accepted_correctly = 0
    total_absolute_error = 0.0
    matched_points = 0

    for case in build_cases():
        points = trace_curve(case.image, PLOT, color_hex="#DC1414", config=TraceConfig())
        quality = assess_trace_quality(points, PLOT)
        accepted = bool(quality["is_reliable"])
        should_accept = case.expected_curve is not None
        correct_decision = accepted == should_accept
        if correct_decision:
            accepted_correctly += 1

        errors = [
            abs(point.y_px - case.expected_curve[point.x_px])
            for point in points
            if case.expected_curve is not None and point.x_px in case.expected_curve
        ]
        mae = float(np.mean(errors)) if errors else None
        if errors:
            total_absolute_error += sum(errors)
            matched_points += len(errors)
        rows.append({
            "case": case.name,
            "expected": "accept" if should_accept else "reject",
            "decision": "accept" if accepted else "reject",
            "correct_decision": correct_decision,
            "coverage_ratio": quality["coverage_ratio"],
            "mae_px": round(mae, 3) if mae is not None else None,
        })

    decision_accuracy = accepted_correctly / len(rows)
    mean_absolute_error = total_absolute_error / matched_points if matched_points else float("inf")
    accepted_maes = [row["mae_px"] for row in rows if row["expected"] == "accept"]
    max_case_mae = max(float(value) for value in accepted_maes if value is not None)
    passed = decision_accuracy == 1.0 and mean_absolute_error <= 3.0 and max_case_mae <= 3.0
    return {
        "passed": passed,
        "decision_accuracy": round(decision_accuracy, 4),
        "mean_absolute_error_px": round(mean_absolute_error, 4),
        "max_case_mae_px": round(max_case_mae, 4),
        "cases": rows,
    }


def main() -> int:
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
