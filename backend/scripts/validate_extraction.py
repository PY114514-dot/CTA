"""Validation harness: run the improved chart-extraction pipeline on a real
fund report image and report geometric + calibration accuracy.

Ground truth comes from the report's own performance table (OCR-extracted):
each fund's cumulative return. We validate:
  1. Chart detection: are all 3 green curve regions found?
  2. Color refinement: does K-means isolate the green fund curve?
  3. Continuity tracing: is the traced curve smooth (no jumps to the gray
     benchmark), monotonic in X, with no large Y spikes?
  4. Calibration gap: does the tight crop include axis labels?
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.multi_product_report import extract_multi_product_report  # noqa: E402
from app.services.chart_extractor.pixel_tracer import (  # noqa: E402
    TraceConfig,
    build_color_mask,
    detect_plot_area,
    refine_colors_kmeans,
    trace_curve,
)

IMAGE_PATH = Path(__file__).resolve().parents[2] / "data" / "uploads" / (
    "15a3363680d34897ba3c_博衍基金_九溪CTA系列产品周报.png"
)
OUT_DIR = Path(__file__).resolve().parent / "validation_out"


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite fails on non-ASCII paths on Windows; use imencode + tofile."""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))


def smoothness_metrics(points) -> dict:
    """Compute max inter-point Y jump and X monotonicity."""
    if len(points) < 2:
        return {"n": len(points), "max_jump_px": 0, "x_monotonic": True, "backtracks": 0}
    ys = np.array([p.y_px for p in points], dtype=float)
    xs = np.array([p.x_px for p in points], dtype=float)
    jumps = np.abs(np.diff(ys))
    backtracks = int(np.sum(np.diff(xs) < 0))
    return {
        "n": len(points),
        "max_jump_px": float(jumps.max()),
        "mean_jump_px": float(jumps.mean()),
        "x_monotonic": backtracks == 0,
        "backtracks": backtracks,
        "y_range_px": (float(ys.min()), float(ys.max())),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    content = IMAGE_PATH.read_bytes()
    img = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # --- Step 1: report-level extraction (metrics + curve candidates) ---
    report = extract_multi_product_report(content)
    print(f"\n[1] Detected {len(report.disclosed_metrics)} products, "
          f"{len(report.product_curve_candidates)} curve candidates")
    for m in report.disclosed_metrics:
        print(f"    {m.product_id}: cumulative={m.cumulative_return:.2%} "
              f"annualized={m.annualized_return:.2%} mdd={m.maximum_drawdown:.2%}")

    if not report.product_curve_candidates:
        print("!! No curve candidates detected — chart detection FAILED")
        return

    # --- Step 2-3: per-chart K-means + continuity tracing ---
    config = TraceConfig()
    for idx, cand in enumerate(report.product_curve_candidates):
        x0, y0 = int(cand.left_ratio * w), int(cand.top_ratio * h)
        x1, y1 = int(cand.right_ratio * w), int(cand.bottom_ratio * h)
        crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if crop.size == 0:
            print(f"\n[chart {idx+1}] empty crop, skipped")
            continue
        ch, cw = crop.shape[:2]

        plot_area = detect_plot_area(crop)
        # K-means refinement starting from a generic "green" hint
        specs = [{"name": f"fund_{idx+1}", "color_hex": "#00A651", "color_name": "green",
                  "is_benchmark": False}]
        refined = refine_colors_kmeans(crop, plot_area, specs)
        refined_color = refined[0]["color_hex"]

        points = trace_curve(crop, plot_area, color_hex=refined_color, config=config)
        m = smoothness_metrics(points)

        print(f"\n[chart {idx+1}] crop={cw}x{ch} plot_area="
              f"({plot_area.left},{plot_area.top})-({plot_area.right},{plot_area.bottom})")
        print(f"    K-means color: #00A651 -> {refined_color}")
        print(f"    traced points: {m['n']}")
        print(f"    max Y jump: {m['max_jump_px']:.1f}px  mean: {m['mean_jump_px']:.2f}px")
        print(f"    X monotonic: {m['x_monotonic']}  backtracks: {m['backtracks']}")
        print(f"    Y pixel range: {m['y_range_px']}")

        # --- Save overlay for visual inspection ---
        overlay = crop.copy()
        for p in points:
            cv2.circle(overlay, (p.x_px, p.y_px), 2, (0, 0, 255), -1)
        cv2.rectangle(overlay, (plot_area.left, plot_area.top),
                      (plot_area.right, plot_area.bottom), (255, 0, 0), 2)
        imwrite_unicode(OUT_DIR / f"chart{idx+1}_overlay.png", overlay)
        imwrite_unicode(OUT_DIR / f"chart{idx+1}_crop.png", crop)

        # --- Save traced points as CSV ---
        csv_lines = ["x_px,y_px"]
        csv_lines += [f"{p.x_px},{p.y_px}" for p in points]
        (OUT_DIR / f"chart{idx+1}_points.csv").write_text("\n".join(csv_lines), encoding="utf-8")

        # --- Check: did the mask also pick up the gray benchmark? ---
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green_mask = build_color_mask(hsv, refined_color, "green")
        region = np.zeros_like(green_mask)
        region[plot_area.top:plot_area.bottom, plot_area.left:plot_area.right] = 255
        green_px = int((green_mask & region).sum() / 255)
        print(f"    green-mask pixels in plot: {green_px} "
              f"(coverage {green_px / max(1, (plot_area.right-plot_area.left)):.1f} px/col avg)")

    print(f"\nOverlays + CSV saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
