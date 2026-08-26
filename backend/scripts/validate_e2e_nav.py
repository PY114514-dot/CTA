"""End-to-end validation of the CV-only NAV extraction path.

Runs the *production* code path (material_ingestion._extract_points_cv_only)
on a real weekly-report image and checks the reconstructed NAV curve against
the report's own disclosed metrics (cumulative return, max drawdown).

Before the fix, the CV-only path returned ZERO points (no axis anchors ->
values never calibrated -> sanitize discarded everything). After the fix the
curve is anchored to the disclosed start/end values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.multi_product_report import extract_multi_product_report  # noqa: E402
from app.services.material_ingestion import _extract_points_cv_only  # noqa: E402

IMAGE_PATH = Path(__file__).resolve().parents[2] / "data" / "uploads" / (
    "15a3363680d34897ba3c_博衍基金_九溪CTA系列产品周报.png"
)
OUT_DIR = Path(__file__).resolve().parent / "validation_out"


def max_drawdown(navs: np.ndarray) -> float:
    running_max = np.maximum.accumulate(navs)
    return float(np.min(navs / running_max - 1.0))


def annualized(nav_start: float, nav_end: float, days: int) -> float:
    if days <= 0 or nav_start <= 0:
        return 0.0
    return float((nav_end / nav_start) ** (365.25 / days) - 1.0)


def analyze(points: list[dict]) -> dict:
    dates = [p["observation_date"] for p in points]
    navs = np.array([p["nav"] for p in points], dtype=float)
    d0 = np.datetime64(dates[0])
    d1 = np.datetime64(dates[-1])
    days = int((d1 - d0) / np.timedelta64(1, "D"))
    return {
        "n": len(points),
        "start_date": dates[0],
        "end_date": dates[-1],
        "days": days,
        "nav_start": float(navs[0]),
        "nav_end": float(navs[-1]),
        "cumulative": float(navs[-1] / navs[0] - 1.0),
        "annualized": annualized(navs[0], navs[-1], days),
        "mdd": max_drawdown(navs),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    content = IMAGE_PATH.read_bytes()
    img = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    report = extract_multi_product_report(content)
    metrics_by_id = {m.product_id: m for m in report.disclosed_metrics}
    print(f"\nDetected {len(report.disclosed_metrics)} products, "
          f"{len(report.product_curve_candidates)} curve candidates\n")

    header = f"{'product':<10} {'n_pts':>6} {'period':>23} " \
             f"{'cum(disc)':>9} {'cum(calc)':>9} {'ann(disc)':>9} {'ann(calc)':>9} " \
             f"{'mdd(disc)':>9} {'mdd(calc)':>9}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for idx, cand in enumerate(report.product_curve_candidates):
        metric = metrics_by_id.get(cand.product_id or "")
        if metric is None:
            print(f"[chart {idx+1}] no matching metric, skipped")
            continue

        x0, y0 = int(cand.left_ratio * w), int(cand.top_ratio * h)
        x1, y1 = int(cand.right_ratio * w), int(cand.bottom_ratio * h)
        crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if crop.size == 0:
            print(f"[chart {idx+1}] empty crop")
            continue
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            continue

        disclosed = {
            "start_date": metric.start_date,
            "end_date": metric.end_date,
            "cumulative_return": metric.cumulative_return,
        }
        # The production CV-only path:
        points = _extract_points_cv_only(buf.tobytes(), disclosed)

        if len(points) < 3:
            print(f"{metric.product_id:<10} {'FAILED — only ' + str(len(points)) + ' points'}")
            all_ok = False
            continue

        a = analyze(points)
        cum_err = abs(a["cumulative"] - metric.cumulative_return)
        status = "OK" if cum_err < 0.02 else "CHECK"
        if cum_err >= 0.02:
            all_ok = False

        print(f"{metric.product_id:<10} {a['n']:>6} "
              f"{a['start_date'] + '→' + a['end_date']:>23} "
              f"{metric.cumulative_return:>8.2%} {a['cumulative']:>9.2%} "
              f"{metric.annualized_return:>8.2%} {a['annualized']:>9.2%} "
              f"{metric.maximum_drawdown:>8.2%} {a['mdd']:>9.2%}  [{status}]")

        # Save the reconstructed NAV series as CSV
        csv_lines = ["date,nav"] + [f"{p['observation_date']},{p['nav']:.6f}" for p in points]
        (OUT_DIR / f"nav_{metric.product_id}.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    print("-" * len(header))
    print(f"\nNAV CSVs saved to: {OUT_DIR}")
    print("\nRESULT:", "PASS — CV-only path now yields calibrated NAV curves"
          if all_ok else "NEEDS REVIEW — see CHECK rows above")


if __name__ == "__main__":
    main()
