"""Run the single-product VLM/CV validation manifest and save auditable JSON."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.config import load_local_environment
from app.services.chart_extractor.pipeline import extract_single_image


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("single_product_validation_20.txt")
OUTPUT = Path(os.getenv(
    "SINGLE_PRODUCT_VALIDATION_OUTPUT",
    str(Path(__file__).with_name("validation_out") / "single_product_vlm_validation_20.json"),
))
# Provide a prior validation JSON via this variable to replay the exact VLM
# structure captured for every image. This measures CV changes without VLM
# response variance being counted as a regression or improvement.
STRUCTURE_SNAPSHOT_RAW = os.getenv("SINGLE_PRODUCT_VALIDATION_STRUCTURE_SNAPSHOT", "").strip()
STRUCTURE_SNAPSHOT = Path(STRUCTURE_SNAPSHOT_RAW) if STRUCTURE_SNAPSHOT_RAW else None
PER_IMAGE_TIMEOUT_SECONDS = 210


def _paths() -> list[str]:
    return [
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


async def main() -> None:
    load_local_environment()
    # This is a real-network validation.  Persist after every image so an API
    # timeout never discards the preceding completed samples, and allow a
    # later invocation to resume the same manifest.
    existing: dict[str, dict[str, object]] = {}
    if OUTPUT.exists() and os.getenv("SINGLE_PRODUCT_VALIDATION_FORCE") != "1":
        try:
            existing = {str(row["file"]): row for row in json.loads(OUTPUT.read_text(encoding="utf-8")).get("rows", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}
    replay_structures: dict[str, dict[str, object]] = {}
    if STRUCTURE_SNAPSHOT is not None:
        if not STRUCTURE_SNAPSHOT.exists():
            raise FileNotFoundError(f"Structure snapshot does not exist: {STRUCTURE_SNAPSHOT}")
        snapshot = json.loads(STRUCTURE_SNAPSHOT.read_text(encoding="utf-8"))
        replay_structures = {
            str(row["file"]): row["structure_snapshot"]
            for row in snapshot.get("rows", [])
            if isinstance(row, dict) and isinstance(row.get("structure_snapshot"), dict)
        }
        print(f"replaying_structures={len(replay_structures)} source={STRUCTURE_SNAPSHOT}", flush=True)
    rows: list[dict[str, object]] = []

    def save() -> None:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "sample_count": len(rows),
                    "success_count": sum(row["status"] == "ok" for row in rows),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    for index, relative_path in enumerate(_paths(), start=1):
        if relative_path in existing:
            rows.append(existing[relative_path])
            print(f"resumed={index} {relative_path}", flush=True)
            continue
        try:
            result = await asyncio.wait_for(
                extract_single_image(
                    (ROOT / relative_path).read_bytes(),
                    use_vlm=relative_path not in replay_structures,
                    structure_override=replay_structures.get(relative_path),
                ),
                timeout=PER_IMAGE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            row = {
                "index": index, "file": relative_path, "status": "error",
                "error": f"VLM/CV processing exceeded {PER_IMAGE_TIMEOUT_SECONDS}s", "category": "manual_review",
            }
            rows.append(row)
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        structure = result.structure
        calibrated_points = sum(
            1
            for curve in result.curves
            for point in curve.points
            if point.date and point.value is not None
        )
        row = {
            "index": index,
            "file": relative_path,
            "status": "ok" if not result.error else "error",
            "error": result.error,
            "plot_bbox_1000": structure.plot_bbox_1000 if structure else [],
            "vlm_curve_count": len(structure.curves) if structure else 0,
            "x_tick_count": len(structure.x_ticks) if structure else 0,
            "y_range": structure.y_range if structure else [],
            # Keep the full contract response needed to reproduce CV tracing.
            # It is intentionally saved per image rather than separately so
            # a validation artifact remains independently auditable.
            "structure_snapshot": structure.model_dump() if structure else None,
            "traced_curve_count": len(result.curves),
            "calibrated_point_count": calibrated_points,
            "confidence": result.confidence,
            "curve_quality": [curve.quality for curve in result.curves],
        }
        if result.error:
            row["category"] = "manual_review"
        elif calibrated_points >= 3:
            row["category"] = "automatic"
        elif structure and structure.plot_bbox_1000:
            row["category"] = "date_endpoint_fallback"
        else:
            row["category"] = "manual_review"
        rows.append(row)
        save()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    save()
    print(f"saved={OUTPUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
