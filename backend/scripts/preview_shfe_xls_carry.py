"""Create a read-only coverage preview from downloaded SHFE contract reports."""

from __future__ import annotations

import json
import sys
import csv
from collections import Counter
from datetime import date
from pathlib import Path

BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
if str(BACKEND_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIRECTORY))

from app.services.shfe_xls_carry_import import build_shfe_carry_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = PROJECT_ROOT / "上期所"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cta_factors" / "shfe_xls_carry_preview.json"
CANDIDATE_PATH = PROJECT_ROOT / "data" / "cta_factors" / "shfe_xls_carry_candidates.csv"


def main() -> None:
    rows = build_shfe_carry_rows(SOURCE_DIRECTORY, start=date(2016, 1, 1), end=date(2024, 12, 31))
    dates = sorted({row["observation_date"] for row in rows})
    counts = Counter(row["observation_date"] for row in rows)
    valid_dates = [day for day, count in counts.items() if count >= 4]
    invalid_dates = sorted(day.isoformat() for day, count in counts.items() if count < 4)
    with CANDIDATE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_PATH.write_text(json.dumps({
        "source": str(SOURCE_DIRECTORY),
        "status": "preview_only",
        "date_range": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "row_count": len(rows),
        "weekly_observation_count": len(dates),
        "minimum_symbols_per_week": min(counts.values()),
        "eligible_weekly_observation_count": len(valid_dates),
        "ineligible_weekly_observation_count": len(invalid_dates),
        "ineligible_weeks": invalid_dates,
        "candidate_rows_path": str(CANDIDATE_PATH),
        "symbols": sorted({row["symbol"] for row in rows}),
        "rule": "contract delivery-month midpoint derived from SHFE contract code; not an exchange-published expiry date",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
