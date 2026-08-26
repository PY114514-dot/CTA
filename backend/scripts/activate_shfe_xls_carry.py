"""Activate reviewed SHFE-report Carry candidates after the preview has passed."""

from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
if str(BACKEND_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIRECTORY))

from app.services.cta_factor_bundle import (  # noqa: E402
    REAL_CARRY_PATH,
    load_real_carry_rows,
    save_real_carry_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATH = PROJECT_ROOT / "data" / "cta_factors" / "shfe_xls_carry_candidates.csv"
HISTORICAL_END = date(2024, 12, 31)


def main() -> None:
    if not CANDIDATE_PATH.is_file():
        raise RuntimeError("缺少预览候选文件；请先运行 preview_shfe_xls_carry.py")
    with CANDIDATE_PATH.open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    counts = Counter(row["observation_date"] for row in candidates)
    historical = [row for row in candidates if counts[row["observation_date"]] >= 4]
    recent = [
        row for row in load_real_carry_rows()
        if row["observation_date"] > HISTORICAL_END
    ]
    backup_directory = REAL_CARRY_PATH.parent / "archive"
    backup_directory.mkdir(exist_ok=True)
    backup = backup_directory / f"term_structure_carry_before_shfe_xls_{datetime.now():%Y%m%d_%H%M%S}.csv"
    shutil.copy2(REAL_CARRY_PATH, backup)
    result = save_real_carry_rows(
        historical + recent,
        source="SHFE downloaded contract reports (2016-2024) + existing reviewed SHFE carry inputs (2025+)",
        data_version="shfe-contract-reports-weekly-2016-01-01-2024-12-31-plus-reviewed-2025",
        continuous_contract_rule=(
            "named near/far contracts; zero-OI excluded; at least four symbols per week; "
            "2016-2024 delivery-month midpoint derived from contract code, 2025+ uses reviewed expiry inputs"
        ),
    )
    print({"backup": str(backup), **result})


if __name__ == "__main__":
    main()
