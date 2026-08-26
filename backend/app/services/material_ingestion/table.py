"""Tabular material ingestion (CSV/XLSX, long or wide format)."""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.services import product_store as store

from .helpers import _find_column, _find_or_create_product, _infer_frequency


def _ingest_nav_table(session: Any, file_id: str, content: bytes, filename: str, suffix: str) -> int:
    frame = pd.read_csv(BytesIO(content)) if suffix == ".csv" else pd.read_excel(BytesIO(content))
    return _ingest_nav_frame(session, file_id, frame, filename)


def _ingest_nav_frame(
    session: Any,
    file_id: str,
    frame: pd.DataFrame,
    filename: str,
    *,
    location: str | None = None,
) -> int:
    """Persist a disclosed NAV table only when its dates and values are usable."""
    date_column = _find_column(frame, ("date", "日期", "净值日期", "估值日期", "观察日"))
    if date_column is None:
        # Heuristic: first column mostly parseable as dates.
        for col in frame.columns:
            if pd.to_datetime(frame[col], errors="coerce", format="mixed").notna().mean() > 0.7:
                date_column = col
                break
    if date_column is None:
        return 0

    nav_column = _find_column(frame, ("nav", "净值", "单位净值", "累计净值", "acc_nav"))
    if nav_column is not None:
        # Long format: single explicit NAV column.
        value_columns = [nav_column]
    else:
        # Wide format: every numeric column (except date) is one product.
        value_columns = [
            col for col in frame.columns
            if col != date_column and pd.api.types.is_numeric_dtype(frame[col])
        ]
        if not value_columns:
            return 0

    dates_series = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    total_rows = 0
    for col in value_columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        mask = dates_series.notna() & values.notna() & (values > 0)
        parsed = pd.DataFrame({"date": dates_series[mask], "nav": values[mask]}).drop_duplicates("date").sort_values("date")
        if len(parsed) < 2:
            continue
        dates = [value.date() for value in parsed["date"].tolist()]
        col_name = str(col).strip()
        product_name = Path(filename).stem if nav_column is not None else col_name
        if not product_name or product_name.startswith("Unnamed"):
            product_name = Path(filename).stem
        product = _find_or_create_product(session, product_name, None, dates[0], dates[-1])
        fragment = store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="table",
            content_text=f"自动导入 {len(parsed)} 条净值记录（列: {col_name}）",
            content_data={"date_column": str(date_column), "nav_column": col_name, "rows": len(parsed), "method": "表格解析（Pandas）", "review_status": "pending", "location": location},
            ocr_confidence=1.0,
            product_id=product.id,
        )
        frequency = _infer_frequency(dates)
        total_rows += store.add_nav_observations(
            session,
            product.id,
            [{"observation_date": item_date, "nav": float(value)} for item_date, value in zip(dates, parsed["nav"].tolist(), strict=True)],
            source_file_id=file_id,
            source_fragment_id=fragment.id,
            frequency=frequency,
        )
    if total_rows == 0:
        return 0
    return total_rows


