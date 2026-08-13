"""Versioned storage for externally disclosed CTA factor observations.

The source report is preserved alongside parsed observations.  These values
are an external benchmark, never a replacement for internally computed factor
returns.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from urllib.parse import quote
from datetime import date, datetime
from pathlib import Path

from app.config import EXTERNAL_FACTOR_LIBRARY_DIRECTORY

SOURCE = "guotai_junan_cta"  # legacy identifier retained for existing snapshots
HEADERS = ["as_of_date", "source", "factor_id", "source_factor_name", "horizon", "return_pct", "source_document_id", "source_sha256", "page", "extraction_method", "confidence", "verified", "revision", "ingested_at"]
HORIZONS = ("1w", "1m", "3m", "6m", "1y")
# Anchor text that marks the start of the factor-performance table (issue #8).
TABLE_HEADER = "近一周"
FACTOR_DEFINITIONS = [
    {"factor_id": "short_term_trend_20", "display_name": "短期时序动量", "category": "量价", "lookback_days": 20, "internal_mapping": None},
    {"factor_id": "long_term_trend_240", "display_name": "长期时序动量", "category": "量价", "lookback_days": 240, "internal_mapping": "trend_120_250"},
    {"factor_id": "long_term_cross_section_momentum_240", "display_name": "长期截面动量", "category": "量价", "lookback_days": 240, "internal_mapping": "cross_section_momentum"},
    {"factor_id": "basis", "display_name": "基差", "category": "基本面", "lookback_days": None, "internal_mapping": None},
    {"factor_id": "skewness", "display_name": "偏度", "category": "量价", "lookback_days": None, "internal_mapping": None},
    {"factor_id": "warehouse_receipt", "display_name": "仓单", "category": "基本面", "lookback_days": None, "internal_mapping": None},
    {"factor_id": "inventory", "display_name": "库存", "category": "基本面", "lookback_days": None, "internal_mapping": None},
    {"factor_id": "profit", "display_name": "利润", "category": "基本面", "lookback_days": None, "internal_mapping": None},
]
_NAME_TO_ID = {item["display_name"]: item["factor_id"] for item in FACTOR_DEFINITIONS}


class ExternalFactorLibraryError(ValueError):
    pass


def _paths() -> tuple[Path, Path, Path]:
    root = EXTERNAL_FACTOR_LIBRARY_DIRECTORY
    return root, root / "source_documents", root / "observations.csv"


def _ensure_storage() -> tuple[Path, Path, Path]:
    root, docs, observations = _paths()
    docs.mkdir(parents=True, exist_ok=True)
    definitions = root / "factor_definitions.json"
    if not definitions.exists():
        definitions.write_text(json.dumps(FACTOR_DEFINITIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    return root, docs, observations


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise ExternalFactorLibraryError(f"PDF 文本提取失败：{error}") from error


def parse_gtja_factor_table(text: str) -> tuple[str, list[dict]]:
    """Parse table 3 from native PDF text or PaddleOCR Markdown."""
    date_match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not date_match:
        raise ExternalFactorLibraryError("未识别到报告日期，无法入库。")
    as_of_date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
    start = text.find(TABLE_HEADER)
    if start < 0:
        raise ExternalFactorLibraryError(f"未识别到“{TABLE_HEADER}”表头。")
    block = text[start:start + 2500]
    observations: list[dict] = []
    # Longest-name-first so a shorter display_name can never shadow a longer
    # one that shares its prefix, e.g. "长期" vs "长期截面" (issue #7).
    for display_name, factor_id in sorted(_NAME_TO_ID.items(), key=lambda kv: len(kv[0]), reverse=True):
        match = re.search(
            rf"{re.escape(display_name)}\s+([-+]?\d+(?:\.\d+)?)\s*%?\s+([-+]?\d+(?:\.\d+)?)\s*%?\s+([-+]?\d+(?:\.\d+)?)\s*%?\s+([-+]?\d+(?:\.\d+)?)\s*%?\s+([-+]?\d+(?:\.\d+)?)\s*%?",
            block,
        )
        if not match:
            continue
        for horizon, value in zip(HORIZONS, match.groups(), strict=True):
            observations.append({"factor_id": factor_id, "source_factor_name": display_name, "horizon": horizon, "return_pct": float(value)})
    if len(observations) != len(FACTOR_DEFINITIONS) * len(HORIZONS):
        found = len(observations) // len(HORIZONS)
        raise ExternalFactorLibraryError(f"因子表仅识别到 {found}/8 个因子；请人工核验或使用 PaddleOCR 后重试。")
    return as_of_date, observations


def detect_organization(text: str) -> str:
    """Return a human-readable issuer name from common CTA report headers."""
    match = re.search(r"([\u4e00-\u9fa5]{2,12}(?:期货|证券|投资|基金|资产))", text[:3000])
    return match.group(1) if match else "未识别机构"


def organization_slug(organization_name: str) -> str:
    digest = hashlib.sha1(organization_name.strip().encode("utf-8")).hexdigest()[:10]
    return f"org_{digest}"


def ingest_external_document(file_bytes: bytes, filename: str, text: str, extraction_method: str, organization_name: str | None = None) -> dict:
    """Persist source document and parsed table as a versioned weekly snapshot."""
    as_of_date, parsed = parse_gtja_factor_table(text)
    organization_name = organization_name.strip() if organization_name else detect_organization(text)
    source = organization_slug(organization_name)
    root, documents, observations_path = _ensure_storage()
    digest = hashlib.sha256(file_bytes).hexdigest()
    document_id = digest[:16]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "source.bin"
    source_path = documents / f"{source}_{document_id}_{safe_name}"
    if not source_path.exists():
        source_path.write_bytes(file_bytes)

    existing = _read_rows(observations_path)
    same_document = any(row["source_sha256"] == digest for row in existing)
    if same_document:
        return {"organization_name": organization_name, "source": source, "as_of_date": as_of_date, "source_document_id": document_id, "rows_added": 0, "revision": max(int(row["revision"]) for row in existing if row["source_sha256"] == digest), "duplicate": True}
    revisions = [int(row["revision"]) for row in existing if row["as_of_date"] == as_of_date and row["source"] == source]
    revision = max(revisions, default=0) + 1
    now = datetime.now().isoformat(timespec="seconds")
    rows = [{
        "as_of_date": as_of_date, "source": source, "source_document_id": document_id,
        "source_sha256": digest, "page": "8", "extraction_method": extraction_method,
        "confidence": "1.0" if extraction_method == "pdf_text" else "0.75", "verified": "false",
        "revision": str(revision), "ingested_at": now, **item,
    } for item in parsed]
    _append_rows(observations_path, rows)
    with (root / "ingest_log.jsonl").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"as_of_date": as_of_date, "organization_name": organization_name, "source": source, "document_id": document_id, "revision": revision, "rows_added": len(rows), "filename": filename, "ingested_at": now}, ensure_ascii=False) + "\n")
    return {"organization_name": organization_name, "source": source, "as_of_date": as_of_date, "source_document_id": document_id, "rows_added": len(rows), "revision": revision, "duplicate": False}


def ingest_gtja_document(file_bytes: bytes, filename: str, text: str, extraction_method: str) -> dict:
    """Backward-compatible wrapper for the original Guotai Junan route."""
    return ingest_external_document(file_bytes, filename, text, extraction_method, "国泰君安期货")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_rows(path: Path, rows: list[dict]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def latest_observations() -> list[dict[str, str]]:
    _, _, path = _ensure_storage()
    rows = _read_rows(path)
    latest_revision: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["as_of_date"], row["source"])
        latest_revision[key] = max(latest_revision.get(key, 0), int(row["revision"]))
    return [row for row in rows if int(row["revision"]) == latest_revision[(row["as_of_date"], row["source"])]]


def latest_snapshot_date() -> str | None:
    dates = [row["as_of_date"] for row in latest_observations()]
    return max(dates) if dates else None


def latest_snapshot_metadata(product_start: date | None = None, product_end: date | None = None) -> dict:
    """Return the latest snapshot that was knowable on the product end date.

    The imported GTJA report is a dated multi-horizon snapshot, not a complete
    factor return time series.  This metadata is intentionally suitable for
    comparison and audit only; callers must not treat it as regression input.
    """
    rows = latest_observations()
    if not rows:
        return {
            "available": False,
            "data_role": "reference_snapshot_only",
            "warnings": ["尚未导入外部因子周报。"],
        }

    available_dates = sorted({row["as_of_date"] for row in rows})
    eligible_dates = [value for value in available_dates if product_end is None or value <= product_end.isoformat()]
    if not eligible_dates:
        latest_available = available_dates[-1]
        return {
            "available": False,
            "data_role": "reference_snapshot_only",
            "product_start": product_start.isoformat() if product_start else None,
            "product_end": product_end.isoformat() if product_end else None,
            "selected_as_of_date": None,
            "latest_available_as_of_date": latest_available,
            "date_relation": "no_snapshot_on_or_before_product_end",
            "usable_for_regression": False,
            "warnings": [
                f"外部因子最新快照截至 {latest_available}，晚于产品净值截止日 {product_end.isoformat()}；"
                "不存在可对齐的历史快照，因此未纳入本次参考。",
            ],
        }

    target = eligible_dates[-1]
    snapshot = [row for row in rows if row["as_of_date"] == target]
    factor_ids = {row.get("factor_id", "") for row in snapshot if row.get("factor_id")}
    horizons = sorted({row.get("horizon", "") for row in snapshot if row.get("horizon")})
    verified_rows = sum(row.get("verified") == "true" for row in snapshot)
    try:
        snapshot_date = date.fromisoformat(target)
    except ValueError:
        snapshot_date = None

    warnings: list[str] = []
    relation = "unknown"
    if snapshot_date and product_start and product_end:
        if snapshot_date > product_end:
            relation = "reference_after_product_end"
            warnings.append("外部因子为产品截止日后的披露快照，仅作市场背景参考；该类快照不参与连续回归。")
        elif snapshot_date < product_start:
            relation = "reference_before_product_start"
            warnings.append("外部因子为产品起始日前的披露快照，仅作市场背景参考；该类快照不参与连续回归。")
        else:
            relation = "inside_product_range"
    if verified_rows < len(snapshot):
        warnings.append("外部因子快照尚有未核验观测，不能作为已确认基准。")

    return {
        "available": True,
        "data_role": "reference_snapshot_only",
        "source": sorted({row.get("source", "") for row in snapshot if row.get("source")}),
        # Kept for existing clients; it is the selected, date-aligned snapshot,
        # never blindly the newest external report.
        "latest_as_of_date": target,
        "selected_as_of_date": target,
        "latest_available_as_of_date": available_dates[-1],
        "factor_count": len(factor_ids),
        "observation_count": len(snapshot),
        "horizons": horizons,
        "verified": verified_rows == len(snapshot) and bool(snapshot),
        "verified_rows": verified_rows,
        "product_start": product_start.isoformat() if product_start else None,
        "product_end": product_end.isoformat() if product_end else None,
        "date_relation": relation,
        "usable_for_regression": False,
        "warnings": warnings,
    }


def verify_snapshot(as_of_date: str, source: str | None = None, verified_by: str = "user") -> int:
    """Mark the current revision of a report snapshot as manually checked."""
    _, _, path = _ensure_storage()
    rows = _read_rows(path)
    candidates = [r for r in rows if r["as_of_date"] == as_of_date and (source is None or r["source"] == source)]
    if not candidates:
        raise ExternalFactorLibraryError("未找到对应报告日的数据。")
    latest: dict[str, int] = {}
    for row in candidates:
        latest[row["source"]] = max(latest.get(row["source"], 0), int(row["revision"]))
    changed = 0
    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        if row["as_of_date"] == as_of_date and row["source"] in latest and int(row["revision"]) == latest[row["source"]]:
            if row.get("verified") != "true":
                row["verified"] = "true"
                row["verified_by"] = verified_by
                row["verified_at"] = now
                changed += 1
    fields = list(dict.fromkeys(HEADERS + ["verified_by", "verified_at"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return changed


def snapshot_rows(as_of_date: str | None = None) -> list[dict[str, str]]:
    target = as_of_date or latest_snapshot_date()
    return [row for row in latest_observations() if row["as_of_date"] == target]


def _snapshot_matrix(as_of_date: str | None = None) -> tuple[str, list[dict[str, str]]]:
    target = as_of_date or latest_snapshot_date()
    if target is None:
        raise ExternalFactorLibraryError("尚未导入任何周报因子数据。")
    values = {(row["factor_id"], row["horizon"]): row["return_pct"] for row in snapshot_rows(target)}
    rows = []
    for factor in FACTOR_DEFINITIONS:
        row = {"因子": factor["display_name"]}
        for horizon, label in zip(HORIZONS, ("近一周", "近一月", "近三月", "近半年", "近一年"), strict=True):
            value = values.get((factor["factor_id"], horizon))
            row[label] = f"{float(value):.2f}%" if value is not None else ""
        rows.append(row)
    return target, rows


def export_latest_csv() -> tuple[str, bytes]:
    target, rows = _snapshot_matrix()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["因子", "近一周", "近一月", "近三月", "近半年", "近一年"])
    writer.writeheader()
    writer.writerows(rows)
    return target, buffer.getvalue().encode("utf-8-sig")


def export_latest_json() -> tuple[str, bytes]:
    root, _, _ = _ensure_storage()
    target, rows = _snapshot_matrix()
    payload = {"来源": "国泰君安期货研究", "报告日": target, "因子表现": rows, "因子定义": json.loads((root / "factor_definitions.json").read_text(encoding="utf-8"))}
    return target, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_latest_xlsx() -> tuple[str, bytes]:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError as error:
        raise ExternalFactorLibraryError("未安装 openpyxl，无法导出 Excel。") from error
    target, rows = _snapshot_matrix()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "CTA因子表现"
    sheet.append(["国泰君安 CTA 因子表现", target])
    sheet.append(["数据来源：国泰君安期货研究"])
    headers = ["因子", "近一周", "近一月", "近三月", "近半年", "近一年"]
    sheet.append(headers)
    for row in rows:
        sheet.append([row[header] for header in headers])
    sheet.merge_cells("A1:F1")
    sheet.merge_cells("A2:F2")
    sheet["A1"].font = Font(name="Arial", bold=True, size=14, color="1F4E78")
    sheet["A2"].font = Font(name="Arial", italic=True, color="666666")
    for cell in sheet[3]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4F81BD")
        cell.alignment = Alignment(horizontal="center")
    for row in sheet.iter_rows(min_row=4, max_row=11, min_col=2, max_col=6):
        for cell in row:
            cell.alignment = Alignment(horizontal="center")
    for column, width in {"A": 22, "B": 14, "C": 14, "D": 14, "E": 14, "F": 14}.items():
        sheet.column_dimensions[column].width = width
    output = io.BytesIO(); workbook.save(output)
    return target, output.getvalue()


def content_disposition(extension: str, as_of_date: str) -> str:
    chinese_name = f"国泰君安-CTA因子-{as_of_date.replace('-', '')}.{extension}"
    return f"attachment; filename=guotai-junan-cta-{as_of_date.replace('-', '')}.{extension}; filename*=UTF-8''{quote(chinese_name)}"
