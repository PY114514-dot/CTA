"""Resumable, single-worker CTA material import for the local workbench."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import UPLOAD_DIRECTORY  # noqa: E402
from app.database import SessionFactory, init_db  # noqa: E402
from app.models import RawFile  # noqa: E402
from app.services import product_store as store  # noqa: E402
from app.services.material_ingestion import ingest_uploaded_material  # noqa: E402


def _records() -> list[RawFile]:
    with SessionFactory() as session:
        return list(session.scalars(select(RawFile).order_by(RawFile.uploaded_at)).all())


def _write_status(log_path: Path, total: int, records: list[RawFile], remaining: int) -> None:
    payload = {
        "total_supported_sources": total,
        "registered": len(records),
        "remaining": remaining,
        "statuses": dict(Counter(record.parsing_status for record in records)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _register_and_ingest(
    path: Path,
    nature: str,
    product_name: str | None,
    manager_name: str | None,
    existing: RawFile | None,
) -> None:
    content = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with SessionFactory() as session:
        if existing is None:
            record = store.register_file(
                session,
                filename=path.name,
                file_hash=hashlib.sha256(content).hexdigest(),
                mime_type=mime_type,
                size_bytes=len(content),
                source=nature,
            )
        else:
            record = session.get(RawFile, existing.id)
            if record is None:
                raise RuntimeError(f"无法恢复资料记录：{existing.id}")
            record.filename = path.name
            record.mime_type = mime_type
            record.size_bytes = len(content)
            record.source = nature
            record.parsing_status = "processing"
            record.parsing_error = None
            session.commit()
        record.ingestion_context = {
            "material_nature": nature,
            "product_name_hint": product_name,
            "manager_name_hint": manager_name,
        }
        UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
        storage_name = f"{record.id}_{path.name.replace('/', '_')}"
        (UPLOAD_DIRECTORY / storage_name).write_bytes(content)
        record.storage_path = storage_name
        session.commit()
        file_id = record.id
    ingest_uploaded_material(file_id, content, path.name, mime_type, nature, product_name, manager_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    init_db()

    manifest = pd.read_csv(args.source / "分类清单.csv", encoding="utf-8-sig")
    candidates = [
        (
            args.source / Path(str(row["分类后路径"]).replace("\\", "/")),
            str(row["资料性质"]),
            str(row["产品名称"]) if pd.notna(row["产品名称"]) else None,
            str(row["私募公司"]) if pd.notna(row["私募公司"]) else None,
            str(row["来源哈希"]),
        )
        for _, row in manifest.iterrows()
    ]
    missing = [path for path, *_ in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"分类清单中有 {len(missing)} 个文件不存在：{missing[0]}")

    while True:
        records = _records()
        by_hash = {record.file_hash: record for record in records}
        remaining = [
            item for item in candidates
            if item[4] not in by_hash or by_hash[item[4]].parsing_status not in {"completed", "completed_no_nav"}
        ]
        _write_status(args.log, len(candidates), records, len(remaining))
        if not remaining:
            return
        path, nature, product_name, manager_name, file_hash = remaining[0]
        try:
            _register_and_ingest(path, nature, product_name, manager_name, by_hash.get(file_hash))
        except Exception as error:
            # Ingestion records its own concise failure where possible. Keep
            # the batch moving so one malformed document cannot strand 174.
            print(json.dumps({"failed_source": str(path), "error": str(error)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
