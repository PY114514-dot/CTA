"""Top-level material ingestion entry points and dispatcher."""

from __future__ import annotations

import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.database import SessionFactory
from app.models import DocumentFragment, NavCandidateVersion, NavObservation, ProductEntity
from app.services import product_store as store

from .helpers import (
    _empty_extraction_audit,
    _infer_frequency,
    _is_multi_product_material,
    _merge_extraction_audit,
    _normalise_manifest_product_name,
)
from .image import _ingest_report_image, _recover_disclosed_nav_table
from .office import _ingest_office_document
from .pdf import _ingest_pdf
from .table import _ingest_nav_table

_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
_TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}
_OFFICE_SUFFIXES = {".docx", ".pptx"}


def _env_ingest_concurrency() -> int:
    """Parse-slot budget; VLM/OCR-heavy jobs thrash CPU and quotas past ~4."""
    try:
        value = int(os.getenv("INGEST_MAX_CONCURRENCY", "4"))
    except ValueError:
        return 4
    return max(1, min(value, 16))


_INGEST_MAX_CONCURRENCY = _env_ingest_concurrency()
_INGEST_SEMAPHORE = threading.BoundedSemaphore(_INGEST_MAX_CONCURRENCY)
_INGEST_ACTIVE_LOCK = threading.Lock()
_INGEST_ACTIVE_FILE_IDS: set[str] = set()


def ingestion_queue_snapshot() -> dict[str, Any]:
    """Current parse-slot state for the upload workflow UI."""
    with _INGEST_ACTIVE_LOCK:
        active_file_ids = sorted(_INGEST_ACTIVE_FILE_IDS)
    return {
        "max_concurrency": _INGEST_MAX_CONCURRENCY,
        "active_count": len(active_file_ids),
        "active_file_ids": active_file_ids,
    }


def ingest_uploaded_material(
    file_id: str,
    content: bytes,
    filename: str,
    mime_type: str | None,
    material_nature: str | None = None,
    product_name_hint: str | None = None,
    manager_name_hint: str | None = None,
) -> None:
    """Parse one uploaded file within a bounded concurrency budget.

    BackgroundTasks fans every upload onto the anyio threadpool; without a cap
    a batch of 98 files launches that many VLM/OCR jobs at once.  The semaphore
    serialises the heavy work while queued files keep ``parsing_status =
    "processing"`` in the database, so the queue endpoint can report how many
    are actually running versus waiting.
    """
    with _INGEST_SEMAPHORE:
        with _INGEST_ACTIVE_LOCK:
            _INGEST_ACTIVE_FILE_IDS.add(file_id)
        try:
            _ingest_uploaded_material_impl(
                file_id, content, filename, mime_type,
                material_nature, product_name_hint, manager_name_hint,
            )
        finally:
            with _INGEST_ACTIVE_LOCK:
                _INGEST_ACTIVE_FILE_IDS.discard(file_id)


def _ingest_uploaded_material_impl(
    file_id: str,
    content: bytes,
    filename: str,
    mime_type: str | None,
    material_nature: str | None = None,
    product_name_hint: str | None = None,
    manager_name_hint: str | None = None,
) -> None:
    """Parse one uploaded file and persist reviewable evidence in the KB."""
    with SessionFactory() as session:
        store.update_file_status(session, file_id, "processing")
        # A reparse supersedes only unreviewed candidates from this same file.
        # Keep published/reviewed evidence intact, but do not show stale
        # alternatives alongside the new extraction.
        for candidate in session.execute(
            select(NavCandidateVersion).where(
                NavCandidateVersion.source_file_id == file_id,
                NavCandidateVersion.status == "pending",
            )
        ).scalars():
            candidate.status = "discarded"
        reviewed_fragment_ids = {
            fragment_id
            for fragment_id in session.execute(
                select(NavObservation.source_fragment_id).where(
                    NavObservation.source_file_id == file_id,
                    NavObservation.review_status == "reviewed",
                    NavObservation.source_fragment_id.is_not(None),
                )
            ).scalars()
        }
        for fragment in session.execute(
            select(DocumentFragment).where(
                DocumentFragment.file_id == file_id,
                DocumentFragment.fragment_type.in_(("chart", "chart_traced")),
            )
        ).scalars():
            if fragment.id not in reviewed_fragment_ids:
                fragment.content_data = {**(fragment.content_data or {}), "superseded": True}
        session.commit()
        try:
            suffix = Path(filename).suffix.lower()
            context_only = material_nature in {"CTA策略介绍", "CTA研究/市场资料"}
            visual_nav_allowed = not context_only
            if not visual_nav_allowed:
                audit = _ingest_context_material(session, file_id, content, filename, suffix, mime_type)
            elif mime_type in _IMAGE_TYPES or suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                audit = _ingest_report_image(
                    session, file_id, content, filename,
                    product_name_hint=product_name_hint,
                    manager_name_hint=manager_name_hint,
                )
            elif suffix in _TABLE_SUFFIXES:
                rows = _ingest_nav_table(session, file_id, content, filename, suffix)
                audit = {"configured": False, "attempted": False, "succeeded": False, "methods": ["Pandas 表格解析"], "nav_rows": rows}
            elif suffix in _OFFICE_SUFFIXES:
                rows = _ingest_office_document(session, file_id, content, filename, suffix)
                audit = {"configured": False, "attempted": False, "succeeded": False, "methods": ["DOCX/PPTX XML 提取"], "nav_rows": rows}
            elif mime_type == "application/pdf" or suffix == ".pdf":
                audit = _ingest_pdf(
                    session,
                    file_id,
                    content,
                    filename,
                    product_name_hint=product_name_hint,
                )
            else:
                raise ValueError("暂不支持该文件类型的自动解析；请上传 PDF、图片、XLSX、CSV、DOCX 或 PPTX")
        except Exception as error:  # Background jobs must surface a concise, user-visible status.
            # Preserve the failure, but also persist the configured extraction
            # path so the UI can distinguish a VLM outage from a bad document.
            audit = locals().get("audit") or _empty_extraction_audit()
            audit = {**audit, "status": "failed", "error": str(error)[:300]}
            store.update_file_extraction_audit(session, file_id, audit)
            store.update_file_status(session, file_id, "failed", str(error)[:300])
        else:
            store.update_file_extraction_audit(session, file_id, audit)
            candidate_count = session.execute(
                select(NavCandidateVersion.id).where(NavCandidateVersion.source_file_id == file_id)
            ).scalars().all()
            nav_count = session.execute(
                select(NavObservation.id).where(NavObservation.source_file_id == file_id)
            ).scalars().all()
            if candidate_count or nav_count:
                store.update_file_status(session, file_id, "completed")
            else:
                charts = session.execute(
                    select(DocumentFragment.id).where(
                        DocumentFragment.file_id == file_id,
                        DocumentFragment.fragment_type.in_(("chart", "chart_traced")),
                    )
                ).scalars().all()
                detail = "已定位图表区域，但未提取到可校准的净值曲线；请打开原图选择曲线颜色或手动框选坐标轴。" if charts else "未定位到可提取的净值图表；资料已保留，可在原图中手动框选。"
                store.update_file_status(session, file_id, "completed_no_nav", detail)


def _ingest_context_material(
    session: Any,
    file_id: str,
    content: bytes,
    filename: str,
    suffix: str,
    mime_type: str | None,
) -> dict[str, Any]:
    """Retain strategy/research evidence without pretending charts are fund NAV."""
    if suffix in _OFFICE_SUFFIXES:
        _ingest_office_document(session, file_id, content, filename, suffix)
        method = "DOCX/PPTX XML 提取（上下文资料）"
    elif mime_type == "application/pdf" or suffix == ".pdf":
        import fitz

        document = fitz.open(stream=content, filetype="pdf")
        text_parts: list[str] = []
        text_length = 0
        for page_index in range(min(document.page_count, 80)):
            page_text = document.load_page(page_index).get_text("text")
            if page_text:
                text_parts.append(page_text)
                text_length += len(page_text)
            if text_length >= 20000:
                break
        document.close()
        text = "\n".join(text_parts).strip()
        if not text:
            raise ValueError("上下文 PDF 无可读取文本层；未将图表误作产品净值")
        store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="text",
            content_text=text[:20000],
            content_data={"method": "PDF 文本层提取（上下文资料）", "review_status": "pending"},
            ocr_confidence=0.9,
            product_id=None,
        )
        method = "PDF 文本层提取（上下文资料）"
    else:
        # The classification manifest has already established that this is
        # context, not a product-performance image. Keep the source reviewable
        # and avoid an expensive VLM call that cannot produce formal NAV.
        store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="image",
            content_data={"method": "分类清单分流（上下文图片）", "review_status": "pending"},
            product_id=None,
        )
        method = "分类清单分流（上下文图片）"
    return {"configured": False, "attempted": False, "succeeded": False, "methods": [method]}


def recover_pending_disclosed_nav(file_id: str) -> dict[str, Any]:
    """Run only the exact-table recovery lane for one unresolved image.

    The operation is idempotent: an existing candidate, including a reviewed
    or published one, prevents another machine version from being appended.
    """
    from app.services.upload_storage import resolve_upload_path

    with SessionFactory() as session:
        from app.models import RawFile

        raw_file = session.get(RawFile, file_id)
        if raw_file is None:
            return {"file_id": file_id, "outcome": "missing_file"}
        existing = session.execute(
            select(NavCandidateVersion.id).where(NavCandidateVersion.source_file_id == file_id)
        ).scalar()
        if existing is not None:
            return {"file_id": file_id, "outcome": "candidate_already_exists"}
        context = raw_file.ingestion_context or {}
        if _is_multi_product_material(raw_file.filename, context.get("product_name_hint")):
            audit = dict(raw_file.extraction_audit or {})
            workflow = dict(audit.get("workflow") or {})
            workflow.update({"binding": "needs_review", "trace": "blocked_on_binding", "next_action": "confirm_product_binding"})
            audit["workflow"] = workflow
            store.update_file_extraction_audit(session, file_id, audit)
            return {"file_id": file_id, "outcome": "needs_product_binding"}
        suffix = Path(raw_file.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            return {"file_id": file_id, "outcome": "unsupported_media"}
        path = resolve_upload_path(raw_file.storage_path)
        if path is None or not path.is_file():
            return {"file_id": file_id, "outcome": "missing_upload"}

        products = session.execute(
            select(ProductEntity).join(DocumentFragment, DocumentFragment.product_id == ProductEntity.id)
            .where(DocumentFragment.file_id == file_id).distinct()
        ).scalars().all()
        product = products[0] if len(products) == 1 else None
        if product is None:
            hint = _normalise_manifest_product_name(context.get("product_name_hint"))
            if hint:
                product = session.execute(
                    select(ProductEntity).where(ProductEntity.standard_name == hint)
                ).scalars().first()
        if product is None:
            audit = dict(raw_file.extraction_audit or {})
            workflow = dict(audit.get("workflow") or {})
            workflow.update({"identity": "needs_review", "next_action": "confirm_product_binding"})
            audit["workflow"] = workflow
            store.update_file_extraction_audit(session, file_id, audit)
            return {"file_id": file_id, "outcome": "needs_product_binding"}

        points, recovery_audit = _recover_disclosed_nav_table(path.read_bytes())
        audit = _merge_extraction_audit(raw_file.extraction_audit or {}, recovery_audit)
        if len(points) < 3:
            workflow = dict((raw_file.extraction_audit or {}).get("workflow") or {})
            workflow.update({"trace": "needs_calibration", "next_action": "recover_chart_calibration"})
            audit["workflow"] = workflow
            store.update_file_extraction_audit(session, file_id, audit)
            return {"file_id": file_id, "outcome": "no_reliable_table", "rows": len(points)}

        fragment = store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="table",
            content_text=f"披露表格恢复 {len(points)} 个候选净值点",
            content_data={
                "method": "VLM 披露净值表读取",
                "review_status": "pending",
                "workflow_stage": "table_recovery",
                "num_points": len(points),
                "selected_variant": recovery_audit.get("selected_variant"),
            },
            ocr_confidence=0.9,
            product_id=product.id,
        )
        candidate = store.create_nav_candidate_version(
            session,
            product.id,
            file_id,
            points,
            source_fragment_id=fragment.id,
            frequency=_infer_frequency([date.fromisoformat(point["observation_date"]) for point in points]),
            confidence=0.9,
        )
        fragment.content_data = {**(fragment.content_data or {}), "candidate_version_id": candidate.id}
        session.commit()
        audit["workflow"] = {
            **dict((raw_file.extraction_audit or {}).get("workflow") or {}),
            "identity": "resolved",
            "trace": "candidate_created",
            "next_action": "review_candidate",
        }
        store.update_file_extraction_audit(session, file_id, audit)
        store.update_file_status(session, file_id, "completed")
        return {
            "file_id": file_id,
            "outcome": "candidate_created",
            "candidate_id": candidate.id,
            "product_name": product.standard_name,
            "rows": len(points),
        }


