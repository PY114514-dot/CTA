"""File and fragment route handlers for the knowledge-base API."""

from __future__ import annotations

from email.header import decode_header
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import UPLOAD_DIRECTORY
from app.database import get_session
from app.models import DocumentFragment, NavObservation
from app.services import product_store as store
from app.services.material_ingestion import ingest_uploaded_material, ingestion_queue_snapshot
from app.services.review_page_locator import ReviewPageLocation, locate_review_page
from app.services.upload_storage import resolve_upload_path
from app.services.futures_weekly_sqlite_import import import_futures_weekly_sqlite

from .helpers import _is_pdf_record, _review_page_for_product
from .schemas import (
    BatchCurveBindRequest,
    FileBindRequest,
    FileMaterialNatureUpdate,
    FileRegisterRequest,
    FileStatusUpdate,
    FragmentCreate,
)

router = APIRouter()


def _decode_upload_filename(filename: str | None) -> str:
    """Return a usable filename when a multipart client RFC-2047 encodes it.

    Browsers normally send Unicode filenames directly, while PowerShell and a
    few enterprise upload clients send ``=?utf-8?...?=``.  Parsing must use
    the decoded suffix, otherwise a valid PNG is incorrectly rejected before
    it reaches the image pipeline.
    """
    raw = filename or "unnamed"
    try:
        chunks: list[str] = []
        for value, encoding in decode_header(raw):
            chunks.append(value.decode(encoding or "utf-8", errors="replace") if isinstance(value, bytes) else value)
        decoded = "".join(chunks).strip()
        return decoded or "unnamed"
    except Exception:
        return raw


# ---------------------------------------------------------------------------
# File endpoints
# ---------------------------------------------------------------------------


@router.post("/files")
def register_file(body: FileRegisterRequest, session: Session = Depends(get_session)):
    record = store.register_file(session, **body.model_dump())
    return {"id": record.id, "version": record.version, "parsing_status": record.parsing_status}


@router.get("/files")
def list_files(
    parsing_status: str | None = None,
    pending_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    files = store.list_files(
        session,
        parsing_status=parsing_status,
        pending_only=pending_only,
        limit=limit,
        offset=offset,
    )
    payload = []
    for f in files:
        is_bulk_sqlite = f.source == "futures_weekly_sqlite"
        active_fragments = [
            fragment for fragment in f.fragments
            if not (fragment.content_data or {}).get("superseded")
        ]
        unresolved_fragment_ids = {
            fragment.id
            for fragment in active_fragments
            if not is_bulk_sqlite
            and (fragment.product_id is None
            or (fragment.content_data or {}).get("binding_status") == "unmatched"
            )
        }
        linked_fragments = [
            fragment
            for fragment in active_fragments
            if fragment.product_id is not None and fragment.id not in unresolved_fragment_ids
        ]
        linked_observations = [
            observation
            for observation in f.nav_observations
            if observation.product_id is not None
            and observation.source_fragment_id not in unresolved_fragment_ids
        ]
        payload.append({
            "id": f.id, "filename": f.filename, "file_hash": f.file_hash,
            "mime_type": f.mime_type, "size_bytes": f.size_bytes,
            "source": f.source, "report_period": f.report_period,
            "version": f.version, "parsing_status": f.parsing_status,
            "parsing_error": f.parsing_error, "uploaded_at": f.uploaded_at.isoformat() if f.uploaded_at else None,
            "extraction_audit": f.extraction_audit,
            "workflow": (f.extraction_audit or {}).get("workflow", {}),
            "workflow_errors": (f.extraction_audit or {}).get("errors", []),
            "ingestion_context": f.ingestion_context,
            "fragment_count": len(active_fragments),
            "linked_product_ids": list(dict.fromkeys(
                [fragment.product_id for fragment in linked_fragments]
                + [observation.product_id for observation in linked_observations]
            )),
            "linked_product_names": list(dict.fromkeys(
                [fragment.product.standard_name for fragment in linked_fragments if fragment.product is not None]
                + [observation.product.standard_name for observation in linked_observations if observation.product is not None]
            ))[:8],
            "nav_count": len(f.nav_observations),
            "reviewed_nav_count": sum(
                1 for observation in f.nav_observations if observation.review_status == "reviewed"
            ),
            "unbound_fragment_count": len(unresolved_fragment_ids),
            "unbound_curve_count": len({
                (fragment.content_data or {}).get("curve_index")
                for fragment in active_fragments
                if (fragment.content_data or {}).get("binding_status") == "unmatched"
                and (fragment.content_data or {}).get("curve_index") is not None
            }),
            "curve_bindings": [
                {
                    "fragment_id": fragment.id,
                    "curve_index": (fragment.content_data or {}).get("curve_index"),
                    "product_id": fragment.product_id,
                    "product_name": fragment.product.standard_name if fragment.product is not None else None,
                    "binding_status": (fragment.content_data or {}).get("binding_status", "unmatched"),
                    "binding_confidence": (fragment.content_data or {}).get("binding_confidence", 0.0),
                    "binding_evidence": (fragment.content_data or {}).get("binding_evidence", []),
                    "legend_label": (fragment.content_data or {}).get("legend_label"),
                    "layout_product_name": (fragment.content_data or {}).get("layout_product_name"),
                    "color_hex": (fragment.content_data or {}).get("color_hex"),
                    "bbox": fragment.bbox,
                    "page_number": fragment.page_number or (fragment.content_data or {}).get("source_page"),
                }
                for fragment in active_fragments
                if fragment.fragment_type == "chart" and fragment.bbox
                and (fragment.content_data or {}).get("curve_index") is not None
            ],
        })
    return payload


@router.get("/files/ingestion-queue")
def get_ingestion_queue(session: Session = Depends(get_session)):
    """Explain the async parse queue: slot budget, active jobs, queued files.

    ``processing_count`` counts every file whose status is ``processing``,
    including jobs still waiting on the semaphore.  ``active_count`` is the
    subset actually executing inside a parse slot right now, so the UI can
    show "识别中 4 · 排队 94" instead of a single opaque number.
    """
    snapshot = ingestion_queue_snapshot()
    processing_count = session.execute(
        select(func.count()).select_from(store.RawFile).where(
            store.RawFile.parsing_status == "processing"
        )
    ).scalar_one()
    return {
        **snapshot,
        "processing_count": processing_count,
        "queued_count": max(0, processing_count - snapshot["active_count"]),
    }


@router.patch("/files/{file_id}/material-nature")
def update_file_material_nature(
    file_id: str,
    body: FileMaterialNatureUpdate,
    session: Session = Depends(get_session),
):
    """Classify an existing source without deleting its preserved evidence."""
    record = session.get(store.RawFile, file_id)
    if record is None:
        raise HTTPException(404, "文件不存在")
    record.ingestion_context = {
        **(record.ingestion_context or {}),
        "material_nature": body.material_nature,
    }
    session.commit()
    return {"id": record.id, "material_nature": body.material_nature}


@router.post("/files/{file_id}/bind")
def bind_file(file_id: str, body: FileBindRequest, session: Session = Depends(get_session)):
    """Bind only unresolved evidence from a source file to a product.

    A filename/OCR guess is never silently promoted to a confirmed product.
    ``product_id`` links to an existing entity; ``product_name`` creates a
    new pending entity, which still needs normal product and NAV review before
    it can enter research.
    """
    product_id = (body.product_id or "").strip()
    product_name = (body.product_name or "").strip()
    if not product_id and not product_name:
        raise HTTPException(422, "请选择已有产品或填写新产品名称")
    if session.get(store.RawFile, file_id) is None:
        raise HTTPException(404, "文件不存在")

    if product_id:
        product = store.get_product(session, product_id)
        if product is None:
            raise HTTPException(404, "目标产品不存在")
        if product.confirmation_status == "rejected":
            raise HTTPException(409, "不能绑定到已标记为不是产品的记录")
    else:
        product = store.create_product(
            session,
            standard_name=product_name,
            manager_name=(body.manager_name or "").strip() or None,
            strategy=(body.strategy or "").strip() or None,
        )

    result = store.bind_file_to_product(session, file_id, product.id)
    if result is None:
        raise HTTPException(404, "文件或目标产品不存在")
    return result


@router.post("/files/{file_id}/curve-bindings")
def bind_file_curves(file_id: str, body: BatchCurveBindRequest, session: Session = Depends(get_session)):
    """Apply separate, auditable product choices for a multi-product report.

    This deliberately accepts fragment IDs rather than curve indexes alone so
    a stale browser cannot accidentally bind a similarly numbered curve from
    another page or a later reparse.
    """
    if session.get(store.RawFile, file_id) is None:
        raise HTTPException(404, "文件不存在")
    fragment_ids = [choice.fragment_id for choice in body.bindings]
    if len(fragment_ids) != len(set(fragment_ids)):
        raise HTTPException(422, "同一条曲线只能确认一次")
    targets = {choice.product_id: store.get_product(session, choice.product_id) for choice in body.bindings}
    invalid_targets = [product_id for product_id, product in targets.items() if product is None or product.confirmation_status == "rejected"]
    if invalid_targets:
        raise HTTPException(422, "存在无效或已拒绝的目标产品")

    fragments = {
        fragment.id: fragment
        for fragment in session.execute(
            select(DocumentFragment).where(
                DocumentFragment.file_id == file_id,
                DocumentFragment.id.in_(fragment_ids),
            )
        ).scalars().all()
    }
    if len(fragments) != len(fragment_ids):
        raise HTTPException(404, "存在不属于该文件的曲线候选")

    moved_nav = 0
    for choice in body.bindings:
        fragment = fragments[choice.fragment_id]
        if fragment.fragment_type not in {"chart", "chart_traced"} or (fragment.content_data or {}).get("curve_index") is None:
            raise HTTPException(422, "只能绑定图表曲线候选")
        product = targets[choice.product_id]
        old_product_id = fragment.product_id
        content_data = dict(fragment.content_data or {})
        fragment.product_id = product.id
        content_data.update({
            "binding_status": "matched",
            "binding_reason": "用户在多产品曲线绑定中确认",
            "bound_by": "user:batch-curve-binding",
        })
        fragment.content_data = content_data

        # Keep curve observations with the exact evidence fragment.  Do not
        # overwrite a target's existing date; table data remains authoritative.
        observations = session.execute(
            select(NavObservation).where(NavObservation.source_fragment_id == fragment.id)
        ).scalars().all()
        for observation in observations:
            duplicate = session.execute(
                select(NavObservation.id).where(
                    NavObservation.product_id == product.id,
                    NavObservation.observation_date == observation.observation_date,
                    NavObservation.id != observation.id,
                )
            ).scalar()
            if duplicate is None:
                observation.product_id = product.id
                moved_nav += 1
            else:
                session.delete(observation)
        # Some historic imports stored observations against a temporary
        # placeholder without recording the source fragment.
        if old_product_id and old_product_id != product.id:
            for observation in session.execute(
                select(NavObservation).where(
                    NavObservation.source_file_id == file_id,
                    NavObservation.product_id == old_product_id,
                    NavObservation.source_fragment_id.is_(None),
                )
            ).scalars().all():
                observation.product_id = product.id
                moved_nav += 1
    session.commit()
    return {"file_id": file_id, "bound_curves": len(body.bindings), "moved_nav": moved_nav}


@router.get("/files/{file_id}/content")
def get_file_content(file_id: str, session: Session = Depends(get_session)):
    """Serve an already-uploaded source file for the calibration workspace."""
    record = session.get(store.RawFile, file_id)
    if record is None or not record.storage_path:
        raise HTTPException(404, "未找到可用于校准的原始文件")
    try:
        path = resolve_upload_path(record.storage_path)
    except ValueError as exc:
        raise HTTPException(400, "原始文件存储路径无效") from exc
    assert path is not None
    if not path.is_file():
        raise HTTPException(404, "原始文件已不在本地存储中")
    return FileResponse(path, media_type=record.mime_type or "application/octet-stream", filename=record.filename)


@router.get("/files/{file_id}/review-page")
def get_file_review_page(
    file_id: str,
    product_id: str,
    session: Session = Depends(get_session),
):
    """Choose the one source page that the single-image VLM may inspect."""
    record = session.get(store.RawFile, file_id)
    product = store.get_product(session, product_id)
    if record is None:
        raise HTTPException(404, "来源文件不存在")
    if product is None:
        raise HTTPException(404, "产品不存在")
    has_fragment = session.execute(
        select(DocumentFragment.id).where(
            DocumentFragment.file_id == file_id,
            DocumentFragment.product_id == product_id,
        ).limit(1)
    ).scalar_one_or_none()
    has_nav = session.execute(
        select(NavObservation.id).where(
            NavObservation.source_file_id == file_id,
            NavObservation.product_id == product_id,
        ).limit(1)
    ).scalar_one_or_none()
    if has_fragment is None and has_nav is None:
        raise HTTPException(409, "该来源文件未关联到当前产品")
    if not _is_pdf_record(record):
        return {
            "is_pdf": False,
            "page_number": None,
            "confidence": 1.0,
            "reason": "来源为图片，无需 PDF 选页",
            "selection_method": "original_image",
        }

    location = _review_page_for_product(product=product, record=record, session=session)
    return {
        "is_pdf": True,
        "page_number": location.page_number,
        "confidence": location.confidence,
        "reason": location.reason,
        "selection_method": location.selection_method,
    }


@router.get("/files/{file_id}/preview")
def get_file_preview(file_id: str, page: int = 0, session: Session = Depends(get_session)):
    """Return a browser-displayable image for source review.

    Images are returned unchanged.  PDFs are rendered page-by-page so the
    calibration workspace can always show the source evidence instead of
    asking users to upload a file they already provided.
    """
    record = session.get(store.RawFile, file_id)
    if record is None or not record.storage_path:
        raise HTTPException(404, "未找到可预览的原始文件")
    try:
        path = resolve_upload_path(record.storage_path)
    except ValueError as exc:
        raise HTTPException(400, "原始文件存储路径无效") from exc
    assert path is not None
    if not path.is_file():
        raise HTTPException(404, "原始文件已不在本地存储中")
    suffix = path.suffix.lower()
    if suffix != ".pdf" and (record.mime_type or "").lower() != "application/pdf":
        return FileResponse(path, media_type=record.mime_type or "application/octet-stream", filename=record.filename)
    if page < 0:
        raise HTTPException(422, "页码不能为负数")
    try:
        from app.services.chart_extractor.pdf_renderer import render_single_page

        rendered = render_single_page(path.read_bytes(), page_index=page)
    except IndexError:
        raise HTTPException(404, "PDF 中不存在该页") from None
    except Exception as exc:
        raise HTTPException(422, f"PDF 页面渲染失败：{exc}") from exc
    return Response(
        content=rendered.image_bytes,
        media_type="image/png",
        # HTTP headers are latin-1.  Keep the preview filename ASCII instead
        # of letting a Chinese source filename turn a valid rendered page into
        # a 500 response.
        headers={"Content-Disposition": f'inline; filename="source-preview-p{page + 1}.png"'},
    )


@router.patch("/files/{file_id}/status")
def update_file_status(file_id: str, body: FileStatusUpdate, session: Session = Depends(get_session)):
    record = store.update_file_status(session, file_id, body.parsing_status, body.parsing_error)
    if record is None:
        raise HTTPException(404, "File not found")
    return {"id": record.id, "parsing_status": record.parsing_status}


@router.post("/files/upload")
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    material_nature: str | None = Form(default=None),
    original_filename: str | None = Form(default=None),
    product_name_hint: str | None = Form(default=None),
    manager_name_hint: str | None = Form(default=None),
    session: Session = Depends(get_session),
):
    """Register an upload then automatically run it through the KB ingestion pipeline."""
    content = await file.read()
    filename = _decode_upload_filename(original_filename or file.filename)
    file_hash = store.compute_file_hash(content)
    record = store.register_file(
        session,
        filename=filename,
        file_hash=file_hash,
        mime_type=file.content_type,
        size_bytes=len(content),
        source=material_nature or "upload",
    )
    record.ingestion_context = {
        "material_nature": material_nature,
        "product_name_hint": product_name_hint,
        "manager_name_hint": manager_name_hint,
    }
    session.commit()
    # Persist to disk so the file can be re-parsed later without re-upload.
    storage_name = f"{record.id}{Path(filename).suffix.lower()}"
    try:
        UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
        path = resolve_upload_path(storage_name)
        assert path is not None
        path.write_bytes(content)
        record.storage_path = storage_name
        session.commit()
    except Exception:
        pass  # disk persistence is best-effort; ingestion still uses in-memory content
    store.update_file_status(session, record.id, "processing")
    background_tasks.add_task(
        ingest_uploaded_material, record.id, content, filename, record.mime_type,
        material_nature, product_name_hint, manager_name_hint,
    )
    return {"id": record.id, "file_hash": file_hash, "size_bytes": len(content), "version": record.version, "parsing_status": "processing"}


@router.post("/files/import-futures-weekly-sqlite")
async def import_futures_sqlite(file: UploadFile, limit: int = Form(default=100), confirm_products: bool = Form(default=False), session: Session = Depends(get_session)):
    if not (file.filename or "").lower().endswith((".sqlite", ".sqlite3", ".db")):
        raise HTTPException(415, "请上传 SQLite 数据库文件")
    content = await file.read()
    if not content: raise HTTPException(422, "上传文件为空")
    filename = _decode_upload_filename(file.filename)
    record = store.register_file(session, filename=filename, file_hash=store.compute_file_hash(content), mime_type="application/vnd.sqlite3", size_bytes=len(content), source="futures_weekly_sqlite")
    storage_name = f"{record.id}.sqlite"; path = resolve_upload_path(storage_name); assert path is not None
    path.write_bytes(content); record.storage_path = storage_name; session.commit()
    try: return import_futures_weekly_sqlite(session, path=path, raw_file=record, limit=limit, confirm_products=confirm_products)
    except ValueError as error:
        record.parsing_status = "failed"; record.parsing_error = str(error); session.commit(); raise HTTPException(422, str(error)) from error


@router.post("/files/{file_id}/reparse")
def reparse_file(file_id: str, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """Re-run an unresolved source; published human-reviewed NAV is immutable by default."""
    record = session.get(store.RawFile, file_id)
    if record is None:
        raise HTTPException(404, "File not found")
    if not record.storage_path:
        raise HTTPException(409, "该文件未落盘，无法重新解析；请重新上传")
    try:
        path = resolve_upload_path(record.storage_path)
    except ValueError as exc:
        raise HTTPException(400, "存储文件路径无效") from exc
    assert path is not None
    if not path.is_file():
        raise HTTPException(409, "存储文件丢失，请重新上传")
    has_reviewed_nav = session.execute(
        select(NavObservation.id).where(
            NavObservation.source_file_id == file_id,
            NavObservation.review_status == "reviewed",
        ).limit(1)
    ).scalar()
    if has_reviewed_nav is not None:
        raise HTTPException(409, "该资料的净值已人工审核并发布；不会自动重新识别。请在产品页明确发起重新复核。")
    content = path.read_bytes()
    store.update_file_status(session, file_id, "processing")
    context = record.ingestion_context or {}
    background_tasks.add_task(
        ingest_uploaded_material, record.id, content, record.filename, record.mime_type,
        context.get("material_nature") or record.source,
        context.get("product_name_hint"), context.get("manager_name_hint"),
    )
    return {"id": record.id, "parsing_status": "processing"}


@router.delete("/files/{file_id}")
def delete_file(file_id: str, session: Session = Depends(get_session)):
    """Delete an uploaded file and the source-dependent research records."""
    record = session.get(store.RawFile, file_id)
    if record is None:
        raise HTTPException(404, "File not found")
    try:
        disk_path = resolve_upload_path(record.storage_path)
    except ValueError as exc:
        raise HTTPException(400, "存储文件路径无效") from exc
    result = store.delete_file(session, file_id)
    if result is False:
        raise HTTPException(404, "File not found")
    # Remove physical file if it was stored
    if disk_path is not None and disk_path.is_file():
        disk_path.unlink(missing_ok=True)
    return {"id": file_id, "deleted": True, **result}


# ---------------------------------------------------------------------------
# Fragment endpoints
# ---------------------------------------------------------------------------


@router.post("/fragments")
def create_fragment(body: FragmentCreate, session: Session = Depends(get_session)):
    fragment = store.add_fragment(session, **body.model_dump())
    return {"id": fragment.id, "file_id": fragment.file_id, "fragment_type": fragment.fragment_type}


@router.get("/fragments")
def list_fragments(file_id: str, session: Session = Depends(get_session)):
    fragments = store.get_fragments_for_file(session, file_id)
    return [
        {
            "id": f.id, "file_id": f.file_id, "fragment_type": f.fragment_type,
            "page_number": f.page_number, "bbox": f.bbox,
            "content_text": f.content_text[:200] if f.content_text else None,
            "ocr_confidence": f.ocr_confidence, "product_id": f.product_id,
        }
        for f in fragments
    ]
