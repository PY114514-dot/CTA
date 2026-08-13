"""Product knowledge-base HTTP interface (P0 data layer).

Endpoints for material registration, product CRUD, NAV management,
structured facts, agent run audit and decision records.
"""

from __future__ import annotations

from io import BytesIO
from datetime import date
from email.header import decode_header
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import UPLOAD_DIRECTORY
from app.database import get_session
from app.models import DocumentFragment, NavCandidateVersion, NavObservation
from app.services import product_store as store
from app.services.material_ingestion import ingest_uploaded_material
from app.services.nav_quality import assess_machine_nav_review, assess_nav_quality
from app.services.review_page_locator import ReviewPageLocation, locate_review_page
from app.services.upload_storage import resolve_upload_path

router = APIRouter(prefix="/api/kb", tags=["产品知识库"])
# Keep the development reloader aware of the knowledge-base route module.


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
# Request / Response schemas
# ---------------------------------------------------------------------------


class FileRegisterRequest(BaseModel):
    filename: str
    file_hash: str
    mime_type: str | None = None
    size_bytes: int | None = None
    source: str | None = None
    report_period: str | None = None


class FileStatusUpdate(BaseModel):
    parsing_status: str
    parsing_error: str | None = None


class FileBindRequest(BaseModel):
    """Human choice used to resolve an uploaded file's product identity.

    Existing products are selected by ``product_id``.  When the source file
    contains a new product, the UI sends ``product_name`` and we create a
    pending entity before attaching the file evidence to it.
    """

    product_id: str | None = None
    product_name: str | None = None
    manager_name: str | None = None
    strategy: str | None = None


class CurveBindingChoice(BaseModel):
    """One explicit user confirmation for one chart curve."""

    fragment_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)


class BatchCurveBindRequest(BaseModel):
    bindings: list[CurveBindingChoice] = Field(min_length=1, max_length=32)


class FragmentCreate(BaseModel):
    file_id: str
    fragment_type: str
    page_number: int | None = None
    bbox: dict[str, Any] | None = None
    content_text: str | None = None
    content_data: dict[str, Any] | None = None
    ocr_confidence: float | None = None
    product_id: str | None = None


class ProductCreate(BaseModel):
    standard_name: str
    manager_name: str | None = None
    strategy: str | None = None
    inception_date: date | None = None
    nav_frequency: str | None = None
    status: str = "unknown"
    notes: str | None = None


class ProductConfirm(BaseModel):
    confirmed_by: str = "user"


class ProductUpdate(BaseModel):
    standard_name: str | None = None
    manager_name: str | None = None
    strategy: str | None = None
    inception_date: date | None = None
    close_date: date | None = None
    nav_frequency: str | None = None
    status: str | None = None
    notes: str | None = None


class AliasCreate(BaseModel):
    alias: str
    alias_type: str = "name"
    source_file_id: str | None = None


class MergeRequest(BaseModel):
    source_id: str
    target_id: str


class NavPointInput(BaseModel):
    observation_date: date
    nav: float = Field(gt=0)
    acc_nav: float | None = None


class NavBulkCreate(BaseModel):
    product_id: str
    points: list[NavPointInput]
    source_file_id: str | None = None
    source_fragment_id: str | None = None
    frequency: str | None = None


class NavReviewRequest(BaseModel):
    observation_ids: list[str]
    status: str = "reviewed"
    reviewed_by: str = "user"


class MachineNavReviewRequest(BaseModel):
    """Optional product subset for a batch machine review run."""

    product_ids: list[str] | None = None


class NavReplaceRequest(BaseModel):
    points: list[NavPointInput] = Field(min_length=2)
    frequency: str | None = None
    source_file_id: str | None = None
    source_fragment_id: str | None = None
    reviewed_by: str = "user"

    @classmethod
    def _ordered_dates(cls, points: list[NavPointInput]) -> bool:
        dates = [point.observation_date for point in points]
        return dates == sorted(dates) and len(dates) == len(set(dates))

    @model_validator(mode="after")
    def validate_dates(self) -> "NavReplaceRequest":
        if not self._ordered_dates(self.points):
            raise ValueError("净值日期必须严格递增且不重复")
        return self


class NavCandidatePublishRequest(BaseModel):
    """Explicit publication decision for an extracted NAV candidate."""

    mode: str = Field(pattern="^(replace|merge_missing|discard)$")
    reviewed_by: str = "user"


class FactCreate(BaseModel):
    product_id: str
    field_name: str
    field_value: str
    source_file_id: str | None = None
    source_fragment_id: str | None = None
    confidence: float | None = None


class AgentRunCreate(BaseModel):
    user_query: str
    session_id: str | None = None
    plan: dict[str, Any] | None = None


class AgentRunComplete(BaseModel):
    phase: str = "complete"
    tools_used: list[str] | None = None
    citations: list[dict[str, Any]] | None = None
    reflection: dict[str, Any] | None = None
    answer: str | None = None
    data_snapshot_id: str | None = None
    duration_ms: float | None = None


class ToolInvocationCreate(BaseModel):
    run_id: str
    tool_name: str
    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    status: str = "ok"
    error_message: str | None = None
    duration_ms: float | None = None


class SnapshotCreate(BaseModel):
    label: str | None = None
    content: dict[str, Any]


class DecisionCreate(BaseModel):
    decision_type: str = "recommendation"
    title: str | None = None
    content: dict[str, Any]
    run_id: str | None = None
    data_snapshot_id: str | None = None
    status: str = "draft"


class DecisionReview(BaseModel):
    status: str
    reviewer: str | None = None
    veto_reason: str | None = None
    adjustment: dict[str, Any] | None = None


def _product_readiness(product: Any) -> tuple[bool, str]:
    """Explain exactly why a product can or cannot enter research."""
    name = (product.standard_name or "").strip()
    nav_count = len(product.nav_observations)
    reviewed_count = sum(1 for observation in product.nav_observations if observation.review_status == "reviewed")
    if product.confirmation_status == "rejected":
        return False, "已标记为不是产品"
    if not name or name in {"未命名产品", "待确认产品"}:
        return False, "需要先确认产品名称"
    if nav_count < 2:
        return False, "净值点不足 2 条"
    if any(observation.source_file_id is None for observation in product.nav_observations):
        return False, "存在未关联原始文件的净值，不能用于正式研究"
    if reviewed_count < nav_count:
        return False, f"净值待复核（{reviewed_count}/{nav_count}）"
    if product.confirmation_status != "confirmed":
        return False, "产品身份尚未确认"
    quality = assess_nav_quality(product.nav_observations, product.facts)
    if quality["blocking"]:
        return False, "净值与材料披露冲突，需人工复核"
    return True, "净值与产品身份均已复核"


def _product_nav_confidence(product: Any) -> float | None:
    """Aggregate source confidence for the NAV evidence shown in the UI."""
    values = [
        float(fragment.ocr_confidence)
        for fragment in product.fragments
        if fragment.ocr_confidence is not None
        and fragment.fragment_type in {"chart", "chart_traced", "image", "table"}
    ]
    return round(sum(values) / len(values), 3) if values else None


def _machine_nav_review(product: Any) -> dict[str, Any]:
    return assess_machine_nav_review(
        product.nav_observations,
        product.facts,
        source_confidence=_product_nav_confidence(product),
        declared_frequency=product.nav_frequency,
    )


def _confidence_band(value: float | None) -> str:
    """Return the user-facing confidence band used by the workbench.

    The score is evidence confidence (OCR/VLM/CV), not a probability that a
    product will make money.  A missing score means that no extraction method
    reported a confidence value yet.
    """
    if value is None:
        return "unknown"
    if value >= 0.85:
        return "high"
    if value >= 0.60:
        return "medium"
    return "low"


def _product_payload(product: Any, *, include_detail: bool = False) -> dict[str, Any]:
    """Build the common product response without duplicating confidence logic."""
    research_ready, readiness_reason = _product_readiness(product)
    confidence = _product_nav_confidence(product)
    # A PDF's first page is often a disclaimer.  The chart-tracing fragment
    # records the actual source page, so return it with the product card and
    # let calibration open the evidence page rather than page 1.
    source_preview_pages: dict[str, int] = {}
    for fragment in product.fragments:
        if fragment.file_id is None or fragment.fragment_type not in {"chart", "chart_traced"}:
            continue
        stored_page = fragment.page_number or (fragment.content_data or {}).get("source_page")
        if isinstance(stored_page, int) and stored_page > 0:
            source_preview_pages.setdefault(fragment.file_id, stored_page)
    payload: dict[str, Any] = {
        "id": product.id,
        "standard_name": product.standard_name,
        "manager_name": product.manager_name,
        "strategy": product.strategy,
        "inception_date": product.inception_date.isoformat() if product.inception_date else None,
        "nav_frequency": product.nav_frequency,
        "status": product.status,
        "confirmation_status": product.confirmation_status,
        "merged_into_id": product.merged_into_id,
        "nav_count": len(product.nav_observations),
        "fact_count": len(product.facts),
        "nav_start": min((o.observation_date for o in product.nav_observations), default=None).isoformat() if product.nav_observations else None,
        "nav_end": max((o.observation_date for o in product.nav_observations), default=None).isoformat() if product.nav_observations else None,
        "reviewed_nav_count": sum(1 for o in product.nav_observations if o.review_status == "reviewed"),
        "source_files": list(dict.fromkeys(
            [o.source_file.filename for o in product.nav_observations if o.source_file is not None]
            + [fragment.file.filename for fragment in product.fragments if fragment.file is not None]
        ))[:3],
        "created_at": product.created_at.isoformat() if product.created_at else None,
        "fragment_count": len(product.fragments),
        "nav_methods": list(dict.fromkeys(
            str((fragment.content_data or {}).get("method"))
            for fragment in product.fragments
            if (fragment.content_data or {}).get("method")
        )),
        "source_file_ids": list(dict.fromkeys(
            [o.source_file_id for o in product.nav_observations if o.source_file_id]
            + [fragment.file_id for fragment in product.fragments if fragment.file_id]
        )),
        "source_preview_pages": source_preview_pages,
        "research_ready": research_ready,
        "readiness_reason": readiness_reason,
        "nav_confidence": confidence,
        "nav_confidence_band": _confidence_band(confidence),
        "nav_confidence_requires_review": confidence is not None and confidence < 0.85,
        "nav_quality": assess_nav_quality(product.nav_observations, product.facts),
        "machine_nav_review": _machine_nav_review(product),
    }
    if include_detail:
        payload.update({
            "close_date": product.close_date.isoformat() if product.close_date else None,
            "confirmed_by": product.confirmed_by,
            "confirmed_at": product.confirmed_at.isoformat() if product.confirmed_at else None,
            "notes": product.notes,
            "aliases": [{"id": a.id, "alias": a.alias, "alias_type": a.alias_type} for a in product.aliases],
        })
    return payload


def _is_pdf_record(record: Any) -> bool:
    return (record.mime_type or "").lower() == "application/pdf" or Path(record.filename or "").suffix.lower() == ".pdf"


def _review_page_for_product(
    *,
    product: Any,
    record: Any,
    session: Session,
) -> ReviewPageLocation:
    """Locate a reviewable PDF page without ever treating the cover as proof.

    A pre-existing chart fragment is the strongest evidence.  Otherwise the
    PDF text layer is ranked locally before the browser renders a page and
    sends it to the single-image VLM.
    """
    fragments = session.execute(
        select(DocumentFragment).where(
            DocumentFragment.file_id == record.id,
            DocumentFragment.product_id == product.id,
            DocumentFragment.fragment_type.in_(("chart", "chart_traced")),
        )
    ).scalars().all()
    def fragment_quality(fragment: DocumentFragment) -> tuple[int, int, int, float, float]:
        data = fragment.content_data or {}
        created_at = fragment.created_at.timestamp() if fragment.created_at else 0.0
        return (
            int(fragment.fragment_type == "chart_traced"),
            int(data.get("review_status") == "reviewed"),
            int(data.get("binding_status") == "matched"),
            float(fragment.ocr_confidence or 0.0),
            created_at,
        )

    fragments.sort(key=fragment_quality, reverse=True)
    known_pages = list(dict.fromkeys(
        fragment.page_number or (fragment.content_data or {}).get("source_page")
        for fragment in fragments
        if isinstance(fragment.page_number or (fragment.content_data or {}).get("source_page"), int)
    ))
    if known_pages:
        return locate_review_page(
            page_texts=[],
            product_names=[product.standard_name, *(alias.alias for alias in product.aliases)],
            filename=record.filename,
            known_page_numbers=known_pages,
        )

    if not record.storage_path:
        return ReviewPageLocation(None, 0.0, "原始 PDF 不存在，无法定位业绩页", "not_found")
    try:
        path = resolve_upload_path(record.storage_path)
    except ValueError:
        return ReviewPageLocation(None, 0.0, "原始 PDF 存储路径无效", "not_found")
    assert path is not None
    if not path.is_file():
        return ReviewPageLocation(None, 0.0, "原始 PDF 不存在，无法定位业绩页", "not_found")
    try:
        from pypdf import PdfReader

        page_texts = [page.extract_text() or "" for page in PdfReader(BytesIO(path.read_bytes())).pages]
    except Exception as exc:
        return ReviewPageLocation(None, 0.0, f"无法读取 PDF 文本层：{str(exc)[:120]}", "not_found")
    return locate_review_page(
        page_texts=page_texts,
        product_names=[product.standard_name, *(alias.alias for alias in product.aliases)],
        filename=record.filename,
    )


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
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    files = store.list_files(session, parsing_status=parsing_status, limit=limit, offset=offset)
    payload = []
    for f in files:
        unresolved_fragment_ids = {
            fragment.id
            for fragment in f.fragments
            if fragment.product_id is None
            or (fragment.content_data or {}).get("binding_status") == "unmatched"
        }
        linked_fragments = [
            fragment
            for fragment in f.fragments
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
            "fragment_count": len(f.fragments),
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
                for fragment in f.fragments
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
                    "color_hex": (fragment.content_data or {}).get("color_hex"),
                    "bbox": fragment.bbox,
                }
                for fragment in f.fragments
                if fragment.fragment_type in {"chart", "chart_traced"}
                and (fragment.content_data or {}).get("curve_index") is not None
            ],
        })
    return payload


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


# ---------------------------------------------------------------------------
# Product endpoints
# ---------------------------------------------------------------------------


@router.post("/products")
def create_product(body: ProductCreate, session: Session = Depends(get_session)):
    product = store.create_product(session, **body.model_dump())
    return {"id": product.id, "standard_name": product.standard_name, "confirmation_status": product.confirmation_status}


@router.get("/products")
def list_products(
    confirmation_status: str | None = None,
    strategy: str | None = None,
    manager_name: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    products = store.list_products(
        session,
        confirmation_status=confirmation_status,
        strategy=strategy,
        manager_name=manager_name,
        search=search,
        limit=limit,
        offset=offset,
    )
    return [_product_payload(product) for product in products]


@router.get("/products/{product_id}")
def get_product(product_id: str, session: Session = Depends(get_session)):
    product = store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    return _product_payload(product, include_detail=True)


@router.post("/products/{product_id}/confirm")
def confirm_product(product_id: str, body: ProductConfirm, session: Session = Depends(get_session)):
    product = store.confirm_product(session, product_id, body.confirmed_by)
    if product is None:
        raise HTTPException(404, "Product not found")
    return {"id": product.id, "confirmation_status": product.confirmation_status}


@router.post("/products/{product_id}/reject")
def reject_product(product_id: str, session: Session = Depends(get_session)):
    product = store.reject_product(session, product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    return {"id": product.id, "confirmation_status": product.confirmation_status}


@router.patch("/products/{product_id}")
def update_product(product_id: str, body: ProductUpdate, session: Session = Depends(get_session)):
    product = store.update_product(session, product_id, body.model_dump(exclude_unset=True))
    if product is None:
        raise HTTPException(404, "Product not found")
    return {"id": product.id, "standard_name": product.standard_name, "manager_name": product.manager_name, "strategy": product.strategy}


@router.delete("/products/{product_id}")
def delete_product(product_id: str, session: Session = Depends(get_session)):
    deleted = store.delete_product(session, product_id)
    if not deleted:
        raise HTTPException(404, "Product not found")
    return {"id": product_id, "deleted": True}


@router.post("/products/{product_id}/aliases")
def add_alias(product_id: str, body: AliasCreate, session: Session = Depends(get_session)):
    alias = store.add_alias(session, product_id, body.alias, body.alias_type, body.source_file_id)
    return {"id": alias.id, "alias": alias.alias}


@router.post("/products/merge")
def merge_products(body: MergeRequest, session: Session = Depends(get_session)):
    target = store.merge_products(session, body.source_id, body.target_id)
    if target is None:
        raise HTTPException(404, "Source or target product not found")
    return {"id": target.id, "standard_name": target.standard_name, "merged": True}


# ---------------------------------------------------------------------------
# NAV endpoints
# ---------------------------------------------------------------------------


@router.post("/nav")
def add_nav(body: NavBulkCreate, session: Session = Depends(get_session)):
    points = [p.model_dump() for p in body.points]
    inserted = store.add_nav_observations(
        session,
        body.product_id,
        points,
        source_file_id=body.source_file_id,
        source_fragment_id=body.source_fragment_id,
        frequency=body.frequency,
    )
    return {"product_id": body.product_id, "inserted": inserted, "total_submitted": len(body.points)}


@router.get("/nav/{product_id}")
def get_nav_series(product_id: str, reviewed_only: bool = False, session: Session = Depends(get_session)):
    observations = store.get_nav_series(session, product_id, reviewed_only=reviewed_only)
    return [
        {
            "id": o.id, "observation_date": o.observation_date.isoformat(),
            "nav": o.nav, "acc_nav": o.acc_nav, "frequency": o.frequency,
            "source_file_id": o.source_file_id, "review_status": o.review_status,
        }
        for o in observations
    ]


@router.get("/products/{product_id}/nav")
def get_product_nav_series(product_id: str, reviewed_only: bool = False, session: Session = Depends(get_session)):
    """Resource-oriented alias for clients that treat NAV as product data.

    Keep ``/nav/{product_id}`` for existing clients, while exposing the more
    discoverable nested route used by the product detail workflow.
    """
    return get_nav_series(product_id, reviewed_only=reviewed_only, session=session)


@router.get("/products/{product_id}/nav-candidates")
def list_nav_candidates(product_id: str, session: Session = Depends(get_session)):
    """List unapproved image/PDF extractions and their overlap with official NAV."""
    product = store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    official = store.get_nav_series(session, product_id)
    official_by_date = {item.observation_date.isoformat(): item.nav for item in official}
    result = []
    for candidate in store.list_nav_candidate_versions(session, product_id):
        points = candidate.points or []
        overlap = [point for point in points if str(point["observation_date"]) in official_by_date]
        differences = [
            abs(float(point["nav"]) / official_by_date[str(point["observation_date"])] - 1)
            for point in overlap if official_by_date[str(point["observation_date"])] > 0
        ]
        result.append({
            "id": candidate.id, "source_file_id": candidate.source_file_id,
            "source_fragment_id": candidate.source_fragment_id, "frequency": candidate.frequency,
            "confidence": candidate.confidence, "status": candidate.status,
            "point_count": len(points), "start_date": str(points[0]["observation_date"]) if points else None,
            "end_date": str(points[-1]["observation_date"]) if points else None,
            "overlap_count": len(overlap), "missing_date_count": len(points) - len(overlap),
            "max_overlap_difference": round(max(differences), 6) if differences else None,
            "points": points,
        })
    return result


@router.post("/nav-candidates/{candidate_id}/publish")
def publish_nav_candidate(candidate_id: str, body: NavCandidatePublishRequest, session: Session = Depends(get_session)):
    """Publish a reviewed extraction, or discard it without touching official NAV."""
    candidate = session.get(NavCandidateVersion, candidate_id)
    if candidate is None:
        raise HTTPException(404, "候选净值版本不存在")
    if candidate.status != "pending":
        raise HTTPException(409, "该候选版本已处理，不能重复发布")
    if body.mode == "discard":
        candidate.status = "discarded"
        session.commit()
        return {"candidate_id": candidate_id, "mode": body.mode, "published": 0}
    points = candidate.points or []
    if len(points) < 2:
        raise HTTPException(422, "候选版本净值点不足")
    if body.mode == "replace":
        published = store.replace_nav_observations(
            session, candidate.product_id, points, source_file_id=candidate.source_file_id,
            frequency=candidate.frequency, reviewed_by=body.reviewed_by,
        )
    else:
        existing_dates = {item.observation_date.isoformat() for item in store.get_nav_series(session, candidate.product_id)}
        missing = [point for point in points if str(point["observation_date"]) not in existing_dates]
        published = store.add_nav_observations(
            session, candidate.product_id, missing, source_file_id=candidate.source_file_id,
            source_fragment_id=candidate.source_fragment_id, frequency=candidate.frequency,
        )
        observations = session.execute(select(NavObservation).where(
            NavObservation.product_id == candidate.product_id,
            NavObservation.source_file_id == candidate.source_file_id,
            NavObservation.review_status == "pending",
        )).scalars().all()
        store.review_nav_observations(session, [item.id for item in observations], "reviewed", body.reviewed_by)
    candidate.status = "published"
    from datetime import datetime
    candidate.published_at = datetime.now()
    finalization = store.finalize_manual_nav_review(
        session,
        file_id=candidate.source_file_id,
        product_id=candidate.product_id,
        fragment_id=candidate.source_fragment_id,
        reviewed_by=body.reviewed_by,
    )
    session.commit()
    return {"candidate_id": candidate_id, "mode": body.mode, "published": published, "product_id": candidate.product_id, "review_finalization": finalization}


@router.put("/products/{product_id}/nav")
def replace_nav_series(product_id: str, body: NavReplaceRequest, session: Session = Depends(get_session)):
    """Save a manually calibrated NAV curve and mark it reviewed."""
    product = store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    if body.source_file_id and session.get(store.RawFile, body.source_file_id) is None:
        raise HTTPException(404, "来源文件不存在")
    if body.source_fragment_id:
        fragment = session.get(DocumentFragment, body.source_fragment_id)
        if fragment is None or fragment.file_id != body.source_file_id:
            raise HTTPException(422, "来源片段不属于该来源文件")
        if fragment.product_id not in {None, product_id}:
            raise HTTPException(422, "来源片段不属于当前产品")
    count = store.replace_nav_observations(
        session,
        product_id,
        [point.model_dump() for point in body.points],
        source_file_id=body.source_file_id,
        frequency=body.frequency,
        reviewed_by=body.reviewed_by,
    )
    if body.frequency:
        product.nav_frequency = body.frequency
        product.close_date = max((point.observation_date for point in body.points), default=product.close_date)
        session.commit()
    review_finalization = None
    if body.source_file_id:
        try:
            review_finalization = store.finalize_manual_nav_review(
                session,
                file_id=body.source_file_id,
                product_id=product_id,
                fragment_id=body.source_fragment_id,
                reviewed_by=body.reviewed_by,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return {
        "product_id": product_id,
        "saved": count,
        "review_status": "reviewed",
        "review_finalization": review_finalization,
    }


@router.patch("/nav/review")
def review_nav(body: NavReviewRequest, session: Session = Depends(get_session)):
    updated = store.review_nav_observations(session, body.observation_ids, body.status, body.reviewed_by)
    return {"updated": updated}


@router.post("/machine-review/products")
def machine_review_products(body: MachineNavReviewRequest, session: Session = Depends(get_session)):
    """Mark trustworthy CV/VLM curves as machine-reviewed only.

    This is a queue-prioritisation aid, not human confirmation. Formal
    screening and allocation continue to require ``reviewed`` NAV points.
    """
    candidates = (
        [store.get_product(session, product_id) for product_id in body.product_ids]
        if body.product_ids is not None
        else store.list_products(session, limit=100)
    )
    results: list[dict[str, Any]] = []
    for product in candidates:
        if product is None or not product.nav_observations:
            continue
        review = _machine_nav_review(product)
        updated = 0
        if review["machine_reviewed"]:
            pending_ids = [observation.id for observation in product.nav_observations if observation.review_status == "pending"]
            if pending_ids:
                updated = store.review_nav_observations(
                    session, pending_ids, "machine_reviewed", reviewed_by="agent:nav-quality-v1"
                )
        results.append({
            "product_id": product.id,
            "product_name": product.standard_name,
            "updated": updated,
            **review,
        })
    return {
        "machine_reviewed": sum(1 for item in results if item["machine_reviewed"]),
        "human_review_required": sum(1 for item in results if not item["machine_reviewed"]),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Structured Fact endpoints
# ---------------------------------------------------------------------------


@router.post("/facts")
def create_fact(body: FactCreate, session: Session = Depends(get_session)):
    fact = store.add_fact(session, **body.model_dump())
    return {"id": fact.id, "field_name": fact.field_name, "confidence": fact.confidence}


@router.get("/facts/{product_id}")
def list_facts(product_id: str, current_only: bool = True, session: Session = Depends(get_session)):
    facts = store.get_facts_for_product(session, product_id, current_only=current_only)
    return [
        {
            "id": f.id, "field_name": f.field_name, "field_value": f.field_value,
            "confidence": f.confidence, "extraction_version": f.extraction_version,
            "confirmation_status": f.confirmation_status,
            "source_file_id": f.source_file_id, "source_fragment_id": f.source_fragment_id,
        }
        for f in facts
    ]


# ---------------------------------------------------------------------------
# Agent Run audit endpoints
# ---------------------------------------------------------------------------


@router.post("/runs")
def create_run(body: AgentRunCreate, session: Session = Depends(get_session)):
    run = store.create_agent_run(session, user_query=body.user_query, session_id=body.session_id, plan=body.plan)
    return {"id": run.id, "phase": run.phase}


@router.patch("/runs/{run_id}/complete")
def complete_run(run_id: str, body: AgentRunComplete, session: Session = Depends(get_session)):
    run = store.complete_agent_run(session, run_id, **body.model_dump())
    if run is None:
        raise HTTPException(404, "Agent run not found")
    return {"id": run.id, "phase": run.phase, "duration_ms": run.duration_ms}


@router.post("/runs/{run_id}/tools")
def add_tool_invocation(run_id: str, body: ToolInvocationCreate, session: Session = Depends(get_session)):
    invocation = store.add_tool_invocation(
        session, run_id,
        tool_name=body.tool_name,
        input_summary=body.input_summary,
        output_summary=body.output_summary,
        status=body.status,
        error_message=body.error_message,
        duration_ms=body.duration_ms,
    )
    return {"id": invocation.id, "tool_name": invocation.tool_name, "status": invocation.status}


@router.get("/runs")
def list_runs(session_id: str | None = None, limit: int = 20, offset: int = 0, session: Session = Depends(get_session)):
    runs = store.list_agent_runs(session, session_id=session_id, limit=limit, offset=offset)
    return [
        {
            "id": r.id, "session_id": r.session_id, "user_query": r.user_query[:100],
            "phase": r.phase, "tools_used": r.tools_used,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]


# ---------------------------------------------------------------------------
# Snapshot & Decision endpoints
# ---------------------------------------------------------------------------


@router.post("/snapshots")
def create_snapshot(body: SnapshotCreate, session: Session = Depends(get_session)):
    snapshot = store.create_snapshot(session, label=body.label, content=body.content)
    return {"id": snapshot.id, "label": snapshot.label}


@router.post("/decisions")
def create_decision(body: DecisionCreate, session: Session = Depends(get_session)):
    decision = store.create_decision(session, **body.model_dump())
    return {"id": decision.id, "decision_type": decision.decision_type, "status": decision.status}


@router.patch("/decisions/{decision_id}/review")
def review_decision(decision_id: str, body: DecisionReview, session: Session = Depends(get_session)):
    decision = store.review_decision(
        session, decision_id,
        status=body.status, reviewer=body.reviewer,
        veto_reason=body.veto_reason, adjustment=body.adjustment,
    )
    if decision is None:
        raise HTTPException(404, "Decision not found")
    return {"id": decision.id, "status": decision.status, "reviewer": decision.reviewer}


@router.get("/decisions")
def list_decisions(status: str | None = None, limit: int = 20, offset: int = 0, session: Session = Depends(get_session)):
    decisions = store.list_decisions(session, status=status, limit=limit, offset=offset)
    return [
        {
            "id": d.id, "decision_type": d.decision_type, "title": d.title,
            "status": d.status, "reviewer": d.reviewer,
            "veto_reason": d.veto_reason,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in decisions
    ]
