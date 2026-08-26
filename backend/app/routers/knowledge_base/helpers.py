"""Shared helper functions used by multiple knowledge-base route modules."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DocumentFragment
from app.services import product_store as store
from app.services.nav_quality import assess_machine_nav_review, assess_nav_quality
from app.services.review_page_locator import ReviewPageLocation, locate_review_page
from app.services.upload_storage import resolve_upload_path


def _is_direct_sqlite_nav(product: Any, is_direct: bool | None = None) -> bool:
    """Structured SQLite rows are source data, not image-traced candidates."""
    if is_direct is not None:
        return bool(is_direct)
    observations = product.nav_observations
    return bool(observations) and all(
        observation.source_file is not None
        and observation.source_file.source == "futures_weekly_sqlite"
        for observation in observations
    )


def _research_workflow(product: Any, stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the single, evidence-based workflow state for a product.

    stats 为可选的预聚合净值统计（点数/复核数/是否有缺失来源/是否全部
    直连 SQLite），由列表页批量聚合查询提供，避免逐只物化净值点。
    """
    name = (product.standard_name or "").strip()
    if stats is not None:
        nav_count = int(stats["nav_count"])
        reviewed_count = int(stats["reviewed_count"])
        has_missing_source = not bool(stats["all_have_source"])
        is_direct = bool(stats["all_direct"])
    else:
        nav_count = len(product.nav_observations)
        reviewed_count = sum(1 for observation in product.nav_observations if observation.review_status == "reviewed")
        has_missing_source = any(observation.source_file_id is None for observation in product.nav_observations)
        is_direct = None
    if product.confirmation_status == "rejected":
        return {"stage": "rejected", "label": "已标记为非产品", "next_action": None, "blocking_reasons": ["该记录不会进入研究上下文"], "completed_steps": {}}
    if not name or name in {"未命名产品", "待确认产品"}:
        return {"stage": "needs_identity", "label": "待确认产品名称", "next_action": "edit_identity", "blocking_reasons": ["需要先确认产品名称"], "completed_steps": {"identity": False}}
    if nav_count < 2:
        return {"stage": "needs_nav", "label": "待补净值", "next_action": "calibrate_nav", "blocking_reasons": ["净值点不足 2 条"], "completed_steps": {"identity": True, "nav": False}}
    if has_missing_source:
        return {"stage": "needs_nav_source", "label": "净值缺少原始来源", "next_action": "calibrate_nav", "blocking_reasons": ["存在未关联原始文件的净值，不能用于正式研究"], "completed_steps": {"identity": True, "nav": True, "source": False}}
    if reviewed_count < nav_count:
        return {"stage": "needs_review", "label": "待复核净值", "next_action": "calibrate_nav", "blocking_reasons": [f"净值待复核（{reviewed_count}/{nav_count}）"], "completed_steps": {"identity": True, "nav": True, "source": True, "review": False}}
    if product.confirmation_status != "confirmed":
        return {"stage": "needs_confirmation", "label": "待确认产品", "next_action": "confirm_identity", "blocking_reasons": ["产品身份尚未确认"], "completed_steps": {"identity": True, "nav": True, "source": True, "review": True, "confirmation": False}}
    if _is_direct_sqlite_nav(product, is_direct):
        return {"stage": "research_ready", "label": "可研究", "next_action": None, "blocking_reasons": [], "completed_steps": {"identity": True, "nav": True, "source": True, "review": True, "confirmation": True, "quality": True}}
    quality = assess_nav_quality(product.nav_observations, product.facts)
    if quality["blocking"]:
        return {"stage": "needs_quality_review", "label": "曲线与披露冲突", "next_action": "calibrate_nav", "blocking_reasons": ["净值与材料披露冲突，需人工复核"], "completed_steps": {"identity": True, "nav": True, "source": True, "review": True, "confirmation": True, "quality": False}}
    return {"stage": "research_ready", "label": "可研究", "next_action": None, "blocking_reasons": [], "completed_steps": {"identity": True, "nav": True, "source": True, "review": True, "confirmation": True, "quality": True}}


def _product_readiness(product: Any, stats: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Explain exactly why a product can or cannot enter research."""
    workflow = _research_workflow(product, stats)
    return workflow["stage"] == "research_ready", workflow["blocking_reasons"][0] if workflow["blocking_reasons"] else "净值与产品身份均已复核"


def _product_nav_confidence(product: Any) -> float | None:
    """Aggregate source confidence for the NAV evidence shown in the UI."""
    values = [
        float(fragment.ocr_confidence)
        for fragment in product.fragments
        if fragment.ocr_confidence is not None
        and fragment.fragment_type in {"chart", "chart_traced", "image", "table"}
    ]
    return round(sum(values) / len(values), 3) if values else None


def _machine_nav_review(product: Any, is_direct: bool | None = None) -> dict[str, Any]:
    if _is_direct_sqlite_nav(product, is_direct):
        return {
            "status": "direct_source", "machine_reviewed": True, "confidence": 1.0,
            "reasons": [], "scope": "可进入研究", "method": "结构化 SQLite 周频净值，跳过图像曲线校验",
            "reviewed_at": None,
        }
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


def _product_payload(
    product: Any,
    *,
    include_detail: bool = False,
    nav_stats: dict[str, Any] | None = None,
    nav_source_names: list[str] | None = None,
    nav_source_file_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build the common product response without duplicating confidence logic.

    列表页批量取数时传入 nav_stats 等预聚合值，跳过逐只物化净值点；
    详情页不传，走原有懒加载路径。
    """
    if nav_stats is not None:
        research_ready, readiness_reason = _product_readiness(product, nav_stats)
        research_workflow = _research_workflow(product, nav_stats)
        nav_count = int(nav_stats["nav_count"])
        reviewed_nav_count = int(nav_stats["reviewed_count"])
        nav_start = nav_stats["nav_start"]
        nav_end = nav_stats["nav_end"]
        is_direct = bool(nav_stats["all_direct"])
    else:
        research_ready, readiness_reason = _product_readiness(product)
        research_workflow = _research_workflow(product)
        nav_count = len(product.nav_observations)
        reviewed_nav_count = sum(1 for o in product.nav_observations if o.review_status == "reviewed")
        nav_start = min((o.observation_date for o in product.nav_observations), default=None)
        nav_end = max((o.observation_date for o in product.nav_observations), default=None)
        is_direct = None
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
    if nav_source_names is None:
        nav_source_names = [
            o.source_file.filename for o in product.nav_observations if o.source_file is not None
        ]
    if nav_source_file_ids is None:
        nav_source_file_ids = [o.source_file_id for o in product.nav_observations if o.source_file_id]
    payload: dict[str, Any] = {
        "id": product.id,
        "standard_name": product.standard_name,
        "manager_name": product.manager_name,
        "strategy": product.strategy,
        "strategy_disclosure": product.strategy_disclosure,
        "inception_date": product.inception_date.isoformat() if product.inception_date else None,
        "close_date": product.close_date.isoformat() if product.close_date else None,
        "nav_frequency": product.nav_frequency,
        "status": product.status,
        "confirmation_status": product.confirmation_status,
        "merged_into_id": product.merged_into_id,
        "nav_count": nav_count,
        "fact_count": len(product.facts),
        "nav_start": nav_start.isoformat() if nav_start else None,
        "nav_end": nav_end.isoformat() if nav_end else None,
        "reviewed_nav_count": reviewed_nav_count,
        "source_files": list(dict.fromkeys(
            nav_source_names
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
            nav_source_file_ids
            + [fragment.file_id for fragment in product.fragments if fragment.file_id]
        )),
        "source_preview_pages": source_preview_pages,
        "research_ready": research_ready,
        "readiness_reason": readiness_reason,
        "research_workflow": research_workflow,
        "nav_confidence": confidence,
        "nav_confidence_band": _confidence_band(confidence),
        "nav_confidence_requires_review": confidence is not None and confidence < 0.85,
        "nav_quality": (
            {"status": "direct_source", "blocking": False, "reasons": [], "source": "futures_weekly_sqlite"}
            if _is_direct_sqlite_nav(product, is_direct)
            else assess_nav_quality(product.nav_observations, product.facts)
        ),
        "machine_nav_review": _machine_nav_review(product, is_direct),
    }
    if include_detail:
        payload.update({
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
