"""NAV, NAV candidate, and machine review route handlers for the knowledge-base API."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import DocumentFragment, NavCandidateVersion, NavObservation
from app.services import product_store as store

from .helpers import _machine_nav_review
from .schemas import (
    MachineNavReviewRequest,
    NavBulkCreate,
    NavBulkQuery,
    NavCandidatePublishRequest,
    NavReplaceRequest,
    NavReviewRequest,
)

router = APIRouter()


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


@router.post("/nav/bulk")
def get_nav_series_bulk(body: NavBulkQuery, session: Session = Depends(get_session)):
    """单次请求返回多只产品的净值序列，供排名/评分宇宙构建器批量取数。

    返回结构与 /nav/{product_id} 完全一致，仅按 product_id 分组，避免
    前端对 700+ 只产品逐个发请求造成浏览器连接池排队。

    批次 13（体验）：响应可达 34MB/18 万点，FastAPI 默认对返回 dict 先做
    jsonable_encoder 全量遍历（约 2-4s）再序列化。数据已是 JSON 安全
    类型（StrEnum 是 str 子类），直接预序列化交给 Response 发送：注意
    必须用 Response 而非 JSONResponse——后者会对 str 再做一次 json.dumps
    造成双重编码。
    """
    grouped = store.get_nav_series_bulk(session, body.product_ids, reviewed_only=body.reviewed_only)
    payload = json.dumps(
        {
            product_id: [
                {
                    "id": o.id, "observation_date": o.observation_date.isoformat(),
                    "nav": o.nav, "acc_nav": o.acc_nav, "frequency": o.frequency,
                    "source_file_id": o.source_file_id, "review_status": o.review_status,
                }
                for o in observations
            ]
            for product_id, observations in grouped.items()
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return Response(content=payload, media_type="application/json")


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
