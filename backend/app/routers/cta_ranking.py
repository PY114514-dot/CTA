"""HTTP seam for the read-only CODEX CTA ranking engine and snapshots."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import DataSnapshot
from app.schemas import (
    CtaRankingRequest,
    CtaRankingResponse,
    CtaRankingSnapshotResponse,
    CtaRankingSnapshotSummary,
)
from app.services import product_store
from app.services.codex_cta_ranking import rank_cta_products

router = APIRouter(prefix="/api/cta-ranking", tags=["CODEX CTA 排名"])


@router.post("/rank", response_model=CtaRankingResponse)
def create_cta_ranking(request: CtaRankingRequest) -> CtaRankingResponse:
    """Calculate a ranking snapshot without writing allocations or products."""
    try:
        return rank_cta_products(request)
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/snapshots", response_model=CtaRankingSnapshotResponse)
def persist_cta_ranking_snapshot(
    request: CtaRankingRequest,
    session: Session = Depends(get_session),
) -> CtaRankingSnapshotResponse:
    """Calculate and persist one immutable, idempotent weekly snapshot."""
    try:
        ranking = rank_cta_products(request)
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    label = (
        f"codex-cta:{ranking.model_version}:"
        f"{ranking.as_of_date.isoformat()}:{ranking.nav_fingerprint}:"
        f"{ranking.attribution_evidence_fingerprint}"
    )
    existing = session.execute(
        select(DataSnapshot).where(DataSnapshot.label == label).limit(1)
    ).scalars().first()
    snapshot = existing or product_store.create_snapshot(
        session,
        label=label,
        content=ranking.model_dump(mode="json"),
    )
    return CtaRankingSnapshotResponse(
        snapshot_id=snapshot.id,
        created_at=snapshot.created_at.isoformat() if snapshot.created_at else None,
        ranking=CtaRankingResponse.model_validate(snapshot.content),
    )


@router.get("/snapshots", response_model=list[CtaRankingSnapshotSummary])
def list_cta_ranking_snapshots(
    limit: int = 12,
    session: Session = Depends(get_session),
) -> list[CtaRankingSnapshotSummary]:
    """List immutable ranking snapshots, newest first.

    只读取摘要所需的标量字段，不做 CtaRankingResponse 全量校验——
    每个快照含 700+ 只产品的完整排名，全量解析会让列表接口随快照数线性变慢。
    """
    limit = max(1, min(limit, 52))
    snapshots = session.execute(
        select(DataSnapshot)
        .where(DataSnapshot.label.like("codex-cta:%"))
        .order_by(DataSnapshot.created_at.desc())
        .limit(52)
    ).scalars().all()
    result: list[CtaRankingSnapshotSummary] = []
    for snapshot in snapshots:
        content = snapshot.content or {}
        try:
            as_of_date = date.fromisoformat(str(content.get("as_of_date", "")))
        except ValueError:
            continue
        model_version = str(content.get("model_version", ""))
        if not model_version:
            continue
        result.append(CtaRankingSnapshotSummary(
            snapshot_id=snapshot.id,
            label=snapshot.label,
            model_version=model_version,
            as_of_date=as_of_date,
            created_at=snapshot.created_at.isoformat() if snapshot.created_at else None,
            nav_fingerprint=str(content.get("nav_fingerprint", "")),
            attribution_evidence_fingerprint=str(content.get("attribution_evidence_fingerprint", "")),
            universe_size=int(content.get("universe_size", 0)),
            eligible_count=int(content.get("eligible_count", 0)),
        ))
        if len(result) == limit:
            break
    return result


@router.get("/snapshots/{snapshot_id}", response_model=CtaRankingSnapshotResponse)
def get_cta_ranking_snapshot(
    snapshot_id: str,
    session: Session = Depends(get_session),
) -> CtaRankingSnapshotResponse:
    """Return one immutable ranking snapshot for historical inspection."""
    snapshot = session.get(DataSnapshot, snapshot_id)
    if snapshot is None or not (snapshot.label or "").startswith("codex-cta:"):
        raise HTTPException(status_code=404, detail="CODEX 排名快照不存在")
    try:
        ranking = CtaRankingResponse.model_validate(snapshot.content)
    except ValueError as error:
        raise HTTPException(status_code=500, detail="CODEX 排名快照格式无效") from error
    return CtaRankingSnapshotResponse(
        snapshot_id=snapshot.id,
        created_at=snapshot.created_at.isoformat() if snapshot.created_at else None,
        ranking=ranking,
    )


@router.delete("/snapshots/{snapshot_id}", status_code=204)
def delete_cta_ranking_snapshot(
    snapshot_id: str,
    session: Session = Depends(get_session),
) -> None:
    """Delete one persisted ranking snapshot chosen from history."""
    snapshot = session.get(DataSnapshot, snapshot_id)
    if snapshot is None or not (snapshot.label or "").startswith("codex-cta:"):
        raise HTTPException(status_code=404, detail="CODEX 排名快照不存在")
    session.delete(snapshot)
    session.commit()
