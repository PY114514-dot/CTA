"""HTTP seam for the read-only CODEX CTA ranking engine and snapshots."""

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
    """List immutable ranking snapshots, newest first."""
    limit = max(1, min(limit, 52))
    snapshots = session.execute(
        select(DataSnapshot)
        .where(DataSnapshot.label.like("codex-cta:%"))
        .order_by(DataSnapshot.created_at.desc())
        .limit(limit)
    ).scalars().all()
    result: list[CtaRankingSnapshotSummary] = []
    for snapshot in snapshots:
        content = snapshot.content or {}
        result.append(CtaRankingSnapshotSummary(
            snapshot_id=snapshot.id,
            label=snapshot.label,
            model_version=str(content.get("model_version", "")),
            as_of_date=content.get("as_of_date"),
            created_at=snapshot.created_at.isoformat() if snapshot.created_at else None,
            nav_fingerprint=str(content.get("nav_fingerprint", "")),
            attribution_evidence_fingerprint=str(content.get("attribution_evidence_fingerprint", "")),
            universe_size=int(content.get("universe_size", 0)),
            eligible_count=int(content.get("eligible_count", 0)),
        ))
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
