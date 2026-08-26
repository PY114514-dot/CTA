"""Refresh the current read-only CODEX ranking from research-ready weekly NAVs."""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ConfirmationStatus, DataSnapshot, NavObservation, ProductEntity, ReviewStatus
from app.schemas import (
    CtaRankingAttributionEvidence,
    CtaRankingProductInput,
    CtaRankingRequest,
    NetAssetValuePoint,
)
from app.services.codex_cta_ranking import rank_cta_products
from app.services.cta_attribution_evidence import load_latest_phase_d_evidence_payload
from app.services.product_store import create_snapshot


def refresh_weekly_ranking(session: Session) -> dict[str, int]:
    products = session.execute(select(ProductEntity).where(
        ProductEntity.confirmation_status == ConfirmationStatus.CONFIRMED,
        ProductEntity.nav_frequency == "weekly",
    )).scalars().all()
    observations = session.execute(select(
        NavObservation.product_id, NavObservation.observation_date, NavObservation.nav,
    ).where(
        NavObservation.review_status == ReviewStatus.REVIEWED,
        NavObservation.product_id.in_([product.id for product in products]),
    ).order_by(NavObservation.product_id, NavObservation.observation_date)).all() if products else []
    points: dict[str, list[NetAssetValuePoint]] = defaultdict(list)
    for product_id, observation_date, nav in observations:
        if nav > 0:
            points[product_id].append(NetAssetValuePoint(observation_date=observation_date, net_asset_value=nav))
    inputs = [CtaRankingProductInput(product_id=product.id, product_name=product.standard_name, nav_points=points[product.id], frequency="weekly", strategy=product.strategy) for product in products if len(points[product.id]) >= 2]
    if not inputs:
        return {"universe_size": 0, "eligible_count": 0}
    # 载入已冻结的 Phase-D 证据（只读，绝不在此重新计算），与排名截点对齐：
    # 证据自身的净值截点晚于宇宙最新净值日的快照会被排名内部按未来信息剔除。
    universe_cutoff = max(
        point.observation_date for product in inputs for point in product.nav_points
    )
    evidence_payload = load_latest_phase_d_evidence_payload(
        session,
        product_ids={product.id for product in products},
        as_of_date=universe_cutoff,
    )
    evidence = {
        product_id: CtaRankingAttributionEvidence.model_validate(payload)
        for product_id, payload in evidence_payload.items()
    }
    ranking = rank_cta_products(CtaRankingRequest(products=inputs, attribution_evidence=evidence))
    label = f"codex-cta:{ranking.model_version}:{ranking.as_of_date.isoformat()}:{ranking.nav_fingerprint}:{ranking.attribution_evidence_fingerprint}"
    existing = session.execute(select(DataSnapshot).where(DataSnapshot.label == label).limit(1)).scalars().first()
    if existing is None:
        create_snapshot(session, label=label, content=ranking.model_dump(mode="json"))
    return {
        "universe_size": ranking.universe_size,
        "eligible_count": ranking.eligible_count,
        "phase_d_evidence_products": len(evidence),
    }
