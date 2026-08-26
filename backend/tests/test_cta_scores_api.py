from datetime import date, datetime, timedelta

import pytest
from fastapi import HTTPException

from app.routers.cta_scores import _latest_score_snapshot, get_allocation_score, get_product_score_profile
from app.schemas import CtaPortfolioPosition, CtaProductScoreRequest, CtaRankingProductInput, DataFrequency, NetAssetValuePoint


def _product(product_id: str) -> CtaRankingProductInput:
    nav = 1.0
    points = [NetAssetValuePoint(observation_date=date(2024, 1, 5), net_asset_value=nav)]
    for index in range(20):
        nav *= 1.0 + (0.004 if index % 2 == 0 else -0.001)
        points.append(NetAssetValuePoint(observation_date=date(2024, 1, 5) + timedelta(days=7 * (index + 1)), net_asset_value=nav))
    return CtaRankingProductInput(product_id=product_id, product_name=product_id, nav_points=points, frequency=DataFrequency.WEEKLY)


def test_profile_endpoint_rejects_mixed_frequencies_with_422() -> None:
    daily = _product("daily")
    daily.frequency = DataFrequency.DAILY
    request = CtaProductScoreRequest(products=[_product("one"), daily])

    with pytest.raises(HTTPException, match="common product frequency") as error:
        get_product_score_profile(request)
    assert error.value.status_code == 422


def test_profile_endpoint_keeps_allocation_unavailable_without_portfolio() -> None:
    response = get_product_score_profile(CtaProductScoreRequest(products=[_product("one")]));

    assert response["products"][0]["summary"]["quality_status"] == "insufficient_data"
    assert response["products"][0]["detail"]["confidence"]["score"] is not None
    assert response["allocation"]["status"] == "not_available"


def test_allocation_endpoint_rejects_missing_explicit_portfolio() -> None:
    with pytest.raises(HTTPException, match="当前组合") as error:
        get_allocation_score(CtaProductScoreRequest(products=[_product("one")]))
    assert error.value.status_code == 422


def test_allocation_endpoint_returns_only_explicit_allocation_case() -> None:
    request = CtaProductScoreRequest(
        products=[_product("current"), _product("candidate")],
        current_portfolio=[CtaPortfolioPosition(product_id="current", weight=1.0)],
        candidate_product_id="candidate",
        candidate_weight=0.2,
    )

    response = get_allocation_score(request)

    assert response["allocation"]["status"] == "available"
    assert response["allocation"]["candidate_product_id"] == "candidate"


def test_score_snapshot_is_persisted_and_retrievable() -> None:
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base, get_session
    from app.main import app

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    app.dependency_overrides[get_session] = lambda: session_factory()
    payload = CtaProductScoreRequest(products=[_product("persisted")]).model_dump(mode="json")
    try:
        with TestClient(app) as client:
            created = client.post("/api/cta-scores/snapshots", json=payload)
            latest = client.get("/api/cta-scores/snapshots/latest")
            product = client.get("/api/cta-scores/snapshots/latest/persisted")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert created.status_code == 200
    assert latest.status_code == 200
    assert latest.json()["snapshot_id"] == created.json()["snapshot_id"]
    assert latest.json()["report"]["products"][0]["product_id"] == "persisted"
    assert "detail" not in latest.json()["report"]["products"][0]
    assert product.status_code == 200
    assert product.json()["item"]["product_id"] == "persisted"


def test_latest_snapshot_prefers_most_recent_regardless_of_universe_size() -> None:
    """回归：旧快照产品数更多时（749 vs 722）也不能压过更新固化的快照。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models import DataSnapshot

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with session_factory() as session:
        older = DataSnapshot(
            label="cta-product-score:v2:2026-05-28:legacy",
            content={"universe_size": 749, "products": [{"product_id": "old"}]},
            created_at=datetime(2026, 5, 28, 10, 0, 0),
        )
        newer = DataSnapshot(
            label="cta-product-score:v3.0:2026-07-10:current",
            content={"universe_size": 722, "products": [{"product_id": "new"}]},
            created_at=datetime(2026, 7, 10, 10, 0, 0),
        )
        session.add_all([older, newer])
        session.commit()

        selected = _latest_score_snapshot(session)
        assert selected is not None
        assert selected.content["products"][0]["product_id"] == "new"
