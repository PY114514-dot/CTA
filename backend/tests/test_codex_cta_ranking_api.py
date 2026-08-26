"""HTTP contract tests for the CODEX ranking route."""

from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_session
from app.main import app
from app.models import DataSnapshot


def _nav_points(rate: float) -> list[dict[str, object]]:
    nav = 1.0
    points: list[dict[str, object]] = [
        {"observation_date": "2024-01-05", "net_asset_value": nav}
    ]
    for index in range(1, 53):
        nav *= 1.0 + rate
        points.append({
            "observation_date": (date(2024, 1, 5) + timedelta(days=index * 7)).isoformat(),
            "net_asset_value": nav,
        })
    return points


def test_cta_ranking_route_returns_read_only_snapshot() -> None:
    payload = {
        "products": [
            {
                "product_id": "cta-a",
                "product_name": "CTA A",
                "nav_points": _nav_points(0.004),
                "frequency": "weekly",
            },
            {
                "product_id": "cta-b",
                "product_name": "CTA B",
                "nav_points": _nav_points(0.001),
                "frequency": "weekly",
            },
        ],
        "as_of_date": "2024-12-31",
    }

    with TestClient(app) as client:
        response = client.post("/api/cta-ranking/rank", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ranking_only"] is True
    assert body["eligible_count"] == 2
    assert body["rankings"][0]["rank"] == 1
    assert body["method_provenance"]["ranking"]["allocation_written"] is False
    assert body["method_provenance"]["ranking"]["fof_marginal_diversification_calculated"] is False


def test_cta_ranking_snapshot_is_idempotent_and_retrievable() -> None:
    payload = {
        "products": [
            {
                "product_id": "snapshot-a",
                "product_name": "Snapshot A",
                "nav_points": _nav_points(0.003),
                "frequency": "weekly",
            },
            {
                "product_id": "snapshot-b",
                "product_name": "Snapshot B",
                "nav_points": _nav_points(0.001),
                "frequency": "weekly",
            },
        ],
        "as_of_date": "2024-12-31",
    }

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    app.dependency_overrides[get_session] = lambda: session_factory()
    try:
        with TestClient(app) as client:
            first = client.post("/api/cta-ranking/snapshots", json=payload)
            second = client.post("/api/cta-ranking/snapshots", json=payload)
            listed = client.get("/api/cta-ranking/snapshots?limit=52")
            detail = client.get(
                f"/api/cta-ranking/snapshots/{first.json()['snapshot_id']}"
            )
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["snapshot_id"] == second.json()["snapshot_id"]
    assert listed.status_code == 200
    assert any(
        item["snapshot_id"] == first.json()["snapshot_id"]
        for item in listed.json()
    )
    assert detail.status_code == 200
    assert detail.json()["ranking"]["ranking_only"] is True


def test_cta_ranking_snapshot_can_be_deleted_from_history() -> None:
    payload = {
        "products": [
            {"product_id": "delete-a", "product_name": "Delete A", "nav_points": _nav_points(0.003), "frequency": "weekly"},
        ],
        "as_of_date": "2024-12-31",
    }
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    app.dependency_overrides[get_session] = lambda: session_factory()
    try:
        with TestClient(app) as client:
            created = client.post("/api/cta-ranking/snapshots", json=payload)
            snapshot_id = created.json()["snapshot_id"]
            deleted = client.delete(f"/api/cta-ranking/snapshots/{snapshot_id}")
            missing = client.get(f"/api/cta-ranking/snapshots/{snapshot_id}")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert created.status_code == 200
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_cta_ranking_history_hides_snapshots_with_obsolete_score_contract() -> None:
    payload = {
        "products": [
            {"product_id": "valid", "product_name": "Valid", "nav_points": _nav_points(0.003), "frequency": "weekly"},
        ],
        "as_of_date": "2024-12-31",
    }
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    app.dependency_overrides[get_session] = lambda: session_factory()
    try:
        with session_factory() as session:
            obsolete = DataSnapshot(label="codex-cta:obsolete", content={"model_version": "obsolete"})
            session.add(obsolete)
            session.commit()
            obsolete_id = obsolete.id
        with TestClient(app) as client:
            valid = client.post("/api/cta-ranking/snapshots", json=payload)
            listed = client.get("/api/cta-ranking/snapshots")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert valid.status_code == 200
    assert listed.status_code == 200
    snapshot_ids = {item["snapshot_id"] for item in listed.json()}
    assert valid.json()["snapshot_id"] in snapshot_ids
    assert obsolete_id not in snapshot_ids
