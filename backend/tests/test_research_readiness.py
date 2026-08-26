"""Regression tests for formal-research data quality gates."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import ConfirmationStatus, ReviewStatus
from app.routers.knowledge_base import _product_readiness, _research_workflow
from app.services.fof_allocation_intent import is_screening_candidate
from app.services import product_store as store


def _session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_machine_reviewed_nav_is_not_a_formal_screening_candidate() -> None:
    session = _session()
    source = store.register_file(session, filename="curve.png", file_hash="machine-status")
    product = store.create_product(session, standard_name="可疑曲线产品")
    product.confirmation_status = ConfirmationStatus.CONFIRMED
    store.add_nav_observations(
        session,
        product.id,
        [
            {"observation_date": "2026-01-02", "nav": 1.0},
            {"observation_date": "2026-01-09", "nav": 1.01},
        ],
        source_file_id=source.id,
        frequency="weekly",
    )
    observations = store.get_nav_series(session, product.id)
    store.review_nav_observations(
        session,
        [observation.id for observation in observations],
        ReviewStatus.MACHINE_REVIEWED,
        "agent:nav-quality-v1",
    )
    product = store.get_product(session, product.id)
    assert product is not None

    ready, reason = _product_readiness(product)
    assert ready is False
    assert "待复核" in reason
    assert is_screening_candidate(product) is False


def test_reviewed_nav_without_source_is_not_research_ready() -> None:
    session = _session()
    product = store.create_product(session, standard_name="无来源净值产品")
    product.confirmation_status = ConfirmationStatus.CONFIRMED
    store.add_nav_observations(
        session,
        product.id,
        [
            {"observation_date": "2026-01-02", "nav": 1.0},
            {"observation_date": "2026-01-09", "nav": 1.01},
        ],
        frequency="weekly",
    )
    observations = store.get_nav_series(session, product.id)
    store.review_nav_observations(session, [observation.id for observation in observations], ReviewStatus.REVIEWED, "analyst")
    product = store.get_product(session, product.id)
    assert product is not None

    ready, reason = _product_readiness(product)
    assert ready is False
    assert "未关联原始文件" in reason
    assert is_screening_candidate(product) is False
    workflow = _research_workflow(product)
    assert workflow["stage"] == "needs_nav_source"
    assert workflow["next_action"] == "calibrate_nav"
