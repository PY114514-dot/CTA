from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.routers.cta_fama_audit import ProductAuditUpdate, ResearchAuditUpdate, get_audit, update_global_audit, update_product_audit
from app.services import product_store


def test_fama_audit_persists_global_and_product_facts() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        product = product_store.create_product(session, standard_name="CTA")
        global_audit = update_global_audit(ResearchAuditUpdate(
            exit_history_complete=True,
            survivorship_audit_passed=True,
            backfill_audit_passed=False,
            same_source_deduplicated=True,
            oos_state_segments=3,
        ), session)
        product_audit = update_product_audit(product.id, ProductAuditUpdate(
            source_group="manager-a",
            point_in_time_verified=True,
        ), session)
        loaded = get_audit(session)

    assert global_audit["oos_state_segments"] == 3
    assert product_audit["source_group"] == "manager-a"
    assert loaded["global"]["exit_history_complete"] is True
    assert loaded["products"][0]["point_in_time_verified"] is True
