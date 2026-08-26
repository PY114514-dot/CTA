"""Regression coverage for stable browser-to-backend history IDs."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.routers.history import HistoryRecordCreate, create_history, delete_history


def test_client_history_id_can_be_deleted_after_creation() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        created = create_history(
            HistoryRecordCreate(
                id="1724169600000-abc123",
                frequency="weekly",
                metrics={},
            ),
            session,
        )

        assert created.id == "1724169600000-abc123"
        assert delete_history(created.id, session) == {"status": "deleted"}
    finally:
        session.close()
