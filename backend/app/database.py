"""Database engine, session factory and declarative base.

Uses SQLite for zero-config local persistence. The database file lives under
``data/fof_agent/fof.db`` (Git-ignored data directory). All models inherit
from ``Base`` declared here.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Resolve data directory relative to the backend root (backend/../data/fof_agent).
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND_ROOT.parent / "data" / "fof_agent"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR / 'fof.db'}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
    pool_timeout=60,
)


# Enable WAL mode and foreign keys for SQLite.
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record):  # noqa: ANN001
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    # Batch material ingestion can legitimately hold a write transaction while
    # an external VLM request is in flight.  Waiting is safer than surfacing a
    # transient "database is locked"/pool timeout as a permanent parse failure.
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base for all FOF data-layer models."""


def get_session() -> Session:
    """FastAPI dependency that yields a scoped session."""
    session = SessionFactory()
    try:
        yield session  # type: ignore[misc]
    finally:
        session.close()


def init_db() -> None:
    """Create all tables. Called once at application startup."""
    # Import models so they register on Base.metadata before create_all.
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
