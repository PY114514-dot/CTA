"""Database engine, session factory and declarative base.

Defaults to zero-config SQLite with the database file under
``data/fof_agent/fof.db`` (Git-ignored data directory).  Setting the
``DATABASE_URL`` environment variable switches to any other SQLAlchemy
backend (e.g. ``postgresql+psycopg://...``); see ``docs/系统总览.md`` for the
migration direction.  All models inherit from ``Base`` declared here.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Resolve data directory relative to the backend root (backend/../data/fof_agent).
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND_ROOT.parent / "data" / "fof_agent"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATABASE_URL = f"sqlite:///{DATA_DIR / 'fof.db'}"
DATABASE_URL = os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL

# check_same_thread is a SQLite-only connection argument; other backends
# (e.g. psycopg) reject unknown connect_args outright.
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    pool_pre_ping=True,
    pool_timeout=60,
)


if _IS_SQLITE:
    # Enable WAL mode and foreign keys for SQLite.  PostgreSQL enforces both
    # natively, so the PRAGMA listener is only registered for the SQLite path.
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


def _alembic_config():
    """Alembic configuration pointing at the bundled migration scripts."""
    from alembic.config import Config

    config = Config(str(_BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_BACKEND_ROOT / "migrations"))
    return config


def init_db() -> None:
    """Create and migrate the schema. Called once at application startup.

    Three paths (ADR-0001 phase 2):

    * Greenfield (no tables): ``create_all`` materialises the full current
      schema in one shot (indexes included), then the alembic version is
      stamped at head.
    * Pre-alembic database (tables exist, no alembic_version): create_all
      repairs any missing tables, two one-time shims add the only ad-hoc
      evolution that predates alembic (strategy_disclosure column and the
      data_snapshots.label index), then head is stamped.  Schema evolution
      no longer flows through shims after this transition.
    * Versioned database: ``alembic upgrade head`` applies only revisions
      not yet applied.
    """
    from alembic import command

    import app.models  # noqa: F401  # register models on Base.metadata

    config = _alembic_config()
    inspector = inspect(engine)
    has_tables = inspector.has_table("product_entities")
    has_version = inspector.has_table("alembic_version")

    if has_tables and not has_version:
        # Pre-alembic database: bring it up to the current schema once, then
        # record the version.  From here on alembic owns schema evolution.
        Base.metadata.create_all(bind=engine)
        with engine.begin() as connection:
            connection_inspector = inspect(connection)
            columns = {column["name"] for column in connection_inspector.get_columns("product_entities")}
            if "strategy_disclosure" not in columns:
                connection.execute(text("ALTER TABLE product_entities ADD COLUMN strategy_disclosure JSON"))
            indexes = {index["name"] for index in connection_inspector.get_indexes("data_snapshots")}
            if "ix_data_snapshots_label" not in indexes:
                connection.execute(text("CREATE INDEX ix_data_snapshots_label ON data_snapshots (label)"))
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.stamp(config, "head")
    elif not has_tables:
        Base.metadata.create_all(bind=engine)
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.stamp(config, "head")
    else:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
