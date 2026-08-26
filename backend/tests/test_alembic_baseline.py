"""Alembic baseline tests (ADR-0001 phase 2).

These tests exercise the three init_db() paths without a live PostgreSQL
server:

* greenfield databases get create_all + a head stamp;
* pre-alembic legacy databases get the one-time column/index shims, keep
  their data, and are stamped at head;
* versioned databases are left to ``alembic upgrade head`` (no-op at head).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

import app.database as database

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _head_revision() -> str:
    """Current head of the bundled migration scripts."""
    config = Config(str(_BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_BACKEND_ROOT / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


def _version(engine) -> str | None:
    with engine.connect() as connection:
        return connection.execute(text("SELECT version_num FROM alembic_version")).scalar()


def test_greenfield_init_stamps_head_and_creates_phase2_schema(monkeypatch, tmp_path: Path) -> None:
    """A brand-new database gets the full schema plus a head stamp, idempotently."""
    engine = create_engine(f"sqlite:///{tmp_path / 'green.db'}")
    monkeypatch.setattr(database, "engine", engine)

    database.init_db()
    database.init_db()  # second run must be a no-op

    inspector = inspect(engine)
    assert inspector.has_table("alembic_version")
    assert _version(engine) == _head_revision()
    assert "strategy_disclosure" in {c["name"] for c in inspector.get_columns("product_entities")}
    assert "ix_data_snapshots_label" in {i["name"] for i in inspector.get_indexes("data_snapshots")}


def test_legacy_database_is_repaired_and_stamped(monkeypatch, tmp_path: Path) -> None:
    """A pre-alembic database keeps its data while gaining the missing
    strategy_disclosure column, the label index and a head stamp."""
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        # Deliberately incomplete shapes, the way a pre-alembic database that
        # predates both ad-hoc evolutions would look.
        connection.execute(text(
            "CREATE TABLE product_entities (product_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
        ))
        connection.execute(text(
            "CREATE TABLE data_snapshots (id TEXT PRIMARY KEY, label VARCHAR(256))"
        ))
        connection.execute(text(
            "INSERT INTO data_snapshots (id, label) VALUES ('snap-1', 'cta_attribution:2026-01')"
        ))
    monkeypatch.setattr(database, "engine", engine)

    database.init_db()

    inspector = inspect(engine)
    assert "strategy_disclosure" in {c["name"] for c in inspector.get_columns("product_entities")}
    assert "ix_data_snapshots_label" in {i["name"] for i in inspector.get_indexes("data_snapshots")}
    assert _version(engine) == _head_revision()
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM data_snapshots")).scalar() == 1
        preserved = connection.execute(text(
            "INSERT INTO product_entities (product_id, name) VALUES ('legacy-1', '历史产品')"
        ))
        assert preserved.rowcount == 1


def test_versioned_database_upgrade_keeps_foreign_rows_intact(monkeypatch, tmp_path: Path) -> None:
    """Once stamped, a second init_db() must not re-run baseline DDL."""
    engine = create_engine(f"sqlite:///{tmp_path / 'versioned.db'}")
    monkeypatch.setattr(database, "engine", engine)
    database.init_db()
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO product_entities (id, standard_name, confirmation_status, status) "
            "VALUES ('prod-1', '测试产品', 'pending', 'unknown')"
        ))

    database.init_db()  # versioned path: upgrade head, which is a no-op at head

    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM product_entities")).scalar() == 1


def test_baseline_migration_declares_jsonb_variants() -> None:
    """The baseline revision must render JSONB on PostgreSQL, not plain JSON."""
    revision_dir = _BACKEND_ROOT / "migrations" / "versions"
    migration_files = list(revision_dir.glob("*.py"))
    assert len(migration_files) == 1, "baseline must be the only revision"
    source = migration_files[0].read_text(encoding="utf-8")
    assert "with_variant(postgresql.JSONB(), 'postgresql')" in source
    # 20 JSON columns across the models, all variant-declared.
    assert source.count("with_variant(postgresql.JSONB(), 'postgresql')") == 20
    assert "ix_data_snapshots_label" in source
    # The autogenerate renderer emits a bare Text() that is never imported;
    # the post-processing step must have removed it.
    assert "astext_type=" not in source
