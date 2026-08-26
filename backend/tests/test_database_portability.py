"""Dialect-portability tests for the two persistence stores.

These tests keep the ADR-0001 promises honest without a live PostgreSQL
server: they verify the inspector-based migration path, init_db idempotency
and the dialect branch of the INSERT-OR-IGNORE helper on SQLite.  A real
PostgreSQL round trip remains covered by TEST_PG_URL when available.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, inspect, text

import app.database as database
from app.services.fof_library import FofLibraryStore


def test_init_db_creates_and_migrates_schema_idempotently(monkeypatch, tmp_path: Path) -> None:
    """init_db() must add strategy_disclosure on first run and be a no-op after."""
    engine = create_engine(f"sqlite:///{tmp_path / 'portable.db'}")
    monkeypatch.setattr(database, "engine", engine)

    database.init_db()

    with engine.connect() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("product_entities")}
    assert "strategy_disclosure" in columns

    # Second run: the column already exists, so the ALTER branch must be skipped.
    database.init_db()
    with engine.connect() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("product_entities")}
    assert "strategy_disclosure" in columns


def test_init_db_migrates_legacy_table_without_strategy_disclosure(monkeypatch, tmp_path: Path) -> None:
    """A pre-existing table missing strategy_disclosure is patched, not dropped."""
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE product_entities (product_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
        ))
    monkeypatch.setattr(database, "engine", engine)

    database.init_db()

    with engine.connect() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("product_entities")}
        preserved = connection.execute(text(
            "INSERT INTO product_entities (product_id, name) VALUES ('legacy-1', '历史产品')"
        ))
        assert preserved.rowcount == 1
    assert "strategy_disclosure" in columns


def test_insert_ignore_uses_or_ignore_on_sqlite() -> None:
    connection = MagicMock()
    connection.dialect.name = "sqlite"

    FofLibraryStore._insert_ignore(connection, "product_materials", "product_id, material_id, linked_at",
                                   {"product_id": "p", "material_id": "m", "linked_at": "now"})

    statement = str(connection.execute.call_args[0][0])
    assert "OR IGNORE" in statement
    assert "ON CONFLICT" not in statement


def test_insert_ignore_uses_on_conflict_on_postgresql() -> None:
    connection = MagicMock()
    connection.dialect.name = "postgresql"

    FofLibraryStore._insert_ignore(connection, "product_materials", "product_id, material_id, linked_at",
                                   {"product_id": "p", "material_id": "m", "linked_at": "now"})

    statement = str(connection.execute.call_args[0][0])
    assert "ON CONFLICT DO NOTHING" in statement
    assert "OR IGNORE" not in statement
