"""Alembic environment for the FOF platform backend.

Migrations target the ORM metadata registered in ``app.models``.  The
database URL is derived from ``app.database.DATABASE_URL`` (the DATABASE_URL
env var or the zero-config SQLite default) so there is a single source of
truth with the application.  Two execution modes are supported:

* CLI: ``python -m alembic ...`` resolves the URL from the environment and
  opens its own connection (NullPool, closed after each command).
* Startup: ``init_db()`` passes a live connection from the application engine
  via ``config.attributes["connection"]`` and owns the surrounding
  transaction.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make `app` importable regardless of the invoking directory.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from app.database import Base, DATABASE_URL  # noqa: E402
import app.models  # noqa: E402, F401  # register every model on Base.metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _configure_context(connection) -> None:  # noqa: ANN001
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Batch mode keeps future SQLite ALTERs (add/drop/alter column)
        # working within the dialect's limited DDL support.  On PostgreSQL
        # the native ALTER path is used.
        render_as_batch=connection.dialect.name == "sqlite",
    )


def run_migrations_offline() -> None:
    """Generate SQL without a live connection (e.g. ``--sql`` output)."""
    context.configure(
        url=os.environ.get("DATABASE_URL") or DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    provided = config.attributes.get("connection", None)
    if provided is not None:
        # Startup path: init_db() already opened and wrapped the connection.
        _configure_context(provided)
        with context.begin_transaction():
            context.run_migrations()
        return

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = os.environ.get("DATABASE_URL") or DATABASE_URL
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        _configure_context(connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
