"""FastAPI entry point for the private CTA research MVP.

This module is intentionally thin: it creates the application, wires
middleware, and mounts domain routers.  All business logic lives in
``app/routers/`` and ``app/services/``.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import DEVELOPMENT_CORS_ORIGINS
from app.database import init_db
from app.routers import (
    analysis,
    cta_attribution,
    cta_ranking,
    chat,
    extract,
    external_factor_library,
    factor_library,
    fof_library,
    fof,
    investment_committee,
    knowledge_base,
    market_data,
    nav,
    report_parser,
)

# ---------------------------------------------------------------------------
# Logging (issue #47: previously unconfigured)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(title="Private CTA Research API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEVELOPMENT_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(nav.router)
app.include_router(fof.router)
app.include_router(fof_library.router)
app.include_router(market_data.router)
app.include_router(analysis.router)
app.include_router(cta_attribution.router)
app.include_router(cta_ranking.router)
app.include_router(factor_library.router)
app.include_router(external_factor_library.router)
app.include_router(report_parser.router)
app.include_router(extract.router)
app.include_router(knowledge_base.router)
app.include_router(chat.router)
app.include_router(investment_committee.router)


# ---------------------------------------------------------------------------
# Startup: ensure database tables exist
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _startup_init_db() -> None:
    init_db()


# ---------------------------------------------------------------------------
# Health check (kept inline — single trivial endpoint)
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["系统"])
def read_health() -> dict[str, str]:
    """Small readiness endpoint for local development and deployment checks."""
    return {"status": "ok"}
