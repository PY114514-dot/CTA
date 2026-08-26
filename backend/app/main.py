"""FastAPI entry point for the private CTA research MVP.

This module is intentionally thin: it creates the application, wires
middleware, and mounts domain routers.  All business logic lives in
``app/routers/`` and ``app/services/``.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import DEVELOPMENT_CORS_ORIGINS
from app.database import init_db
from app.routers import (
    analysis,
    cta_attribution,
    cta_factors,
    cta_fama,
    cta_fama_audit,
    cta_ranking,
    cta_scores,
    cta_style_risk,
    chat,
    extract,
    external_factor_library,
    factor_library,
    fof_library,
    fof,
    history,
    investment_committee,
    knowledge_base,
    market_data,
    multi_asset_factors,
    nav,
    report_parser,
    product_archive,
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

# 批次 13（体验）：净值批量端点单次响应可达 33MB，回环传输约 3s 是页面
# 加载的大头。GZip 把数字型 JSON 压到约 1/8，传输与前端解析同时受益。
# minimum_size=1KB 避免给小 JSON 白白付压缩 CPU；level 6 在压缩率与
# CPU 之间取平衡。放在 CORS 之后注册，即位于中间件栈最外层。
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(nav.router)
app.include_router(fof.router)
app.include_router(fof_library.router)
app.include_router(market_data.router)
app.include_router(multi_asset_factors.router)
app.include_router(analysis.router)
app.include_router(cta_attribution.router)
app.include_router(cta_factors.router)
app.include_router(cta_fama.router)
app.include_router(cta_fama_audit.router)
app.include_router(cta_ranking.router)
app.include_router(cta_scores.router)
app.include_router(cta_style_risk.router)
app.include_router(factor_library.router)
app.include_router(external_factor_library.router)
app.include_router(report_parser.router)
app.include_router(product_archive.router)
app.include_router(extract.router)
app.include_router(knowledge_base.router)
app.include_router(chat.router)
app.include_router(history.router)
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
