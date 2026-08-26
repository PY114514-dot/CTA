"""Product knowledge-base HTTP interface (P0 data layer).

Endpoints for material registration, product CRUD, NAV management,
structured facts, agent run audit and decision records.
"""

from __future__ import annotations

from fastapi import APIRouter

from .agent_runs import router as agent_runs_router
from .decisions import router as decisions_router
from .facts import router as facts_router
from .files import router as files_router
from .helpers import _product_readiness, _research_workflow
from .nav_routes import router as nav_router
from .products import router as products_router

router = APIRouter(prefix="/api/kb", tags=["产品知识库"])

# Include all sub-routers.  Each sub-module creates a plain APIRouter()
# without prefix or tags; the parent router above owns those.
router.include_router(files_router)
router.include_router(products_router)
router.include_router(nav_router)
router.include_router(facts_router)
router.include_router(agent_runs_router)
router.include_router(decisions_router)
