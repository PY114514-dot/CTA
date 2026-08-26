"""Agent run audit route handlers for the knowledge-base API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_session
from app.services import product_store as store

from .schemas import AgentRunComplete, AgentRunCreate, ToolInvocationCreate

router = APIRouter()


# ---------------------------------------------------------------------------
# Agent Run audit endpoints
# ---------------------------------------------------------------------------


@router.post("/runs")
def create_run(body: AgentRunCreate, session: Session = Depends(get_session)):
    run = store.create_agent_run(session, user_query=body.user_query, session_id=body.session_id, plan=body.plan)
    return {"id": run.id, "phase": run.phase}


@router.patch("/runs/{run_id}/complete")
def complete_run(run_id: str, body: AgentRunComplete, session: Session = Depends(get_session)):
    run = store.complete_agent_run(session, run_id, **body.model_dump())
    if run is None:
        raise HTTPException(404, "Agent run not found")
    return {"id": run.id, "phase": run.phase, "duration_ms": run.duration_ms}


@router.post("/runs/{run_id}/tools")
def add_tool_invocation(run_id: str, body: ToolInvocationCreate, session: Session = Depends(get_session)):
    invocation = store.add_tool_invocation(
        session, run_id,
        tool_name=body.tool_name,
        input_summary=body.input_summary,
        output_summary=body.output_summary,
        status=body.status,
        error_message=body.error_message,
        duration_ms=body.duration_ms,
    )
    return {"id": invocation.id, "tool_name": invocation.tool_name, "status": invocation.status}


@router.get("/runs")
def list_runs(session_id: str | None = None, limit: int = 20, offset: int = 0, session: Session = Depends(get_session)):
    runs = store.list_agent_runs(session, session_id=session_id, limit=limit, offset=offset)
    return [
        {
            "id": r.id, "session_id": r.session_id, "user_query": r.user_query[:100],
            "phase": r.phase, "tools_used": r.tools_used,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]
