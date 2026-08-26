"""Fact route handlers for the knowledge-base API."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.services import product_store as store

from .schemas import FactCreate

router = APIRouter()


# ---------------------------------------------------------------------------
# Structured Fact endpoints
# ---------------------------------------------------------------------------


@router.post("/facts")
def create_fact(body: FactCreate, session: Session = Depends(get_session)):
    fact = store.add_fact(session, **body.model_dump())
    return {"id": fact.id, "field_name": fact.field_name, "confidence": fact.confidence}


@router.get("/facts/{product_id}")
def list_facts(product_id: str, current_only: bool = True, session: Session = Depends(get_session)):
    facts = store.get_facts_for_product(session, product_id, current_only=current_only)
    return [
        {
            "id": f.id, "field_name": f.field_name, "field_value": f.field_value,
            "confidence": f.confidence, "extraction_version": f.extraction_version,
            "confirmation_status": f.confirmation_status,
            "source_file_id": f.source_file_id, "source_fragment_id": f.source_fragment_id,
        }
        for f in facts
    ]
