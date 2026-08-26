"""Persistent, human-entered governance facts for CTA-Fama research."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import CtaFamaProductAudit, CtaFamaResearchAudit, ProductEntity

router = APIRouter(prefix="/api/cta-fama/audit", tags=["CTA-Fama 数据治理"])


class ResearchAuditUpdate(BaseModel):
    exit_history_complete: bool
    survivorship_audit_passed: bool
    backfill_audit_passed: bool
    same_source_deduplicated: bool
    oos_state_segments: int = Field(ge=0)
    reviewed_by: str = Field(default="user", min_length=1, max_length=128)


class ProductAuditUpdate(BaseModel):
    source_group: str | None = Field(default=None, max_length=120)
    point_in_time_verified: bool
    reviewed_by: str = Field(default="user", min_length=1, max_length=128)


def _global_payload(audit: CtaFamaResearchAudit | None) -> dict:
    return {
        "exit_history_complete": bool(audit and audit.exit_history_complete),
        "survivorship_audit_passed": bool(audit and audit.survivorship_audit_passed),
        "backfill_audit_passed": bool(audit and audit.backfill_audit_passed),
        "same_source_deduplicated": bool(audit and audit.same_source_deduplicated),
        "oos_state_segments": audit.oos_state_segments if audit else 0,
        "reviewed_by": audit.reviewed_by if audit else None,
        "updated_at": audit.updated_at.isoformat() if audit and isinstance(audit.updated_at, datetime) else None,
    }


@router.get("")
def get_audit(session: Session = Depends(get_session)) -> dict:
    audit = session.get(CtaFamaResearchAudit, "default")
    products = session.query(CtaFamaProductAudit).all()
    return {
        "global": _global_payload(audit),
        "products": [{
            "product_id": item.product_id,
            "source_group": item.source_group,
            "point_in_time_verified": item.point_in_time_verified,
            "reviewed_by": item.reviewed_by,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        } for item in products],
    }


@router.put("/global")
def update_global_audit(body: ResearchAuditUpdate, session: Session = Depends(get_session)) -> dict:
    audit = session.get(CtaFamaResearchAudit, "default") or CtaFamaResearchAudit(id="default")
    for field, value in body.model_dump().items():
        setattr(audit, field, value)
    session.add(audit)
    session.commit()
    session.refresh(audit)
    return _global_payload(audit)


@router.put("/products/{product_id}")
def update_product_audit(product_id: str, body: ProductAuditUpdate, session: Session = Depends(get_session)) -> dict:
    if session.get(ProductEntity, product_id) is None:
        raise HTTPException(404, "产品不存在")
    audit = session.get(CtaFamaProductAudit, product_id) or CtaFamaProductAudit(product_id=product_id)
    audit.source_group = body.source_group.strip() if body.source_group else None
    audit.point_in_time_verified = body.point_in_time_verified
    audit.reviewed_by = body.reviewed_by
    session.add(audit)
    session.commit()
    session.refresh(audit)
    return {
        "product_id": audit.product_id,
        "source_group": audit.source_group,
        "point_in_time_verified": audit.point_in_time_verified,
        "reviewed_by": audit.reviewed_by,
        "updated_at": audit.updated_at.isoformat() if audit.updated_at else None,
    }
