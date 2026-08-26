"""Read-only CTA style-risk endpoint based on confirmed, reviewed NAV."""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import ConfirmationStatus
from app.services import product_store
from app.services.attribution_applicability import select_attribution_contract
from app.services.cta_style_risk import evaluate_style_risk
from app.services.dynamic_beta import DYNAMIC_FACTOR_NAMES, prepare_factor_matrix

router = APIRouter(prefix="/api/cta-style-risk", tags=["CTA 风格风险"])


@router.get("/products/{product_id}")
def get_cta_style_risk(product_id: str, session: Session = Depends(get_session)) -> dict:
    product = product_store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    if product.confirmation_status != ConfirmationStatus.CONFIRMED:
        raise HTTPException(422, "仅已确认产品可进行 CTA 风格风险评价")
    observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
    if len(observations) < 20:
        raise HTTPException(422, "至少需要 20 条已审核净值才能进行 CTA 风格风险评价")
    applicability = select_attribution_contract(product, len(observations))
    if applicability["status"] not in {"applicable", "observe_only"}:
        raise HTTPException(422, applicability["reason"])
    frequency = product.nav_frequency or observations[0].frequency or "weekly"
    if frequency not in {"daily", "weekly", "monthly"}:
        raise HTTPException(422, "净值频率必须为 daily、weekly 或 monthly")
    nav = np.asarray([float(item.nav) for item in observations], dtype=float)
    try:
        aligned = prepare_factor_matrix(nav[1:] / nav[:-1] - 1.0, [item.observation_date for item in observations[1:]], frequency, DYNAMIC_FACTOR_NAMES)
        report = evaluate_style_risk(aligned["returns"], aligned["factor_returns"], aligned["dates"], aligned["factor_names"], rolling_window={"daily": 60, "weekly": 26, "monthly": 12}[frequency])
    except ValueError as error:
        raise HTTPException(422, f"CTA 风格风险数据不足：{error}") from error
    return {"product_id": product.id, "product_name": product.standard_name, "frequency": frequency,
            "data_contract": {"confirmation_status": product.confirmation_status, "reviewed_nav_count": len(observations), "source": "confirmed + reviewed NAV only", "parsing_triggered": False},
            "alignment": {"factor_names": aligned["factor_names"], "aligned_observation_count": aligned["observation_count"], "start_date": aligned["dates"][0].isoformat(), "end_date": aligned["dates"][-1].isoformat()},
            **report, "warnings": aligned["warnings"] + report["warnings"]}
