"""CTA-Fama research-gate endpoint; it never emits a production factor."""

from fastapi import APIRouter

from app.schemas import CtaFamaReadinessRequest
from app.services.cta_fama_readiness import evaluate_cta_fama_readiness

router = APIRouter(prefix="/api/cta-fama", tags=["CTA-Fama 研究门"])


@router.post("/readiness")
def get_cta_fama_readiness(request: CtaFamaReadinessRequest) -> dict:
    return evaluate_cta_fama_readiness(request)
