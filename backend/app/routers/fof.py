"""FOF recommendation-agent HTTP interface."""

from typing import Any

from fastapi import APIRouter

from app.schemas import FofRecommendationRequest, FofRecommendationResponse
from app.services.fof_agent import FofRecommendationAgent
from app.services.fof_agent.tools import TOOL_DEFINITIONS

router = APIRouter(tags=["FOF 智能体"])


@router.get("/api/fof/tools")
def list_fof_tools() -> list[dict[str, Any]]:
    """List callable tools exposed by the FOF agent controller."""
    return [
        {"name": tool.name, "description": tool.description, "input_contract": tool.input_contract}
        for tool in TOOL_DEFINITIONS
    ]


@router.post("/api/fof/recommend", response_model=FofRecommendationResponse)
def recommend_fof_products(request: FofRecommendationRequest) -> FofRecommendationResponse:
    """Create a traceable FOF candidate ranking and constrained initial weights."""
    return FofRecommendationAgent().recommend(request)
