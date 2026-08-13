"""Conversational FOF Agent with LLM-powered understanding.

This module provides a natural language interface for the FOF Agent,
allowing users to express their investment needs in plain language.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from app.schemas import (
    DataFrequency,
    FofFundInput,
    FofRecommendationRequest,
    FofRecommendationResponse,
    FofRiskProfile,
    NetAssetValuePoint,
)

from .agent import FofRecommendationAgent
from .llm_policy import LlmPolicy, LlmPolicyConfig, HybridPolicy
from .memory import FofSessionMemory
from .runtime import FofAgentRuntime, FofRulePolicy

logger = logging.getLogger(__name__)


def get_llm_config_from_env() -> LlmPolicyConfig | None:
    """Load LLM configuration from environment variables.

    Environment variables:
    - LLM_API_BASE: API base URL (e.g., https://api.deepseek.com/v1)
    - LLM_API_KEY: API key
    - LLM_MODEL: Model name (default: deepseek-v4-flash)
    """
    api_base = os.getenv("LLM_API_BASE", "")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "deepseek-v4-flash")

    if api_base and api_key:
        return LlmPolicyConfig(
            api_base=api_base,
            api_key=api_key,
            model=model,
        )
    return None


@dataclass
class UserIntent:
    """Parsed user intent from natural language."""

    raw_query: str
    risk_profile: FofRiskProfile | None = None
    max_investment: float | None = None
    investment_horizon: str | None = None  # "short", "medium", "long"
    specific_funds: list[str] = field(default_factory=list)
    additional_requirements: str | None = None


class ConversationalFofAgent:
    """Conversational FOF Agent that understands natural language."""

    def __init__(
        self,
        memory_directory: Path | None = None,
        llm_config: LlmPolicyConfig | None = None,
        use_llm: bool | None = None,
    ) -> None:
        self.memory = FofSessionMemory(memory_directory or Path("data/fof_agent"))

        # Auto-detect LLM config from environment if not provided
        if llm_config is None:
            llm_config = get_llm_config_from_env()

        self.llm_config = llm_config

        # Auto-enable LLM if config is available, otherwise disable
        if use_llm is None:
            use_llm = llm_config is not None

        self.use_llm = use_llm

        if use_llm and not llm_config:
            logger.warning("LLM requested but no config available, falling back to rules")

    def _create_agent_with_policy(self) -> tuple[FofRecommendationAgent, HybridPolicy]:
        """Create agent with appropriate policy."""

        if self.llm_config and self.use_llm:
            policy = HybridPolicy(llm_config=self.llm_config, use_llm=True)
        else:
            policy = FofRulePolicy()

        # We'll need to modify the agent to accept custom policy
        # For now, use the default agent
        agent = FofRecommendationAgent(
            memory_directory=self.memory.directory,
            enable_cache=True,
        )

        return agent, policy

    async def chat(
        self,
        user_message: str,
        funds: list[FofFundInput],
        context: dict | None = None,
    ) -> dict[str, any]:
        """
        Process a natural language message and generate a recommendation.

        Example:
            user_message = "我想要一个稳健的产品，不想承担太大风险"
            funds = [fund1, fund2, fund3]

            Returns:
                {
                    "intent": UserIntent(...),
                    "recommendation": FofRecommendationResponse(...),
                    "explanation": "Based on your request for a stable product..."
                }
        """

        # Step 1: Parse user intent
        intent = await self._parse_intent(user_message)

        # Step 2: Determine risk profile
        risk_profile = intent.risk_profile or FofRiskProfile.BALANCED

        # Step 3: Build recommendation request
        request = FofRecommendationRequest(
            funds=funds,
            risk_profile=risk_profile,
            session_id=f"chat-{uuid4().hex[:12]}",
            max_recommendations=5,
            max_single_fund_weight=0.4,
        )

        # Step 4: Run agent
        agent = FofRecommendationAgent(memory_directory=self.memory.directory)
        result = agent.recommend(request)

        # Step 5: Generate explanation
        explanation = self._generate_explanation(intent, result)

        return {
            "intent": {
                "raw_query": intent.raw_query,
                "detected_risk_profile": risk_profile.value,
            },
            "recommendation": result,
            "explanation": explanation,
        }

    async def _parse_intent(self, query: str) -> UserIntent:
        """Parse user query to extract intent."""

        # Simple keyword-based parsing
        # In production, use LLM for this

        intent = UserIntent(raw_query=query)

        # Detect risk profile from keywords
        query_lower = query.lower()

        # Conservative keywords
        conservative_keywords = ["稳健", "保守", "安全", "低风险", "保本", "stable", "conservative", "safe"]
        if any(kw in query_lower for kw in conservative_keywords):
            intent.risk_profile = FofRiskProfile.CONSERVATIVE

        # Growth keywords
        growth_keywords = ["成长", "进取", "高收益", "高风险", "激进", "growth", "aggressive", "high return"]
        if any(kw in query_lower for kw in growth_keywords):
            intent.risk_profile = FofRiskProfile.GROWTH

        # Balanced is default
        if intent.risk_profile is None:
            intent.risk_profile = FofRiskProfile.BALANCED

        # Detect investment horizon
        if "短期" in query or "short" in query_lower:
            intent.investment_horizon = "short"
        elif "长期" in query or "long" in query_lower:
            intent.investment_horizon = "long"
        else:
            intent.investment_horizon = "medium"

        # Extract specific fund mentions
        # This would use NER in production

        return intent

    def _generate_explanation(
        self,
        intent: UserIntent,
        result: FofRecommendationResponse,
    ) -> str:
        """Generate natural language explanation."""

        if not result.recommendations:
            return "Sorry, no products match your requirements."

        lines = [
            f"Based on your request: '{intent.raw_query}'",
        ]

        # Risk profile explanation
        risk_explanations = {
            FofRiskProfile.CONSERVATIVE: "I selected stable products focusing on drawdown control",
            FofRiskProfile.BALANCED: "I selected balanced products considering both return and risk",
            FofRiskProfile.GROWTH: "I selected growth products seeking higher returns",
        }

        lines.append(risk_explanations.get(intent.risk_profile, ""))

        # Recommendations
        lines.append("\nRecommended products:")
        for item in result.recommendations:
            lines.append(
                f"  - {item.fund_name}: allocation {item.weight:.0%}"
            )

        # Rationale
        lines.append("\nRationale:")
        for item in result.recommendations:
            lines.append(f"  - {item.rationale}")

        # Warnings
        if result.reflection.warnings:
            lines.append("\nWarnings:")
            for warning in result.reflection.warnings:
                lines.append(f"  - {warning}")

        # Disclaimer
        lines.append(
            "\nThis recommendation is for reference only and does not constitute"
            " actual investment advice. Please invest carefully."
        )

        return "\n".join(lines)


# Example usage
async def example():
    """Example of using the conversational agent."""

    # Sample funds
    funds = [
        FofFundInput(
            fund_id="fund_001",
            fund_name="稳健一号",
            frequency=DataFrequency.MONTHLY,
            nav_points=[
                NetAssetValuePoint(observation_date=f"2024-{m:02d}-01", net_asset_value=1.0 + m * 0.002)
                for m in range(1, 13)
            ],
        ),
        FofFundInput(
            fund_id="fund_002",
            fund_name="进取成长",
            frequency=DataFrequency.MONTHLY,
            nav_points=[
                NetAssetValuePoint(observation_date=f"2024-{m:02d}-01", net_asset_value=1.0 + m * 0.005)
                for m in range(1, 13)
            ],
        ),
    ]

    # Create agent (without LLM for demo)
    agent = ConversationalFofAgent(use_llm=False)

    # User asks for conservative product
    response = await agent.chat(
        user_message="我想要一个稳健的产品，不想承担太大风险",
        funds=funds,
    )

    print(response["explanation"])
    print("\n---")
    print(f"Detected risk profile: {response['intent']['detected_risk_profile']}")
    print(f"Recommended funds: {len(response['recommendation'].recommendations)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())
