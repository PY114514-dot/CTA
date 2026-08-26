"""Tests for LLM-powered policy and conversational agent."""

import asyncio

import pytest
from pathlib import Path

from app.schemas import (
    DataFrequency,
    FofFundInput,
    FofRiskProfile,
    NetAssetValuePoint,
)

from app.services.fof_agent.llm_policy import (
    LlmPolicy,
    LlmPolicyConfig,
    HybridPolicy,
)
from app.services.fof_agent.conversational import ConversationalFofAgent, UserIntent
from app.services.fof_agent.runtime import FofAgentRuntime, FofRulePolicy, AgentState
from app.services.fof_agent.tools import ResilientToolRegistry


def _fund(fund_id: str, name: str, values: list[float]) -> FofFundInput:
    return FofFundInput(
        fund_id=fund_id,
        fund_name=name,
        frequency=DataFrequency.MONTHLY,
        nav_points=[
            NetAssetValuePoint(observation_date=f"2024-{m:02d}-01", net_asset_value=v)
            for m, v in enumerate(values, start=1)
        ],
    )


class TestLlmPolicy:
    """Tests for LLM-powered policy."""

    def test_llm_policy_config(self):
        """Test LLM policy configuration."""
        config = LlmPolicyConfig(
            api_base="https://api.example.com",
            api_key="test-key",
            model="gpt-4",
        )

        assert config.api_base == "https://api.example.com"
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"

    def test_hybrid_policy_fallback(self):
        """Test that HybridPolicy falls back to rules when LLM is disabled."""
        # Create hybrid policy without LLM config
        policy = HybridPolicy(use_llm=False)

        # Should use rule policy
        assert policy.use_llm is False
        assert policy.llm_policy is None

    def test_llm_policy_never_controls_the_financial_tool_plan(self):
        config = LlmPolicyConfig(api_base="https://api.example.com", api_key="test-key")
        policy = LlmPolicy(config)
        request = FofRecommendationRequest(
            funds=[
                _fund("fund-a", "A", [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]),
                _fund("fund-b", "B", [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]),
            ],
            risk_profile=FofRiskProfile.BALANCED,
        )
        state = AgentState(request=request, registry=ResilientToolRegistry())

        actions = asyncio.run(policy.initial_plan(state))

        assert [action.tool_name for action in actions][-2:] == ["build_fof_allocation", "review_recommendation"]
        assert all(action.tool_name != "web_search" for action in actions)


class TestConversationalAgent:
    """Tests for conversational FOF agent."""

    @pytest.mark.asyncio
    async def test_parse_conservative_intent(self):
        """Test parsing conservative intent from user message."""
        agent = ConversationalFofAgent(use_llm=False)

        intent = await agent._parse_intent("I want a stable product with low risk")

        assert intent.risk_profile == FofRiskProfile.CONSERVATIVE
        assert intent.raw_query == "I want a stable product with low risk"

    @pytest.mark.asyncio
    async def test_parse_growth_intent(self):
        """Test parsing growth intent from user message."""
        agent = ConversationalFofAgent(use_llm=False)

        intent = await agent._parse_intent("I want high returns, can tolerate high risk")

        assert intent.risk_profile == FofRiskProfile.GROWTH

    @pytest.mark.asyncio
    async def test_parse_balanced_intent(self):
        """Test parsing balanced intent (default)."""
        agent = ConversationalFofAgent(use_llm=False)

        # Without explicit keywords, should default to balanced
        intent = await agent._parse_intent("I want to invest my money")

        assert intent.risk_profile == FofRiskProfile.BALANCED

    @pytest.mark.asyncio
    async def test_chat_returns_recommendation(self):
        """Test that chat returns structured recommendation."""
        agent = ConversationalFofAgent(use_llm=False)

        funds = [
            _fund("fund1", "Stable Fund", [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]),
            _fund("fund2", "Growth Fund", [1.0, 1.02, 1.05, 1.03, 1.08, 1.12, 1.10, 1.15]),
        ]

        response = await agent.chat(
            user_message="I want a stable product",
            funds=funds,
        )

        # Should have intent, recommendation, and explanation
        assert "intent" in response
        assert "recommendation" in response
        assert "explanation" in response

        # Intent should detect conservative
        assert response["intent"]["detected_risk_profile"] == "conservative"


class TestAsyncPolicy:
    """Tests for async policy support."""

    def test_rule_policy_has_async_methods(self):
        """Test that FofRulePolicy has async methods."""
        policy = FofRulePolicy()

        assert hasattr(policy, "initial_plan_async")
        assert hasattr(policy, "revised_plan_async")

    @pytest.mark.asyncio
    async def test_rule_policy_async_initial_plan(self):
        """Test async initial plan."""
        from app.services.fof_agent.runtime import AgentState

        policy = FofRulePolicy()

        # Create a minimal state (need at least 2 funds)
        request = FofRecommendationRequest(
            funds=[
                _fund("test1", "Test1", [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]),
                _fund("test2", "Test2", [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]),
            ],
            risk_profile=FofRiskProfile.BALANCED,
        )

        state = AgentState(request=request, registry=ResilientToolRegistry())

        # Call async version
        actions = await policy.initial_plan_async(state)

        assert len(actions) > 0
        assert actions[0].tool_name == "calculate_fund_nav_metrics"


# Import for test
from app.schemas import FofRecommendationRequest
