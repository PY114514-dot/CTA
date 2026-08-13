"""Deprecated compatibility layer for the legacy conversational FOF Agent.

The production workbench uses deterministic tool planning and calculation.
An LLM may explain audited output, but must not choose financial tools or
generate recommendations.  These classes remain import-compatible for older
scripts while delegating every plan to :class:`FofRulePolicy`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.schemas import FofRecommendationRequest
from app.services.fof_agent.runtime import AgentState, AgentAction, FofRulePolicy
from app.services.fof_agent.tools import TOOL_DEFINITIONS
from app.services.fof_agent.tools import WebSearchResult

logger = logging.getLogger(__name__)


@dataclass
class LlmPolicyConfig:
    """Configuration for LLM policy."""

    api_base: str
    api_key: str
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: float = 30.0


class LlmPolicy:
    """Compatibility policy that never lets an LLM control a tool plan."""

    def __init__(self, config: LlmPolicyConfig) -> None:
        self.config = config
        self._rule_policy = FofRulePolicy()

    async def initial_plan(self, state: AgentState) -> list[AgentAction]:
        """Return the fixed, auditable plan used by the production runtime."""
        return self._rule_policy.initial_plan(state)

    async def revised_plan(self, state: AgentState) -> list[AgentAction]:
        """Use the same guarded deterministic replan as the production runtime."""
        return self._rule_policy.revised_plan(state)

    def _build_context(
        self,
        request: FofRecommendationRequest,
        reflection_results: Any = None,
        current_recommendations: list[Any] = None,
    ) -> str:
        """Build context prompt for the LLM."""

        # Available tools
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in TOOL_DEFINITIONS
        ])

        # Fund information
        funds_info = []
        for fund in request.funds:
            funds_info.append(
                f"  - {fund.fund_name} ({fund.fund_id}): "
                f"{len(fund.nav_points)} observations, "
                f"frequency: {fund.frequency.value}"
            )

        context = f"""You are a FOF (Fund of Funds) investment advisor AI. Your task is to analyze
the user's request and decide which tools to use to generate a recommendation.

## Available Tools
{tools_desc}

## User Request
- Risk Profile: {request.risk_profile.value}
- Max Single Fund Weight: {request.max_single_fund_weight:.0%}
- Max Recommendations: {request.max_recommendations}

## Candidate Funds
{chr(10).join(funds_info)}

## Current Situation
"""

        if reflection_results:
            context += f"""
## Previous Reflection Results
- Passed: {reflection_results.passed}
- Checks: {reflection_results.checks}
- Warnings: {reflection_results.warnings}
"""

        if current_recommendations:
            rec_info = "\n".join([
                f"  - {r.fund_name}: weight {r.weight:.1%}"
                for r in current_recommendations
            ])
            context += f"""
## Current Recommendations
{rec_info}
"""

        context += """
## Your Task
Based on the above information, decide which tools to call and in what order.
Output your plan in JSON format:

```json
{
  "reasoning": "Your reasoning for this plan",
  "actions": [
    {"tool": "tool_name", "reason": "why this tool", "priority": 1},
    ...
  ]
}
```

Prioritize:
1. First, search for market information relevant to the risk profile
2. Calculate metrics for all candidate funds
3. Score and rank funds
4. Build allocation
5. Review recommendations

Output only valid JSON, no other text:
"""

        return context

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the LLM API."""

        messages = [
            {
                "role": "system",
                "content": "You are a professional FOF investment advisor. Be analytical and precise."
            },
            {"role": "user", "content": prompt}
        ]

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.config.api_base.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()

                content = response.json()["choices"][0]["message"]["content"]

                # Try to parse JSON from response
                return self._extract_json(content)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback to default plan
            return self._default_plan()

    def _extract_json(self, content: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""

        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        import re
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback
        logger.warning("Could not parse JSON from LLM response, using default plan")
        return self._default_plan()

    def _default_plan(self) -> dict[str, Any]:
        """Default plan when LLM fails."""

        return {
            "reasoning": "Using default plan due to LLM failure",
            "actions": [
                {"tool": "calculate_fund_nav_metrics", "reason": "Calculate metrics for all funds", "priority": 1},
                {"tool": "score_fund_candidate", "reason": "Score each fund", "priority": 2},
                {"tool": "build_fof_allocation", "reason": "Build allocation", "priority": 3},
                {"tool": "review_recommendation", "reason": "Review final recommendations", "priority": 4},
            ]
        }

    def _parse_actions(
        self,
        plan: dict[str, Any],
        request: FofRecommendationRequest,
    ) -> list[AgentAction]:
        """Parse LLM response into AgentAction objects."""

        actions = []

        # Get action list from plan
        action_list = plan.get("actions", [])

        # Sort by priority
        action_list.sort(key=lambda x: x.get("priority", 999))

        for action_spec in action_list:
            tool_name = action_spec.get("tool", "")

            # Map tool name to actual action
            if tool_name == "web_search":
                # Search based on risk profile
                query = self._generate_search_query(request.risk_profile.value)
                actions.append(AgentAction(
                    tool_name="web_search",
                    arguments={"query": query, "max_results": 5}
                ))

            elif tool_name == "calculate_fund_nav_metrics":
                # Calculate for each fund
                for fund in request.funds:
                    actions.append(AgentAction(
                        tool_name="calculate_fund_nav_metrics",
                        arguments={"fund": fund, "risk_free_rate": 0.015}
                    ))

            elif tool_name == "score_fund_candidate":
                # Score each fund
                for fund in request.funds:
                    actions.append(AgentAction(
                        tool_name="score_fund_candidate",
                        arguments={"fund": fund}
                    ))

            elif tool_name == "build_fof_allocation":
                actions.append(AgentAction(
                    tool_name="build_fof_allocation",
                    arguments={}
                ))

            elif tool_name == "review_recommendation":
                actions.append(AgentAction(
                    tool_name="review_recommendation",
                    arguments={}
                ))

        # Ensure we have a complete plan (in case LLM misses something)
        if not any(a.tool_name == "build_fof_allocation" for a in actions):
            actions.append(AgentAction(
                tool_name="build_fof_allocation",
                arguments={}
            ))

        if not any(a.tool_name == "review_recommendation" for a in actions):
            actions.append(AgentAction(
                tool_name="review_recommendation",
                arguments={}
            ))

        return actions

    def _generate_search_query(self, risk_profile: str) -> str:
        """Generate search query based on risk profile."""

        queries = {
            "conservative": "稳健型基金 2024 业绩排名 CTA",
            "balanced": "平衡型基金 2024 业绩 对冲",
            "growth": "成长型基金 2024 量化",
        }

        return queries.get(risk_profile, "CTA基金 2024 业绩")


class HybridPolicy:
    """Hybrid policy that uses both LLM and rules.

    Retained for old callers; planning is always deterministic.
    """

    def __init__(
        self,
        llm_config: LlmPolicyConfig | None = None,
        use_llm: bool = True,
    ) -> None:
        self.use_llm = False
        self.llm_policy = None
        self.rule_policy = FofRulePolicy()
        if use_llm and llm_config:
            logger.warning("Legacy LLM planning is disabled; using deterministic FOF rules")

    async def initial_plan(self, state: AgentState) -> list[AgentAction]:
        """Return the fixed deterministic tool plan."""
        return self.rule_policy.initial_plan(state)

    async def revised_plan(self, state: AgentState) -> list[AgentAction]:
        """Return the guarded deterministic replan."""
        return self.rule_policy.revised_plan(state)
