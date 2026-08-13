"""Independent state-machine runtime for the FOF agent.

It borrows the useful boundaries of the reference projects without depending
on either project: a planner selects registered tools, working memory holds
only the active run, durable memory provides bounded recall, and a reflection
gate can trigger a revised plan before a result is released.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from app.schemas import (
    FofAgentEvent,
    FofAllocationItem,
    FofCandidateEvaluation,
    FofRecommendationRequest,
    FofReflection,
    PerformanceMetrics,
)

from .tools import ResilientToolRegistry as FofToolRegistry


class AgentPhase(StrEnum):
    PLAN = "plan"
    ACT = "act"
    REFLECT = "reflect"
    REPLAN = "replan"
    COMPLETE = "complete"


@dataclass(frozen=True)
class AgentAction:
    """A planner-produced request to execute one registered tool."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass
class WorkingMemory:
    """Run-scoped memory. It is discarded after the response is produced."""

    metrics_by_fund: dict[str, PerformanceMetrics] = field(default_factory=dict)
    evaluations: list[FofCandidateEvaluation] = field(default_factory=list)
    recommendations: list[FofAllocationItem] = field(default_factory=list)
    reflection: FofReflection | None = None
    observations: list[str] = field(default_factory=list)
    web_search_results: dict[str, list[dict]] = field(default_factory=dict)  # query -> results


@dataclass
class AgentState:
    request: FofRecommendationRequest
    registry: FofToolRegistry
    phase: AgentPhase = AgentPhase.PLAN
    actions: list[AgentAction] = field(default_factory=list)
    iteration: int = 0
    plan_revisions: int = 0
    memory: WorkingMemory = field(default_factory=WorkingMemory)
    events: list[FofAgentEvent] = field(default_factory=list)

    def emit(self, action: str, status: str, detail: str) -> None:
        self.events.append(FofAgentEvent(
            phase=self.phase.value, action=action, status=status,
            iteration=self.iteration, detail=detail,
        ))


class AgentPolicy(Protocol):
    """Replaceable decision layer; an LLM policy may implement this later."""

    def initial_plan(self, state: AgentState) -> list[AgentAction]: ...

    def revised_plan(self, state: AgentState) -> list[AgentAction]: ...

    # Optional async methods for LLM-powered policies
    async def initial_plan_async(self, state: AgentState) -> list[AgentAction]:
        """Async version of initial_plan. Falls back to sync if not implemented."""
        return self.initial_plan(state)

    async def revised_plan_async(self, state: AgentState) -> list[AgentAction]:
        """Async version of revised_plan. Falls back to sync if not implemented."""
        return self.revised_plan(state)


class FofRulePolicy:
    """Safe deterministic policy that schedules only registered FOF tools."""

    def initial_plan(self, state: AgentState) -> list[AgentAction]:
        actions: list[AgentAction] = []
        for fund in state.request.funds:
            actions.append(AgentAction("calculate_fund_nav_metrics", {
                "fund": fund, "risk_free_rate": state.request.annual_risk_free_rate,
            }))
        for fund in state.request.funds:
            actions.append(AgentAction("score_fund_candidate", {"fund": fund}))
        actions.extend([
            AgentAction("build_fof_allocation", {}),
            AgentAction("review_recommendation", {}),
        ])
        return actions

    def revised_plan(self, state: AgentState) -> list[AgentAction]:
        """A failed reflection cannot fabricate data; release a guarded result."""
        return [AgentAction("finalize_with_warnings", {})]

    async def initial_plan_async(self, state: AgentState) -> list[AgentAction]:
        """Async version - just delegate to sync version."""
        return self.initial_plan(state)

    async def revised_plan_async(self, state: AgentState) -> list[AgentAction]:
        """Async version - just delegate to sync version."""
        return self.revised_plan(state)


class FofAgentRuntime:
    """Plan → act → reflect → replan state machine with a maximum of one replan."""

    def __init__(self, registry: FofToolRegistry, policy: AgentPolicy | None = None) -> None:
        self.registry = registry
        self.policy = policy or FofRulePolicy()

    def run(self, request: FofRecommendationRequest) -> AgentState:
        state = AgentState(request=request, registry=self.registry)

        # Use sync version for sync run
        state.actions = self.policy.initial_plan(state)

        state.emit("create_plan", "ok", f"已创建包含 {len(state.actions)} 个动作的初始计划。")
        while state.actions:
            action = state.actions.pop(0)
            state.iteration += 1
            if action.tool_name == "finalize_with_warnings":
                state.phase = AgentPhase.COMPLETE
                state.emit(action.tool_name, "ok", "反思未通过，已按保护性策略输出带警告的结果。")
                continue
            self._execute_action(state, action)
            if action.tool_name == "review_recommendation":
                state.phase = AgentPhase.REFLECT
                reflection = state.memory.reflection
                if reflection and not reflection.passed and state.plan_revisions < 1:
                    state.phase = AgentPhase.REPLAN
                    state.plan_revisions += 1

                    # Use sync version for sync run
                    state.actions = self.policy.revised_plan(state)

                    state.emit("revise_plan", "warning", "反思发现不可用结果，已生成保护性修订计划。")
                else:
                    state.phase = AgentPhase.COMPLETE
        if state.phase != AgentPhase.COMPLETE:
            state.phase = AgentPhase.COMPLETE
        return state

    async def run_async(self, request: FofRecommendationRequest) -> AgentState:
        """Async version of run for use with async policies."""
        state = AgentState(request=request, registry=self.registry)

        # Use async version for async run
        if hasattr(self.policy, 'initial_plan_async'):
            state.actions = await self.policy.initial_plan_async(state)
        else:
            state.actions = self.policy.initial_plan(state)

        state.emit("create_plan", "ok", f"已创建包含 {len(state.actions)} 个动作的初始计划。")

        while state.actions:
            action = state.actions.pop(0)
            state.iteration += 1
            if action.tool_name == "finalize_with_warnings":
                state.phase = AgentPhase.COMPLETE
                state.emit(action.tool_name, "ok", "反思未通过，已按保护性策略输出带警告的结果。")
                continue
            self._execute_action(state, action)
            if action.tool_name == "review_recommendation":
                state.phase = AgentPhase.REFLECT
                reflection = state.memory.reflection
                if reflection and not reflection.passed and state.plan_revisions < 1:
                    state.phase = AgentPhase.REPLAN
                    state.plan_revisions += 1

                    if hasattr(self.policy, 'revised_plan_async'):
                        state.actions = await self.policy.revised_plan_async(state)
                    else:
                        state.actions = self.policy.revised_plan(state)

                    state.emit("revise_plan", "warning", "反思发现不可用结果，已生成保护性修订计划。")
                else:
                    state.phase = AgentPhase.COMPLETE

        if state.phase != AgentPhase.COMPLETE:
            state.phase = AgentPhase.COMPLETE

        return state

    def _execute_action(self, state: AgentState, action: AgentAction) -> None:
        state.phase = AgentPhase.ACT
        try:
            if action.tool_name == "calculate_fund_nav_metrics":
                result = self.registry.execute(action.tool_name, **action.arguments)
                state.memory.metrics_by_fund[action.arguments["fund"].fund_id] = result
            elif action.tool_name == "score_fund_candidate":
                fund = action.arguments["fund"]
                result = self.registry.execute(action.tool_name,
                    fund=fund, metrics=state.memory.metrics_by_fund[fund.fund_id],
                    risk_profile=state.request.risk_profile,
                )
                state.memory.evaluations.append(result)
            elif action.tool_name == "build_fof_allocation":
                eligible = sorted(
                    (item for item in state.memory.evaluations if item.eligible),
                    key=lambda item: item.score, reverse=True,
                )[:state.request.max_recommendations]
                weights = self.registry.execute(action.tool_name,
                    evaluations=eligible, max_weight=state.request.max_single_fund_weight,
                )
                selected = {item.fund_id: item for item in eligible}
                state.memory.recommendations = [
                    FofAllocationItem(
                        fund_id=fund_id, fund_name=selected[fund_id].fund_name, weight=weight,
                        rationale=(f"综合评分 {selected[fund_id].score:.2f}；由评分和逆波动生成初始权重，"
                                   f"单产品权重不高于 {state.request.max_single_fund_weight:.0%}。"),
                    )
                    for fund_id, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)
                ]
                result = state.memory.recommendations
            elif action.tool_name == "review_recommendation":
                result = self.registry.execute(action.tool_name,
                    evaluations=state.memory.evaluations,
                    recommendations=state.memory.recommendations,
                    max_weight=state.request.max_single_fund_weight,
                )
                state.memory.reflection = result
            elif action.tool_name == "web_search":
                query = action.arguments.get("query", "")
                max_results = action.arguments.get("max_results", 5)
                result = self.registry.execute(action.tool_name, query=query, max_results=max_results)
                # Store search results in memory for later reference
                state.memory.web_search_results[query] = [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "relevance_score": r.relevance_score,
                    }
                    for r in result
                ]
            else:
                raise ValueError(f"Policy requested unsupported action: {action.tool_name}")
        except Exception as error:
            state.emit(action.tool_name, "error", f"{type(error).__name__}: {error}")
            raise
        state.memory.observations.append(f"{action.tool_name} completed")
        state.emit(action.tool_name, "ok", "已执行注册工具并写入短期工作记忆。")
