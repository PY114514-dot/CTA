"""Bounded LangGraph workflow for an auditable FOF configuration run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.orm import Session

from app.services.fof_allocation_intent import AllocationIntent
from app.services.fof_allocation_research import AllocationResearchResult, build_allocation_research


class AllocationAgentState(TypedDict, total=False):
    query: str
    intent: AllocationIntent
    plan: list[str]
    result: AllocationResearchResult
    validation: dict[str, Any]


class AllocationApprovalState(TypedDict, total=False):
    """Closed approval state: a human decision is required before scoring."""

    decision_id: str
    content: dict[str, Any]
    human_response: dict[str, Any]
    score: dict[str, Any]
    version: dict[str, Any]


@dataclass(frozen=True)
class AllocationAgentResult:
    research: AllocationResearchResult
    plan: list[str]
    validation: dict[str, Any]


@dataclass(frozen=True)
class AllocationApprovalResult:
    waiting_for_human: bool
    approved: bool | None = None
    score: dict[str, Any] | None = None
    version: dict[str, Any] | None = None


def run_allocation_agent(session: Session, query: str, intent: AllocationIntent) -> AllocationAgentResult:
    """Execute a fixed four-node graph; it has no open-ended ReAct loop."""

    def plan(_: AllocationAgentState) -> AllocationAgentState:
        return {"plan": ["检索候选产品", "确定性筛选与配置", "约束校验", "生成研究解读"]}

    def compute(state: AllocationAgentState) -> AllocationAgentState:
        return {"result": build_allocation_research(session, state["query"], state["intent"])}

    def validate(state: AllocationAgentState) -> AllocationAgentState:
        result = state["result"]
        allocations = (result.draft or {}).get("allocations", [])
        max_products = state["intent"].max_products
        valid = max_products is None or len(allocations) <= max_products
        validation = {
            "passed": valid,
            "allocation_count": len(allocations),
            "max_products": max_products,
        }
        if not valid:
            raise ValueError("配置结果超过产品数量上限，已阻止输出。")
        return {"validation": validation}

    graph = StateGraph(AllocationAgentState)
    graph.add_node("plan", plan)
    graph.add_node("compute", compute)
    graph.add_node("validate", validate)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "compute")
    graph.add_edge("compute", "validate")
    graph.add_edge("validate", END)
    state = graph.compile().invoke({"query": query, "intent": intent})
    return AllocationAgentResult(
        research=state["result"], plan=state["plan"], validation=state["validation"],
    )


def score_allocation_draft(content: dict[str, Any]) -> dict[str, Any]:
    """Return an explainable research-quality score from frozen result data."""
    allocations = content.get("allocations") or []
    metrics = ((content.get("portfolio") or {}).get("metrics") or {})
    annual_return = metrics.get("annualized_return")
    volatility = metrics.get("annualized_volatility")
    drawdown = metrics.get("maximum_drawdown")
    sharpe = metrics.get("sharpe_ratio")
    weights = [float(item.get("weight", 0)) for item in allocations]
    valid_weights = bool(weights) and all(0 < weight <= 1 for weight in weights) and abs(sum(weights) - 1) <= 0.001

    def bounded(value: Any, ceiling: float) -> float:
        return max(0.0, min(float(value or 0) / ceiling, 1.0))

    constraints = content.get("interpreted_constraints") or {}
    checks = [valid_weights]
    if constraints.get("max_products") is not None:
        checks.append(len(weights) <= int(constraints["max_products"]))
    if constraints.get("min_annualized_return") is not None:
        checks.append((annual_return or -1) >= constraints["min_annualized_return"])
    if constraints.get("max_annualized_volatility") is not None:
        checks.append((volatility or float("inf")) <= constraints["max_annualized_volatility"])
    components = {
        "收益效率": round(25 * bounded(annual_return, 0.15), 1),
        "回撤控制": round(25 * max(0.0, 1.0 - abs(float(drawdown or 0)) / 0.20), 1),
        "风险调整收益": round(20 * bounded(sharpe, 1.0), 1),
        "分散度": round(15 * min(len(weights) / 3, 1.0) * (1.0 - max(weights, default=1.0) / 2), 1),
        "约束满足": round(15 * sum(checks) / len(checks), 1),
    }
    return {
        "total": round(sum(components.values()), 1),
        "components": components,
        "rules": "收益上限15%、回撤阈值20%、夏普目标1；分散度由产品数量和最大权重计算；约束项逐条校验。",
        "inputs": {"annualized_return": annual_return, "annualized_volatility": volatility, "maximum_drawdown": drawdown, "sharpe_ratio": sharpe, "product_count": len(weights), "largest_weight": max(weights, default=None)},
    }


def _approval_graph():
    def route_from_start(state: AllocationApprovalState) -> Literal["await_human", "score", "end"]:
        response = state.get("human_response")
        if response is None:
            return "await_human"
        return "score" if response.get("approved") else "end"

    def await_human(_: AllocationApprovalState) -> AllocationApprovalState:
        return {"human_response": interrupt({"kind": "allocation_approval", "message": "等待人工确认配置草案"})}

    def route_after_human(state: AllocationApprovalState) -> Literal["score", "end"]:
        return "score" if state.get("human_response", {}).get("approved") else "end"

    def score(state: AllocationApprovalState) -> AllocationApprovalState:
        return {"score": score_allocation_draft(state["content"])}

    def version(_: AllocationApprovalState) -> AllocationApprovalState:
        return {"version": {"number": 1, "status": "approved", "tracking_enabled": True}}

    graph = StateGraph(AllocationApprovalState)
    graph.add_node("await_human", await_human)
    graph.add_node("score", score)
    graph.add_node("version", version)
    graph.add_conditional_edges(START, route_from_start, {"await_human": "await_human", "score": "score", "end": END})
    graph.add_conditional_edges("await_human", route_after_human, {"score": "score", "end": END})
    graph.add_edge("score", "version")
    graph.add_edge("version", END)
    return graph.compile(checkpointer=MemorySaver())


_APPROVAL_GRAPH = _approval_graph()


def request_allocation_approval(decision_id: str, content: dict[str, Any]) -> AllocationApprovalResult:
    state = _APPROVAL_GRAPH.invoke({"decision_id": decision_id, "content": content}, config={"configurable": {"thread_id": f"allocation-approval:{decision_id}"}})
    return AllocationApprovalResult(waiting_for_human="__interrupt__" in state)


def complete_allocation_approval(decision_id: str, *, approved: bool, content: dict[str, Any] | None = None) -> AllocationApprovalResult:
    config = {"configurable": {"thread_id": f"allocation-approval:{decision_id}"}}
    try:
        state = _APPROVAL_GRAPH.invoke(Command(resume={"approved": approved}), config=config)
    except Exception:
        # The database remains the durable approval record. If the web process
        # restarted while waiting, rehydrate the same closed graph from it.
        if content is None:
            raise
        state = _APPROVAL_GRAPH.invoke({"decision_id": decision_id, "content": content, "human_response": {"approved": approved}}, config=config)
    if not approved:
        return AllocationApprovalResult(waiting_for_human=False, approved=False)
    return AllocationApprovalResult(waiting_for_human=False, approved=True, score=state.get("score"), version=state.get("version"))
