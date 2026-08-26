"""Small allow-list for the production Agent's auditable tool plans."""

from __future__ import annotations


ALLOWED_TOOLS: dict[str, tuple[str, ...]] = {
    "recommend": ("interpret_allocation_constraints", "search_products", "optimize_fof_allocation", "llm_research_synthesis"),
    "screen": ("search_products", "screen_products", "llm_research_synthesis"),
    "compare": ("search_products", "compare_products", "llm_comparison_interpretation"),
    "nav": ("search_products", "calculate_nav_metrics", "analyze_attribution_risk", "llm_research_synthesis"),
    "describe": ("search_products", "llm_research_synthesis"),
}


def build_plan(intent: str) -> list[str]:
    """Return the only tools the executor may invoke for the classified task."""
    return list(ALLOWED_TOOLS.get(intent, ALLOWED_TOOLS["describe"]))


def validate_requested_tools(intent: str, requested_tools: object) -> tuple[list[str], list[str]]:
    """Accept LLM tool suggestions only when they are on the fixed allow-list.

    The returned list is audit metadata, never authority to run a new tool.
    Execution remains owned by the deterministic intent handler.
    """
    allowed = set(build_plan(intent))
    requested = requested_tools if isinstance(requested_tools, list) else []
    accepted = [name for name in requested if isinstance(name, str) and name in allowed]
    rejected = [name for name in requested if isinstance(name, str) and name not in allowed]
    return accepted, rejected
