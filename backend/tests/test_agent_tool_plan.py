from app.services.agent_tools import validate_requested_tools


def test_llm_tool_suggestions_cannot_expand_allocation_permissions() -> None:
    accepted, rejected = validate_requested_tools(
        "recommend",
        ["interpret_allocation_constraints", "optimize_fof_allocation", "approve_decision", "delete_product"],
    )

    assert accepted == ["interpret_allocation_constraints", "optimize_fof_allocation"]
    assert rejected == ["approve_decision", "delete_product"]
