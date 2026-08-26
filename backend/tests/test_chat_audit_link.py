"""Audit-link regression coverage for allocation chat responses."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.routers.chat import ChatAllocationDraft, _handle_selected_product_overview


def test_allocation_draft_hides_internal_audit_fields_from_agent_display() -> None:
    draft = ChatAllocationDraft(
        goal="测试配置",
    )

    assert draft.portfolio is None
    assert "snapshot_id" not in draft.model_dump()


def test_selected_product_overview_prioritizes_nav_risk_and_attribution() -> None:
    """A reviewed product conclusion must not degrade into a manager-material template."""
    session = Mock()
    session.get.return_value = SimpleNamespace(
        standard_name="已复核产品",
        facts=[],
    )
    tool_calls = []

    with patch(
        "app.routers.chat._handle_nav_query",
        return_value="**已复核产品**：累计收益 12.00% | 最大回撤 -3.00%",
    ) as nav_query, patch(
        "app.routers.chat._handle_attribution_summary",
        return_value="**归因与风险评价：** R² 0.31；主要统计关联为短期动量。",
    ) as attribution_summary:
        content = _handle_selected_product_overview(session, ["product-1"], tool_calls)

    nav_query.assert_called_once_with(session, ["product-1"], tool_calls)
    attribution_summary.assert_called_once_with(session, "product-1", tool_calls)
    assert "累计收益 12.00%" in content
    assert "归因与风险评价" in content
    assert "管理人资料" not in content
    assert "工具计算结果为空" not in content
