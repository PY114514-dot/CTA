"""Regression tests for natural-language FOF allocation intent parsing."""

from unittest.mock import Mock

from app.routers import chat
from app.routers.chat import _detect_intent, _handle_compare_query, _handle_recommend_query, _needs_allocation_clarification
from app.services.allocation_agent_graph import AllocationAgentResult
from app.services.fof_allocation_research import AllocationResearchResult
from app.services.fof_allocation_intent import parse_allocation_intent, strategy_matches


def test_commodity_cta_matches_persisted_internal_strategy_id() -> None:
    intent = parse_allocation_intent("我看好商品 CTA，请给初始配置")

    assert intent.strategy_keywords == ("commodity_cta",)
    assert strategy_matches("commodity_cta", intent.strategy_keywords)
    assert strategy_matches("商品 CTA", intent.strategy_keywords)
    assert not strategy_matches("trend_cta", intent.strategy_keywords)


def test_generic_cta_matches_cta_variants_but_not_equity_quant() -> None:
    intent = parse_allocation_intent("请给 CTA 做配置")

    assert intent.strategy_keywords == ("cta",)
    assert strategy_matches("commodity_cta", intent.strategy_keywords)
    assert strategy_matches("trend_cta", intent.strategy_keywords)
    assert strategy_matches("arbitrage_cta", intent.strategy_keywords)
    assert not strategy_matches("equity_quant", intent.strategy_keywords)


def test_drawdown_preference_without_number_does_not_invent_limit() -> None:
    intent = parse_allocation_intent("我想控制回撤，做一个 FOF 配置")

    assert intent.drawdown_preference is True
    assert intent.max_drawdown is None


def test_explicit_drawdown_limit_is_parsed() -> None:
    intent = parse_allocation_intent("希望最大回撤控制在 10% 内")

    assert intent.drawdown_preference is True
    assert intent.max_drawdown == 0.10


def test_semantic_constraints_support_preferred_product_count() -> None:
    intent = parse_allocation_intent(
        "我希望组合尽量少，最好不超过五只",
        {
            "max_products": 5,
            "min_annualized_return": 0.10,
            "max_annualized_volatility": 0.15,
        },
    )

    assert intent.max_products == 5
    assert intent.min_annualized_return == 0.10
    assert intent.max_annualized_volatility == 0.15


def test_compare_preset_with_configuration_role_is_not_reclassified_as_recommendation() -> None:
    assert _detect_intent("请比较我已选的产品：收益、回撤、相关性和适合的配置角色。") == "compare"


def test_constraint_only_reply_starts_allocation_without_selected_product() -> None:
    assert _detect_intent("年化收益至少 20%，年化波动不超过 20%，最多 8 只，优先商品 CTA") == "recommend"


def test_return_target_starts_allocation_without_risk_or_capacity_limit() -> None:
    intent = parse_allocation_intent("帮我配置一个年化百分之二十的产品，没有波动率要求")

    assert intent.min_annualized_return == 0.20
    assert not _needs_allocation_clarification(intent)


def test_compare_requires_two_selected_products_and_never_enters_allocation() -> None:
    tool_calls = []

    content = _handle_compare_query(Mock(), ["one-product"], tool_calls)

    assert content == "请选择至少 2 个产品进行比较。"
    assert tool_calls == []


def test_recommendation_keeps_snapshot_sources_out_of_chat_citations(monkeypatch) -> None:
    research = AllocationResearchResult(
        content="配置完成",
        citations=[{"file_id": "source", "filename": "全量资料.pdf", "page_number": None, "fragment_id": "fragment", "fragment_type": "text", "snippet": "审计资料", "confidence": 0.9}],
    )
    monkeypatch.setattr(chat, "run_allocation_agent", lambda *_args: AllocationAgentResult(
        research=research, plan=["计算", "校验"], validation={"passed": True},
    ))
    citations = []

    content, draft = _handle_recommend_query(Mock(), "做配置", [], citations)

    assert content == "配置完成"
    assert draft is None
    assert citations == []
