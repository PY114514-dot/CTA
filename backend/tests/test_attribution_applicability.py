from types import SimpleNamespace

from app.services.attribution_applicability import select_attribution_contract


def test_equity_product_is_not_sent_to_cta_contract() -> None:
    result = select_attribution_contract(SimpleNamespace(standard_name="沪深300指数增强", strategy=None), 120)
    assert result["product_pool"] == "equity_quant"
    assert result["status"] == "not_covered"


def test_named_cta_is_observational_until_strategy_is_confirmed() -> None:
    result = select_attribution_contract(SimpleNamespace(standard_name="商品 CTA 一号", strategy=None), 120)
    assert result["status"] == "observe_only"


def test_confirmed_cta_strategy_is_applicable() -> None:
    result = select_attribution_contract(SimpleNamespace(standard_name="策略一号", strategy="commodity_cta"), 120)
    assert result["status"] == "applicable"
