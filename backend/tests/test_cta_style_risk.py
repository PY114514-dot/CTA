from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from app.routers import cta_style_risk
from app.services.cta_style_risk import evaluate_style_risk


def test_style_risk_recovers_synthetic_beta_and_keeps_drift_causal() -> None:
    rng = np.random.default_rng(7)
    factors = rng.normal(0, 0.01, size=(30, 2))
    returns = 0.6 * factors[:, 0] - 0.2 * factors[:, 1] + rng.normal(0, 0.0001, 30)
    dates = [date(2024, 1, 1) + timedelta(days=index * 7) for index in range(30)]
    report = evaluate_style_risk(returns, factors, dates, ["trend", "carry"], rolling_window=20)
    exposure = {item["factor_name"]: item for item in report["factor_exposures"]}
    assert abs(exposure["trend"]["beta"] - 0.6) < 0.02
    assert abs(exposure["carry"]["beta"] + 0.2) < 0.02
    assert report["style_drift"]["causal"] is True
    assert len(report["style_drift"]["paths"]) == 11
    assert report["risk"]["idiosyncratic_variance"] >= 0


def test_style_risk_rejects_products_outside_the_cta_contract(monkeypatch) -> None:
    product = SimpleNamespace(id="equity", confirmation_status="confirmed")
    observations = [SimpleNamespace(nav=1.0, observation_date=date(2024, 1, 1), frequency="weekly")] * 20
    monkeypatch.setattr(cta_style_risk.product_store, "get_product", lambda *_args: product)
    monkeypatch.setattr(cta_style_risk.product_store, "get_nav_series", lambda *_args, **_kwargs: observations)
    monkeypatch.setattr(
        cta_style_risk,
        "select_attribution_contract",
        lambda *_args, **_kwargs: {"status": "not_covered", "reason": "股票产品不能使用商品 CTA 因子。"},
    )

    with pytest.raises(HTTPException, match="不能使用商品 CTA") as error:
        cta_style_risk.get_cta_style_risk("equity", None)

    assert error.value.status_code == 422
