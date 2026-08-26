from datetime import date, timedelta

from app.schemas import (
    CtaPortfolioPosition,
    CtaProductScoreRequest,
    CtaRankingProductInput,
    DataFrequency,
    NetAssetValuePoint,
)
from app.services.cta_scores import build_product_score_report


def _product(product_id: str, returns: list[float], *, exposure: dict[str, float] | None = None) -> CtaRankingProductInput:
    nav = 1.0
    points = [NetAssetValuePoint(observation_date=date(2024, 1, 5), net_asset_value=nav)]
    for index, value in enumerate(returns, start=1):
        nav *= 1.0 + value
        points.append(NetAssetValuePoint(
            observation_date=date(2024, 1, 5) + timedelta(days=7 * index),
            net_asset_value=nav,
        ))
    return CtaRankingProductInput(
        product_id=product_id,
        product_name=product_id,
        nav_points=points,
        frequency=DataFrequency.WEEKLY,
        factor_exposures=exposure or {},
    )


def test_product_scores_keep_quality_and_confidence_separate() -> None:
    request = CtaProductScoreRequest(products=[
        _product("stable", [0.004, 0.003, 0.005, 0.002] * 16, exposure={"trend": 0.4}),
        _product("volatile", [0.02, -0.03, 0.015, -0.02] * 16, exposure={"trend": 0.9}),
    ])

    result = build_product_score_report(request)

    stable = next(item for item in result["products"] if item["product_id"] == "stable")
    assert result["model_version"] == "cta-product-score-v3.0"
    assert stable["summary"]["quality_score"] is not None
    assert stable["summary"]["confidence_score"] is not None
    assert stable["summary"]["quality_score"] != stable["summary"]["confidence_score"]
    assert stable["detail"]["quality_explanation"]["model_version"] == "codex-cta-score-v1.1"
    assert result["allocation"]["status"] == "not_available"


def test_product_score_request_accepts_full_sqlite_universe() -> None:
    """The scoring API accepts the current full weekly SQLite product universe."""
    product = _product("weekly", [0.001] * 4)

    request = CtaProductScoreRequest(products=[product.model_copy(update={"product_id": str(index)}) for index in range(749)])

    assert len(request.products) == 749


def test_product_report_encapsulates_summary_scores_and_drill_down_detail(monkeypatch) -> None:
    # Pin the CTA contract loader to empty so the unavailable path is
    # deterministic regardless of the general factor-library cache.
    from app.services import cta_scores

    monkeypatch.setattr(cta_scores, "get_factor_series", lambda *_args, **_kwargs: None)
    request = CtaProductScoreRequest(products=[
        _product("stable", [0.004, 0.003, 0.005, 0.002] * 16, exposure={"trend": 0.4}),
    ])

    result = build_product_score_report(request)

    item = result["products"][0]
    assert set(item.keys()) == {"product_id", "product_name", "summary", "detail"}
    summary = item["summary"]
    assert summary["quality_status"] in {"eligible", "short_sample", "insufficient_data"}
    assert summary["confidence_band"] in {"high", "medium", "low"}
    # 收益分与风险暴露分是 v3 主线新增的绝对尺度评判标准，与证据分并存。
    assert summary["return_score"] is not None
    assert summary["return_band"] in {"high", "medium", "low"}
    assert summary["risk_score"] is not None
    assert summary["risk_band"] in {"high", "medium", "low"}
    # Factor cache is empty in this test module, so attribution degrades
    # gracefully: headline numbers are null and the detail carries a status.
    assert summary["attribution_quality_score"] is None
    assert summary["attribution_quality_band"] is None
    assert summary["headline"]["r_squared"] is None
    assert summary["headline"]["annualized_return"] is not None
    assert summary["headline"]["maximum_drawdown"] is not None
    assert summary["headline"]["var_95"] is not None
    assert set(summary["headline"].keys()) == {
        "annualized_return",
        "cumulative_return",
        "maximum_drawdown",
        "annualized_volatility",
        "var_95",
        "r_squared",
        "oos_r_squared",
        "annualized_alpha",
        "alpha_t_stat",
        "trending_regime_r_squared",
        "choppy_regime_r_squared",
    }
    detail = item["detail"]
    assert set(detail.keys()) == {
        "quality_explanation",
        "confidence",
        "returns_analysis",
        "risk_exposure",
        "return_score",
        "risk_score",
        "attribution",
        "quality_history",
    }
    assert detail["attribution"]["status"] == "unavailable"
    assert detail["attribution"]["warnings"]
    assert detail["returns_analysis"]["status"] == "available"
    assert detail["risk_exposure"]["status"] == "available"
    assert detail["risk_exposure"]["factor_exposure_status"] == "unavailable"
    assert detail["return_score"]["score"] is not None
    assert isinstance(detail["quality_history"], list)


def test_product_summary_populates_headline_from_factor_regression(monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    from app.services import factor_library

    rng = np.random.default_rng(7)
    start = date(2024, 1, 5)
    n = 64
    # Trending half: volatile trend factor the product loads on.  Choppy
    # half: quiet factor and a product that decouples from it, with choppy
    # volatility balanced to the trending variance.
    trend = rng.normal(0.0, 0.01, n)
    trend[40:] = rng.normal(0.0, 0.001, 24)
    product_returns = 1.2 * trend + rng.normal(0.0, 0.001, n)
    product_returns[40:] = rng.normal(0.0, 0.012, 24)
    # Factor dates align with the product RETURN dates (2024-01-12 + 7i),
    # since _product's first return lands one week after its first NAV point.
    dates = [start + timedelta(days=7 * i) for i in range(1, n + 1)]
    monkeypatch.setattr(
        factor_library,
        "get_all_factors",
        lambda: [{"name": "trend", "display_name": "趋势", "category": "量价", "factor_group": "trend"}],
    )
    monkeypatch.setattr(
        factor_library,
        "get_factor_series",
        lambda name, risk_profile="baseline": pd.Series(trend, index=pd.to_datetime(dates)),
    )

    result = build_product_score_report(CtaProductScoreRequest(products=[
        _product("trendy", list(product_returns), exposure={"trend": 0.5}),
    ]))

    summary = result["products"][0]["summary"]
    assert summary["headline"]["r_squared"] is not None
    assert summary["headline"]["r_squared"] > 0.5
    assert summary["headline"]["oos_r_squared"] is not None
    assert summary["attribution_quality_score"] is not None
    assert summary["attribution_quality_band"] in {"high", "medium", "low"}
    detail = result["products"][0]["detail"]
    assert detail["attribution"]["status"] == "available"
    assert detail["attribution"]["factors"][0]["name"] == "trend"
    assert detail["attribution"]["regime"] is not None
    assert (
        detail["attribution"]["regime"]["trending"]["r_squared"]
        > detail["attribution"]["regime"]["choppy"]["r_squared"]
    )


def test_confidence_uses_regression_evidence_instead_of_request_exposures() -> None:
    result = build_product_score_report(CtaProductScoreRequest(products=[
        _product("unverified", [0.004, 0.003, 0.005, 0.002] * 16),
    ]))

    confidence = result["products"][0]["detail"]["confidence"]
    assert confidence["components"]["beta_uncertainty"] is not None
    assert confidence["components"]["factor_alignment_coverage"] is not None
    # v2 起 OOS 稳定性是连续分（0-100），不再只有 0/100 两档。
    assert confidence["components"]["oos_stability"] is not None
    assert 0.0 <= confidence["components"]["oos_stability"] <= 100.0
    assert "beta_uncertainty" in confidence["weights"]
    assert "oos_stability" in confidence["weights"]


def test_quality_score_report_includes_history_and_change_reasons() -> None:
    request = CtaProductScoreRequest(products=[
        _product("stable", [0.004, 0.003, 0.005, 0.002] * 20, exposure={"trend": 0.4}),
        _product("volatile", [0.02, -0.03, 0.015, -0.02] * 20, exposure={"trend": 0.9}),
    ])

    result = build_product_score_report(request)

    stable = next(item for item in result["products"] if item["product_id"] == "stable")
    history = stable["detail"]["quality_history"]
    assert len(history) >= 2
    assert history[-1]["as_of_date"] == result["as_of_date"].isoformat()
    assert history[-1]["quality_score"] == stable["summary"]["quality_score"]
    assert "change_reasons" in history[-1]


def test_allocation_score_requires_explicit_portfolio_and_candidate_weight() -> None:
    current = _product("current", [0.005, -0.002, 0.004, 0.001] * 16, exposure={"trend": 0.5, "carry": 0.2})
    candidate = _product("candidate", [-0.002, 0.004, -0.001, 0.003] * 16, exposure={"trend": -0.1, "carry": 0.1})
    request = CtaProductScoreRequest(
        products=[current, candidate],
        current_portfolio=[CtaPortfolioPosition(product_id="current", weight=1.0)],
        candidate_product_id="candidate",
        candidate_weight=0.25,
    )

    result = build_product_score_report(request)

    allocation = result["allocation"]
    assert allocation["status"] == "available"
    assert allocation["candidate_product_id"] == "candidate"
    assert allocation["metrics"]["full_sample_correlation"] is not None
    assert allocation["metrics"]["marginal_volatility_change"] is not None
    assert allocation["metrics"]["factor_exposure_overlap"] is not None
    assert allocation["score"] is not None


def test_allocation_complementarity_uses_distinct_crisis_and_whipsaw_samples() -> None:
    current = _product("current", [0.02, -0.02] * 10)
    candidate = _product("candidate", [0.05] * 20)
    request = CtaProductScoreRequest(
        products=[current, candidate],
        current_portfolio=[CtaPortfolioPosition(product_id="current", weight=1.0)],
        candidate_product_id="candidate",
        candidate_weight=0.25,
    )

    result = build_product_score_report(request)

    # Crisis returns are -2%, while the alternating series changes sign on
    # every later observation.  The two independent complementarity deltas
    # are therefore 7% and 5%, producing an 80-point component.
    component = result["allocation"]["components"]["crisis_whipsaw_complementarity"]
    assert 80.0 < component < 81.0


def test_allocation_contract_rejects_unlisted_position() -> None:
    product = _product("current", [0.001] * 20)
    try:
        CtaProductScoreRequest(
            products=[product],
            current_portfolio=[CtaPortfolioPosition(product_id="missing", weight=1.0)],
            candidate_product_id="current",
            candidate_weight=0.2,
        )
    except ValueError as error:
        assert "产品池" in str(error)
    else:
        raise AssertionError("expected an explicit portfolio membership error")
