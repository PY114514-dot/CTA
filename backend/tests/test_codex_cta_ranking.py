"""Behavioural tests for the CODEX CTA ranking seam."""

from datetime import date, timedelta

from app.schemas import (
    CtaRankingAttributionEvidence,
    CtaRankingProductInput,
    CtaRankingRequest,
    DataFrequency,
    NetAssetValuePoint,
)
from app.services.codex_cta_ranking import rank_cta_products


def _weekly_product(product_id: str, weekly_returns: list[float]) -> CtaRankingProductInput:
    nav = 1.0
    points = [NetAssetValuePoint(observation_date=date(2024, 1, 5), net_asset_value=nav)]
    for index, weekly_return in enumerate(weekly_returns, start=1):
        nav *= 1.0 + weekly_return
        points.append(
            NetAssetValuePoint(
                observation_date=date(2024, 1, 5) + timedelta(days=7 * index),
                net_asset_value=nav,
            )
        )
    return CtaRankingProductInput(
        product_id=product_id,
        product_name=product_id,
        nav_points=points,
        frequency=DataFrequency.WEEKLY,
    )


def test_codex_ranks_stable_positive_product_above_drawdown_product() -> None:
    stable = _weekly_product("stable", [0.004, 0.003, 0.005, 0.002] * 16)
    drawdown = _weekly_product("drawdown", [0.015, -0.025, 0.01, -0.03] * 16)

    result = rank_cta_products(
        CtaRankingRequest(
            products=[drawdown, stable],
            as_of_date=date(2024, 12, 31),
        )
    )

    assert result.ranking_only is True
    assert result.model_version == "codex-cta-score-v1.1"
    assert result.rankings[0].product_id == "stable"
    assert result.rankings[0].rank == 1
    assert result.rankings[0].score > result.rankings[1].score
    assert round(sum(result.rankings[0].dimension_scores.values()), 6) > 0
    assert sum(result.dimension_weights.values()) == 100


def test_codex_as_of_date_excludes_future_nav_backfill() -> None:
    stable = _weekly_product("stable", [0.004, 0.003, 0.005, 0.002] * 20)
    drawdown = _weekly_product("drawdown", [0.015, -0.025, 0.01, -0.03] * 20)
    future_points = list(stable.nav_points)
    future_points[-1] = future_points[-1].model_copy(
        update={"net_asset_value": future_points[-1].net_asset_value * 4.0}
    )
    backfilled = stable.model_copy(update={"nav_points": future_points})
    request = CtaRankingRequest(
        products=[drawdown, stable],
        as_of_date=date(2024, 12, 31),
    )
    baseline = rank_cta_products(request)
    changed_future = rank_cta_products(request.model_copy(update={
        "products": [drawdown, backfilled],
    }))

    assert changed_future.as_of_date == baseline.as_of_date
    assert changed_future.nav_fingerprint == baseline.nav_fingerprint
    assert [
        (item.product_id, item.rank, item.score)
        for item in changed_future.rankings
    ] == [
        (item.product_id, item.rank, item.score)
        for item in baseline.rankings
    ]


def test_codex_short_sample_is_returned_without_a_formal_rank() -> None:
    short = _weekly_product("short", [0.005] * 20)
    stable = _weekly_product("stable", [0.004, 0.003, 0.005, 0.002] * 16)

    result = rank_cta_products(
        CtaRankingRequest(
            products=[short, stable],
            as_of_date=date(2024, 12, 31),
        )
    )

    short_result = next(item for item in result.rankings if item.product_id == "short")
    assert short_result.eligible is False
    assert short_result.rank is None
    assert short_result.score is None
    assert short_result.status == "insufficient_data"
    assert result.eligible_count == 1


def test_uncovered_market_dimensions_are_omitted_from_the_score() -> None:
    products = [
        _weekly_product("stable", [0.004, 0.003, 0.005, 0.002] * 16),
        _weekly_product("drawdown", [0.015, -0.025, 0.01, -0.03] * 16),
    ]

    result = rank_cta_products(CtaRankingRequest(products=products, as_of_date=date(2024, 12, 31)))

    tail = next(dimension for dimension in result.rankings[0].dimensions if dimension.dimension == "tail")
    assert tail.raw_score is None
    assert tail.adjusted_score is None
    assert tail.covered_metric_count == 0
    assert "tail" not in result.rankings[0].dimension_scores


def _phase_d_evidence(*, sensitivity_stable: bool, stable_improvement: bool) -> CtaRankingAttributionEvidence:
    return CtaRankingAttributionEvidence(
        source="immutable_phase_d_snapshot",
        snapshot_id="snapshot-d",
        model_version="cta-attribution-phase-d-v1.0",
        nav_value_signature="signature",
        as_of_date=date(2024, 12, 31),
        observation_count=64,
        status="available",
        evaluated_segments=5,
        minimum_segments=3,
        stable_improvement=stable_improvement,
        sensitivity_stable=sensitivity_stable,
        selected_from_sensitivity=False,
    )


def test_phase_d_evidence_enters_robustness_without_directly_scoring_nonlinear_uplift() -> None:
    products = [
        _weekly_product("stable-evidence", [0.003, 0.001, 0.002, 0.004] * 16),
        _weekly_product("unstable-evidence", [0.002, 0.001, 0.003, 0.002] * 16),
        _weekly_product("no-evidence", [0.001, 0.002, 0.001, 0.003] * 16),
    ]
    evidence = {
        "stable-evidence": _phase_d_evidence(sensitivity_stable=True, stable_improvement=True),
        "unstable-evidence": _phase_d_evidence(sensitivity_stable=False, stable_improvement=True),
    }
    request = CtaRankingRequest(
        products=products,
        as_of_date=date(2024, 12, 31),
        attribution_evidence=evidence,
    )

    result = rank_cta_products(request)

    stable = next(item for item in result.rankings if item.product_id == "stable-evidence")
    unstable = next(item for item in result.rankings if item.product_id == "unstable-evidence")
    missing = next(item for item in result.rankings if item.product_id == "no-evidence")
    stable_dimension = next(item for item in stable.dimensions if item.dimension == "robustness")
    unstable_dimension = next(item for item in unstable.dimensions if item.dimension == "robustness")
    missing_dimension = next(item for item in missing.dimensions if item.dimension == "robustness")

    assert stable_dimension.metric_values["oos_validation"] == 100.0
    assert unstable_dimension.metric_values["oos_validation"] == 50.0
    assert "oos_validation" not in missing_dimension.metric_values
    assert result.method_provenance["robustness"]["nonlinear_increment_direct_score"] is False
    assert result.method_provenance["robustness"]["missing_phase_d_evidence_policy"] == "omit_metric"


def test_phase_d_evidence_is_not_used_when_insufficient() -> None:
    product = _weekly_product("insufficient-evidence", [0.002, 0.001, 0.003, 0.002] * 16)
    evidence = _phase_d_evidence(sensitivity_stable=True, stable_improvement=False).model_copy(update={
        "status": "insufficient",
        "evaluated_segments": 1,
    })

    result = rank_cta_products(CtaRankingRequest(
        products=[product],
        as_of_date=date(2024, 12, 31),
        attribution_evidence={product.product_id: evidence},
    ))
    dimension = next(item for item in result.rankings[0].dimensions if item.dimension == "robustness")

    assert "oos_validation" not in dimension.metric_values
    assert any("Phase D" in warning for warning in result.rankings[0].warnings)
