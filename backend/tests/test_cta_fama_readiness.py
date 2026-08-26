from datetime import date, timedelta

from app.schemas import CtaFamaProductInput, CtaFamaReadinessRequest, DataFrequency, NetAssetValuePoint
from app.services.cta_fama_readiness import evaluate_cta_fama_readiness


def _product(index: int, *, exited: bool = False, point_in_time: bool = True) -> CtaFamaProductInput:
    points = [
        NetAssetValuePoint(
            observation_date=date(2019, 1, 4) + timedelta(days=7 * offset),
            net_asset_value=1.0 + 0.0001 * offset,
        )
        for offset in range(53)
    ]
    return CtaFamaProductInput(
        product_id=f"P{index}",
        frequency=DataFrequency.WEEKLY,
        nav_points=points,
        source_group=f"manager-{index // 2}",
        exit_date=points[-1].observation_date if exited else None,
        point_in_time_verified=point_in_time,
    )


def test_cta_fama_gate_is_research_ready_only_when_all_gates_pass() -> None:
    request = CtaFamaReadinessRequest(
        products=[_product(index, exited=index == 0) for index in range(30)],
        exit_history_complete=True,
        survivorship_audit_passed=True,
        backfill_audit_passed=True,
        same_source_deduplicated=True,
        oos_state_segments=3,
    )

    result = evaluate_cta_fama_readiness(request)

    assert result["status"] == "ready_for_research"
    assert result["production_ready"] is False
    assert result["gates"]["cross_section"]["status"] == "passed"
    assert result["gates"]["exit_history"]["status"] == "passed"
    assert result["policy"] == "research_design_only"


def test_cta_fama_gate_blocks_short_survivor_only_universe() -> None:
    request = CtaFamaReadinessRequest(
        products=[_product(index, exited=False, point_in_time=False) for index in range(5)],
        exit_history_complete=False,
        survivorship_audit_passed=False,
        backfill_audit_passed=False,
        same_source_deduplicated=False,
        oos_state_segments=0,
    )

    result = evaluate_cta_fama_readiness(request)

    assert result["status"] == "blocked"
    assert result["production_ready"] is False
    assert result["gates"]["cross_section"]["status"] == "failed"
    assert result["gates"]["exit_history"]["status"] == "failed"
    assert any("幸存者" in warning or "退出" in warning for warning in result["warnings"])
