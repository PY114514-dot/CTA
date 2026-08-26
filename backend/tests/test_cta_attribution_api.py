"""HTTP contract tests for the reviewed-NAV CTA Phase-A endpoint."""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.routers.cta_attribution import (
    CtaAttributionSnapshotRequest,
    create_cta_attribution_snapshot,
    evaluate_dynamic_product,
    evaluate_regime_product,
    evaluate_reviewed_product,
    get_cta_attribution_snapshot,
    list_latest_phase_d_evidence,
    list_cta_attribution_snapshots,
)
from app.services import product_store
from app.services.cta_factor_bundle import CTA_FACTOR_NAMES


@pytest.fixture()
def session(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def regression(**_kwargs):
        return SimpleNamespace(
            factors=[SimpleNamespace(
                name="trend", display_name="趋势", factor_group="trend", beta=0.4,
                t_stat=2.2, p_value=0.03, bootstrap_ci_low=0.1, bootstrap_ci_high=0.7,
                contribution_pct=42.0, mean_return_contribution=0.0042,
            )],
            r_squared=0.36, adj_r_squared=0.31, annualized_alpha=0.02,
            out_of_sample={"r_squared": 0.1}, warnings=[],
            factor_risk_contributions={"trend": 0.015},
        )

    monkeypatch.setattr("app.routers.cta_attribution.run_factor_regression", regression)
    with factory() as database_session:
        yield database_session


def _add_nav(session, product_id: str, *, reviewed: bool) -> None:
    points = []
    nav = 1.0
    for index in range(20):
        nav *= 1.001
        points.append({"observation_date": date(2024, 1, 5) + timedelta(days=index * 7), "nav": nav})
    product_store.add_nav_observations(session, product_id, points, frequency="weekly")
    if reviewed:
        observations = product_store.get_nav_series(session, product_id)
        product_store.review_nav_observations(session, [row.id for row in observations], "reviewed", "tester")


def test_phase_a_rejects_unconfirmed_product(session) -> None:
    product = product_store.create_product(session, standard_name="待确认 CTA")
    _add_nav(session, product.id, reviewed=True)

    with pytest.raises(HTTPException, match="已确认") as error:
        evaluate_reviewed_product(product.id, session)

    assert error.value.status_code == 422


def test_phase_a_rejects_unreviewed_nav(session) -> None:
    product = product_store.create_product(session, standard_name="未审核 CTA")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=False)

    with pytest.raises(HTTPException, match="已审核净值") as error:
        evaluate_reviewed_product(product.id, session)

    assert error.value.status_code == 422


def test_phase_a_uses_only_reviewed_nav_and_returns_separate_tables(session) -> None:
    product = product_store.create_product(session, standard_name="已审核 CTA")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)

    body = evaluate_reviewed_product(product.id, session)
    assert body["data_contract"]["parsing_triggered"] is False
    assert body["data_contract"]["reviewed_nav_count"] == 20
    assert body["attribution"]["factor_exposure"][0]["beta"] == 0.4
    assert body["attribution"]["return_contribution"][0]["mean_return_contribution"] == 0.0042
    assert body["attribution"]["euler_risk_contribution"][0]["component_risk_contribution"] == 0.015


def test_phase_a_uses_the_registered_cta_factor_bundle(monkeypatch, session) -> None:
    product = product_store.create_product(session, standard_name="CTA 因子合同")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)
    from app.routers import cta_attribution

    captured: dict[str, object] = {}
    original = cta_attribution.run_factor_regression

    def regression(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cta_attribution, "run_factor_regression", regression)

    body = cta_attribution.evaluate_reviewed_product(product.id, session)

    assert captured["factor_names"] == list(CTA_FACTOR_NAMES)
    assert callable(captured["factor_series_loader"])
    assert body["factor_bundle"]["bundle_version"] == "cta_factor_bundle_v1"


def test_phase_a_keeps_commodity_arbitrage_candidate_out_of_formal_model_before_admission(monkeypatch, session) -> None:
    product = product_store.create_product(session, standard_name="商品套利", strategy="商品套利")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)
    from app.routers import cta_attribution

    captured: dict[str, object] = {}
    original = cta_attribution.run_factor_regression

    def regression(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cta_attribution, "run_factor_regression", regression)
    cta_attribution.evaluate_reviewed_product(product.id, session)

    assert captured["factor_names"] == list(CTA_FACTOR_NAMES)


def test_phase_b_rejects_unconfirmed_product(session) -> None:
    product = product_store.create_product(session, standard_name="待确认动态 CTA")
    _add_nav(session, product.id, reviewed=True)

    with pytest.raises(HTTPException, match="已确认") as error:
        evaluate_dynamic_product(product.id, session)

    assert error.value.status_code == 422


def test_phase_b_returns_causal_dynamic_and_rolling_beta(monkeypatch, session) -> None:
    product = product_store.create_product(session, standard_name="动态 CTA")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)

    aligned = {
        "returns": __import__("numpy").array([0.001] * 20),
        "factor_returns": __import__("numpy").array([[0.002] for _ in range(20)]),
        "dates": [date(2024, 1, 5) + timedelta(days=index * 7) for index in range(20)],
        "factor_names": ["trend"],
        "warnings": ["测试因子缓存"],
        "observation_count": 20,
    }
    monkeypatch.setattr("app.routers.cta_attribution.prepare_factor_matrix", lambda *args, **kwargs: aligned)
    monkeypatch.setattr("app.routers.cta_attribution.compute_rolling_beta", lambda *args, **kwargs: {"windows": [{"window": 26, "status": "insufficient", "n_periods": 0, "paths": [], "warnings": ["样本不足"]}]})
    monkeypatch.setattr("app.routers.cta_attribution.filter_dynamic_beta", lambda *args, **kwargs: {
        "method": "causal_random_walk_kalman_filter",
        "factor_names": ["trend"],
        "parameters": {"smoother_used": False},
        "long_run_style_beta": {"trend": 0.5},
        "beta_range": {"trend": 0.2},
        "paths": [{"date": "2024-02-16", "betas": {"trend": 0.6}}],
        "warnings": ["未使用未来数据"],
    })

    body = evaluate_dynamic_product(product.id, session)

    assert body["data_contract"]["parsing_triggered"] is False
    assert body["data_contract"]["reviewed_nav_count"] == 20
    assert body["alignment"]["factor_cache_profile"] == "baseline"
    assert body["kalman_beta"]["parameters"]["smoother_used"] is False
    assert body["rolling_beta"]["windows"][0]["status"] == "insufficient"
    assert body["warnings"]


def test_phase_c_returns_auditable_regime_results_without_parsing(monkeypatch, session) -> None:
    product = product_store.create_product(session, standard_name="状态 CTA")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)
    aligned = {
        "returns": __import__("numpy").array([0.001] * 20),
        "factor_returns": __import__("numpy").array([[0.002] for _ in range(20)]),
        "dates": [date(2024, 1, 5) + timedelta(days=index * 7) for index in range(20)],
        "factor_names": ["trend"],
        "warnings": [],
        "observation_count": 20,
    }
    monkeypatch.setattr("app.routers.cta_attribution.prepare_factor_matrix", lambda *args, **kwargs: aligned)
    monkeypatch.setattr("app.routers.cta_attribution.classify_observable_regimes", lambda *args, **kwargs: {
        "method": "observable_lagged_regime_rules",
        "parameters": {"uses_future_data": False, "hmm_used": False},
        "states": ["normal"] * 20,
        "state_counts": {"normal": 20, "crisis": 0, "whipsaw": 0, "insufficient": 0},
        "features": [],
        "warnings": [],
    })
    monkeypatch.setattr("app.routers.cta_attribution.run_regime_attribution", lambda *args, **kwargs: {
        "method": "conditional_ols_by_observable_state",
        "parameters": {"min_regime_observations": 20},
        "regimes": [{"state": "normal", "status": "available", "observation_count": 20, "conditional_beta": {"trend": 0.5}, "crisis_alpha_candidate": None, "conditional_correlation": 0.3}],
        "warnings": [],
    })

    body = evaluate_regime_product(product.id, session)

    assert body["data_contract"]["parsing_triggered"] is False
    assert body["state_proxy"]["source"] == "aligned baseline factor basket"
    assert body["regime_rules"]["parameters"]["uses_future_data"] is False
    assert body["regime_attribution"]["regimes"][0]["conditional_beta"]["trend"] == 0.5


def test_phase_d_returns_oos_nonlinear_increment_without_parsing(monkeypatch, session) -> None:
    product = product_store.create_product(session, standard_name="非线性 CTA")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)
    aligned = {
        "returns": __import__("numpy").array([0.001] * 20),
        "factor_returns": __import__("numpy").array([[0.002, 0.01] for _ in range(20)]),
        "dates": [date(2024, 1, 5) + timedelta(days=index * 7) for index in range(20)],
        "factor_names": ["short_term_trend_20", "volatility_state"],
        "warnings": ["测试因子缓存"],
        "observation_count": 20,
    }
    monkeypatch.setattr("app.routers.cta_attribution.prepare_factor_matrix", lambda *args, **kwargs: aligned)
    monkeypatch.setattr("app.routers.cta_attribution.evaluate_momentum_volatility_increment", lambda *args, **kwargs: {
        "method": "pre_registered_oos_momentum_x_volatility_nested_ols",
        "frequency": "weekly",
        "factor_names": ["momentum", "volatility", "momentum_x_volatility"],
        "status": "insufficient",
        "parameters": {"uses_future_data": False, "training_scheme": "expanding_window", "standardization": "train_window_only"},
        "segments": [],
        "summary": {"stable_improvement": False, "conclusion": "未发现可靠的动量 × 波动率样本外增量", "mean_r2_delta": None},
        "sensitivity": {"window_sensitivity": [], "threshold_sensitivity": [], "model_parameter_sensitivity": []},
        "warnings": ["样本不足"],
    })

    from app.routers.cta_attribution import evaluate_nonlinear_product

    body = evaluate_nonlinear_product(product.id, session)

    assert body["data_contract"]["parsing_triggered"] is False
    assert body["data_contract"]["reviewed_nav_count"] == 20
    assert body["alignment"]["factor_cache_profile"] == "baseline"
    assert body["nonlinear_increment"]["summary"]["conclusion"] == "未发现可靠的动量 × 波动率样本外增量"


def test_attribution_snapshot_is_immutable_and_idempotent(session) -> None:
    product = product_store.create_product(session, standard_name="快照 CTA")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)
    request = CtaAttributionSnapshotRequest(product_id=product.id, phase="phase-a")

    first = create_cta_attribution_snapshot(request, session)
    second = create_cta_attribution_snapshot(request, session)
    listed = list_cta_attribution_snapshots(product_id=product.id, session=session)
    detail = get_cta_attribution_snapshot(first["snapshot_id"], session)

    assert first["snapshot_id"] == second["snapshot_id"]
    assert second["idempotent"] is True
    assert len(listed) == 1
    assert detail["nav_fingerprint"] == first["nav_fingerprint"]
    assert detail["results"]["data_contract"]["parsing_triggered"] is False
    evidence = detail["evidence_package"]
    assert evidence["purpose"] == "reviewable_attribution_evidence"
    assert evidence["data_lineage"]["reviewed_observation_count"] == 20
    assert evidence["data_lineage"]["source_unlinked_observation_count"] == 20
    assert evidence["claims"][0]["claim"].endswith("统计暴露。")
    assert evidence["llm_handoff"]["images_included"] is False


def test_phase_d_snapshot_is_immutable_and_idempotent(monkeypatch, session) -> None:
    product = product_store.create_product(session, standard_name="非线性快照 CTA")
    product_store.confirm_product(session, product.id)
    _add_nav(session, product.id, reviewed=True)
    monkeypatch.setattr("app.routers.cta_attribution.evaluate_nonlinear_product", lambda product_id, session: {
        "product_id": product_id,
        "product_name": "非线性快照 CTA",
        "frequency": "weekly",
        "data_contract": {"confirmation_status": "confirmed", "reviewed_nav_count": 20, "source": "confirmed + reviewed NAV only", "parsing_triggered": False},
        "alignment": {"factor_cache_profile": "baseline", "factor_names": ["trend"], "aligned_observation_count": 19, "start_date": "2024-01-12", "end_date": "2024-05-10"},
        "nonlinear_increment": {"factor_names": ["trend"], "parameters": {"uses_future_data": False}, "summary": {"conclusion": "未发现可靠非线性增量"}, "warnings": []},
        "warnings": [],
    })

    request = CtaAttributionSnapshotRequest(product_id=product.id, phase="phase-d")
    first = create_cta_attribution_snapshot(request, session)
    second = create_cta_attribution_snapshot(request, session)
    detail = get_cta_attribution_snapshot(first["snapshot_id"], session)

    assert first["snapshot_id"] == second["snapshot_id"]
    assert second["idempotent"] is True
    assert detail["phase"] == "phase-d"
    assert detail["results"]["nonlinear_increment"]["summary"]["conclusion"] == "未发现可靠非线性增量"


def test_latest_phase_d_evidence_reads_frozen_snapshot_and_applies_cutoff(session) -> None:
    product = product_store.create_product(session, standard_name="排名证据 CTA")
    product_store.create_snapshot(
        session,
        label=f"cta-attribution:phase-d:{product.id}:navsig:factorsig",
        content={
            "snapshot_type": "cta_dynamic_attribution",
            "product_id": product.id,
            "model_version": "cta-attribution-phase-d-v1.0",
            "as_of_date": "2024-06-28",
            "nav_fingerprint": "navsig",
            "parameters": {"reviewed_nav_count": 64},
            "results": {
                "nonlinear_increment": {
                    "status": "available",
                    "parameters": {"min_segments": 3},
                    "summary": {"evaluated_delta_count": 5, "stable_improvement": True},
                    "sensitivity": {
                        "window_sensitivity": [{"status": "available", "stable_improvement": True}],
                        "threshold_sensitivity": [{"stable_improvement": True}],
                        "model_parameter_sensitivity": [{"status": "available", "stable_improvement": True}],
                        "selection_policy": {"selected_from_sensitivity": False},
                    },
                },
            },
        },
    )

    before_cutoff = list_latest_phase_d_evidence(
        product_ids=product.id,
        as_of_date=date(2024, 6, 27),
        session=session,
    )
    evidence = list_latest_phase_d_evidence(
        product_ids=product.id,
        as_of_date=date(2024, 12, 31),
        session=session,
    )

    assert before_cutoff == {}
    assert evidence[product.id]["source"] == "immutable_phase_d_snapshot"
    assert evidence[product.id]["snapshot_id"]
    assert evidence[product.id]["sensitivity_stable"] is True
    assert evidence[product.id]["evaluated_segments"] == 5
