from datetime import date, timedelta

import numpy as np

from app.services.candidate_factor_gate import evaluate_candidate_factor
from app.routers.cta_attribution import _candidate_factor_names_for_product


def _dates(count: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=7 * index) for index in range(count)]


def test_gate_admits_a_stable_incremental_candidate() -> None:
    rng = np.random.default_rng(7)
    candidate = rng.normal(0.0, 0.01, 110)
    baseline = rng.normal(0.0, 0.01, 110)
    returns = 0.8 * candidate + rng.normal(0.0, 0.001, 110)

    result = evaluate_candidate_factor(returns, np.column_stack([baseline, candidate]), _dates(110), ["baseline", "candidate"], "candidate")

    assert result["admitted"] is True
    assert result["summary"]["evaluated_segments"] == 4


def test_gate_keeps_unstable_candidate_as_research_only() -> None:
    rng = np.random.default_rng(8)
    baseline = rng.normal(0.0, 0.01, 110)
    candidate = rng.normal(0.0, 0.01, 110)
    returns = 0.7 * baseline + rng.normal(0.0, 0.002, 110)

    result = evaluate_candidate_factor(returns, np.column_stack([baseline, candidate]), _dates(110), ["baseline", "candidate"], "candidate")

    assert result["admitted"] is False
    assert result["status"] == "candidate"


def test_gate_requires_candidate_model_to_have_non_negative_average_oos_fit(monkeypatch) -> None:
    returns = np.linspace(-0.01, 0.01, 110)
    factors = np.column_stack([returns, returns])
    monkeypatch.setattr("app.services.candidate_factor_gate._oos_r2", lambda *_args: -0.1)

    result = evaluate_candidate_factor(returns, factors, _dates(110), ["baseline", "candidate"], "candidate")

    assert result["admitted"] is False


def test_explicit_disclosure_limits_calendar_spread_candidate() -> None:
    assert _candidate_factor_names_for_product("商品套利", {"arbitrage_type": "跨品种"}) == ()
    assert _candidate_factor_names_for_product("商品套利", {"arbitrage_type": "跨期"}) == ("calendar_spread_momentum",)
