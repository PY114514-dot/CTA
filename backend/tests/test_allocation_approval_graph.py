from app.services.allocation_agent_graph import (
    complete_allocation_approval,
    request_allocation_approval,
    score_allocation_draft,
)


def _draft() -> dict:
    return {
        "allocations": [{"weight": 0.5}, {"weight": 0.5}],
        "portfolio": {"metrics": {
            "annualized_return": 0.12,
            "annualized_volatility": 0.10,
            "maximum_drawdown": -0.08,
            "sharpe_ratio": 1.1,
        }},
        "interpreted_constraints": {
            "max_products": 5,
            "min_annualized_return": 0.10,
            "max_annualized_volatility": 0.15,
        },
    }


def test_allocation_approval_interrupts_before_scoring() -> None:
    draft = _draft()
    started = request_allocation_approval("approval-test", draft)
    assert started.waiting_for_human is True

    completed = complete_allocation_approval("approval-test", approved=True, content=draft)
    assert completed.approved is True
    assert completed.score is not None
    assert completed.score["total"] > 0
    assert completed.version == {"number": 1, "status": "approved", "tracking_enabled": True}


def test_allocation_score_penalizes_failed_constraints() -> None:
    draft = _draft()
    draft["interpreted_constraints"]["max_products"] = 1
    score = score_allocation_draft(draft)
    assert score["components"]["约束满足"] < 15
