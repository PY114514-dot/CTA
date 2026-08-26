"""Build reviewable evidence for CTA attribution results.

The package deliberately contains computed facts and provenance, not an LLM
judgement.  A presentation layer may turn it into prose, but cannot use it to
alter NAV observations or attribution outputs.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DataSnapshot, DocumentFragment, RawFile


def build_attribution_evidence_package(
    session: Session,
    *,
    phase: str,
    observations: list[Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact data lineage and bounded claims for one result."""
    file_ids = {item.source_file_id for item in observations if item.source_file_id}
    fragment_ids = {item.source_fragment_id for item in observations if item.source_fragment_id}
    files = {
        item.id: item
        for item in session.execute(select(RawFile).where(RawFile.id.in_(file_ids))).scalars()
    } if file_ids else {}
    fragments = {
        item.id: item
        for item in session.execute(select(DocumentFragment).where(DocumentFragment.id.in_(fragment_ids))).scalars()
    } if fragment_ids else {}
    linked_count = sum(1 for item in observations if item.source_file_id or item.source_fragment_id)
    claims = _claims_for_phase(phase, result)
    return {
        "purpose": "reviewable_attribution_evidence",
        "data_lineage": {
            "reviewed_observation_ids": [item.id for item in observations],
            "reviewed_observation_count": len(observations),
            "date_range": {
                "start": observations[0].observation_date.isoformat(),
                "end": observations[-1].observation_date.isoformat(),
            },
            "source_linked_observation_count": linked_count,
            "source_unlinked_observation_count": len(observations) - linked_count,
            "source_files": [
                {
                    "file_id": item.id,
                    "filename": item.filename,
                    "file_hash": item.file_hash,
                    "version": item.version,
                    "report_period": item.report_period,
                }
                for item in files.values()
            ],
            "source_fragments": [
                {
                    "fragment_id": item.id,
                    "file_id": item.file_id,
                    "page_number": item.page_number,
                    "fragment_type": item.fragment_type,
                    "bbox": item.bbox,
                }
                for item in fragments.values()
            ],
        },
        "claims": claims,
        "llm_handoff": {
            "allowed_use": "Explain the supplied computed claims and their stated limitations.",
            "forbidden_use": [
                "Do not calculate, replace, infer, or approve NAV observations.",
                "Do not convert statistical exposure into actual holdings, trades, or manager skill.",
                "Do not make investment recommendations from this package.",
            ],
            "images_included": False,
            "image_policy": "Only send a separately selected source crop when a human asks the model to interpret visible chart semantics; never use an image as numeric ground truth.",
        },
    }


def _claims_for_phase(phase: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    if phase == "phase-a":
        return _static_claims(result)
    if phase == "phase-b":
        return _dynamic_claims(result)
    if phase == "phase-c":
        return _regime_claims(result)
    if phase == "phase-d":
        return _nonlinear_claims(result)
    return []


def _static_claims(result: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = result.get("baseline", {})
    out_of_sample = baseline.get("out_of_sample", {}).get("r_squared")
    claims: list[dict[str, Any]] = []
    for item in result.get("attribution", {}).get("factor_exposure", [])[:3]:
        beta = item.get("beta")
        p_value = item.get("hac_p_value")
        lower, upper = item.get("bootstrap_ci_low"), item.get("bootstrap_ci_high")
        ci_excludes_zero = lower is not None and upper is not None and (lower > 0 or upper < 0)
        confidence = "medium" if p_value is not None and p_value < 0.05 and ci_excludes_zero and out_of_sample is not None else "low"
        direction = "正向" if (beta or 0) >= 0 else "负向"
        claims.append({
            "claim": f"当前样本中，产品收益与{item.get('display_name', item.get('factor_name'))}呈{direction}统计暴露。",
            "confidence": confidence,
            "supporting_evidence": [
                f"Beta={_number(beta)}，HAC p={_number(p_value)}",
                f"bootstrap CI=[{_number(lower)}, {_number(upper)}]",
                f"样本内 R²={_number(baseline.get('r_squared'))}；样本外 R²={_number(out_of_sample)}",
            ],
            "counter_evidence": [
                "公开因子回归仅描述统计关系，不证明真实持仓或交易行为。",
                "样本外拟合、因子共线性和未覆盖的交易成本可能削弱解释力。",
            ],
        })
    return claims


def _dynamic_claims(result: dict[str, Any]) -> list[dict[str, Any]]:
    kalman = result.get("kalman_beta", {})
    ranges = kalman.get("beta_range", {})
    if not ranges:
        return []
    factor, value = max(ranges.items(), key=lambda item: abs(item[1] or 0))
    return [{
        "claim": f"{factor} 的因果过滤 Beta 在样本内出现了可观测变化。",
        "confidence": "low",
        "supporting_evidence": [f"Beta 路径范围={_number(value)}", "Kalman 过滤未使用未来数据。"],
        "counter_evidence": ["路径变化可能来自估计误差或代理因子变化，不等同于管理人主动择时。"],
    }]


def _regime_claims(result: dict[str, Any]) -> list[dict[str, Any]]:
    regimes = result.get("regime_attribution", {}).get("regimes", [])
    available = [item for item in regimes if item.get("status") == "available"]
    if not available:
        return []
    strongest = max(available, key=lambda item: item.get("observation_count", 0))
    return [{
        "claim": f"{strongest.get('state')} 状态下的条件统计结果可供复核。",
        "confidence": "low",
        "supporting_evidence": [
            f"状态样本数={strongest.get('observation_count', 0)}",
            f"条件相关={_number(strongest.get('conditional_correlation'))}",
        ],
        "counter_evidence": ["状态由公开因子代理和滞后规则定义，不是产品真实持仓状态。"],
    }]


def _nonlinear_claims(result: dict[str, Any]) -> list[dict[str, Any]]:
    increment = result.get("nonlinear_increment", {})
    summary = increment.get("summary", {})
    stable = bool(summary.get("stable_improvement", False))
    return [{
        "claim": str(summary.get("conclusion") or "未形成非线性结论"),
        "confidence": "medium" if stable else "low",
        "supporting_evidence": [
            f"连续 OOS 评估段={summary.get('evaluated_delta_count', 0)}",
            f"平均 R² 增量={_number(summary.get('mean_r2_delta'))}",
        ],
        "counter_evidence": [
            "敏感性分析仅用于诊断，不能作为挑选模型或扩大结论的依据。",
            "非线性预测增量不等同于收益归因或可交易优势。",
        ],
    }]


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _phase_d_sensitivity_is_stable(sensitivity: dict) -> bool:
    """Require every reported sensitivity scenario to be available and stable."""
    if not sensitivity:
        return False
    if "stable_across_pre_registered_scenarios" in sensitivity:
        return bool(sensitivity["stable_across_pre_registered_scenarios"])
    scenarios = []
    for key in ("window_sensitivity", "threshold_sensitivity", "model_parameter_sensitivity"):
        family = sensitivity.get(key, [])
        if not isinstance(family, list) or not family:
            return False
        scenarios.extend(item for item in family if isinstance(item, dict))
    if len(scenarios) == 0:
        return False
    return all(
        item.get("status") in {None, "available"}
        and bool(item.get("stable_improvement", False))
        for item in scenarios
    )


def load_latest_phase_d_evidence_payload(
    session: Session,
    product_ids: set[str] | None = None,
    as_of_date: date | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the newest frozen Phase-D evidence per product, never recomputing.

    The selection rule is identical to the HTTP endpoint
    ``GET /api/cta-attribution/snapshots/latest-phase-d``: only the newest
    snapshot whose own NAV cutoff is not after the requested cutoff is kept.
    Ranking callers feed the returned payload into
    ``CtaRankingAttributionEvidence.model_validate``.
    """
    statement = select(DataSnapshot).where(DataSnapshot.label.like("cta-attribution:phase-d:%"))
    snapshots = session.execute(
        statement.order_by(DataSnapshot.created_at.desc())
    ).scalars().all()
    selected: dict[str, tuple[date, DataSnapshot]] = {}
    for snapshot in snapshots:
        content = snapshot.content or {}
        product_id = str(content.get("product_id", ""))
        if product_ids is not None and product_id not in product_ids:
            continue
        snapshot_date_raw = content.get("as_of_date")
        if not product_id or not snapshot_date_raw:
            continue
        try:
            snapshot_date = date.fromisoformat(str(snapshot_date_raw))
        except ValueError:
            continue
        if as_of_date is not None and snapshot_date > as_of_date:
            continue
        previous = selected.get(product_id)
        if previous is not None and snapshot_date <= previous[0]:
            continue
        selected[product_id] = (snapshot_date, snapshot)

    from app.services.cta_attribution_snapshot import MODEL_VERSIONS

    result: dict[str, dict[str, Any]] = {}
    for product_id, (snapshot_date, snapshot) in selected.items():
        content = snapshot.content or {}
        nonlinear = content.get("results", {}).get("nonlinear_increment", {})
        summary = nonlinear.get("summary", {})
        sensitivity = nonlinear.get("sensitivity", {})
        selection_policy = sensitivity.get("selection_policy", {})
        parameters = nonlinear.get("parameters", {})
        result[product_id] = {
            "source": "immutable_phase_d_snapshot",
            "snapshot_id": snapshot.id,
            "model_version": content.get("model_version", MODEL_VERSIONS["phase-d"]),
            "nav_value_signature": content.get("nav_fingerprint", ""),
            "as_of_date": snapshot_date.isoformat(),
            "observation_count": int(content.get("parameters", {}).get("reviewed_nav_count", 0)),
            "status": str(nonlinear.get("status", "insufficient")),
            "evaluated_segments": int(summary.get("evaluated_delta_count", 0)),
            "minimum_segments": int(parameters.get("min_segments", 1)),
            "stable_improvement": bool(summary.get("stable_improvement", False)),
            "sensitivity_stable": _phase_d_sensitivity_is_stable(sensitivity),
            "selected_from_sensitivity": bool(selection_policy.get("selected_from_sensitivity", False)),
        }
    return result
