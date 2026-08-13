"""Read-only Phase-A CTA evaluation from confirmed, reviewed NAV."""

from datetime import date

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import ConfirmationStatus, DataSnapshot
from app.services import product_store
from app.services.cta_attribution_snapshot import MODEL_VERSIONS, build_snapshot_content
from app.services.cta_evaluation import build_attribution_tables, evaluate_nav_path
from app.services.dynamic_beta import (
    DYNAMIC_FACTOR_NAMES,
    compute_rolling_beta,
    filter_dynamic_beta,
    prepare_factor_matrix,
)
from app.services.regime_attribution import classify_observable_regimes, run_regime_attribution
from app.services.nonlinear_attribution import evaluate_nonlinear_increment
from app.services.factor_library.factor_regression import run_factor_regression

router = APIRouter(prefix="/api/cta-attribution", tags=["CTA 动态归因"])


class CtaAttributionSnapshotRequest(BaseModel):
    """Select one reviewed-NAV attribution phase to freeze."""

    product_id: str = Field(min_length=1, max_length=80)
    phase: str = Field(pattern=r"^phase-[abcd]$")


def _phase_function(phase: str):
    return {
        "phase-a": evaluate_reviewed_product,
        "phase-b": evaluate_dynamic_product,
        "phase-c": evaluate_regime_product,
        "phase-d": evaluate_nonlinear_product,
    }.get(phase)


@router.get("/products/{product_id}/phase-a")
def evaluate_reviewed_product(product_id: str, session: Session = Depends(get_session)) -> dict:
    """Never parses files: evaluates only confirmed, human-reviewed NAV."""
    product = product_store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    if product.confirmation_status != ConfirmationStatus.CONFIRMED:
        raise HTTPException(422, "仅已确认产品可进行 CTA 评价")
    observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
    if len(observations) < 20:
        raise HTTPException(422, "至少需要 20 条已审核净值才能进行 Phase A 评价")
    frequency = product.nav_frequency or observations[0].frequency or "weekly"
    if frequency not in {"daily", "weekly", "monthly"}:
        raise HTTPException(422, "净值频率必须为 daily、weekly 或 monthly")
    dates = [item.observation_date for item in observations]
    navs = [float(item.nav) for item in observations]
    returns = [right / left - 1.0 for left, right in zip(navs, navs[1:])]
    try:
        regression = run_factor_regression(
            product_returns=np.asarray(returns), product_dates=dates[1:], frequency=frequency,
            rolling_window={"daily": 60, "weekly": 26, "monthly": 12}[frequency],
            oos_train_window={"daily": 60, "weekly": 26, "monthly": 24}[frequency],
            minimum_observations=19,
        )
    except Exception as error:
        raise HTTPException(500, f"CTA 因子基线计算失败：{error}") from error
    return {
        "product_id": product.id, "product_name": product.standard_name, "frequency": frequency,
        "data_contract": {"confirmation_status": product.confirmation_status, "reviewed_nav_count": len(observations),
                          "source": "confirmed + reviewed NAV only", "parsing_triggered": False},
        "performance_path": evaluate_nav_path(navs, dates, frequency),
        "attribution": build_attribution_tables(regression),
        "baseline": {"r_squared": regression.r_squared, "adj_r_squared": regression.adj_r_squared,
                     "annualized_alpha_candidate": regression.annualized_alpha, "out_of_sample": regression.out_of_sample,
                     "warnings": regression.warnings},
        "warnings": ["收益贡献、因子暴露和 Euler 风险贡献为不同统计对象，不代表真实持仓或真实 P&L。",
                     "当前阶段未纳入持仓、成交、费用、容量或交易成本数据。"],
    }


@router.get("/products/{product_id}/phase-b")
def evaluate_dynamic_product(product_id: str, session: Session = Depends(get_session)) -> dict:
    """Read-only Rolling/Kalman attribution over confirmed, reviewed NAV."""
    product = product_store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    if product.confirmation_status != ConfirmationStatus.CONFIRMED:
        raise HTTPException(422, "仅已确认产品可进行动态 CTA 归因")
    observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
    if len(observations) < 20:
        raise HTTPException(422, "至少需要 20 条已审核净值才能进行 Phase B 动态归因")
    frequency = product.nav_frequency or observations[0].frequency or "weekly"
    if frequency not in {"daily", "weekly", "monthly"}:
        raise HTTPException(422, "净值频率必须为 daily、weekly 或 monthly")
    dates = [item.observation_date for item in observations]
    navs = np.asarray([float(item.nav) for item in observations], dtype=float)
    returns = navs[1:] / navs[:-1] - 1.0
    return_dates = dates[1:]
    try:
        aligned = prepare_factor_matrix(returns, return_dates, frequency, DYNAMIC_FACTOR_NAMES)
        rolling = compute_rolling_beta(
            aligned["returns"], aligned["factor_returns"], aligned["dates"], aligned["factor_names"],
            windows=[{"daily": 60, "weekly": 26, "monthly": 12}[frequency],
                     {"daily": 120, "weekly": 52, "monthly": 24}[frequency]],
            minimum_window={"daily": 60, "weekly": 26, "monthly": 12}[frequency],
        )
        kalman = filter_dynamic_beta(
            aligned["returns"], aligned["factor_returns"], aligned["dates"], aligned["factor_names"],
        )
    except ValueError as error:
        raise HTTPException(422, f"动态 Beta 数据不足：{error}") from error
    except Exception as error:
        raise HTTPException(500, f"动态 Beta 计算失败：{error}") from error
    return {
        "product_id": product.id,
        "product_name": product.standard_name,
        "frequency": frequency,
        "data_contract": {
            "confirmation_status": product.confirmation_status,
            "reviewed_nav_count": len(observations),
            "source": "confirmed + reviewed NAV only",
            "parsing_triggered": False,
        },
        "alignment": {
            "factor_cache_profile": "baseline",
            "factor_names": aligned["factor_names"],
            "aligned_observation_count": aligned["observation_count"],
            "start_date": aligned["dates"][0].isoformat(),
            "end_date": aligned["dates"][-1].isoformat(),
        },
        "rolling_beta": rolling,
        "kalman_beta": kalman,
        "warnings": aligned["warnings"] + [
            "Rolling Beta 仅用于风格漂移监控；Kalman Beta 是因果过滤估计。",
            "动态贡献、长期风格 Beta 和漂移告警均不代表真实持仓或管理人确定性择时收益。",
        ],
    }


@router.get("/products/{product_id}/phase-c")
def evaluate_regime_product(product_id: str, session: Session = Depends(get_session)) -> dict:
    """Read-only observable-state attribution over confirmed, reviewed NAV."""
    product = product_store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    if product.confirmation_status != ConfirmationStatus.CONFIRMED:
        raise HTTPException(422, "仅已确认产品可进行状态 CTA 归因")
    observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
    if len(observations) < 20:
        raise HTTPException(422, "至少需要 20 条已审核净值才能进行 Phase C 状态归因")
    frequency = product.nav_frequency or observations[0].frequency or "weekly"
    if frequency not in {"daily", "weekly", "monthly"}:
        raise HTTPException(422, "净值频率必须为 daily、weekly 或 monthly")
    dates = [item.observation_date for item in observations]
    navs = np.asarray([float(item.nav) for item in observations], dtype=float)
    returns = navs[1:] / navs[:-1] - 1.0
    return_dates = dates[1:]
    try:
        aligned = prepare_factor_matrix(returns, return_dates, frequency, DYNAMIC_FACTOR_NAMES)
        # The first aligned factor is a deterministic market proxy from the
        # same baseline basket; it is not a disclosed product position.
        market_proxy = aligned["factor_returns"][:, 0]
        regime_rules = classify_observable_regimes(market_proxy, aligned["dates"], frequency)
        regime_attribution = run_regime_attribution(
            aligned["returns"], aligned["factor_returns"], aligned["dates"], aligned["factor_names"],
            regime_rules, frequency, market_proxy=market_proxy,
        )
    except ValueError as error:
        raise HTTPException(422, f"状态归因数据不足：{error}") from error
    except Exception as error:
        raise HTTPException(500, f"状态归因计算失败：{error}") from error
    return {
        "product_id": product.id,
        "product_name": product.standard_name,
        "frequency": frequency,
        "data_contract": {
            "confirmation_status": product.confirmation_status,
            "reviewed_nav_count": len(observations),
            "source": "confirmed + reviewed NAV only",
            "parsing_triggered": False,
        },
        "alignment": {
            "factor_cache_profile": "baseline",
            "factor_names": aligned["factor_names"],
            "aligned_observation_count": aligned["observation_count"],
            "start_date": aligned["dates"][0].isoformat(),
            "end_date": aligned["dates"][-1].isoformat(),
        },
        "state_proxy": {
            "source": "aligned baseline factor basket",
            "market_proxy_factor": aligned["factor_names"][0],
            "interpretation": "公开因子篮子的市场代理，不是产品真实持仓或真实市场状态标签。",
        },
        "regime_rules": regime_rules,
        "regime_attribution": regime_attribution,
        "warnings": aligned["warnings"] + [
            "Crisis Alpha 是危机条件下扣除公开因子后的统计残差，不代表管理人技能。",
            "条件 Beta 与条件相关性仅对当前公开代理和状态规则成立。",
        ],
    }


@router.get("/products/{product_id}/phase-d")
def evaluate_nonlinear_product(product_id: str, session: Session = Depends(get_session)) -> dict:
    """Read-only causal OOS nonlinear increment over confirmed, reviewed NAV."""
    product = product_store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    if product.confirmation_status != ConfirmationStatus.CONFIRMED:
        raise HTTPException(422, "仅已确认产品可进行非线性 CTA 归因")
    observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
    if len(observations) < 20:
        raise HTTPException(422, "至少需要 20 条已审核净值才能进行 Phase D 非线性归因")
    frequency = product.nav_frequency or observations[0].frequency or "weekly"
    if frequency not in {"daily", "weekly", "monthly"}:
        raise HTTPException(422, "净值频率必须为 daily、weekly 或 monthly")
    dates = [item.observation_date for item in observations]
    navs = np.asarray([float(item.nav) for item in observations], dtype=float)
    returns = navs[1:] / navs[:-1] - 1.0
    return_dates = dates[1:]
    train_window = {"daily": 60, "weekly": 26, "monthly": 12}[frequency]
    test_window = {"daily": 20, "weekly": 12, "monthly": 6}[frequency]
    try:
        aligned = prepare_factor_matrix(returns, return_dates, frequency, DYNAMIC_FACTOR_NAMES)
        nonlinear = evaluate_nonlinear_increment(
            aligned["returns"], aligned["factor_returns"], aligned["dates"], aligned["factor_names"],
            frequency=frequency, train_window=train_window, test_window=test_window,
            max_segments=5, min_segments=3, min_r2_uplift=0.01,
        )
    except ValueError as error:
        raise HTTPException(422, f"非线性增量数据不足：{error}") from error
    except Exception as error:
        raise HTTPException(500, f"非线性增量计算失败：{error}") from error
    return {
        "product_id": product.id,
        "product_name": product.standard_name,
        "frequency": frequency,
        "data_contract": {
            "confirmation_status": product.confirmation_status,
            "reviewed_nav_count": len(observations),
            "source": "confirmed + reviewed NAV only",
            "parsing_triggered": False,
        },
        "alignment": {
            "factor_cache_profile": "baseline",
            "factor_names": aligned["factor_names"],
            "aligned_observation_count": aligned["observation_count"],
            "start_date": aligned["dates"][0].isoformat(),
            "end_date": aligned["dates"][-1].isoformat(),
        },
        "nonlinear_increment": nonlinear,
        "warnings": aligned["warnings"] + [
            "非线性层只报告固定因子集合上的连续 OOS 预测增量，不将训练集拟合或模型重要性写成收益归因。",
            "无稳定 OOS 增益时保留线性基线，不升级为非线性结论。",
        ],
    }


@router.post("/snapshots")
def create_cta_attribution_snapshot(
    body: CtaAttributionSnapshotRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Compute one phase and persist an immutable, idempotent result."""
    product = product_store.get_product(session, body.product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    phase_function = _phase_function(body.phase)
    if phase_function is None:  # Defensive guard for future route changes.
        raise HTTPException(422, "不支持的 CTA 归因阶段")
    try:
        result = phase_function(body.product_id, session)
    except HTTPException:
        raise
    observations = product_store.get_nav_series(session, body.product_id, reviewed_only=True)
    if not observations:
        raise HTTPException(422, "没有可用于快照的已审核净值")
    label, model_version, content = build_snapshot_content(
        phase=body.phase, product=product, observations=observations, result=result,
    )
    existing = session.execute(
        select(DataSnapshot).where(DataSnapshot.label == label).limit(1)
    ).scalars().first()
    snapshot = existing or product_store.create_snapshot(session, label=label, content=content)
    frozen = snapshot.content
    return {
        "snapshot_id": snapshot.id,
        "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
        "snapshot_type": "cta_dynamic_attribution",
        "phase": body.phase,
        "model_version": model_version,
        "product_id": frozen.get("product_id"),
        "as_of_date": frozen.get("as_of_date"),
        "nav_fingerprint": frozen.get("nav_fingerprint"),
        "factor_data_version": frozen.get("factor_data_version"),
        "result": frozen.get("results", {}),
        "idempotent": existing is not None,
    }


@router.get("/snapshots")
def list_cta_attribution_snapshots(
    product_id: str | None = None,
    phase: str | None = None,
    limit: int = 20,
    session: Session = Depends(get_session),
) -> list[dict]:
    """List frozen CTA attribution results without recomputing them."""
    limit = max(1, min(limit, 100))
    statement = select(DataSnapshot).where(DataSnapshot.label.like("cta-attribution:%"))
    if product_id:
        statement = statement.where(DataSnapshot.label.like(f"cta-attribution:%:{product_id}:%"))
    if phase:
        if phase not in MODEL_VERSIONS:
            raise HTTPException(422, "不支持的 CTA 归因阶段")
        statement = statement.where(DataSnapshot.label.like(f"cta-attribution:{phase}:%"))
    snapshots = session.execute(
        statement.order_by(DataSnapshot.created_at.desc()).limit(limit)
    ).scalars().all()
    return [
        {
            "snapshot_id": item.id,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "snapshot_type": item.content.get("snapshot_type"),
            "phase": item.content.get("phase"),
            "model_version": item.content.get("model_version"),
            "product_id": item.content.get("product_id"),
            "product_name": item.content.get("product_name"),
            "as_of_date": item.content.get("as_of_date"),
            "nav_fingerprint": item.content.get("nav_fingerprint"),
            "factor_data_version": item.content.get("factor_data_version"),
        }
        for item in snapshots
    ]


@router.get("/snapshots/latest-phase-d")
def list_latest_phase_d_evidence(
    product_ids: str | None = None,
    as_of_date: date | None = None,
    session: Session = Depends(get_session),
) -> dict[str, dict]:
    """Return frozen Phase-D evidence for ranking; never recomputes Phase D.

    ``product_ids`` is a comma-separated list supplied by the ranking caller.
    Only the newest snapshot whose own NAV cutoff is not after the requested
    ranking cutoff is returned for each product.
    """
    requested_ids = {
        value.strip() for value in (product_ids or "").split(",") if value.strip()
    }
    cutoff = as_of_date
    statement = select(DataSnapshot).where(DataSnapshot.label.like("cta-attribution:phase-d:%"))
    snapshots = session.execute(
        statement.order_by(DataSnapshot.created_at.desc())
    ).scalars().all()
    selected: dict[str, tuple[date, DataSnapshot]] = {}
    for snapshot in snapshots:
        content = snapshot.content or {}
        product_id = str(content.get("product_id", ""))
        if requested_ids and product_id not in requested_ids:
            continue
        snapshot_date_raw = content.get("as_of_date")
        if not product_id or not snapshot_date_raw:
            continue
        try:
            snapshot_date = date.fromisoformat(str(snapshot_date_raw))
        except ValueError:
            continue
        if cutoff is not None and snapshot_date > cutoff:
            continue
        previous = selected.get(product_id)
        if previous is not None and snapshot_date <= previous[0]:
            continue
        selected[product_id] = (snapshot_date, snapshot)

    result: dict[str, dict] = {}
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


def _phase_d_sensitivity_is_stable(sensitivity: dict) -> bool:
    """Require every reported sensitivity scenario to be available and stable."""
    if not sensitivity:
        return False
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


@router.get("/snapshots/{snapshot_id}")
def get_cta_attribution_snapshot(snapshot_id: str, session: Session = Depends(get_session)) -> dict:
    """Read one frozen CTA attribution result; never reruns a model."""
    snapshot = session.get(DataSnapshot, snapshot_id)
    if snapshot is None or not (snapshot.label or "").startswith("cta-attribution:"):
        raise HTTPException(404, "CTA 归因快照不存在")
    content = snapshot.content or {}
    return {
        "snapshot_id": snapshot.id,
        "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
        **content,
    }
