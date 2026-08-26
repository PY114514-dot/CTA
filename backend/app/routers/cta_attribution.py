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
from app.services.cta_attribution_snapshot import MODEL_VERSIONS
from app.services.cta_attribution_evidence import load_latest_phase_d_evidence_payload
from app.services.phase_d_batch import (
    freeze_attribution_snapshot,
    get_batch_runner,
    list_phase_d_batch_candidates,
)
from app.services.cta_evaluation import build_attribution_tables, evaluate_nav_path
from app.services.cta_factor_bundle import CTA_FACTOR_NAMES, COMMODITY_ARBITRAGE_FACTOR_NAMES, get_cta_factor_bundle, get_factor_series
from app.services.dynamic_beta import (
    DYNAMIC_FACTOR_NAMES,
    INTERACTION_FACTOR_NAMES,
    compute_rolling_beta,
    filter_dynamic_beta,
    prepare_factor_matrix,
)
from app.services.regime_attribution import classify_observable_regimes, run_regime_attribution
from app.services.nonlinear_attribution import evaluate_factor_interaction_increment, evaluate_momentum_volatility_increment
from app.services.factor_library.factor_regression import assess_oos_applicability, run_factor_regression
from app.services.candidate_factor_gate import evaluate_candidate_factor
from app.services.attribution_applicability import select_attribution_contract

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


def _cta_contract_or_error(product, reviewed_nav_count: int) -> dict:
    applicability = select_attribution_contract(product, reviewed_nav_count)
    if applicability["status"] not in {"applicable", "observe_only"}:
        raise HTTPException(422, applicability["reason"])
    return applicability


@router.get("/products/{product_id}/applicability")
def get_model_applicability(product_id: str, session: Session = Depends(get_session)) -> dict:
    """Return the selected factor contract without running a regression."""
    product = product_store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
    reviewed_count = len(observations)
    return {"product_id": product.id, "product_name": product.standard_name,
            "reviewed_nav_count": reviewed_count,
            **select_attribution_contract(product, reviewed_count, as_of_date=observations[-1].observation_date if observations else None)}


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
    applicability = _cta_contract_or_error(product, len(observations))
    frequency = product.nav_frequency or observations[0].frequency or "weekly"
    if frequency not in {"daily", "weekly", "monthly"}:
        raise HTTPException(422, "净值频率必须为 daily、weekly 或 monthly")
    dates = [item.observation_date for item in observations]
    navs = [float(item.nav) for item in observations]
    returns = [right / left - 1.0 for left, right in zip(navs, navs[1:])]
    bundle_cutoff = max(dates[1:])
    factor_bundle = get_cta_factor_bundle(as_of_date=bundle_cutoff)
    candidate_gate = _candidate_gate(product.strategy, product.strategy_disclosure, returns, dates[1:], frequency)
    factor_names = list(CTA_FACTOR_NAMES)
    if candidate_gate and candidate_gate["admitted"]:
        factor_names.append(candidate_gate["candidate_name"])
    try:
        regression = run_factor_regression(
            product_returns=np.asarray(returns), product_dates=dates[1:], frequency=frequency,
            factor_names=factor_names,
            factor_series_loader=lambda name: get_factor_series(name, as_of_date=bundle_cutoff),
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
        "factor_bundle": factor_bundle,
        "baseline": {"r_squared": regression.r_squared, "adj_r_squared": regression.adj_r_squared,
                     "annualized_alpha_candidate": regression.annualized_alpha, "out_of_sample": regression.out_of_sample,
                     "validation": assess_oos_applicability(regression.out_of_sample), "warnings": regression.warnings},
        "candidate_factor_gate": candidate_gate,
        "model_applicability": applicability,
        "warnings": ["收益贡献、因子暴露和 Euler 风险贡献为不同统计对象，不代表真实持仓或真实 P&L。",
                     "当前阶段未纳入持仓、成交、费用、容量或交易成本数据。"],
    }


def _candidate_factor_names_for_product(strategy: str | None, disclosure: dict | None = None) -> tuple[str, ...]:
    arbitrage_type = (disclosure or {}).get("arbitrage_type")
    if arbitrage_type is not None:
        if arbitrage_type not in {"跨期", "期限结构", "混合"}:
            return ()
        return tuple(name for name in COMMODITY_ARBITRAGE_FACTOR_NAMES if name not in CTA_FACTOR_NAMES)
    normalized = (strategy or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"商品套利", "commodity_arbitrage", "commodity_spread_arbitrage"}:
        return tuple(name for name in COMMODITY_ARBITRAGE_FACTOR_NAMES if name not in CTA_FACTOR_NAMES)
    return ()


def _candidate_gate(strategy: str | None, disclosure: dict | None, returns: list[float], dates: list[date], frequency: str) -> dict | None:
    candidates = _candidate_factor_names_for_product(strategy, disclosure)
    if not candidates:
        return None
    from app.services.dynamic_beta import prepare_factor_matrix

    try:
        aligned = prepare_factor_matrix(
            np.asarray(returns), dates, frequency, list(CTA_FACTOR_NAMES + candidates),
        )
    except ValueError as error:
        return {"candidate_name": candidates[0], "status": "insufficient", "admitted": False, "segments": [], "summary": {}, "rule": "至少 3 个连续样本外段。", "conclusion": str(error)}
    return evaluate_candidate_factor(
        aligned["returns"], aligned["factor_returns"], aligned["dates"], aligned["factor_names"], candidates[0],
    )


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
    _cta_contract_or_error(product, len(observations))
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
    _cta_contract_or_error(product, len(observations))
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
    _cta_contract_or_error(product, len(observations))
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
        aligned = prepare_factor_matrix(returns, return_dates, frequency, INTERACTION_FACTOR_NAMES)
        if aligned["factor_names"] != list(INTERACTION_FACTOR_NAMES):
            raise ValueError("动量 × 波动率主效应因子不完整，拒绝交互检验")
        nonlinear = evaluate_momentum_volatility_increment(
            aligned["returns"], aligned["factor_returns"], aligned["dates"],
            frequency=frequency, train_window=train_window, test_window=test_window,
            max_segments=5, min_segments=3, min_r2_uplift=0.01,
        )
    except ValueError as error:
        raise HTTPException(422, f"非线性增量数据不足：{error}") from error
    except Exception as error:
        raise HTTPException(500, f"非线性增量计算失败：{error}") from error

    def scenario_check(
        *,
        key: str,
        title: str,
        description: str,
        factor_names: tuple[str, str],
        interaction_name: str,
        use_prior_drawdown: bool = False,
        source_factor_names: tuple[str, ...] | None = None,
    ) -> dict:
        try:
            requested_factors = source_factor_names or factor_names
            scenario_aligned = prepare_factor_matrix(returns, return_dates, frequency, requested_factors)
            if scenario_aligned["factor_names"] != list(requested_factors):
                raise ValueError("所需因子未完整覆盖")
            scenario_factors = scenario_aligned["factor_returns"]
            if use_prior_drawdown:
                prior_nav = np.r_[1.0, np.cumprod(1.0 + scenario_aligned["returns"])[:-1]]
                prior_peak = np.maximum.accumulate(prior_nav)
                scenario_factors = np.column_stack([scenario_factors[:, 0], prior_nav / prior_peak - 1.0])
            check = evaluate_factor_interaction_increment(
                scenario_aligned["returns"],
                scenario_factors,
                scenario_aligned["dates"],
                factor_names=factor_names,
                interaction_name=interaction_name,
                display_name=title,
                frequency=frequency,
                train_window=train_window,
                test_window=test_window,
                max_segments=5,
                min_segments=3,
                min_r2_uplift=0.01,
            )
            summary = check["summary"]
            return {
                "key": key,
                "title": title,
                "description": description,
                "status": check["status"],
                "stable": summary["stable_improvement"],
                "conclusion": summary["conclusion"],
                "evaluated_segments": summary["evaluated_delta_count"],
                "positive_fraction": summary["positive_delta_fraction"],
                "mean_r2_delta": summary["mean_r2_delta"],
                "details": check,
            }
        except ValueError as error:
            return {
                "key": key,
                "title": title,
                "description": description,
                "status": "insufficient",
                "stable": False,
                "conclusion": f"暂无法验证：{error}",
                "evaluated_segments": 0,
                "positive_fraction": None,
                "mean_r2_delta": None,
                "details": None,
            }

    scenario_checks = [
        scenario_check(
            key="trend_agreement",
            title="趋势协同",
            description="检验长短趋势同时明显时，产品是否出现更稳定的趋势特征。",
            factor_names=("trend", "short_term_trend_20"),
            interaction_name="trend_x_short_term_trend",
        ),
        scenario_check(
            key="trend_carry",
            title="趋势与期限结构",
            description="检验不同期限结构下，趋势特征是否出现稳定变化。",
            factor_names=("trend", "term_structure_carry"),
            interaction_name="trend_x_term_structure_carry",
        ),
        scenario_check(
            key="drawdown_recovery",
            title="回撤后的趋势修复",
            description="检验产品经历自身回撤后，趋势行情是否带来更稳定的修复表现。",
            factor_names=("trend", "prior_drawdown"),
            interaction_name="trend_x_prior_drawdown",
            use_prior_drawdown=True,
            source_factor_names=("trend",),
        ),
    ]
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
        "scenario_checks": scenario_checks,
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
    return freeze_attribution_snapshot(session, body.product_id, body.phase)


class PhaseDBatchRequest(BaseModel):
    """Trigger background Phase-D evidence generation for a product set."""

    product_ids: list[str] = Field(default_factory=list, max_length=2000)
    concurrency: int = Field(default=4, ge=1, le=16)


@router.post("/phase-d/refresh-batch")
def refresh_phase_d_batch(
    body: PhaseDBatchRequest | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """启动后台批量生成 Phase-D 证据快照（幂等，已冻结的产品直接复用）。

    不传 product_ids 时默认为全部 confirmed + weekly 产品。任务在后台
    线程池运行，进度通过 GET /phase-d/batch-status 查询。
    """
    body = body or PhaseDBatchRequest()
    if body.product_ids:
        candidates = body.product_ids
    else:
        candidates = list_phase_d_batch_candidates(session)
    runner = get_batch_runner()
    if runner.status()["status"] == "running":
        return {**runner.status(), "note": "批量任务已在运行，未重复启动。"}
    return runner.start(candidates, concurrency=body.concurrency)


@router.get("/phase-d/batch-status")
def phase_d_batch_status() -> dict:
    """返回后台 Phase-D 批量任务的可见性快照（当前对象/并行/排队）。"""
    return get_batch_runner().status()


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
    ranking cutoff is returned for each product.  The selection logic lives in
    ``cta_attribution_evidence.load_latest_phase_d_evidence_payload`` so the
    ranking refresh path and this endpoint cannot drift apart.
    """
    requested_ids = {
        value.strip() for value in (product_ids or "").split(",") if value.strip()
    }
    return load_latest_phase_d_evidence_payload(
        session,
        product_ids=requested_ids or None,
        as_of_date=as_of_date,
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
