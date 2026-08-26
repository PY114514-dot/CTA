"""HTTP seams for independent CTA scores."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import DataSnapshot
from app.schemas import CtaProductScoreRequest
from app.services.cta_scores import build_product_score_report
from app.services.product_store import create_snapshot

router = APIRouter(prefix="/api/cta-scores", tags=["CTA 产品评分"])


def _validation_message(error: ValidationError) -> str:
    """Extract the readable reason from a request-shape validation failure.

    ``build_product_score_report`` constructs the internal CODEX ranking
    request itself, so shape violations (e.g. mixed product frequencies)
    surface as pydantic errors here instead of at request parsing time.
    """
    for item in error.errors():
        message = str(item.get("msg", ""))
        if message.startswith("Value error, "):
            return message[len("Value error, "):]
    return "请求数据不符合 CODEX 评分要求"


def _build_report(request: CtaProductScoreRequest) -> dict:
    """Build the score report, mapping shape violations to a clean 422."""
    try:
        return build_product_score_report(request)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=_validation_message(error)) from error


@router.post("/profile")
def get_product_score_profile(request: CtaProductScoreRequest) -> dict:
    """Return quality and confidence, plus an unavailable allocation result by default."""
    return _build_report(request)


def _score_snapshot_label(report: dict) -> str:
    return (
        f"cta-product-score:{report['model_version']}:"
        f"{report['as_of_date']}:{report['nav_fingerprint']}"
    )


def _score_report_summary(report: dict) -> dict:
    """Keep the list view small; product detail is fetched on demand."""
    return {
        "model_version": report["model_version"],
        "quality_model_version": report["quality_model_version"],
        "as_of_date": report["as_of_date"],
        "nav_fingerprint": report.get("nav_fingerprint"),
        "universe_size": report.get("universe_size"),
        "products": [
            {"product_id": item["product_id"], "product_name": item["product_name"], "summary": item["summary"]}
            for item in report.get("products", [])
        ],
        "warnings": report.get("warnings", []),
    }


def _latest_score_snapshot(session: Session) -> DataSnapshot | None:
    """Return the most recently persisted snapshot; universe size only breaks ties.

    时间优先：旧版本快照的产品数更多时（如 v2 的 749 只 vs 新 v3 的 722 只），
    仍应返回最近一次固化的结果，否则面板会一直加载过期数据。
    """
    snapshots = session.execute(
        select(DataSnapshot).where(DataSnapshot.label.like("cta-product-score:%"))
    ).scalars().all()
    return max(
        snapshots,
        key=lambda item: (
            item.created_at.timestamp() if item.created_at else 0,
            int((item.content or {}).get("universe_size", 0)),
        ),
        default=None,
    )


@router.post("/snapshots")
def persist_product_score_snapshot(
    request: CtaProductScoreRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Calculate and persist an immutable score report for direct reuse."""
    report = jsonable_encoder(_build_report(request))
    label = _score_snapshot_label(report)
    snapshot = session.execute(
        select(DataSnapshot).where(DataSnapshot.label == label).limit(1)
    ).scalars().first()
    if snapshot is None:
        snapshot = create_snapshot(session, label=label, content=report)
    return {
        "snapshot_id": snapshot.id,
        "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
        "report": _score_report_summary(snapshot.content or {}),
    }


@router.get("/snapshots/latest")
def get_latest_product_score_snapshot(session: Session = Depends(get_session)) -> dict:
    """Return the most recently persisted score snapshot without recalculation."""
    snapshot = _latest_score_snapshot(session)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="暂无已固化的 CTA 产品评分")
    return {
        "snapshot_id": snapshot.id,
        "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
        "report": _score_report_summary(snapshot.content or {}),
    }


@router.get("/snapshots/latest/{product_id}")
def get_latest_product_score_item(product_id: str, session: Session = Depends(get_session)) -> dict:
    """Return one product's persisted score and attribution summary."""
    snapshot = _latest_score_snapshot(session)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="暂无已固化的 CTA 产品评分")
    item = next(
        (value for value in (snapshot.content or {}).get("products", []) if value.get("product_id") == product_id),
        None,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="该产品不在当前 CTA 评分范围内")
    return {
        "as_of_date": snapshot.content.get("as_of_date"),
        "item": item,
    }


@router.post("/allocation")
def get_allocation_score(request: CtaProductScoreRequest) -> dict:
    """Return allocation value only for an explicit current portfolio case."""
    if not request.current_portfolio or request.candidate_product_id is None or request.candidate_weight is None:
        raise HTTPException(422, "配置分必须提供当前组合、候选产品和候选权重")
    report = _build_report(request)
    return {
        "model_version": report["model_version"],
        "quality_model_version": report["quality_model_version"],
        "as_of_date": report["as_of_date"],
        "allocation": report["allocation"],
        "warnings": report["warnings"],
    }
