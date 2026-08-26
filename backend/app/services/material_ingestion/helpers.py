"""Shared helpers for material ingestion: date/NAV normalisation,
audit merging, name heuristics and product lookup."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.models import ProductEntity
from app.services import product_store as store


def _normalise_observation_date(value: Any) -> str | None:
    """Return an ISO date only when the source contains a complete date.

    Visual/OCR output sometimes contains axis labels such as ``21/11`` or
    ``10月``.  Those values do not contain a year, so assigning one would turn
    a chart-reading artefact into fabricated NAV history.  Keep the source
    evidence, but exclude that individual point from the NAV series instead
    of failing the complete material-ingestion job.
    """
    if isinstance(value, date):
        parsed = value
    else:
        if not isinstance(value, str):
            return None
        token = value.strip().replace("/", "-").replace(".", "-")
        if len(token) == 8 and token.isdigit():
            token = f"{token[:4]}-{token[4:6]}-{token[6:]}"
        try:
            parsed = date.fromisoformat(token)
        except ValueError:
            return None
    # A chart OCR artefact can be syntactically valid while still being an
    # impossible NAV date (e.g. 2078-12-02).  NAV history must not contain
    # future observations merely because an axis label was misread.
    if parsed < date(1990, 1, 1) or parsed > date.today():
        return None
    return parsed.isoformat()


def _valid_nav_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only finite positive NAV values with complete plausible dates."""
    result: list[dict[str, Any]] = []
    for point in _normalise_nav_points(points):
        try:
            value = float(point.get("nav"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or value <= 0 or value > 1_000_000:
            continue
        result.append({**point, "nav": value})
    return result


def _normalise_nav_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discard malformed visual dates without discarding the uploaded file."""
    normalised: list[dict[str, Any]] = []
    for point in points:
        observation_date = _normalise_observation_date(point.get("observation_date"))
        nav = point.get("nav")
        if observation_date is None or nav is None:
            continue
        normalised.append({**point, "observation_date": observation_date})
    return normalised


def _visual_method() -> str:
    """Return the actual image method used for provenance labels."""
    provider_type = os.getenv("VLM_PROVIDER", "dashscope")
    configured = bool(os.getenv("DASHSCOPE_API_KEY", "")) if provider_type == "dashscope" else bool(os.getenv("VLM_BASE_URL", ""))
    return "VLM 结构识别 + CV 像素追踪" if configured else "CV 像素追踪（无 VLM）"


def _empty_extraction_audit() -> dict[str, Any]:
    provider = os.getenv("VLM_PROVIDER", "dashscope")
    model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
    configured = bool(os.getenv("DASHSCOPE_API_KEY", "")) if provider == "dashscope" else bool(os.getenv("VLM_BASE_URL", ""))
    return {"configured": configured, "attempted": False, "succeeded": False, "provider": provider, "model": model}


def _merge_extraction_audit(*audits: dict[str, Any]) -> dict[str, Any]:
    """Combine per-page/per-curve provenance into one file-level audit."""
    valid = [item for item in audits if item]
    if not valid:
        return _empty_extraction_audit()
    first = valid[0]
    errors: list[str] = []
    for item in valid:
        if item.get("error"):
            errors.append(str(item["error"]))
        errors.extend(str(error) for error in item.get("errors", []) if error)
    methods: list[str] = []
    for item in valid:
        for method in item.get("methods", []):
            if method not in methods:
                methods.append(method)
    return {
        "configured": any(bool(item.get("configured")) for item in valid),
        "attempted": any(bool(item.get("attempted")) for item in valid),
        "succeeded": any(bool(item.get("succeeded")) for item in valid),
        "provider": next((item.get("provider") for item in valid if item.get("provider")), first.get("provider")),
        "model": next((item.get("model") for item in valid if item.get("model")), first.get("model")),
        "calls": sum(int(item.get("calls", 0)) for item in valid),
        "successful_calls": sum(int(item.get("successful_calls", 0)) for item in valid),
        "failed_calls": sum(int(item.get("failed_calls", 0)) for item in valid),
        "errors": errors[:20],
        "methods": methods,
    }


def _looks_like_strategy_label(value: str) -> bool:
    token = value.replace(" ", "")
    return bool(token) and any(marker in token for marker in ("策略", "产品介绍", "系列"))


def _normalise_manifest_product_name(value: str | None) -> str:
    """Turn a classified material label into a concrete product hint if possible."""
    token = (value or "").strip()
    token = re.sub(
        r"(?:近一年)?(?:产品)?(?:净值)?(?:业绩)?(?:周度报告|周报|月度报告|月报|运行报告)$",
        "",
        token,
    ).strip(" _-—（）()")
    generic = (
        "代表产品", "多产品", "产品总", "产品业绩", "策略代表产品", "期货多策略",
        "低波CTA", "中波CTA", "CTA策略", "CTA系列",
    )
    if len(token.replace(" ", "")) < 3 or any(label in token for label in generic):
        return ""
    if token.endswith("策略") and not re.search(r"\d+号|一期|二期|基金", token):
        return ""
    return token


def _is_multi_product_material(filename: str, product_name_hint: str | None) -> bool:
    token = f"{Path(filename).stem} {product_name_hint or ''}".replace(" ", "")
    return any(marker in token for marker in ("多产品", "产品总", "产品汇总", "产品组合", "产品池"))


def _product_name_from_filename(filename: str) -> str:
    """Use a concrete filename product segment as the primary identity hint.

    Classified source images are commonly named ``管理人_产品名.png``.  This is
    more reliable than chart OCR, which regularly confuses axis labels such
    as “单位净值” for a product title.  Strategy-only files deliberately
    remain unbound and need a human to choose the product.
    """
    stem = Path(filename).stem.strip()
    if not stem:
        return ""
    candidate = stem.rsplit("_", 1)[-1].strip() if "_" in stem else stem
    return candidate if _looks_like_product_name(candidate) else ""


def _manager_name_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem.strip()
    if "_" not in stem:
        return None
    manager = stem.rsplit("_", 1)[0].strip()
    return manager or None


def _looks_like_product_name(value: str) -> bool:
    token = value.replace(" ", "").strip()
    if len(token) < 3 or _looks_like_strategy_label(token):
        return False
    generic_labels = ("单位", "累计", "周度", "产品", "净值", "期末", "本期", "合计")
    return not any(label in token for label in generic_labels)


def _find_or_create_product(
    session: Any,
    name: str,
    strategy: str | None,
    inception: date | None,
    close: date | None,
    manager_name: str | None = None,
) -> ProductEntity:
    cleaned_name = name.strip() or "未命名产品"
    product = session.execute(select(ProductEntity).where(ProductEntity.standard_name == cleaned_name)).scalars().first()
    if product is None:
        product = store.create_product(
            session,
            standard_name=cleaned_name,
            manager_name=manager_name,
            strategy=strategy,
            inception_date=inception,
        )
    elif strategy and not product.strategy:
        product.strategy = strategy
    if manager_name and not product.manager_name:
        product.manager_name = manager_name
    if close and (product.close_date is None or close > product.close_date):
        product.close_date = close
    session.commit()
    session.refresh(product)
    return product


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> Any | None:
    normalized = {str(column).strip().lower().replace(" ", ""): column for column in frame.columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower().replace(" ", ""))
        if found is not None:
            return found
    return None


def _infer_frequency(dates: list[date], *, pixel_trace: bool = False) -> str:
    if len(dates) < 2:
        return "unknown"
    # Pixel tracing emits one point per horizontal pixel.  Those points are
    # interpolation samples, not daily observations; treating them as daily
    # would inflate volatility and Sharpe.  Unless VLM/user metadata provides
    # another frequency, use the product-report default of weekly.
    if pixel_trace and len(dates) >= 30:
        return "weekly"
    gaps = [(right - left).days for left, right in zip(dates, dates[1:])]
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 3:
        return "daily"
    if median_gap <= 10:
        return "weekly"
    return "monthly"


