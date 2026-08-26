"""Report-image ingestion: OCR/table recovery and disclosed NAV reconstruction."""

from __future__ import annotations

import calendar
import json
import os
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from app.models import NavCandidateVersion, ProductEntity
from app.services import product_store as store
from app.services.multi_product_report import extract_multi_product_report

from .curves import _run_async, _trace_image_curves, _trace_whole_image
from .helpers import (
    _find_or_create_product,
    _infer_frequency,
    _is_multi_product_material,
    _looks_like_product_name,
    _manager_name_from_filename,
    _merge_extraction_audit,
    _normalise_manifest_product_name,
    _product_name_from_filename,
    _valid_nav_points,
    _visual_method,
)


def _ingest_report_image(
    session: Any,
    file_id: str,
    content: bytes,
    filename: str = "",
    *,
    product_name_hint: str | None = None,
    manager_name_hint: str | None = None,
) -> dict[str, Any]:
    report = extract_multi_product_report(content)
    # Even when OCR cannot read a product table or trace a curve, keep a
    # reviewable candidate in the workbench.  A completed file must never look
    # like it simply disappeared from the user's product queue.
    # The filename is a valuable identity hint (e.g. manager_product.png).
    # Prefer an existing filename candidate for a one-product report so a
    # later OCR title such as "低波CTA" enriches the same product instead of
    # creating a second, disconnected product record.
    multi_product_material = _is_multi_product_material(filename, product_name_hint) or report.report_scope == "multi_product"
    manifest_name = _normalise_manifest_product_name(product_name_hint)
    filename_name = manifest_name or _product_name_from_filename(filename)
    filename_product = None
    if filename_name and not multi_product_material:
        filename_product = _find_or_create_product(
            session,
            filename_name,
            report.product_identity.strategy,
            None,
            None,
            manager_name_hint or _manager_name_from_filename(filename),
        )
    fallback_product = filename_product
    if not report.disclosed_metrics:
        # Product identity is independent evidence.  A factsheet may have a
        # perfectly readable title while its KPI row is incomplete; do not
        # fall back to an opaque file stem in that case.
        # A filename such as "低波CTA策略" is a material/strategy label, not
        # proof of a distinct fund.  Keep the file reviewable but leave it
        # unbound until the user selects the concrete product.  This prevents
        # one factsheet from producing both a strategy pseudo-product and its
        # actual product.
        identity_name = "" if multi_product_material else (report.product_identity.product_name or "").strip()
        fallback_name = identity_name if _looks_like_product_name(identity_name) else ""
        if fallback_product is None and fallback_name:
            fallback_product = _find_or_create_product(
                session,
                fallback_name,
                report.product_identity.strategy,
                None,
                None,
                report.product_identity.manager_name,
            )
    # Cheap, explicit routing before the costly pixel trace.  A multi-product
    # page must go through curve binding; a page without a product identity is
    # retained as material evidence rather than spending time guessing a line.
    if multi_product_material or (report.report_scope != "single_product" and len(report.disclosed_metrics) > 1):
        material_lane = "multi_product"
        trace_allowed = False
        triage_reason = "文件标记为多产品资料，需先确认曲线与产品归属"
    elif fallback_product is None:
        material_lane = "non_product_or_unbound"
        trace_allowed = False
        triage_reason = "未获得可绑定的单产品身份，未自动追踪曲线"
    else:
        material_lane = "single_product_nav"
        trace_allowed = True
        triage_reason = "单产品身份可绑定，进入净值曲线追踪"
    store.add_fragment(
        session,
        file_id=file_id,
        fragment_type="image",
        content_data={
            "method": _visual_method(),
            "warnings": report.warnings,
            "product_identity": report.product_identity.model_dump(mode="json"),
            "curve_candidates": [item.model_dump() for item in report.product_curve_candidates],
            "vlm_layout_attempted": report.vlm_layout_attempted,
            "vlm_layout_used": report.vlm_layout_used,
            "report_scope": report.report_scope,
            "confidence": report.product_identity.confidence or None,
            "material_lane": material_lane,
            "trace_allowed": trace_allowed,
            "triage_reason": triage_reason,
            "workflow": {
                "stage": "trace" if trace_allowed else "review_binding",
                "identity_source": "classification_manifest" if manifest_name else report.product_identity.method,
                "next_action": "trace_chart" if trace_allowed else "confirm_product_binding",
                "retryable": True,
            },
        },
        ocr_confidence=report.product_identity.confidence or None,
        product_id=fallback_product.id if fallback_product else None,
    )
    products_by_source_id: dict[str, ProductEntity] = {}
    for metric in report.disclosed_metrics:
        name = metric.product_name or metric.product_id
        # A one-product image whose filename carries a concrete product name
        # has stronger identity evidence than OCR of labels such as “单位” or
        # “累计”.  Do not create a product from that noisy label.
        if multi_product_material:
            # A page-level multi-product hint is not enough to bind its only
            # OCR row to one fund; wait for explicit product/curve evidence.
            product = None if len(report.disclosed_metrics) == 1 else _find_or_create_product(
                session, name, metric.strategy, metric.start_date, metric.end_date,
                report.product_identity.manager_name,
            )
        elif len(report.disclosed_metrics) == 1:
            product = filename_product
            if product is None and _looks_like_product_name(name):
                product = _find_or_create_product(
                    session, name, metric.strategy, metric.start_date, metric.end_date,
                    report.product_identity.manager_name,
                )
        else:
            product = _find_or_create_product(
                session, name, metric.strategy, metric.start_date, metric.end_date,
                report.product_identity.manager_name,
            )
        if product is filename_product and product is not None:
            if metric.strategy and not product.strategy:
                product.strategy = metric.strategy
            if metric.start_date and product.inception_date is None:
                product.inception_date = metric.start_date
            if metric.end_date and (product.close_date is None or metric.end_date > product.close_date):
                product.close_date = metric.end_date
            session.commit()
        fragment = store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="table",
            content_text=f"{name}：区间 {metric.start_date} 至 {metric.end_date}",
            content_data={**metric.model_dump(mode="json"), "method": "OCR/表格指标解析"},
            ocr_confidence=0.65,
            product_id=product.id if product is not None else None,
        )
        disclosed_facts = {
            "disclosed_cumulative_return": metric.cumulative_return,
            "disclosed_annualized_return": metric.annualized_return,
        }
        if metric.maximum_drawdown_disclosed:
            disclosed_facts["disclosed_maximum_drawdown"] = metric.maximum_drawdown
        if metric.sharpe_ratio is not None:
            disclosed_facts["disclosed_sharpe_ratio"] = metric.sharpe_ratio
        if product is not None:
            products_by_source_id[metric.product_id] = product
            for field_name, value in disclosed_facts.items():
                store.add_fact(
                    session,
                    product_id=product.id if product else None,
                    field_name=field_name,
                    field_value=str(value),
                    source_file_id=file_id,
                    source_fragment_id=fragment.id,
                    confidence=0.65,
                )

    ordered_products = [products_by_source_id.get(m.product_id) for m in report.disclosed_metrics]
    if not ordered_products and fallback_product is not None:
        ordered_products = [fallback_product]
    for curve in report.product_curve_candidates:
        product = products_by_source_id.get(curve.product_id or "")
        curve_is_bound = product is not None or (
            len(report.disclosed_metrics) == 1 and not multi_product_material
        )
        store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="chart",
            bbox={"left_ratio": curve.left_ratio, "top_ratio": curve.top_ratio, "right_ratio": curve.right_ratio, "bottom_ratio": curve.bottom_ratio},
            content_data={
                "curve_index": curve.curve_index,
                "source_product_id": curve.product_id,
                "method": _visual_method(),
                "binding_status": "matched" if curve_is_bound else "unmatched",
                "binding_reason": None if curve_is_bound else "多产品曲线未返回可验证的产品 ID",
                "binding_confidence": curve.binding_confidence,
                "binding_evidence": curve.binding_evidence,
                "legend_label": curve.legend_label,
                "layout_product_name": curve.layout_product_name,
                "color_hex": curve.color_hex,
            },
            ocr_confidence=0.45,
            product_id=product.id if product else None,
        )

    # --- Curve tracing: convert pixel curves into NAV observations ---
    trace_audits: list[dict[str, Any]] = []
    if trace_allowed and report.product_curve_candidates:
        trace_audits = _trace_image_curves(
            session, file_id, content, report.product_curve_candidates,
            ordered_products, filename, report.disclosed_metrics,
        )
    elif not report.disclosed_metrics and fallback_product is not None and trace_allowed:
        # No table, no candidates — treat the whole image as one NAV chart.
        trace_audits = [_trace_whole_image(session, file_id, content, fallback_product.standard_name if fallback_product else (Path(filename).stem if filename else "未命名产品"))]
    # A few red/green table cells can be mistaken for a short chart trace.
    # Such a low-confidence candidate must not block the stronger monthly
    # return-table reconstruction lane.
    has_reliable_candidate = session.execute(
        select(NavCandidateVersion.id).where(
            NavCandidateVersion.source_file_id == file_id,
            NavCandidateVersion.confidence >= 0.65,
        )
    ).scalar() is not None
    if trace_allowed and not has_reliable_candidate and fallback_product is not None:
        recovery_points, recovery_audit = _recover_disclosed_nav_table(content)
        trace_audits.append(recovery_audit)
        recovery_method = "VLM 披露净值表读取"
        recovery_confidence = 0.9
        if len(recovery_points) < 3:
            recovery_points, recovery_audit = _recover_monthly_nav_from_return_table(content)
            trace_audits.append(recovery_audit)
            recovery_method = "VLM 月收益表读取 + 确定性复利重建"
            recovery_confidence = 0.72
        if len(recovery_points) >= 3:
            fragment = store.add_fragment(
                session,
                file_id=file_id,
                fragment_type="table",
                content_text=f"披露表格恢复 {len(recovery_points)} 个候选净值点",
                content_data={
                    "method": recovery_method,
                    "review_status": "pending",
                    "workflow_stage": "table_recovery",
                    "num_points": len(recovery_points),
                },
                ocr_confidence=recovery_confidence,
                product_id=fallback_product.id,
            )
            candidate = store.create_nav_candidate_version(
                session,
                fallback_product.id,
                file_id,
                recovery_points,
                source_fragment_id=fragment.id,
                frequency=_infer_frequency([
                    date.fromisoformat(point["observation_date"]) for point in recovery_points
                ]),
                confidence=recovery_confidence,
            )
            fragment.content_data = {**(fragment.content_data or {}), "candidate_version_id": candidate.id}
            session.commit()
    audit = _merge_extraction_audit(report.vlm_audit, *trace_audits)
    audit["workflow"] = {
        "identity": "resolved" if fallback_product else "needs_review",
        "layout": "resolved" if report.product_curve_candidates else "not_found",
        "binding": "resolved" if trace_allowed else "needs_review",
        "trace": "candidate_created" if any(
            session.execute(select(NavCandidateVersion.id).where(NavCandidateVersion.source_file_id == file_id)).scalars().all()
        ) else "needs_calibration",
        "next_action": "review_candidate" if session.execute(
            select(NavCandidateVersion.id).where(NavCandidateVersion.source_file_id == file_id)
        ).scalar() else ("confirm_product_binding" if not trace_allowed else "recover_chart_calibration"),
    }
    return audit


_MONTHLY_RETURN_TABLE_PROMPT = """这是一份私募产品业绩报告。请读取明确印刷的逐月收益率表，并读取每行最右侧的全年/今年以来合计以及页面印刷的当前净值，用于交叉校验。
严格输出 JSON：{"monthly_returns":[{"year":2024,"month":1,"return_pct":1.2}],"annual_returns":[{"year":2024,"return_pct":12.5}],"latest_nav":1.3922}
规则：
1. 每项必须来自表格中“年份行 × 月份列”的交叉单元格；空白单元格不要输出。
2. return_pct 使用百分数数值，例如 -1.3% 输出 -1.3。
3. 表格第一列若为“周收益”，它不是月份，绝对不能作为1月或其他月份；月份只对应1月到12月的明确列头。
4. annual_returns 只读取该年份行最右侧“全年”列；latest_nav 只读取“当前净值/最新净值”旁的印刷数字。
5. 无可靠逐月收益表时输出 {"monthly_returns":[],"annual_returns":[],"latest_nav":null}；只输出 JSON。"""


_DISCLOSED_NAV_TABLE_PROMPT = """这是一份私募产品净值/业绩报告。请只读取页面明确印刷的历史净值明细表，不要从曲线估算数据。
严格输出 JSON：{"nav_rows":[{"date":"2026-07-03","unit_nav":1.9810}],"latest_unit_nav":1.9810}
规则：
1. nav_rows 只允许来自同时印刷了完整年月日和“单位净值”的表格行；不要使用复权净值、累计净值或净值变动列。
2. 日期统一为 YYYY-MM-DD，unit_nav 保留原始精度。
3. latest_unit_nav 只读取页面“最新单位净值/单位净值”旁明确印刷的数字；没有则填 null。
4. 不要根据曲线、收益率或前后行推算缺失值。
5. 少于 3 行可靠明细时仍原样输出已有行；没有则输出空数组。只输出 JSON。"""


def _recover_disclosed_nav_table(image_bytes: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read exact dated unit NAV rows; never infer values from chart pixels."""
    from app.config import load_local_environment

    load_local_environment()
    provider_type = os.getenv("VLM_PROVIDER", "dashscope")
    model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
    configured = bool(os.getenv("DASHSCOPE_API_KEY")) if provider_type == "dashscope" else bool(os.getenv("VLM_BASE_URL"))
    audit = {"configured": configured, "attempted": False, "succeeded": False,
             "provider": provider_type, "model": model, "calls": 0,
             "successful_calls": 0, "failed_calls": 0,
             "methods": ["VLM 披露净值表读取"], "stage": "disclosed_nav_table_recovery"}
    if not configured:
        audit["errors"] = ["VLM 未配置，无法读取披露净值表"]
        return [], audit
    try:
        from app.services.chart_extractor.vlm_extractor import create_provider

        provider = create_provider(provider_type, api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                                   base_url=os.getenv("VLM_BASE_URL", ""), model=model)
        audit.update({"attempted": True})
        points: list[dict[str, Any]] = []
        latest_nav: float | None = None
        for variant_name, variant_bytes in _disclosed_nav_image_variants(image_bytes):
            audit["calls"] += 1
            raw = _run_async(provider.extract_structure(variant_bytes, _DISCLOSED_NAV_TABLE_PROMPT))
            audit["successful_calls"] += 1
            variant_points, variant_latest = _parse_disclosed_nav_payload(raw)
            if len(variant_points) > len(points):
                points, latest_nav = variant_points, variant_latest
                audit["selected_variant"] = variant_name
            if len(points) >= 3:
                break
        audit.update({"succeeded": True, "row_count": len(points)})
    except Exception as exc:
        audit.update({"attempted": True, "failed_calls": 1, "errors": [str(exc)[:300]]})
        return [], audit
    if len(points) < 3:
        audit.update({"succeeded": False, "errors": ["披露净值明细不足 3 行"]})
        return [], audit
    if latest_nav is not None and abs(float(points[-1]["nav"]) - latest_nav) / latest_nav > 0.005:
        audit.update({"succeeded": False, "errors": [
            f"历史表最新单位净值 {float(points[-1]['nav']):.6f} 与页面披露 {latest_nav:.6f} 不一致"]})
        return [], audit
    return points, audit


def _disclosed_nav_image_variants(image_bytes: bytes) -> list[tuple[str, bytes]]:
    """Retry tall screenshots on an enlarged lower section where history tables live."""
    variants = [("full_page", image_bytes)]
    try:
        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        if height <= width * 1.2:
            return variants
        top = int(height * 0.68)
        lower = image.crop((0, top, width, height))
        scale = max(2, min(4, round(1800 / max(width, 1))))
        lower = lower.resize((lower.width * scale, lower.height * scale))
        output = BytesIO()
        lower.save(output, format="PNG")
        variants.append(("enlarged_lower_section", output.getvalue()))
    except Exception:
        pass
    return variants


def _parse_disclosed_nav_payload(raw: str) -> tuple[list[dict[str, Any]], float | None]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return [], None
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return [], None
    if not isinstance(payload, dict):
        return [], None
    points = _valid_nav_points([
        {"observation_date": item.get("date"), "nav": item.get("unit_nav")}
        for item in payload.get("nav_rows", []) if isinstance(item, dict)
    ])
    unique = {point["observation_date"]: point for point in points}
    points = [unique[key] for key in sorted(unique)]
    try:
        latest_nav = float(payload.get("latest_unit_nav")) if payload.get("latest_unit_nav") is not None else None
    except (TypeError, ValueError):
        latest_nav = None
    if latest_nav is not None and (not np.isfinite(latest_nav) or latest_nav <= 0 or latest_nav > 1_000_000):
        latest_nav = None
    return points, latest_nav


def _recover_monthly_nav_from_return_table(image_bytes: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover a reviewable low-frequency NAV candidate from a printed return table."""
    from app.config import load_local_environment

    load_local_environment()
    provider_type = os.getenv("VLM_PROVIDER", "dashscope")
    model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
    configured = bool(os.getenv("DASHSCOPE_API_KEY")) if provider_type == "dashscope" else bool(os.getenv("VLM_BASE_URL"))
    audit = {
        "configured": configured,
        "attempted": False,
        "succeeded": False,
        "provider": provider_type,
        "model": model,
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "methods": ["VLM 月收益表读取", "确定性复利重建"],
        "stage": "table_recovery",
    }
    if not configured:
        audit["errors"] = ["VLM 未配置，无法读取月收益表"]
        return [], audit
    try:
        from app.services.chart_extractor.vlm_extractor import create_provider

        provider = create_provider(
            provider_type,
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=os.getenv("VLM_BASE_URL", ""),
            model=model,
        )
        audit.update({"attempted": True, "calls": 1})
        raw = _run_async(provider.extract_structure(image_bytes, _MONTHLY_RETURN_TABLE_PROMPT))
        rows, annual_returns, latest_nav = _parse_monthly_return_payload(raw)
        audit.update({"succeeded": True, "successful_calls": 1, "row_count": len(rows)})
    except Exception as exc:
        audit.update({"attempted": True, "failed_calls": 1, "errors": [str(exc)[:300]]})
        return [], audit
    if len(rows) < 2:
        audit.update({"succeeded": False, "errors": ["未读取到至少 2 个月的可靠收益单元格"]})
        return [], audit
    for year, reported_return in annual_returns.items():
        year_rows = [value for row_year, _, value in rows if row_year == year]
        if not year_rows:
            continue
        compounded = float(np.prod([1 + value for value in year_rows]) - 1)
        if abs(compounded - reported_return) > 0.012:
            audit.update({
                "succeeded": False,
                "failed_calls": 1,
                "errors": [f"{year}年月收益复利 {compounded:.2%} 与披露合计 {reported_return:.2%} 不一致"],
            })
            return [], audit
    first_year, first_month, _ = rows[0]
    baseline_year = first_year if first_month > 1 else first_year - 1
    baseline_month = first_month - 1 if first_month > 1 else 12
    baseline_day = calendar.monthrange(baseline_year, baseline_month)[1]
    points: list[dict[str, Any]] = [{
        "observation_date": date(baseline_year, baseline_month, baseline_day).isoformat(),
        "nav": 1.0,
    }]
    nav = 1.0
    for year, month, return_value in rows:
        nav *= 1 + return_value
        points.append({
            "observation_date": date(year, month, calendar.monthrange(year, month)[1]).isoformat(),
            "nav": round(nav, 8),
        })
    if latest_nav is not None and abs(nav - latest_nav) / latest_nav > 0.025:
        audit.update({
            "succeeded": False,
            "failed_calls": 1,
            "errors": [f"月收益重建终值 {nav:.4f} 与披露当前净值 {latest_nav:.4f} 不一致"],
        })
        return [], audit
    return _valid_nav_points(points), audit


def _parse_monthly_return_rows(raw: str) -> list[tuple[int, int, float]]:
    """Validate VLM table cells; reject future, duplicate and implausible rows."""
    rows, _, _ = _parse_monthly_return_payload(raw)
    return rows


def _parse_monthly_return_payload(
    raw: str,
) -> tuple[list[tuple[int, int, float]], dict[int, float], float | None]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return [], {}, None
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return [], {}, None
    raw_rows = payload.get("monthly_returns", []) if isinstance(payload, dict) else []
    unique: dict[tuple[int, int], float] = {}
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        try:
            year = int(item["year"])
            month = int(item["month"])
            return_value = float(item["return_pct"]) / 100
        except (KeyError, TypeError, ValueError):
            continue
        if not (1990 <= year <= date.today().year and 1 <= month <= 12 and -0.80 <= return_value <= 3.0):
            continue
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        if month_end > date.today():
            continue
        key = (year, month)
        if key in unique and abs(unique[key] - return_value) > 1e-9:
            unique.pop(key, None)
            continue
        unique[key] = return_value
    annual_returns: dict[int, float] = {}
    for item in payload.get("annual_returns", []) if isinstance(payload, dict) else []:
        try:
            year = int(item["year"])
            value = float(item["return_pct"]) / 100
        except (KeyError, TypeError, ValueError):
            continue
        if 1990 <= year <= date.today().year and -0.95 <= value <= 10:
            annual_returns[year] = value
    try:
        latest_nav = float(payload.get("latest_nav")) if payload.get("latest_nav") is not None else None
    except (TypeError, ValueError):
        latest_nav = None
    if latest_nav is not None and (not np.isfinite(latest_nav) or latest_nav <= 0 or latest_nav > 1_000_000):
        latest_nav = None
    rows = [(year, month, unique[(year, month)]) for year, month in sorted(unique)]
    return rows, annual_returns, latest_nav


