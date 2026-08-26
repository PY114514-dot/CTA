"""PDF material ingestion: text layer + chart pipeline + monthly-return reconstruction."""

from __future__ import annotations

import calendar
import logging
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

from .curves import _auto_curve_colors, _run_async
from .helpers import (
    _find_or_create_product,
    _looks_like_product_name,
    _merge_extraction_audit,
    _normalise_manifest_product_name,
    _valid_nav_points,
)

logger = logging.getLogger(__name__)


def _ingest_pdf(
    session: Any,
    file_id: str,
    content: bytes,
    filename: str,
    product_name_hint: str | None = None,
) -> dict[str, Any]:
    stem = Path(filename).stem

    # Phase 1: text layer for context.
    text = ""
    try:
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages).strip()
    except Exception as exc:
        logger.info("PDF text extraction skipped: %s", exc)

    if text:
        store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="text",
            content_text=text[:20000],
            ocr_confidence=0.9,
            product_id=None,
        )

    # Phase 2: chart pipeline for NAV curves.  Text-only PDFs still continue
    # to the deterministic monthly-table lane below; a missing curve color is
    # not evidence that the PDF belongs to the filename's product.
    provider_type = os.getenv("VLM_PROVIDER", "dashscope")
    use_vlm = bool(os.getenv("DASHSCOPE_API_KEY", "")) if provider_type == "dashscope" else bool(os.getenv("VLM_BASE_URL", ""))
    curve_specs = None
    audits: list[dict[str, Any]] = []
    chart_job = None
    if not use_vlm:
        colors = _auto_curve_colors_from_pdf(content)
        if colors:
            curve_specs = [{"name": stem, "color_hex": colors[0], "color_name": "", "is_benchmark": False}]
        elif not text:
            raise ValueError("PDF 无可提取文本且未检测到净值曲线颜色，请在研究工具中手动校准")

    if use_vlm or curve_specs:
        from app.services.chart_extractor import run_extraction

        try:
            chart_job = _run_async(run_extraction(
                pdf_bytes=content,
                filename=filename,
                use_vlm=use_vlm,
                curve_specs=curve_specs,
            ))
        except Exception as exc:
            logger.warning("PDF chart extraction failed: %s", exc)
            if not text:
                raise ValueError(f"PDF 图表提取失败：{exc}") from exc
            audits.append({
                "configured": use_vlm,
                "attempted": use_vlm,
                "succeeded": False,
                "provider": provider_type if use_vlm else None,
                "model": os.getenv("VLM_MODEL") if use_vlm else None,
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "errors": [str(exc)[:300]],
                "methods": ["PDF 文本层提取"],
            })

    traced_products = 0
    traced_targets: list[ProductEntity] = []
    if chart_job is not None:
        for result_index, result in enumerate(chart_job.results):
            audits.append({
                "configured": use_vlm,
                "attempted": bool(result.vlm_attempted),
                "succeeded": bool(result.vlm_succeeded),
                "provider": result.vlm_provider or provider_type if use_vlm else None,
                "model": result.vlm_model or os.getenv("VLM_MODEL", "qwen3-vl-flash") if use_vlm else None,
                "calls": 1 if result.vlm_attempted else 0,
                "successful_calls": 1 if result.vlm_succeeded else 0,
                "failed_calls": 1 if result.vlm_attempted and not result.vlm_succeeded else 0,
                "errors": [result.vlm_error] if result.vlm_error else [],
                "methods": ["VLM 结构识别", "CV 像素追踪"] if use_vlm else ["CV 像素追踪（无 VLM）"],
            })
            source_page = chart_job.regions[result_index].page_index if result_index < len(chart_job.regions) else 0
            for curve in result.curves:
                if curve.is_benchmark:
                    continue
                points = [
                    {"observation_date": p.date, "nav": p.value}
                    for p in curve.points
                    if p.date and p.value is not None and p.value > 0
                ]
                if len(points) < 3:
                    continue
                curve_name = (curve.name or "").strip()
                title = (result.structure.chart_title.strip() if result.structure and result.structure.chart_title else "")
                target_name = curve_name or title or stem
                if not _looks_like_pdf_product_curve(target_name, stem):
                    logger.info("Skipping non-product PDF curve label %r from %s", target_name, filename)
                    continue
                target = _find_or_create_product(session, target_name, None, None, None)
                if target not in traced_targets:
                    traced_targets.append(target)
                fragment = store.add_fragment(
                    session,
                    file_id=file_id,
                    fragment_type="chart_traced",
                    page_number=source_page + 1,
                    content_text=f"PDF 曲线追踪 {len(points)} 个数据点",
                    content_data={"num_points": len(points), "confidence": result.confidence, "source_page": source_page + 1, "method": "VLM 结构识别 + CV 像素追踪" if use_vlm else "CV 像素追踪（无 VLM）", "review_status": "pending"},
                    ocr_confidence=max(0.3, result.confidence * 0.8),
                    product_id=target.id,
                )
                frequency = result.frequency if result.frequency != "unknown" else None
                candidate_version = store.create_nav_candidate_version(
                    session, target.id, file_id, points,
                    source_fragment_id=fragment.id,
                    frequency=frequency,
                    confidence=max(0.3, result.confidence * 0.8),
                )
                fragment.content_data = {**(fragment.content_data or {}), "candidate_version_id": candidate_version.id}
                session.commit()
                traced_products += 1

    reliable_curve = session.execute(
        select(NavCandidateVersion.id).where(
            NavCandidateVersion.source_file_id == file_id,
            NavCandidateVersion.confidence >= 0.65,
        )
    ).scalar() is not None
    monthly_points = _recover_monthly_nav_from_pdf_text(text, filename=filename) if not reliable_curve else []
    recovery_target = traced_targets[0] if len(traced_targets) == 1 else None
    if recovery_target is None and not traced_targets:
        hinted_name = _normalise_manifest_product_name(product_name_hint)
        if hinted_name:
            recovery_target = _find_or_create_product(session, hinted_name, None, None, None)
    if len(monthly_points) >= 3 and recovery_target is not None:
        target = recovery_target
        fragment = store.add_fragment(
            session,
            file_id=file_id,
            fragment_type="table",
            page_number=1,
            content_text=f"PDF 文本层月收益表恢复 {len(monthly_points)} 个候选净值点",
            content_data={
                "method": "PDF 文本层月收益表 + 确定性复利重建",
                "review_status": "pending",
                "workflow_stage": "table_recovery",
                "num_points": len(monthly_points),
            },
            ocr_confidence=0.92,
            product_id=target.id,
        )
        candidate = store.create_nav_candidate_version(
            session,
            target.id,
            file_id,
            monthly_points,
            source_fragment_id=fragment.id,
            frequency="monthly",
            confidence=0.92,
        )
        fragment.content_data = {**(fragment.content_data or {}), "candidate_version_id": candidate.id}
        session.commit()
    elif len(monthly_points) >= 3:
        audits.append({
            "configured": False,
            "attempted": False,
            "succeeded": False,
            "errors": ["PDF 月收益表已读取，但缺少唯一产品绑定；未创建净值候选"],
            "methods": ["PDF 文本层月收益表（待产品绑定）"],
        })

    if traced_products == 0 and not text:
        raise ValueError("PDF 未提取到文本或净值曲线，请在研究工具中手动校准")
    return _merge_extraction_audit(*audits)


def _recover_monthly_nav_from_pdf_text(text: str, filename: str | None = None) -> list[dict[str, Any]]:
    """Rebuild a reviewable NAV candidate from a PDF monthly-return table.

    PDF text extraction often emits the table before the title/metadata and
    may omit the header entirely.  The report period therefore comes from the
    title when available, then from a ``YYMM月报`` filename token.  When the
    share inception date is not present before the table, the first visible
    monthly cell is used and the preceding month-end becomes the baseline.
    """
    if not text or "月收益" not in text:
        # Some CJK PDF fonts are not decoded by pypdf.  The rows themselves
        # are still useful when a report-month token is available in the
        # filename, so do not require the literal section heading in that
        # case.
        if not filename or not re.search(r"月(?:报|度)", Path(filename).stem):
            return []
    report_match = re.search(r"(20\d{2})年\s*(\d{1,2})月", text)
    if report_match:
        report_year, report_month = int(report_match.group(1)), int(report_match.group(2))
    else:
        report_year = report_month = 0
        if filename:
            filename_match = re.search(
                r"(?<!\d)(?:(20)?(\d{2}))(0[1-9]|1[0-2])月(?:报|度)",
                Path(filename).stem,
            )
            if filename_match:
                century, short_year, month = filename_match.groups()
                report_year = int(f"{century or '20'}{short_year}")
                report_month = int(month)
    if not report_year or not report_month:
        return []

    # Dates after the monthly table are usually chart x-axis labels.  Only
    # trust a full date occurring in the metadata prefix; otherwise infer a
    # month-end baseline from the first monthly row below.
    table_start = text.find("月收益")
    metadata_text = text[:table_start] if table_start >= 0 else ""
    inception_match = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", metadata_text)
    nav_match = re.search(r"(?:累计净值|单位净值)\s+([0-9]+(?:\.[0-9]+)?)", text)

    # Keep the longest percentage row per year.  The same PDF commonly has a
    # second quarterly table; selecting the longest row prevents those
    # summary rows from being mistaken for monthly data.
    raw_rows: dict[int, list[float]] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(20\d{2})\s+(.+)$", line.strip())
        if not match:
            continue
        year = int(match.group(1))
        values = [float(value.rstrip("%")) / 100 for value in re.findall(r"[-+]?\d+(?:\.\d+)?%", match.group(2))]
        if not values or year > report_year:
            continue
        if len(values) > len(raw_rows.get(year, [])):
            raw_rows[year] = values
    if not raw_rows:
        return []

    explicit_inception = None
    if inception_match:
        explicit_inception = date(
            int(inception_match.group(1)),
            int(inception_match.group(2)),
            int(inception_match.group(3)),
        )
    first_year = min(raw_rows)
    if explicit_inception is not None:
        inception = explicit_inception
        first_start_month = inception.month + 1
        first_expected_months = 12 - inception.month
    else:
        # When the text layer loses CJK labels, chart x-axis dates remain
        # machine-readable.  They give an unambiguous first monthly cell and
        # avoid mistaking December's return for an annual-total column.
        chart_dates = [
            date(int(year), int(month), int(day))
            for year, month, day in re.findall(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", text)
        ]
        first_chart_date = next((item for item in sorted(chart_dates) if item.year == first_year), None)
        if first_chart_date is not None:
            first_start_month = first_chart_date.month
            first_expected_months = 13 - first_start_month
            inception = date(
                first_chart_date.year,
                first_chart_date.month - 1 if first_chart_date.month > 1 else 12,
                calendar.monthrange(
                    first_chart_date.year if first_chart_date.month > 1 else first_chart_date.year - 1,
                    first_chart_date.month - 1 if first_chart_date.month > 1 else 12,
                )[1],
            )
        else:
        # Prefer treating the final cell as an annual total only when it
        # independently matches the preceding cells' compounded return.
            first_values = raw_rows[first_year]
            annual_candidate = first_values[-1] if len(first_values) >= 2 else None
            month_candidate = first_values[:-1] if annual_candidate is not None else []
            annual_is_proven = bool(
                month_candidate
                and annual_candidate is not None
                and abs(float(np.prod([1 + value for value in month_candidate]) - 1) - annual_candidate) <= 0.012
            )
            month_count = len(month_candidate) if annual_is_proven else len(first_values)
            if month_count < 1 or month_count > 12:
                return []
            first_start_month = 13 - month_count
            baseline_year = first_year if first_start_month > 1 else first_year - 1
            baseline_month = first_start_month - 1 if first_start_month > 1 else 12
            inception = date(baseline_year, baseline_month, calendar.monthrange(baseline_year, baseline_month)[1])
            first_expected_months = month_count

    rows: list[tuple[int, list[float], float | None]] = []
    for year in sorted(raw_rows):
        values = list(raw_rows[year])
        if year < inception.year or year > report_year:
            continue
        expected_months = (
            first_expected_months
            if year == inception.year
            else report_month
            if year == report_year
            else 12
        )
        annual: float | None = None
        if len(values) == expected_months + 1:
            annual = values.pop()
        elif year != first_year and 2 <= len(values) <= expected_months + 1:
            # Some PDF text layers omit visually empty month cells.  A short
            # row is usable only when its final printed value independently
            # proves that the preceding cells are monthly returns.
            possible_annual = values[-1]
            possible_months = values[:-1]
            compounded = float(np.prod([1 + value for value in possible_months]) - 1)
            if abs(compounded - possible_annual) <= 0.012:
                values = possible_months
                annual = possible_annual
        if not values or len(values) > expected_months:
            continue
        if annual is not None:
            compounded = float(np.prod([1 + value for value in values]) - 1)
            if abs(compounded - annual) > 0.012:
                return []
        rows.append((year, values, annual))
    if not rows or rows[0][0] != first_year or rows[-1][0] != report_year:
        return []
    points: list[dict[str, Any]] = [{
        "observation_date": inception.isoformat(),
        "nav": 1.0,
    }]
    nav = 1.0
    for year, values, _ in rows:
        start_month = first_start_month if year == inception.year else 1
        for offset, return_value in enumerate(values):
            month = start_month + offset
            nav *= 1 + return_value
            points.append({
                "observation_date": date(year, month, calendar.monthrange(year, month)[1]).isoformat(),
                "nav": round(nav, 8),
            })
    if nav_match:
        disclosed_nav = float(nav_match.group(1))
        if disclosed_nav <= 0 or abs(nav - disclosed_nav) / disclosed_nav > 0.025:
            return []
    return _valid_nav_points(points)


def _looks_like_pdf_product_curve(name: str, file_stem: str) -> bool:
    """Reject benchmark, sector and chart-label curves before creating products."""
    token = name.replace(" ", "").strip()
    if len(token) < 3:
        return False
    generic = (
        "产品净值", "累计净值", "单位净值", "图例", "超额收益", "基准", "指数",
        "沪深300", "中证", "上证", "国债", "南华", "商品指数", "板块", "行业",
        "农产品", "黑色", "有色", "能化", "贵金属", "策略收益", "组合收益",
    )
    if any(label in token for label in generic):
        return False
    # A model returning the document title instead of a legend has not
    # established a concrete fund identity.  Accept it only when the filename
    # itself looks like a product name under the same conservative gate.
    if token == file_stem.replace(" ", "").strip():
        return _looks_like_product_name(file_stem)
    return _looks_like_product_name(name)


def _auto_curve_colors_from_pdf(pdf_bytes: bytes) -> list[str]:
    """Render first PDF page and auto-detect curve color for CV tracing."""
    from app.services.chart_extractor.pdf_renderer import render_pdf_pages

    try:
        render_result = render_pdf_pages(pdf_bytes, max_pages=1)
    except Exception:
        return []
    if not render_result.pages:
        return []
    return _auto_curve_colors(render_result.pages[0].image_bytes)


