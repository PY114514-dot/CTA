"""Extract product identity, disclosed metrics and NAV chart regions.

The workbench receives complete factsheet images, not only a cropped chart.
Older code assumed that the KPI table always lived in the top 9--25% of the
page and that every saturated horizontal component was a NAV curve.  Real
single-product briefs commonly put the table below the chart and may also
contain a pie chart, benchmark panel and coloured table borders.  This module
therefore uses page-wide OCR plus chart-region geometry, while keeping every
result reviewable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date
from typing import Any

import numpy as np

from app.config import load_local_environment

from app.schemas import (
    MultiProductReportResponse,
    ReportCurveBinding,
    ReportCurveCandidate,
    ReportDisclosedMetrics,
    ReportLegendItem,
    ReportProductCard,
    ReportProductIdentity,
)


logger = logging.getLogger(__name__)

_MULTI_PRODUCT_LAYOUT_PROMPT = """这是一张私募基金多产品周报/净值报告。请只做页面版面与归属识别，不要推算净值。
严格输出 JSON：
{
  "product_cards": [{"product_name":"产品全称", "bbox_1000":[l,t,r,b], "confidence":0.0}],
  "legend_items": [{"label":"图例原文", "color_hex":"#RRGGBB", "bbox_1000":[l,t,r,b], "confidence":0.0}],
  "curves": [{"bbox_1000":[l,t,r,b], "color_hex":"#RRGGBB", "legend_label":"对应图例原文", "product_name":"仅在页面明确写明时填写", "confidence":0.0}]
}
规则：
1. bbox_1000 按整页宽高归一化到 0-1000；无法定位填 []，不要猜。
2. 多产品时不要按上下/左右顺序猜曲线归属；只有图例、同一卡片标题或明确连线支持时，product_name 才能填写。
3. 曲线与产品没有明确关联时 product_name 必须为空字符串。
4. 不要把策略名、基准、指数、最大回撤或表头当作产品名。
5. 只输出 JSON。"""


def extract_multi_product_report(image_bytes: bytes) -> MultiProductReportResponse:
    """Extract reviewable factsheet evidence from one image.

    OCR is intentionally best-effort.  A failed OCR pass must not discard the
    source image or prevent a user from opening the chart calibration view.
    """
    load_local_environment()
    if not image_bytes:
        raise ValueError("image file is empty")
    try:
        import cv2
        import pytesseract
    except ImportError as error:
        raise ValueError("报告识别依赖未安装。请运行 pip install -r requirements.txt") from error

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片")

    height = image.shape[0]
    # Keep the legacy strip for older multi-product reports, but also scan the
    # lower KPI area and the whole page.  The latter is what catches the real
    # Guobiao brief (its product table is below the NAV chart).
    legacy = image[round(height * 0.09):round(height * 0.25), :]
    lower_kpi = image[round(height * 0.28):round(height * 0.58), :]
    whole_chi = _ocr_chinese(image) or ""
    whole_eng = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
    legacy_chi = _ocr_chinese(legacy) or ""
    legacy_eng = pytesseract.image_to_string(legacy, lang="eng", config="--psm 6")
    kpi_chi = _ocr_chinese(lower_kpi) or ""
    kpi_eng = pytesseract.image_to_string(lower_kpi, lang="eng", config="--psm 6")

    all_chi = "\n".join(part for part in (whole_chi, legacy_chi, kpi_chi) if part)
    all_eng = "\n".join(part for part in (whole_eng, legacy_eng, kpi_eng) if part)
    metrics = _parse_disclosed_metrics("\n".join((all_eng, all_chi)))
    if not metrics:
        fallback = _parse_single_product_factsheet(all_chi, all_eng)
        if fallback is not None:
            metrics = [fallback]

    identity = _extract_product_identity(all_chi, all_eng, metrics)
    # Ask VLM for chart layout on every report when configured.  Previously
    # this was limited to pages whose OCR had already found multiple products;
    # a missed OCR identity therefore prevented VLM from ever seeing the
    # chart, silently degrading most batch imports to CV-only detection.
    product_cards, legend_items, vlm_curves, vlm_audit = _vlm_layout(image_bytes, metrics)
    curves = vlm_curves or _curve_candidates(image, metrics)
    bindings = _bind_curves_to_products(curves, metrics, product_cards, legend_items)
    binding_by_curve = {binding.curve_index: binding for binding in bindings}
    for curve in curves:
        binding = binding_by_curve.get(curve.curve_index)
        if binding is None:
            continue
        curve.product_id = binding.product_id
        curve.binding_confidence = binding.confidence
        curve.binding_evidence = binding.evidence
    warnings = [
        "披露指标由图片 OCR 提取；请与原始周报核对。",
        "曲线候选由图表区域检测生成，颜色和日期仍需在净值复核页确认。",
    ]
    if identity.product_name:
        warnings.append(f"已识别产品名称候选：{identity.product_name}（{identity.method}，请确认）。")
    if metrics and not metrics[0].maximum_drawdown_disclosed:
        warnings.append("原始资料未披露最大回撤；界面中的 0 不是已核验的零回撤，不应直接用于筛选。")
    if not metrics:
        warnings.append("未读取到产品披露指标；已保留产品身份和曲线候选，建议打开复核页手动校准。")
    if not curves:
        warnings.append("未定位到足够可信的净值曲线框；建议在图表复核页手动框选。")
    elif len(metrics) > 1 and not any(binding.status == "matched" for binding in bindings):
        warnings.append("已定位多产品曲线，但缺少图例或同一卡片证据，未自动按位置绑定；请在批量绑定页确认。")

    return MultiProductReportResponse(
        disclosed_metrics=metrics,
        product_curve_candidates=curves,
        product_cards=product_cards,
        legend_items=legend_items,
        curve_bindings=bindings,
        product_identity=identity,
        vlm_layout_attempted=bool(vlm_audit.get("attempted")),
        vlm_layout_used=bool(vlm_curves),
        vlm_audit=vlm_audit,
        warnings=warnings,
    )


def _run_async(coro: Any) -> Any:
    """Run the optional page-layout VLM from synchronous ingestion code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _normalise_name(value: str | None) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", (value or "")).lower()


def _ratio_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [float(item) / 1000 for item in value]
    except (TypeError, ValueError):
        return None
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return None
    return left, top, right, bottom


def _vlm_layout(
    image_bytes: bytes,
    metrics: list[ReportDisclosedMetrics],
) -> tuple[list[ReportProductCard], list[ReportLegendItem], list[ReportCurveCandidate], dict[str, Any]]:
    """Ask VLM for chart topology and return auditable call provenance."""
    provider_type = os.getenv("VLM_PROVIDER", "dashscope")
    model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
    configured = bool(os.getenv("DASHSCOPE_API_KEY")) if provider_type == "dashscope" else bool(os.getenv("VLM_BASE_URL"))
    if not configured:
        return [], [], [], {
            "configured": False,
            "attempted": False,
            "succeeded": False,
            "provider": provider_type,
            "model": model,
            "error": "VLM 未配置",
        }
    try:
        from app.services.chart_extractor.vlm_extractor import create_provider

        provider = create_provider(
            provider_type,
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=os.getenv("VLM_BASE_URL", ""),
            model=os.getenv("VLM_MODEL", "qwen3-vl-flash"),
        )
        raw = _run_async(provider.extract_structure(image_bytes, _MULTI_PRODUCT_LAYOUT_PROMPT))
        payload = _parse_layout_json(raw)
    except Exception as exc:
        logger.warning("Report layout VLM unavailable: %s", exc)
        return [], [], [], {
            "configured": True,
            "attempted": True,
            "succeeded": False,
            "provider": provider_type,
            "model": model,
            "error": str(exc)[:300],
        }

    metrics_by_name = {_normalise_name(item.product_name): item for item in metrics if item.product_name}
    cards: list[ReportProductCard] = []
    for item in payload.get("product_cards", []):
        if not isinstance(item, dict):
            continue
        metric = metrics_by_name.get(_normalise_name(str(item.get("product_name", ""))))
        bbox = _ratio_bbox(item.get("bbox_1000"))
        if metric is None or bbox is None:
            continue
        cards.append(ReportProductCard(
            product_id=metric.product_id, product_name=metric.product_name,
            left_ratio=bbox[0], top_ratio=bbox[1], right_ratio=bbox[2], bottom_ratio=bbox[3],
            confidence=_confidence(item.get("confidence")), evidence=["VLM 定位产品信息卡"],
        ))

    legends: list[ReportLegendItem] = []
    for item in payload.get("legend_items", []):
        if not isinstance(item, dict):
            continue
        bbox = _ratio_bbox(item.get("bbox_1000"))
        if bbox is None:
            continue
        legends.append(ReportLegendItem(
            label=str(item.get("label", "")).strip(), color_hex=_valid_color(item.get("color_hex")),
            left_ratio=bbox[0], top_ratio=bbox[1], right_ratio=bbox[2], bottom_ratio=bbox[3],
            confidence=_confidence(item.get("confidence")),
        ))

    curves: list[ReportCurveCandidate] = []
    for index, item in enumerate(payload.get("curves", []), start=1):
        if not isinstance(item, dict):
            continue
        bbox = _ratio_bbox(item.get("bbox_1000"))
        if bbox is None:
            continue
        curves.append(ReportCurveCandidate(
            curve_index=index, left_ratio=bbox[0], top_ratio=bbox[1], right_ratio=bbox[2], bottom_ratio=bbox[3],
            color_hex=_valid_color(item.get("color_hex")),
            legend_label=str(item.get("legend_label", "")).strip() or None,
            layout_product_name=str(item.get("product_name", "")).strip() or None,
            binding_confidence=_confidence(item.get("confidence")),
        ))
    return cards, legends, curves, {
        "configured": True,
        "attempted": True,
        "succeeded": True,
        "provider": provider_type,
        "model": model,
        "curve_count": len(curves),
        "product_card_count": len(cards),
        "legend_count": len(legends),
    }


def _parse_layout_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _valid_color(value: Any) -> str | None:
    token = str(value or "").strip()
    return token.upper() if re.fullmatch(r"#[0-9A-Fa-f]{6}", token) else None


def _bind_curves_to_products(
    curves: list[ReportCurveCandidate],
    metrics: list[ReportDisclosedMetrics],
    cards: list[ReportProductCard],
    legends: list[ReportLegendItem],
) -> list[ReportCurveBinding]:
    """Bind only direct text/color evidence; never use curve/table order."""
    result: list[ReportCurveBinding] = []
    for curve in curves:
        # Single-product reports are the only situation where identity alone
        # is sufficient.  Multi-product reports require a named legend or a
        # VLM assertion that the curve is in the same named card.
        if len(metrics) == 1:
            result.append(ReportCurveBinding(curve_index=curve.curve_index, product_id=metrics[0].product_id,
                status="matched", confidence=max(0.9, curve.binding_confidence), evidence=["单产品报告，曲线归属唯一"]))
            continue
        labels = [curve.legend_label or "", curve.layout_product_name or ""]
        labels.extend(legend.label for legend in legends if curve.color_hex and legend.color_hex == curve.color_hex)
        matched = [metric for metric in metrics if any(
            _normalise_name(metric.product_name) and _normalise_name(metric.product_name) in _normalise_name(label)
            for label in labels
        )]
        if len(matched) == 1:
            evidence = ["VLM 明确给出曲线归属"] if curve.layout_product_name else ["图例文字与产品名称匹配"]
            result.append(ReportCurveBinding(curve_index=curve.curve_index, product_id=matched[0].product_id,
                status="matched", confidence=max(0.85, curve.binding_confidence), evidence=evidence))
        else:
            result.append(ReportCurveBinding(curve_index=curve.curve_index, status="unmatched",
                confidence=curve.binding_confidence, evidence=["未找到唯一的图例/产品卡片关联，不按位置猜测"]))
    return result


def _ocr_chinese(table_image: np.ndarray) -> str | None:
    """Return Chinese OCR text if the language pack is installed."""
    try:
        import pytesseract

        text = pytesseract.image_to_string(table_image, lang="chi_sim", config="--psm 6")
        return text.strip() or None
    except Exception:
        return None


def _enrich_metrics_with_chinese(metrics: list[ReportDisclosedMetrics], text: str) -> None:
    """Match Chinese product names and strategies to numeric rows by row index."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name_candidates: list[str] = []
    strategy_candidates: list[str] = []
    for line in lines:
        if "策略" in line or any(word in line for word in ("CTA", "套利", "对冲", "趋势", "截面", "多资产")):
            strategy_candidates.append(line)
        elif re.search(r"[\u4e00-\u9fa5]{3,}", line) and not re.search(r"\d{4}/\d{2}/\d{2}", line):
            name_candidates.append(line)

    for index, metric in enumerate(metrics):
        if index < len(name_candidates):
            metric.product_name = _clean_product_name(name_candidates[index])
        if index < len(strategy_candidates):
            metric.strategy = _clean_strategy(strategy_candidates[index])


def _parse_disclosed_metrics(text: str) -> list[ReportDisclosedMetrics]:
    """Parse the legacy English table format with explicit dates and drawdown."""
    rows: list[ReportDisclosedMetrics] = []
    pattern = re.compile(
        r"\b([A-Z]{2,}\d+)\b.*?(\d{4}/\d{2}/\d{2})\s+(\d{4}/\d{2}/\d{2})\s+"
        r"(-?\d+(?:\.\d+)?)%\s+(-?\d+(?:\.\d+)?)%\s+(-?\d+(?:\.\d+)?)%"
    )
    for product_id, start, end, cumulative, annualized, drawdown in pattern.findall(text):
        rows.append(
            ReportDisclosedMetrics(
                product_id=product_id,
                start_date=start.replace("/", "-"),
                end_date=end.replace("/", "-"),
                cumulative_return=float(cumulative) / 100,
                annualized_return=float(annualized) / 100,
                maximum_drawdown=float(drawdown) / 100,
                maximum_drawdown_disclosed=True,
            )
        )
    return rows


def _parse_single_product_factsheet(chinese: str, english: str) -> ReportDisclosedMetrics | None:
    """Parse a Chinese factsheet row such as ``产品名 1.7717 -7.46% 70.47% ...``.

    The row normally publishes recent-week, recent-year, annualized and
    excess return, but not maximum drawdown.  We retain the known returns and
    mark the missing drawdown explicitly instead of fabricating a risk value.
    """
    from datetime import timedelta

    combined = _normalise_ocr(f"{chinese}\n{english}")
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    end_date = _find_latest_date(combined)
    if end_date is None:
        # A chart often exposes the end date as an x-axis label even when the
        # KPI heading is unreadable.  Use the last ISO-like date as fallback.
        date_tokens = re.findall(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}", combined)
        if date_tokens:
            end_date = _parse_date_token(date_tokens[-1])
    if end_date is None:
        return None

    # Require four percentages after the NAV value.  This deliberately skips
    # benchmark rows such as 沪深300指数, which have only three percentages.
    number = r"-?\d+(?:\.\d+)?"
    row_pattern = re.compile(
        rf"(?P<name>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9·（）()\- ]{{2,60}}?)"
        rf"\s+1\s*[.,]\s*\d{{3,6}}\s+"
        rf"(?P<week>{number})\s*%\s+"
        rf"(?P<year>{number})\s*%\s+"
        rf"(?P<annual>{number})\s*%\s+"
        rf"(?P<excess>{number})\s*%"
    )
    row_match = None
    for line in lines:
        row_match = row_pattern.search(line)
        if row_match:
            break
    if row_match is None:
        # OCR sometimes drops line breaks in the table.  Search the complete
        # stream as a second pass.
        row_match = row_pattern.search(combined)
    if row_match is None:
        return None

    product_name = _clean_product_name(row_match.group("name"))
    if not product_name or product_name in {"产品/指标", "产品指标"}:
        return None

    start_date = end_date - timedelta(days=365)
    drawdown_match = re.search(r"最大回撤\s*[—:：\s]*(-?\d+(?:\.\d+)?)\s*%", chinese)
    drawdown = float(drawdown_match.group(1)) / 100 if drawdown_match else 0.0
    strategy = _infer_strategy_from_text(chinese)
    return ReportDisclosedMetrics(
        product_id="single_product",
        product_name=product_name,
        start_date=start_date,
        end_date=end_date,
        cumulative_return=float(row_match.group("year")) / 100,
        annualized_return=float(row_match.group("annual")) / 100,
        maximum_drawdown=drawdown,
        maximum_drawdown_disclosed=drawdown_match is not None,
        strategy=strategy,
    )


def _extract_product_identity(
    chinese: str,
    english: str,
    metrics: list[ReportDisclosedMetrics],
) -> ReportProductIdentity:
    """Recover a reviewable identity independently of KPI parsing."""
    # A chart caption is the strongest identity evidence on a single-product
    # factsheet.  It is common for the page heading to describe a strategy
    # (for example, ``低波 CTA 策略产品介绍``), while the chart itself names the
    # investable product (``致远二号-周度净值与回撤``).  Never turn the former
    # into a separate product merely because it is the largest OCR text.
    chart_name = _extract_chart_product_name(chinese)
    metric_name = metrics[0].product_name if len(metrics) == 1 else None
    if _looks_like_strategy_or_series(metric_name):
        metric_name = None
    title_name = None
    for line in chinese.splitlines():
        if "业绩简报" in line:
            before = line.split("业绩简报", 1)[0]
            candidate = _clean_product_name(before)
            if candidate and candidate not in {"历史业绩", "简报"}:
                title_name = candidate
                break
    name = chart_name or metric_name or title_name
    manager_match = re.search(r"([\u4e00-\u9fff]{2,12}(?:投资|资产|基金|资本|期货|管理))", chinese)
    manager = manager_match.group(1) if manager_match else None
    strategy = metrics[0].strategy if len(metrics) == 1 else _infer_strategy_from_text(chinese)
    if chart_name:
        confidence = 0.92
        method = "净值图标题"
    elif metric_name:
        confidence = 0.88
        method = "OCR 表格行 + 标题"
    elif title_name:
        confidence = 0.72
        method = "OCR 标题"
    else:
        confidence = 0.0
        method = "未识别"
    return ReportProductIdentity(
        product_name=name,
        manager_name=manager,
        strategy=strategy,
        confidence=confidence,
        method=method,
    )


def _extract_chart_product_name(text: str) -> str | None:
    """Extract the named product immediately preceding a NAV-chart caption."""
    for line in text.splitlines():
        match = re.search(
            r"(?P<name>[\u4e00-\u9fffA-Za-z0-9()（）·]+(?:[一二三四五六七八九十0-9]+号)?)"
            r"\s*[-—–|｜:：]?\s*(?:周度?|月度?|日度?)?净值(?:与回撤)?",
            line,
        )
        if match:
            candidate = _clean_product_name(match.group("name"))
            if candidate and not _looks_like_strategy_or_series(candidate):
                return candidate
    return None


def _looks_like_strategy_or_series(value: str | None) -> bool:
    """Strategy/series headings are metadata, not a canonical fund name."""
    token = (value or "").replace(" ", "")
    return bool(token) and any(marker in token for marker in ("策略", "产品介绍", "系列"))


def _find_latest_date(text: str) -> date | None:
    match = re.search(r"最新(?:累?计)?净值\s*[:：]?\s*(20\d{6})", text)
    if match:
        token = match.group(1)
        try:
            return date(int(token[:4]), int(token[4:6]), int(token[6:]))
        except ValueError:
            pass
    tokens = re.findall(r"20\d{6}", text)
    for token in reversed(tokens):
        try:
            return date(int(token[:4]), int(token[4:6]), int(token[6:]))
        except ValueError:
            continue
    return None


def _parse_date_token(token: str) -> date | None:
    cleaned = token.replace("/", "-").replace(".", "-")
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _normalise_ocr(text: str) -> str:
    # Tesseract occasionally emits ``o%``/``°%`` for the zero in a percent
    # value.  Correct that local OCR artefact without touching other text.
    return re.sub(r"[oO°](?=\s*%)", "", text)


def _clean_product_name(value: str) -> str:
    cleaned = re.sub(r"^[\s|｜:：、,，>]+|[\s|｜:：、,，]+$", "", value)
    cleaned = re.sub(r"(?:产品/指标|产品指标)$", "", cleaned).strip()
    # A title OCR line may include the company prefix.  Keep the product part
    # when it is followed by the report title marker.
    if "业绩简报" in cleaned:
        cleaned = cleaned.split("业绩简报", 1)[0].strip()
    return cleaned


def _infer_strategy_from_text(text: str) -> str | None:
    """Return a cautious label only when the disclosure is explicit."""
    upper = text.upper()
    if "商品CTA" in upper or "商品 CTA" in upper:
        return "商品 CTA"
    if "股指CTA" in upper or "股指 CTA" in upper:
        return "股指 CTA"
    if "硬科技" in text and ("期货" in text or "衍生品" in text):
        return "权益选股 + 期货及衍生品"
    return None


def _clean_strategy(value: str) -> str:
    return re.sub(r"^[\s|｜:：]+|[\s|｜:：]+$", "", value)


def _curve_candidates(image: np.ndarray, metrics: list[ReportDisclosedMetrics]) -> list[ReportCurveCandidate]:
    """Prefer wide line-chart regions over pie charts and table borders."""
    import cv2

    height, width = image.shape[:2]
    try:
        from app.services.chart_extractor.chart_detector import detect_chart_regions

        encoded_ok, encoded = cv2.imencode(".png", image)
        regions = detect_chart_regions(encoded.tobytes()) if encoded_ok else []
    except Exception:
        regions = []

    # NAV plots are generally wide and shallow.  Pie charts and coloured KPI
    # panels are close to square; excluding them prevents the old false match.
    line_regions = [
        region
        for region in regions
        if region.w >= width * 0.35
        and region.w / max(region.h, 1) >= 2.0
        and region.y < height * 0.65
    ]
    if line_regions:
        line_regions.sort(
            key=lambda region: (
                region.w / max(region.h, 1),
                region.w * region.h,
                -region.y,
            ),
            reverse=True,
        )
        region = line_regions[0]
        pad_x = max(8, int(region.w * 0.025))
        pad_y = max(8, int(region.h * 0.08))
        x0 = max(0, int(region.x) - pad_x)
        y0 = max(0, int(region.y) - pad_y)
        x1 = min(width, int(region.x + region.w) + pad_x)
        y1 = min(height, int(region.y + region.h) + pad_y)
        return [
            ReportCurveCandidate(
                curve_index=1,
                product_id=metrics[0].product_id if len(metrics) == 1 else None,
                left_ratio=x0 / width,
                top_ratio=y0 / height,
                right_ratio=x1 / width,
                bottom_ratio=y1 / height,
            )
        ]

    # Fallback for clean chart-only images where the detector has no Hough
    # region.  Keep only substantial, wide components; thin table rules are
    # intentionally excluded.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
    red = (hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)
    blue = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 135)
    mask = ((green | red | blue) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 50)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [
        stat
        for stat in stats[1:count]
        if stat[cv2.CC_STAT_WIDTH] >= width * 0.35
        and stat[cv2.CC_STAT_HEIGHT] >= height * 0.025
        and stat[cv2.CC_STAT_TOP] >= height * 0.10
    ]
    components.sort(
        key=lambda stat: stat[cv2.CC_STAT_WIDTH] * stat[cv2.CC_STAT_HEIGHT],
        reverse=True,
    )
    curves: list[ReportCurveCandidate] = []
    for index, stat in enumerate(components[: len(metrics) or 1]):
        x, y, component_width, component_height, _ = stat
        curves.append(
            ReportCurveCandidate(
                curve_index=index + 1,
                product_id=metrics[0].product_id if len(metrics) == 1 else None,
                left_ratio=max(0, x - component_width * 0.03) / width,
                top_ratio=max(0, y - component_height * 0.1) / height,
                right_ratio=min(width, x + component_width * 1.03) / width,
                bottom_ratio=min(height, y + component_height * 1.1) / height,
            )
        )
    if not curves and len(metrics) == 1:
        curves.append(
            ReportCurveCandidate(
                curve_index=1,
                product_id=metrics[0].product_id,
                left_ratio=0.08,
                top_ratio=0.14,
                right_ratio=0.92,
                bottom_ratio=0.34,
            )
        )
    return curves
