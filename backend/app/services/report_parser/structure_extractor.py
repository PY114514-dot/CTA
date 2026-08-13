"""Structured extraction from weekly report slices.

Takes rendered slices (images) and extracts structured data:
- Performance metrics table
- Exposure structure (long/short/net/leverage)
- Factor contribution breakdown
- Sector P&L
- Variety x Factor matrix

Uses OCR (Tesseract) for text extraction and pattern matching for
table structure recognition.  This is a rule-based extractor tuned
for the specific weekly report format; it will need adaptation for
different managers' report layouts.
"""

import logging
import re
from datetime import date

from app.services.report_parser.schema import (
    CorrelationMatrix,
    ExposureStructure,
    FactorContribution,
    PerformanceMetrics,
    ProfileSnapshot,
    SectorPnl,
    VarietyFactorRow,
)
from app.services.report_parser.pdf_slicer import SliceInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR helper
# ---------------------------------------------------------------------------

def _ocr_slice(slice_info: SliceInfo) -> str:
    """Run Tesseract OCR on a slice, return raw text."""
    try:
        import pytesseract
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(slice_info.image_bytes))
        text = pytesseract.image_to_string(img, lang="chi_sim+eng", config="--psm 6")
        return text
    except Exception as exc:
        logger.warning("OCR failed for slice %d: %s", slice_info.index, exc)
        return ""


def _ocr_all_slices(slices: list[SliceInfo]) -> list[str]:
    """OCR all slices and return list of text blocks."""
    texts = []
    for s in slices:
        text = _ocr_slice(s)
        texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Pattern-based extractors
# ---------------------------------------------------------------------------

# Common number patterns in Chinese financial reports
_NUM_PATTERN = r"[-+]?\d+\.?\d*%?"
_PCT_PATTERN = r"([-+]?\d+\.?\d*)\s*%"


def _parse_number(text: str) -> float | None:
    """Extract a numeric value from a string like '16.51%' or '-3.64'."""
    text = text.strip().replace(",", "").replace("，", "")
    text = text.rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def extract_product_name(full_text: str) -> str:
    """Try to extract product name from the report header."""
    # Common patterns: "XX资产XX号" or product name in first few lines
    lines = full_text.split("\n")[:10]
    for line in lines:
        # Look for patterns like "XX资产" or "XX计划" or "XX基金"
        match = re.search(r"[\u4e00-\u9fa5]{2,}(?:资产|计划|基金|集合|1号|2号|3号)", line)
        if match:
            return match.group(0)
    # Fallback: first non-empty line
    for line in lines:
        if line.strip() and len(line.strip()) > 2:
            return line.strip()[:30]
    return "未知产品"


def extract_report_date(full_text: str) -> date | None:
    """Extract report date from text."""
    # Patterns: 2024年03月15日, 2024-03-15, 2024/03/15
    patterns = [
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
    ]
    for pat in patterns:
        match = re.search(pat, full_text)
        if match:
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                continue
    return None


def extract_metrics(text: str) -> PerformanceMetrics:
    """Extract performance metrics from OCR text."""
    metrics = PerformanceMetrics()

    # Key-value patterns (label → field)
    patterns = {
        "累积净值|累计净值": "cumulative_nav",
        "年化收益[率]?": "annualized_return",
        "年化波动[率]?": "annualized_vol",
        "最大回撤": "max_drawdown",
        "夏普[比率]?|Sharpe": "sharpe",
        "卡玛[比率]?|Calmar": "calmar",
    }

    for pattern, field_name in patterns.items():
        # Try "label: value" or "label value" patterns
        match = re.search(
            rf"{pattern}\s*[:：]?\s*({_NUM_PATTERN})",
            text,
        )
        if match:
            val = _parse_number(match.group(1))
            if val is not None:
                # Convert percentages to decimals where appropriate
                if field_name in ("annualized_return", "annualized_vol", "max_drawdown"):
                    if abs(val) > 1:  # likely percentage
                        val = val / 100.0
                setattr(metrics, field_name, val)

    return metrics


def extract_exposure(text: str) -> ExposureStructure:
    """Extract long/short exposure structure."""
    exposure = ExposureStructure()

    patterns = {
        "多头敞口|多头占比": "long_exposure",
        "空头敞口|空头占比": "short_exposure",
        "净敞口|净多头": "net_exposure",
        "杠杆[率倍]?": "leverage",
    }

    for pattern, field_name in patterns.items():
        match = re.search(rf"{pattern}\s*[:：]?\s*({_NUM_PATTERN})", text)
        if match:
            val = _parse_number(match.group(1))
            if val is not None:
                if field_name != "leverage" and abs(val) > 1:
                    val = val / 100.0
                setattr(exposure, field_name, val)

    return exposure


def extract_factor_contribution(text: str) -> FactorContribution:
    """Extract per-factor return contribution."""
    contrib = FactorContribution()

    factor_patterns = {
        "利润": "profit",
        "库存": "inventory",
        "基差": "basis",
        "长期规则|趋势": "long_term_rule",
        "量价相关": "volume_price_corr",
        "长期截面|截面": "long_term_cross",
    }

    for pattern, field_name in factor_patterns.items():
        # Look for "因子名 +X.XX%" pattern
        match = re.search(rf"{pattern}\s*[:：]?\s*({_PCT_PATTERN})", text)
        if match:
            val = _parse_number(match.group(1))
            if val is not None:
                setattr(contrib, field_name, val / 100.0)

    return contrib


def extract_sector_pnl(text: str) -> list[SectorPnl]:
    """Extract sector-level P&L breakdown."""
    sectors = []
    known_sectors = ["能化", "农产品", "新能源", "有色", "黑色", "贵金属", "商品指数", "金融"]

    for sector in known_sectors:
        match = re.search(rf"{sector}\s*[:：]?\s*({_PCT_PATTERN})", text)
        if match:
            val = _parse_number(match.group(1))
            if val is not None:
                sectors.append(SectorPnl(
                    sector=sector,
                    pnl_pct=val / 100.0,
                    direction="多" if val > 0 else "空",
                ))

    return sectors


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_profile(
    slices: list[SliceInfo],
    source_file: str = "",
) -> ProfileSnapshot:
    """Run the full extraction pipeline on report slices.

    Parameters
    ----------
    slices : rendered image slices from pdf_slicer
    source_file : original filename for metadata

    Returns
    -------
    ProfileSnapshot with all extractable fields populated.
    """
    warnings: list[str] = []

    # OCR all slices
    logger.info("Running OCR on %d slices...", len(slices))
    texts = _ocr_all_slices(slices)
    full_text = "\n".join(texts)

    if len(full_text.strip()) < 50:
        warnings.append("OCR 输出过少，报告可能为纯图片无文字层，需视觉模型辅助")

    # Extract fields
    product_name = extract_product_name(full_text)
    report_date = extract_report_date(full_text)

    if report_date is None:
        warnings.append("未能识别报告日期")
        report_date = date.today()

    metrics = extract_metrics(full_text)
    exposure = extract_exposure(full_text)
    factor_contrib = extract_factor_contribution(full_text)
    sector_pnl = extract_sector_pnl(full_text)

    # Estimate confidence based on how many fields were extracted
    fields_total = 0
    fields_found = 0

    for val in [metrics.cumulative_nav, metrics.annualized_return, metrics.max_drawdown,
                metrics.sharpe, exposure.long_exposure, exposure.leverage,
                factor_contrib.profit, factor_contrib.basis]:
        fields_total += 1
        if val is not None:
            fields_found += 1

    confidence = fields_found / max(fields_total, 1)

    snapshot = ProfileSnapshot(
        product_name=product_name,
        report_date=report_date,
        metrics=metrics,
        exposure=exposure,
        factor_contribution=factor_contrib,
        sector_pnl=sector_pnl,
        source_file=source_file,
        parse_confidence=round(confidence, 3),
        warnings=warnings,
    )

    logger.info(
        "Extracted profile for '%s' (date=%s, confidence=%.0f%%, %d warnings)",
        product_name, report_date, confidence * 100, len(warnings),
    )
    return snapshot
