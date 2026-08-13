"""Conservative product-name OCR and keyword evidence for research triage."""

from __future__ import annotations

import re

import numpy as np

from app.schemas import (
    ProductImageRecognitionResponse,
    ProductStrategyProfileRequest,
    ProductStrategyProfileResponse,
    StrategyEvidence,
)


def recognize_product_name(image_bytes: bytes) -> ProductImageRecognitionResponse:
    """OCR an image and return a candidate name, never a confirmed identity."""
    if not image_bytes:
        raise ValueError("image file is empty")
    try:
        import cv2
    except ImportError as error:
        raise ValueError("图像识别依赖未安装。请运行 pip install -r requirements.txt") from error
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片；请上传 PNG、JPG 或 JPEG 文件")

    text = _ocr_text(image)
    candidate = _product_name_candidate(text)
    warnings = ["产品名称为 OCR 候选值，必须与产品备案、合同或管理人披露信息核对。"]
    if text is None:
        warnings.append("本机未检测到可用的中文 OCR 语言包；请手动填写产品名称。")
    elif candidate is None:
        warnings.append("OCR 未找到明确产品名称；请手动填写后再生成策略画像。")
    return ProductImageRecognitionResponse(
        product_name_candidate=candidate,
        confidence=0.7 if candidate else 0.0,
        ocr_text=text,
        warnings=warnings,
    )


def build_strategy_profile(request: ProductStrategyProfileRequest) -> ProductStrategyProfileResponse:
    """Derive transparent hypotheses only from supplied name and text evidence."""
    corpus = f"{request.product_name} {request.source_text}".lower()
    strategy = _first_evidence(
        corpus,
        [
            ("趋势 CTA", ("cta", "趋势", "趋势跟踪", "管理期货", "动量"), 0.65),
            ("截面 CTA", ("截面", "横截面", "多空", "价差"), 0.6),
            ("多资产策略", ("多资产", "资产配置", "大类资产配置"), 0.7),
            ("套利策略", ("套利", "期现套利", "跨期套利", "跨品种套利"), 0.65),
            ("市场中性", ("市场中性", "对冲", "alpha", "阿尔法"), 0.65),
            ("宏观对冲", ("宏观", "宏观对冲", "global macro"), 0.65),
            ("股票策略", ("强势股", "股票", "权益", "量化选股"), 0.75),
        ],
        "策略类型未确认",
    )
    factors = _all_evidence(
        corpus,
        [
            ("时序动量", ("趋势", "趋势跟踪", "cta", "动量"), 0.55),
            ("截面动量", ("截面", "横截面", "多空"), 0.55),
            ("股票动量", ("强势股", "动量", "选股"), 0.55),
            ("均值回复", ("均值回复", "反转"), 0.5),
            ("资产配置", ("多资产", "资产配置", "大类资产配置"), 0.55),
        ],
    )
    futures = _all_evidence(
        corpus,
        [
            ("股指期货", ("股指", "if", "ih", "ic", "im"), 0.6),
            ("国债期货", ("国债", "债券期货", "t ", "tf", "ts"), 0.6),
            ("商品期货", ("商品", "黑色", "有色", "能源", "农产品", "化工"), 0.6),
            ("外汇／海外期货", ("外汇", "海外", "美元", "fx"), 0.55),
        ],
    )
    return ProductStrategyProfileResponse(
        product_name=request.product_name,
        strategy_hypothesis=strategy,
        factor_hypotheses=factors,
        futures_categories=futures,
        disclaimer=(
            "该画像只来自产品名称和图片/披露文字的关键词证据，不代表真实因子暴露、"
            "实际持仓或交易品种。需以净值与市场因子序列回归、合同及定期报告核验。"
        ),
    )


def _ocr_text(image: np.ndarray) -> str | None:
    try:
        import pytesseract

        for language in ("chi_sim+eng", "eng"):
            try:
                text = pytesseract.image_to_string(image, lang=language).strip()
                if text:
                    return text
            except Exception:
                continue
    except Exception:
        pass
    return None


# A fund name in a chart legend or performance table usually ends with 号
# (e.g. 博衍留芳1号).  OCR frequently renders the legend dash separator as the
# CJK stroke 一 or an ASCII/Latin dash, so those are stripped from the start.
_FUND_NAME_PATTERN = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}号")
_LEADING_DASH_CHARS = "一-—–·"
# Column-header field names that indicate a table header rather than a product.
_HEADER_FIELD_WORDS = ("编码", "名称", "收益", "年化", "回撤", "波动", "夏普", "日期")


def _product_name_candidate(text: str | None) -> str | None:
    if not text:
        return None
    lines = [re.sub(r"\s+", "", line) for line in text.splitlines()]

    # Prefer an explicit fund-name token ending in 号 (most reliable signal).
    for line in lines:
        for match in _FUND_NAME_PATTERN.finditer(line):
            name = match.group(0).lstrip(_LEADING_DASH_CHARS)
            if any(word in name for word in _HEADER_FIELD_WORDS):
                continue
            if 2 <= len(name) <= 25:
                return name

    # Fall back to a short strategy/product keyword line that is not a table
    # header, a numeric data row, or a description paragraph.
    for line in lines:
        if not 4 <= len(line) <= 25:
            continue
        if line.count("|") >= 2 or any(word in line for word in _HEADER_FIELD_WORDS):
            continue
        if "%" in line or re.search(r"\d{4}[/-]\d", line):
            continue
        if any(punctuation in line for punctuation in "，。；：、"):
            continue
        if any(word in line for word in ("策略", "CTA", "基金", "产品", "增强")):
            return line
    return None


def _first_evidence(corpus: str, rules: list[tuple[str, tuple[str, ...], float]], fallback: str) -> StrategyEvidence:
    matches = _all_evidence(corpus, rules)
    return matches[0] if matches else StrategyEvidence(label=fallback, confidence=0.0, evidence=[])


def _keyword_in(keyword: str, corpus: str) -> bool:
    """Return True when *keyword* occurs in *corpus*.

    Very short ASCII instrument codes (``t``, ``tf``, ``if``, ``ic``, ``fx``,
    ...) must appear as standalone tokens; a plain substring check would match
    OCR noise such as ``ft ask`` for the code ``t``.  Longer keywords
    (``cta``, ``alpha``, ``global macro``) and CJK terms keep substring
    matching because they legitimately abut Chinese characters.
    """
    token = keyword.strip()
    if token.isascii() and len(token) <= 2:
        return re.search(rf"\b{re.escape(token)}\b", corpus) is not None
    return keyword in corpus


def _all_evidence(corpus: str, rules: list[tuple[str, tuple[str, ...], float]]) -> list[StrategyEvidence]:
    results: list[StrategyEvidence] = []
    for label, keywords, confidence in rules:
        matched = [keyword for keyword in keywords if _keyword_in(keyword, corpus)]
        if matched:
            results.append(StrategyEvidence(label=label, confidence=confidence, evidence=[f"关键词：{keyword}" for keyword in matched]))
    return results
