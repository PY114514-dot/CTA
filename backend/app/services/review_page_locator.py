"""Choose the source page shown before single-product chart review.

The chart VLM only receives one rendered image.  It cannot navigate a PDF on
its own, so page choice must happen first and must never silently fall back to
the cover page.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(frozen=True)
class ReviewPageLocation:
    """Auditable page-selection result used by the review UI."""

    page_number: int | None
    confidence: float
    reason: str
    selection_method: str


# A chart-specific phrase should outweigh a generic product-name mention on a
# cover or a team-introduction page.  The lower-tier metrics make performance
# tables reviewable too, but do not by themselves turn a disclaimer into a
# chart candidate.
_PERFORMANCE_SIGNALS: tuple[tuple[str, int], ...] = (
    ("历史业绩", 22),
    ("产品业绩表现", 22),
    ("业绩表现", 13),
    ("净值曲线", 22),
    ("净值走势图", 22),
    ("累计净值", 16),
    ("历史净值", 18),
    ("单位净值", 14),
    ("周度净值", 16),
    ("月度净值", 16),
    ("累计收益", 8),
    ("年化收益", 8),
    ("最大回撤", 7),
    ("夏普比率", 4),
)
_STRONG_SIGNALS = {signal for signal, weight in _PERFORMANCE_SIGNALS if weight >= 14}
_DATE_SUFFIX = re.compile(r"(?:19|20)\d{6}(?:\d{2})?")
_NON_WORD = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def locate_review_page(
    *,
    page_texts: Sequence[str],
    product_names: Sequence[str],
    filename: str,
    known_page_numbers: Sequence[int] = (),
) -> ReviewPageLocation:
    """Return the most defensible performance page, or an explicit miss.

    ``page_number`` is 1-based.  A missing result deliberately stays ``None``
    rather than returning page 1: PDF covers are not chart evidence.
    """
    known: list[int] = []
    for page in known_page_numbers:
        try:
            parsed = int(page)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            known.append(parsed)
    # The caller supplies evidence in quality order (traced/reviewed/high
    # confidence first).  Preserve that order instead of turning "earliest
    # page" into an accidental proxy for evidence quality.
    known = list(dict.fromkeys(known))
    if known:
        return ReviewPageLocation(
            page_number=known[0],
            confidence=0.99,
            reason="已保存的图表追踪证据",
            selection_method="existing_chart_evidence",
        )

    product_tokens = _product_tokens([*product_names, _filename_stem(filename)])
    candidates: list[tuple[int, int, list[str], bool, bool]] = []

    for page_index, raw_text in enumerate(page_texts):
        text = _normalise(raw_text)
        if not text:
            continue
        score = 0
        reasons: list[str] = []
        for signal, weight in _PERFORMANCE_SIGNALS:
            if signal in text:
                score += weight
                reasons.append(signal)
        has_strong_signal = any(signal in text for signal in _STRONG_SIGNALS)
        product_matched = False
        for token in product_tokens:
            if token in text:
                score += 10
                reasons.append("产品名称")
                product_matched = True
                break

        # A cover often contains only the product title.  Require a real
        # performance signal before it can be considered a review page.
        if not has_strong_signal and score < 16:
            continue
        candidates.append((page_index + 1, score, list(dict.fromkeys(reasons)), has_strong_signal, product_matched))

    if not candidates:
        return ReviewPageLocation(
            page_number=None,
            confidence=0.0,
            reason="未在 PDF 文本层找到产品业绩或净值页",
            selection_method="not_found",
        )

    # In a multi-product report a direct target-name match outranks a generic
    # page with more performance vocabulary.  If the document never repeats
    # the product name (common in single-product pitchbooks), retain the best
    # semantic performance page instead of rejecting a useful document.
    matched = [candidate for candidate in candidates if candidate[4]]
    pool = matched or candidates
    best_page, best_score, best_reasons, best_has_strong_signal, _ = max(pool, key=lambda candidate: candidate[1])
    if not best_has_strong_signal and best_score < 24:
        return ReviewPageLocation(
            page_number=None,
            confidence=0.0,
            reason="未在 PDF 文本层找到产品业绩或净值页",
            selection_method="not_found",
        )
    confidence = min(0.95, round(0.50 + best_score / 100, 2))
    return ReviewPageLocation(
        page_number=best_page,
        confidence=confidence,
        reason="、".join(best_reasons[:4]) or "产品业绩关键词",
        selection_method="pdf_text_rank",
    )


def _product_tokens(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        cleaned = _DATE_SUFFIX.sub("", value or "")
        cleaned = _normalise(cleaned)
        if len(cleaned) >= 3:
            tokens.append(cleaned)
    # Preserve order so a canonical product name is tried before filename
    # fragments, while removing duplicate values.
    return list(dict.fromkeys(tokens))


def _normalise(value: str) -> str:
    return _NON_WORD.sub("", value or "").lower()


def _filename_stem(filename: str) -> str:
    # pathlib treats Windows backslashes as ordinary characters on POSIX test
    # runners.  Normalising both separators keeps selection deterministic.
    base = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0]
