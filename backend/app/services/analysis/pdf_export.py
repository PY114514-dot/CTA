"""Small, dependency-free PDF renderer for saved research reports."""

from __future__ import annotations

from pathlib import Path

import pymupdf


_A4_WIDTH = 595
_A4_HEIGHT = 842
_MARGIN_X = 42
_MARGIN_TOP = 48
_MARGIN_BOTTOM = 42
_FONT_SIZE = 9.5
_LINE_HEIGHT = 15
_CHINESE_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")


def _wrap(line: str, width: int = 56) -> list[str]:
    """Wrap plain Markdown text conservatively for an A4 text report."""
    line = line.replace("\t", "  ").strip()
    if not line:
        return [""]
    return [line[index:index + width] for index in range(0, len(line), width)]


def render_markdown_pdf(title: str, markdown: str) -> bytes:
    """Render a Chinese-capable, downloadable PDF from report Markdown.

    The research report is already a reviewable text snapshot.  Rendering that
    snapshot server-side avoids the browser popup/print-dialog dependency that
    prevented PDF exports in embedded browsers.
    """
    document = pymupdf.open()
    page = None
    y = float(_MARGIN_TOP)

    def new_page() -> tuple[object, float]:
        next_page = document.new_page(width=_A4_WIDTH, height=_A4_HEIGHT)
        if _CHINESE_FONT.exists():
            next_page.insert_font(fontname="msyh", fontfile=str(_CHINESE_FONT))
        return next_page, float(_MARGIN_TOP)

    def write_line(text: str, size: float = _FONT_SIZE) -> None:
        nonlocal page, y
        if page is None or y + _LINE_HEIGHT > _A4_HEIGHT - _MARGIN_BOTTOM:
            page, y = new_page()
        font_kwargs = {"fontname": "msyh"} if _CHINESE_FONT.exists() else {"fontname": "helv"}
        page.insert_text((_MARGIN_X, y), text, fontsize=size, color=(0.12, 0.23, 0.33), **font_kwargs)
        y += _LINE_HEIGHT

    write_line(title or "产品净值分析报告", 15)
    y += 6
    for raw_line in markdown.splitlines():
        if raw_line.startswith("## "):
            for part in _wrap(raw_line[3:], 42):
                write_line(part, 12)
            y += 3
            continue
        if raw_line.startswith("### "):
            for part in _wrap(raw_line[4:], 48):
                write_line(part, 10.5)
            continue
        # The Markdown table separators contain no research content.
        if raw_line.startswith("|") and set(raw_line.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        for part in _wrap(raw_line):
            write_line(part)

    document.set_metadata({"title": title or "产品净值分析报告", "author": "私募 CTA 研究平台"})
    result = document.tobytes(garbage=4, deflate=True)
    document.close()
    return result
