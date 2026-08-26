"""Regression coverage for choosing a PDF review page before chart VLM runs."""

from app.services.review_page_locator import locate_review_page


def test_pdf_text_rank_prefers_product_performance_page_over_cover() -> None:
    result = locate_review_page(
        page_texts=[
            "卓银量化选股 2026 年 6 月产品介绍 封面",
            "投资理念与团队介绍",
            "产品业绩表现\n累计收益 26.85%\n年度年化收益 18.20%\n最大回撤 -4.10%",
        ],
        product_names=["卓银量化选股"],
        filename="卓银量化选股20260626.pdf",
    )

    assert result.page_number == 3
    assert result.selection_method == "pdf_text_rank"
    assert "产品业绩表现" in result.reason


def test_pdf_text_rank_does_not_fall_back_to_cover_without_evidence() -> None:
    result = locate_review_page(
        page_texts=["某私募产品封面", "管理人介绍与免责声明"],
        product_names=["某产品"],
        filename="某产品介绍.pdf",
    )

    assert result.page_number is None
    assert result.selection_method == "not_found"


def test_existing_chart_evidence_takes_priority() -> None:
    result = locate_review_page(
        page_texts=["产品业绩表现 累计收益 最大回撤"],
        product_names=["某产品"],
        filename="某产品.pdf",
        known_page_numbers=[8],
    )

    assert result.page_number == 8
    assert result.selection_method == "existing_chart_evidence"


def test_rank_prefers_target_product_page_in_multi_product_pdf() -> None:
    result = locate_review_page(
        page_texts=[
            "产品甲 产品业绩表现 净值曲线 累计收益 年化收益 最大回撤",
            "产品乙 历史业绩 累计收益",
        ],
        product_names=["产品乙"],
        filename="多产品周报.pdf",
    )

    assert result.page_number == 2


def test_existing_evidence_preserves_caller_quality_order() -> None:
    result = locate_review_page(
        page_texts=[],
        product_names=["某产品"],
        filename="某产品.pdf",
        known_page_numbers=[12, 3],
    )

    assert result.page_number == 12
