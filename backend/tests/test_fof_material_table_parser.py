from app.services.fof_material_research import _parse_consolidated_table_text, _product_name_from_title_text, _title_hits_from_ocr_data


def test_consolidated_table_parser_is_manager_agnostic() -> None:
    text = """
基金名称 单位净值 累计净值 上周收益率 今年以来收益率 成立以来收益率 年化收益率
星河稳健一号 1.2345 1.6789 -0.50% 2.30% 67.89% 12.34% 8.90% 15.60%
远山量化对冲 0.9988 1.1223 0.20% -1.10% 12.23% 6.40% 4.30% 9.20%
"""
    rows = _parse_consolidated_table_text(text)
    assert [row["product_name"] for row in rows] == ["星河稳健一号", "远山量化对冲"]
    assert rows[0]["metrics"]["累计收益率"] == "67.89%"
    assert rows[1]["metrics"]["年化收益率"] == "6.40%"


def test_title_parser_only_accepts_visible_title_fields() -> None:
    assert _product_name_from_title_text("量化 CTA（中频） -- 启明星一号") == "启明星一号"
    assert _product_name_from_title_text("产品名称：远山多策略一号") == "远山多策略一号"
    assert _product_name_from_title_text("数据来源：托管平台复核") is None


def test_title_hit_parser_retains_ocr_vertical_anchor() -> None:
    data = {"block_num": [1, 1, 1], "par_num": [1, 1, 1], "line_num": [1, 1, 1], "top": [44, 44, 44], "text": ["量化CTA", "--", "启明星一号"]}
    assert _title_hits_from_ocr_data(data) == [(44, "启明星一号")]
