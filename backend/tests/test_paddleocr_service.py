import json

from app.services.paddleocr_service import _parse_jsonl


def test_parse_jsonl_returns_markdown_pages() -> None:
    payload = "\n".join([
        json.dumps({"result": {"layoutParsingResults": [
            {"markdown": {"text": "# 第一页"}},
            {"markdown": {"text": "第二个版面"}},
        ]}}, ensure_ascii=False),
        "not json",
    ])

    result = _parse_jsonl(payload)

    assert [page.markdown for page in result.pages] == ["# 第一页", "第二个版面"]
    assert result.pages[1].page_number == 2
    assert len(result.warnings) == 1
