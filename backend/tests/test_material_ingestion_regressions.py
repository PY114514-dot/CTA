from datetime import date, timedelta
from io import BytesIO

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import NavCandidateVersion, NavObservation, ProductEntity, RawFile
from app.schemas import MultiProductReportResponse, ReportProductIdentity
from app.services.material_ingestion import (
    _ingest_report_image,
    _ingest_nav_table,
    _ingest_pdf,
    _disclosed_nav_image_variants,
    _is_multi_product_material,
    _looks_like_pdf_product_curve,
    _merge_extraction_audit,
    _normalise_observation_date,
    _normalise_manifest_product_name,
    _parse_disclosed_nav_payload,
    _parse_monthly_return_rows,
    _recover_monthly_nav_from_return_table,
    _recover_monthly_nav_from_pdf_text,
)


def _memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_rejects_incomplete_text_and_implausible_visual_dates() -> None:
    assert _normalise_observation_date("2023") is None
    assert _normalise_observation_date("农产品") is None
    assert _normalise_observation_date("2078-12-02") is None
    assert _normalise_observation_date("2026-01-09") == "2026-01-09"
    assert _normalise_observation_date(date.today() + timedelta(days=91)) is None


def test_table_ingestion_returns_and_persists_nav_count() -> None:
    session = _memory_session()
    raw_file = RawFile(file_hash="a" * 64, filename="测试产品.csv")
    session.add(raw_file)
    session.commit()
    content = "日期,净值\n2026-01-02,1.01\n2026-01-09,1.03\n".encode()

    rows = _ingest_nav_table(session, raw_file.id, content, raw_file.filename, ".csv")

    assert rows == 2
    assert len(session.scalars(select(NavObservation)).all()) == 2


def test_audit_distinguishes_configuration_attempt_and_failure() -> None:
    configured_only = {
        "configured": True,
        "attempted": False,
        "succeeded": False,
        "provider": "dashscope",
        "model": "qwen3-vl-flash",
    }
    failed_call = {
        "configured": True,
        "attempted": True,
        "succeeded": False,
        "calls": 1,
        "failed_calls": 1,
        "errors": ["provider timeout"],
    }

    first = _merge_extraction_audit(configured_only)
    failed = _merge_extraction_audit(configured_only, failed_call)

    assert first["configured"] is True and first["attempted"] is False
    assert failed["attempted"] is True and failed["succeeded"] is False
    assert failed["failed_calls"] == 1
    assert failed["errors"] == ["provider timeout"]


def test_generic_pdf_curve_labels_cannot_create_products() -> None:
    for label in ("产品净值", "图例文字", "超额收益", "沪深300指数", "农产品板块"):
        assert _looks_like_pdf_product_curve(label, "研究报告") is False
    assert _looks_like_pdf_product_curve("致远二号", "研究报告") is True


def test_monthly_return_table_parser_rejects_invalid_cells() -> None:
    raw = """{"monthly_returns":[
      {"year":2025,"month":1,"return_pct":1.2},
      {"year":2025,"month":2,"return_pct":-0.5},
      {"year":2078,"month":3,"return_pct":2.0},
      {"year":2025,"month":13,"return_pct":1.0},
      {"year":2025,"month":4,"return_pct":-99.0}
    ]}"""
    assert _parse_monthly_return_rows(raw) == [(2025, 1, 0.012), (2025, 2, -0.005)]


def test_disclosed_nav_table_requires_complete_plausible_dates_and_unit_nav() -> None:
    points, latest = _parse_disclosed_nav_payload("""{
      "nav_rows": [
        {"date":"2026-06-18","unit_nav":1.9800},
        {"date":"2026-06-26","unit_nav":1.9740},
        {"date":"2026-07-03","unit_nav":1.9810},
        {"date":"2078-01-01","unit_nav":9.9},
        {"date":"2026","unit_nav":1.0}
      ],
      "latest_unit_nav": 1.9810
    }""")

    assert [point["observation_date"] for point in points] == [
        "2026-06-18", "2026-06-26", "2026-07-03",
    ]
    assert latest == 1.981


def test_tall_screenshot_adds_enlarged_lower_table_variant() -> None:
    from PIL import Image

    image = Image.new("RGB", (600, 1200), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    variants = _disclosed_nav_image_variants(buffer.getvalue())

    assert [name for name, _ in variants] == ["full_page", "enlarged_lower_section"]
    with Image.open(BytesIO(variants[1][1])) as lower:
        assert lower.width >= 1800


def test_pdf_text_monthly_returns_rebuild_nav_only_when_totals_agree() -> None:
    text = """2026年2月月度运作报告
产品名称 份额设立日 累计净值 单位净值
某CTA 2025/10/23
月收益
2025 1.00% 2.00% 3.02%
2026 1.00% -1.00% -0.01%
"""
    points = _recover_monthly_nav_from_pdf_text(text)
    assert [point["observation_date"] for point in points] == [
        "2025-10-23", "2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
    ]

    bad = text.replace("3.02%", "8.00%")
    assert _recover_monthly_nav_from_pdf_text(bad) == []


def test_monthly_return_table_rejects_shifted_cells_instead_of_left_shifting(monkeypatch) -> None:
    class FakeProvider:
        async def extract_structure(self, _image: bytes, _prompt: str) -> str:
            return """{
              "monthly_returns": [
                {"year": 2025, "month": 1, "return_pct": 9.0},
                {"year": 2025, "month": 2, "return_pct": 1.0},
                {"year": 2025, "month": 3, "return_pct": 2.0}
              ],
              "annual_returns": [{"year": 2025, "return_pct": 3.02}],
              "latest_nav": null
            }"""

    monkeypatch.setenv("VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLM_BASE_URL", "http://test.invalid")
    monkeypatch.setattr(
        "app.services.chart_extractor.vlm_extractor.create_provider",
        lambda *_args, **_kwargs: FakeProvider(),
    )

    points, audit = _recover_monthly_nav_from_return_table(b"table")

    assert points == []
    assert audit["succeeded"] is False
    assert "复利" in audit["errors"][0]


def test_pdf_text_monthly_returns_uses_filename_period_when_pdf_title_is_unreadable() -> None:
    text = """2023 0.17% -0.48% -0.44% -0.37%
2024 -2.76% -0.30% 4.08% 6.97% 1.39% -3.40% 1.64% -1.73% 14.94% 1.96% 2.05% 0.54% 26.96%
2025 2.94% -0.94% 0.74% 0.43% -0.92% 1.82% 0.57% 0.37% 0.11% 0.91% -0.34% 5.49% 11.57%
2026 4.83% -2.73% -1.76% -0.62% -2.27% -0.98% -3.99%
2023/9/1
2026/6/1
"""
    points = _recover_monthly_nav_from_pdf_text(
        text,
        filename="博润吾同CTA1号A份额2606月报.pdf",
    )
    assert len(points) == 35
    assert points[0] == {"observation_date": "2023-08-31", "nav": 1.0}
    assert points[4]["observation_date"] == "2023-12-31"
    assert points[-1]["observation_date"] == "2026-06-30"


def test_manifest_identity_strips_material_suffix_but_rejects_generic_reports() -> None:
    assert _normalise_manifest_product_name("容成商品趋势系列净值周报") == "容成商品趋势系列"
    assert _normalise_manifest_product_name("量化CTA1号周报") == "量化CTA1号"
    assert _normalise_manifest_product_name("代表产品业绩周报") == ""
    assert _normalise_manifest_product_name("CTA产品总月报") == ""


def test_multi_product_materials_are_routed_to_binding() -> None:
    assert _is_multi_product_material("上海秉昊_多产品周报.jpg", "多产品周报") is True
    assert _is_multi_product_material("某机构_产品汇总.png", None) is True
    assert _is_multi_product_material("艾方_艾方CTA进取1号.jpg", "艾方CTA进取1号") is False


def test_multi_product_filename_blocks_ocr_title_identity() -> None:
    assert _is_multi_product_material("上海秉昊_多产品周报.jpg", "增强配置") is True


def test_multi_product_material_does_not_recover_table_into_hint_product(monkeypatch) -> None:
    session = _memory_session()
    raw_file = RawFile(file_hash="b" * 64, filename="机构_多产品周报.png")
    session.add(raw_file)
    session.commit()

    monkeypatch.setattr(
        "app.services.material_ingestion.image.extract_multi_product_report",
        lambda _content: MultiProductReportResponse(
            disclosed_metrics=[],
            product_curve_candidates=[],
            product_identity=ReportProductIdentity(),
        ),
    )
    monkeypatch.setattr(
        "app.services.material_ingestion.image._trace_whole_image",
        lambda *_args: {"configured": False, "attempted": False, "succeeded": False},
    )

    def unexpected_table_recovery(_content: bytes):
        raise AssertionError("multi-product material must wait for binding before table recovery")

    monkeypatch.setattr(
        "app.services.material_ingestion.image._recover_disclosed_nav_table",
        unexpected_table_recovery,
    )

    _ingest_report_image(
        session,
        raw_file.id,
        b"image",
        raw_file.filename,
        product_name_hint="增强配置",
    )

    assert session.scalars(select(NavCandidateVersion)).all() == []


def test_pdf_monthly_recovery_waits_for_product_binding(monkeypatch) -> None:
    session = _memory_session()
    raw_file = RawFile(file_hash="c" * 64, filename="机构多产品月报.pdf")
    session.add(raw_file)
    session.commit()

    class FakePage:
        def extract_text(self) -> str:
            return """2026年2月月度运作报告
产品名称 份额设立日 累计净值 单位净值
某CTA 2025/10/23
月收益
2025 1.00% 2.00% 3.02%
2026 1.00% -1.00% -0.01%
"""

    class FakeReader:
        def __init__(self, _content: BytesIO) -> None:
            self.pages = [FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", FakeReader)
    monkeypatch.setattr("app.services.material_ingestion.pdf._auto_curve_colors_from_pdf", lambda _content: [])
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("VLM_BASE_URL", raising=False)

    audit = _ingest_pdf(session, raw_file.id, b"pdf", raw_file.filename)

    assert session.scalars(select(NavCandidateVersion)).all() == []
    assert session.scalars(select(ProductEntity)).all() == []
    assert any("待产品绑定" in method for method in audit["methods"])
