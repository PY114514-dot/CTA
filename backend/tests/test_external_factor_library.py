from pathlib import Path

from app.services import external_factor_library as library


def test_gtja_fixture_parses_all_eight_factors() -> None:
    text = (Path(__file__).parent / "fixtures" / "gtja_cta_factor_table.txt").read_text(encoding="utf-8")

    report_date, observations = library.parse_gtja_factor_table(text)

    assert report_date == "2026-07-24"
    assert len(observations) == 40
    assert next(item for item in observations if item["factor_id"] == "profit" and item["horizon"] == "1y")["return_pct"] == 7.32


def test_ingest_is_idempotent_and_export_is_a_wide_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(library, "EXTERNAL_FACTOR_LIBRARY_DIRECTORY", tmp_path)
    text = (Path(__file__).parent / "fixtures" / "gtja_cta_factor_table.txt").read_text(encoding="utf-8")
    source = text.encode("utf-8")

    first = library.ingest_external_document(source, "weekly.pdf", text, "pdf_text", "国泰君安期货")
    second = library.ingest_external_document(source, "weekly.pdf", text, "pdf_text", "国泰君安期货")
    report_date, csv_bytes = library.export_latest_csv()

    assert first["rows_added"] == 40
    assert second["duplicate"] is True
    assert report_date == "2026-07-24"
    assert "source_sha256" not in csv_bytes.decode("utf-8-sig")
    assert "短期时序动量,-0.84%,-2.55%" in csv_bytes.decode("utf-8-sig")
