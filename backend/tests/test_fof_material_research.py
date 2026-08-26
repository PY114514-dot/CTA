from pathlib import Path

from app.services.fof_library import FofLibraryStore
from app.services.fof_material_research import _build_report


def test_report_is_explicitly_evidence_grounded(tmp_path: Path) -> None:
    store = FofLibraryStore(tmp_path)
    material = store.ingest_material(b"image", "周报.png", "image/png")
    evidence = [store.add_evidence(material_id=material["material_id"], claim_type="累计收益", claim_value="162.10%", location_hint="normalized:y=0.1-0.2", confidence=.8)]
    report = _build_report(material, [{"product_name": "启明星一号", "metrics": {"累计收益": "162.10%"}}], "量化CTA", evidence)
    assert "162.10%" in report
    assert "pending" in report
    assert "不能直接形成申购建议" in report
