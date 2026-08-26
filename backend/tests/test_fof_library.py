from pathlib import Path
import sqlite3

from app.services.fof_library import FofLibraryStore
from app.config import persist_local_environment


def test_materials_are_hash_deduplicated_and_linkable_to_a_product(tmp_path: Path) -> None:
    store = FofLibraryStore(tmp_path)
    product = store.create_product("测试产品", manager_name="测试管理人", strategy="cta")
    first = store.ingest_material(
        b"source report", "周报.pdf", "application/pdf", report_date="2026-08-10", parsing_method="pdf_text",
    )
    duplicate = store.ingest_material(b"source report", "周报副本.pdf", "application/pdf")

    store.link_material(product["product_id"], first["material_id"])

    products = store.list_products()
    assert first["duplicate"] is False
    assert duplicate["duplicate"] is True
    assert len(store.list_materials()) == 1
    assert products[0]["material_count"] == 1
    assert first["report_date"] == "2026-08-10"
    assert first["parsing_method"] == "pdf_text"
    assert first["revision"] == 1


def test_existing_library_is_migrated_with_material_audit_defaults(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE materials (
                material_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
                original_filename TEXT NOT NULL, media_type TEXT NOT NULL,
                storage_path TEXT NOT NULL, byte_size INTEGER NOT NULL,
                source_label TEXT, created_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE products (
                product_id TEXT PRIMARY KEY, name TEXT NOT NULL, manager_name TEXT,
                strategy TEXT, verification_status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """INSERT INTO products VALUES (
                'existing-product', '历史产品', NULL, NULL, 'pending', '2026-08-01', '2026-08-01'
            )"""
        )

    store = FofLibraryStore(tmp_path)
    material = store.ingest_material(b"new report", "周报.pdf", "application/pdf")

    assert material["parsing_method"] == "manual_upload"
    assert material["revision"] == 1
    assert store.list_products()[0]["research_status"] == "needs_review"


def test_only_confirmed_products_with_reviewed_navs_can_be_researched(tmp_path: Path) -> None:
    store = FofLibraryStore(tmp_path)
    product = store.create_product("复核产品")
    material = store.ingest_material(b"source report", "report.pdf", "application/pdf")
    store.link_material(product["product_id"], material["material_id"])
    evidence = store.add_evidence(
        material_id=material["material_id"], product_id=product["product_id"], claim_type="单位净值",
        claim_value="1.02", location_hint="page=1", confidence=0.9,
    )

    try:
        store.review_product(product["product_id"], "confirmed", "researchable")
    except ValueError as error:
        assert "至少需要两条净值记录" in str(error)
    else:
        raise AssertionError("未复核的净值不应进入正式研究")

    for date, nav in (("2026-08-01", 1.0), ("2026-08-08", 1.02)):
        store.add_nav_observation(
            product_id=product["product_id"], observation_date=date, net_asset_value=nav, frequency="weekly",
            evidence_id=evidence["evidence_id"], verification_status="confirmed",
        )
    reviewed = store.review_product(product["product_id"], "confirmed", "researchable")

    assert reviewed["research_status"] == "researchable"
    assert [item["product_id"] for item in store.list_researchable_products()] == [product["product_id"]]


def test_local_environment_updates_replace_existing_values(monkeypatch, tmp_path: Path) -> None:
    import app.config as config

    env_file = tmp_path / ".env"
    env_file.write_text("# local settings\nVLM_MODEL=old-model\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path.parent)
    monkeypatch.setattr(config, "persist_local_environment", config.persist_local_environment)

    # The helper's configured location is derived from PROJECT_ROOT/backend.
    backend = tmp_path.parent / "backend"
    backend.mkdir(exist_ok=True)
    (backend / ".env").write_text(env_file.read_text(encoding="utf-8"), encoding="utf-8")
    persist_local_environment({"VLM_MODEL": "qwen3-vl-flash", "VLM_PROVIDER": "dashscope"})

    saved = (backend / ".env").read_text(encoding="utf-8")
    assert "VLM_MODEL=qwen3-vl-flash" in saved
    assert "VLM_PROVIDER=dashscope" in saved
