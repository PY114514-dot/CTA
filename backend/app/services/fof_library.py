"""Versioned local knowledge store for FOF products and research materials."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import FOF_LIBRARY_DIRECTORY

RESEARCH_STATUSES = {"needs_review", "researchable", "unusable"}
VERIFICATION_STATUSES = {"pending", "confirmed", "rejected"}


class FofLibraryStore:
    """SQLite metadata plus hash-addressed immutable material files."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or FOF_LIBRARY_DIRECTORY
        self.database_path = self.root / "knowledge.sqlite3"
        self.materials_path = self.root / "materials"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.materials_path.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY, name TEXT NOT NULL, manager_name TEXT,
                    strategy TEXT, verification_status TEXT NOT NULL DEFAULT 'pending',
                    research_status TEXT NOT NULL DEFAULT 'needs_review', research_status_reason TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS materials (
                    material_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
                    original_filename TEXT NOT NULL, media_type TEXT NOT NULL,
                    storage_path TEXT NOT NULL, byte_size INTEGER NOT NULL,
                    source_label TEXT, report_date TEXT, parsing_method TEXT NOT NULL DEFAULT 'manual_upload',
                    revision INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS product_materials (
                    product_id TEXT NOT NULL REFERENCES products(product_id),
                    material_id TEXT NOT NULL REFERENCES materials(material_id),
                    linked_at TEXT NOT NULL, PRIMARY KEY (product_id, material_id)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY, product_id TEXT REFERENCES products(product_id),
                    material_id TEXT NOT NULL REFERENCES materials(material_id),
                    claim_type TEXT NOT NULL, claim_value TEXT NOT NULL,
                    location_hint TEXT, confidence REAL NOT NULL,
                    verification_status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nav_observations (
                    nav_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(product_id),
                    observation_date TEXT NOT NULL, net_asset_value REAL NOT NULL,
                    frequency TEXT NOT NULL, evidence_id TEXT REFERENCES evidence(evidence_id),
                    verification_status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
                    UNIQUE(product_id, observation_date, frequency)
                );
                CREATE TABLE IF NOT EXISTS parsing_tasks (
                    task_id TEXT PRIMARY KEY, material_id TEXT NOT NULL REFERENCES materials(material_id),
                    task_type TEXT NOT NULL, status TEXT NOT NULL, result_summary TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)
            self._migrate_schema(connection)

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        """Migrate existing local libraries without discarding audit history."""
        material_columns = {row["name"] for row in connection.execute("PRAGMA table_info(materials)")}
        material_migrations = {
            "report_date": "ALTER TABLE materials ADD COLUMN report_date TEXT",
            "parsing_method": "ALTER TABLE materials ADD COLUMN parsing_method TEXT NOT NULL DEFAULT 'manual_upload'",
            "revision": "ALTER TABLE materials ADD COLUMN revision INTEGER NOT NULL DEFAULT 1",
        }
        for column, statement in material_migrations.items():
            if column not in material_columns:
                connection.execute(statement)
        product_columns = {row["name"] for row in connection.execute("PRAGMA table_info(products)")}
        product_migrations = {
            "research_status": "ALTER TABLE products ADD COLUMN research_status TEXT NOT NULL DEFAULT 'needs_review'",
            "research_status_reason": "ALTER TABLE products ADD COLUMN research_status_reason TEXT",
        }
        for column, statement in product_migrations.items():
            if column not in product_columns:
                connection.execute(statement)

    def create_product(self, name: str, manager_name: str | None = None, strategy: str | None = None) -> dict:
        now = _now()
        item = {"product_id": str(uuid4()), "name": name.strip(), "manager_name": manager_name, "strategy": strategy,
                "verification_status": "pending", "research_status": "needs_review", "research_status_reason": None,
                "created_at": now, "updated_at": now}
        if not item["name"]:
            raise ValueError("产品名称不能为空")
        with self._connect() as connection:
            connection.execute("""INSERT INTO products (
                product_id, name, manager_name, strategy, verification_status, research_status,
                research_status_reason, created_at, updated_at
            ) VALUES (
                :product_id, :name, :manager_name, :strategy, :verification_status, :research_status,
                :research_status_reason, :created_at, :updated_at
            )""", item)
        return item

    def list_products(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("""SELECT p.*, COUNT(DISTINCT pm.material_id) AS material_count,
                COUNT(DISTINCT e.evidence_id) AS evidence_count, COUNT(DISTINCT n.nav_id) AS nav_count
                FROM products p
                LEFT JOIN product_materials pm ON p.product_id = pm.product_id
                LEFT JOIN evidence e ON p.product_id = e.product_id
                LEFT JOIN nav_observations n ON p.product_id = n.product_id
                GROUP BY p.product_id ORDER BY p.updated_at DESC""").fetchall()
        return [dict(row) for row in rows]

    def list_researchable_products(self) -> list[dict]:
        return [product for product in self.list_products() if product["research_status"] == "researchable"]

    def review_product(
        self,
        product_id: str,
        verification_status: str,
        research_status: str,
        reason: str | None = None,
    ) -> dict:
        if verification_status not in VERIFICATION_STATUSES:
            raise ValueError("无效的产品确认状态")
        if research_status not in RESEARCH_STATUSES:
            raise ValueError("无效的产品研究状态")
        with self._connect() as connection:
            product = connection.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
            if product is None:
                raise ValueError("产品不存在")
            if research_status == "researchable":
                self._validate_researchable(connection, product_id, verification_status)
            connection.execute(
                """UPDATE products SET verification_status = ?, research_status = ?,
                    research_status_reason = ?, updated_at = ? WHERE product_id = ?""",
                (verification_status, research_status, reason, _now(), product_id),
            )
            return dict(connection.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone())

    @staticmethod
    def _validate_researchable(connection: sqlite3.Connection, product_id: str, verification_status: str) -> None:
        if verification_status != "confirmed":
            raise ValueError("产品身份确认后才能标记为可研究")
        linked_materials = connection.execute(
            "SELECT COUNT(*) FROM product_materials WHERE product_id = ?", (product_id,)
        ).fetchone()[0]
        nav_count, unreviewed_nav_count, unlinked_evidence_count = connection.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN n.verification_status != 'confirmed' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN pm.material_id IS NULL THEN 1 ELSE 0 END)
               FROM nav_observations n
               LEFT JOIN evidence e ON e.evidence_id = n.evidence_id
               LEFT JOIN product_materials pm ON pm.product_id = n.product_id AND pm.material_id = e.material_id
               WHERE n.product_id = ?""",
            (product_id,),
        ).fetchone()
        if not linked_materials:
            raise ValueError("关联原始材料后才能标记为可研究")
        if nav_count < 2:
            raise ValueError("至少需要两条净值记录才能标记为可研究")
        if unreviewed_nav_count:
            raise ValueError("所有净值记录完成复核后才能标记为可研究")
        if unlinked_evidence_count:
            raise ValueError("净值记录必须能回溯到关联的原始材料")

    def get_or_create_product(self, name: str, manager_name: str | None = None, strategy: str | None = None) -> dict:
        """Resolve an OCR candidate conservatively without promoting its status."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM products WHERE name = ? AND COALESCE(manager_name, '') = COALESCE(?, '') ORDER BY created_at LIMIT 1",
                (name.strip(), manager_name),
            ).fetchone()
        return dict(row) if row else self.create_product(name, manager_name, strategy)

    def ingest_material(
        self,
        content: bytes,
        filename: str,
        media_type: str,
        source_label: str | None = None,
        report_date: str | None = None,
        parsing_method: str = "manual_upload",
    ) -> dict:
        if not content:
            raise ValueError("上传文件为空")
        digest = hashlib.sha256(content).hexdigest()
        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM materials WHERE sha256 = ?", (digest,)).fetchone()
            if existing:
                return {**dict(existing), "duplicate": True}
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "material.bin"
        relative_path = Path("materials") / digest[:2] / f"{digest}_{safe_name}"
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        item = {"material_id": str(uuid4()), "sha256": digest, "original_filename": filename,
                "media_type": media_type or "application/octet-stream", "storage_path": str(relative_path),
                "byte_size": len(content), "source_label": source_label, "report_date": report_date,
                "parsing_method": parsing_method or "manual_upload", "revision": 1, "created_at": _now()}
        with self._connect() as connection:
            connection.execute("""INSERT INTO materials (
                material_id, sha256, original_filename, media_type, storage_path, byte_size,
                source_label, report_date, parsing_method, revision, created_at
            ) VALUES (
                :material_id, :sha256, :original_filename, :media_type, :storage_path, :byte_size,
                :source_label, :report_date, :parsing_method, :revision, :created_at
            )""", item)
        return {**item, "duplicate": False}

    def list_materials(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM materials ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def link_material(self, product_id: str, material_id: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT OR IGNORE INTO product_materials VALUES (?, ?, ?)", (product_id, material_id, _now()))

    def get_material(self, material_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,)).fetchone()
        return dict(row) if row else None

    def create_parsing_task(self, material_id: str, task_type: str = "single_material_research") -> dict:
        now = _now()
        item = {"task_id": str(uuid4()), "material_id": material_id, "task_type": task_type,
                "status": "running", "result_summary": None, "created_at": now, "updated_at": now}
        with self._connect() as connection:
            connection.execute("INSERT INTO parsing_tasks VALUES (:task_id, :material_id, :task_type, :status, :result_summary, :created_at, :updated_at)", item)
        return item

    def finish_parsing_task(self, task_id: str, result_summary: str, status: str = "completed") -> None:
        with self._connect() as connection:
            connection.execute("UPDATE parsing_tasks SET status = ?, result_summary = ?, updated_at = ? WHERE task_id = ?", (status, result_summary, _now(), task_id))

    def add_evidence(self, *, material_id: str, claim_type: str, claim_value: str,
                     location_hint: str | None, confidence: float, product_id: str | None = None) -> dict:
        item = {"evidence_id": str(uuid4()), "product_id": product_id, "material_id": material_id,
                "claim_type": claim_type, "claim_value": claim_value, "location_hint": location_hint,
                "confidence": confidence, "verification_status": "pending", "created_at": _now()}
        with self._connect() as connection:
            connection.execute("""INSERT INTO evidence VALUES (:evidence_id, :product_id, :material_id, :claim_type,
                :claim_value, :location_hint, :confidence, :verification_status, :created_at)""", item)
        return item

    def add_nav_observation(
        self,
        *,
        product_id: str,
        observation_date: str,
        net_asset_value: float,
        frequency: str,
        evidence_id: str | None,
        verification_status: str = "pending",
    ) -> dict:
        if net_asset_value <= 0:
            raise ValueError("净值必须为正数")
        if verification_status not in VERIFICATION_STATUSES:
            raise ValueError("无效的净值确认状态")
        item = {"nav_id": str(uuid4()), "product_id": product_id, "observation_date": observation_date,
                "net_asset_value": net_asset_value, "frequency": frequency, "evidence_id": evidence_id,
                "verification_status": verification_status, "created_at": _now()}
        with self._connect() as connection:
            connection.execute("""INSERT INTO nav_observations VALUES (
                :nav_id, :product_id, :observation_date, :net_asset_value, :frequency,
                :evidence_id, :verification_status, :created_at
            )""", item)
        return item

    def list_product_evidence(self, product_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE product_id = ? ORDER BY created_at, claim_type", (product_id,)
            ).fetchall()
        return [dict(row) for row in rows]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
