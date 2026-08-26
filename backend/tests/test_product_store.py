"""Tests for P0 data models: traceability, confirmation workflow and merge."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import (
    ConfirmationStatus,
    DocumentFragment,
    DecisionStatus,
    NavObservation,
    ProductAlias,
    ProductEntity,
    ReviewStatus,
    RunPhase,
)
from app.services import product_store as store


@pytest.fixture()
def session():
    """In-memory SQLite session with all tables created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        yield s


# ---------------------------------------------------------------------------
# File registration & deduplication
# ---------------------------------------------------------------------------


class TestFileRegistration:
    def test_register_and_version(self, session: Session):
        f1 = store.register_file(session, filename="report.pdf", file_hash="abc123")
        assert f1.version == 1
        assert f1.parsing_status == "pending"

        # Same hash → new version.
        f2 = store.register_file(session, filename="report.pdf", file_hash="abc123")
        assert f2.version == 2

    def test_compute_hash(self):
        h = store.compute_file_hash(b"hello world")
        assert len(h) == 64  # SHA-256 hex

    def test_list_pending_files_excludes_completed_and_bulk_sources(self, session: Session):
        pending = store.register_file(session, filename="待识别.pdf", file_hash="pending-file")
        completed = store.register_file(session, filename="已完成.pdf", file_hash="completed-file")
        bulk = store.register_file(
            session, filename="期货周频.sqlite", file_hash="bulk-file", source="futures_weekly_sqlite"
        )
        unbound = store.register_file(session, filename="待绑定.pdf", file_hash="unbound-file")
        review = store.register_file(session, filename="待复核.pdf", file_hash="review-file")
        product = store.create_product(session, standard_name="待复核产品")

        store.update_file_status(session, completed.id, "completed")
        store.update_file_status(session, bulk.id, "completed")
        store.update_file_status(session, unbound.id, "completed")
        store.update_file_status(session, review.id, "completed")
        store.add_fragment(session, file_id=unbound.id, fragment_type="table")
        store.add_nav_observations(
            session, product.id, [{"observation_date": "2026-01-02", "nav": 1.0}], source_file_id=review.id
        )

        assert {item.id for item in store.list_files(session, pending_only=True)} == {
            pending.id, unbound.id, review.id
        }

    def test_delete_removes_source_dependent_research_records(self, session: Session):
        """Deleting a source must not leave unauditable research data behind."""
        session.execute(text("PRAGMA foreign_keys=ON"))
        source = store.register_file(session, filename="source.png", file_hash="delete_hash")
        product = store.create_product(session, standard_name="待删除来源产品")
        fragment = store.add_fragment(
            session, file_id=source.id, fragment_type="table",
            content_text="累计收益 1%", product_id=product.id,
        )
        store.add_nav_observations(
            session, product.id,
            [{"observation_date": "2025-01-01", "nav": 1.0}],
            source_file_id=source.id, source_fragment_id=fragment.id,
        )
        store.add_fact(
            session, product_id=product.id, field_name="aum", field_value="100",
            source_file_id=source.id, source_fragment_id=fragment.id,
        )
        alias = store.add_alias(session, product.id, "来源别名", source_file_id=source.id)

        deleted = store.delete_file(session, source.id)
        assert deleted["storage_path"] is None
        assert deleted["removed_nav_count"] == 1
        assert deleted["removed_fact_count"] == 1
        assert deleted["removed_alias_count"] == 1
        assert store.list_files(session) == []
        assert store.get_nav_series(session, product.id) == []
        assert store.get_facts_for_product(session, product.id) == []
        assert session.get(ProductAlias, alias.id) is None


class TestFileProductBinding:
    def test_merge_drops_duplicate_nav_dates_from_source(self, session: Session):
        source = store.create_product(session, standard_name="策略别名")
        target = store.create_product(session, standard_name="真实产品")
        store.add_nav_observations(session, source.id, [{"observation_date": "2026-01-02", "nav": 1.01}])
        store.add_nav_observations(session, target.id, [{"observation_date": "2026-01-02", "nav": 1.02}])

        store.merge_products(session, source.id, target.id)

        series = store.get_nav_series(session, target.id)
        assert [(row.observation_date.isoformat(), row.nav) for row in series] == [("2026-01-02", 1.02)]

    def test_bind_unmatched_fragment_moves_curve_nav_and_adds_alias(self, session: Session):
        source = store.register_file(session, filename="致远2号_2026Q2.png", file_hash="bind_hash")
        temporary = store.create_product(session, standard_name="致远2号 / 待绑定曲线 1")
        target = store.create_product(session, standard_name="致远2号")
        fragment = store.add_fragment(
            session,
            file_id=source.id,
            fragment_type="chart_traced",
            content_data={"curve_index": 0, "binding_status": "unmatched"},
            product_id=temporary.id,
        )
        store.add_nav_observations(
            session,
            temporary.id,
            [
                {"observation_date": "2026-06-26", "nav": 1.0},
                {"observation_date": "2026-07-03", "nav": 1.02},
            ],
            source_file_id=source.id,
            source_fragment_id=fragment.id,
            frequency="weekly",
        )

        result = store.bind_file_to_product(session, source.id, target.id)

        assert result is not None
        assert result["bound_fragments"] == 1
        assert result["bound_nav"] == 2
        assert result["cleaned_placeholders"] == 1
        refreshed_fragment = session.get(DocumentFragment, fragment.id)
        assert refreshed_fragment is not None
        assert refreshed_fragment.product_id == target.id
        assert refreshed_fragment.content_data["binding_status"] == "matched"
        assert session.get(ProductEntity, temporary.id) is None
        assert [item.product_id for item in store.get_nav_series(session, target.id)] == [target.id, target.id]
        target_full = store.get_product(session, target.id)
        assert target_full is not None
        assert any(alias.alias == "致远2号_2026Q2.png" for alias in target_full.aliases)

    def test_bind_does_not_overwrite_existing_product_fragment(self, session: Session):
        source = store.register_file(session, filename="mixed.png", file_hash="mixed_bind_hash")
        existing = store.create_product(session, standard_name="已确认产品")
        target = store.create_product(session, standard_name="目标产品")
        kept = store.add_fragment(
            session, file_id=source.id, fragment_type="table",
            content_data={"binding_status": "matched"}, product_id=existing.id,
        )
        unmatched = store.add_fragment(
            session, file_id=source.id, fragment_type="chart",
            content_data={"curve_index": 1, "binding_status": "unmatched"}, product_id=None,
        )

        result = store.bind_file_to_product(session, source.id, target.id)

        assert result is not None and result["bound_fragments"] == 1
        assert session.get(DocumentFragment, kept.id).product_id == existing.id
        assert session.get(DocumentFragment, unmatched.id).product_id == target.id

    def test_bind_file_without_fragments_creates_manual_binding_fragment(self, session: Session):
        source = store.register_file(session, filename="无解析结果.pdf", file_hash="manual_bind_hash")
        target = store.create_product(session, standard_name="人工绑定产品")

        first = store.bind_file_to_product(session, source.id, target.id)
        second = store.bind_file_to_product(session, source.id, target.id)

        assert first is not None and first["created_manual_fragment"] is True
        assert second is not None and second["created_manual_fragment"] is False
        fragments = store.get_fragments_for_file(session, source.id)
        assert len(fragments) == 1
        assert fragments[0].fragment_type == "manual_binding"
        product_full = store.get_product(session, target.id)
        assert product_full is not None
        aliases = [alias.alias for alias in product_full.aliases]
        assert aliases.count("无解析结果.pdf") == 1

    def test_bind_keeps_existing_target_nav_on_duplicate_date(self, session: Session):
        source = store.register_file(session, filename="duplicate.png", file_hash="duplicate_bind_hash")
        temporary = store.create_product(session, standard_name="duplicate / 待绑定曲线 1")
        target = store.create_product(session, standard_name="重复日期目标")
        fragment = store.add_fragment(
            session, file_id=source.id, fragment_type="chart_traced",
            content_data={"curve_index": 0, "binding_status": "unmatched"}, product_id=temporary.id,
        )
        store.add_nav_observations(session, target.id, [{"observation_date": "2026-07-03", "nav": 1.10}])
        store.add_nav_observations(
            session, temporary.id,
            [
                {"observation_date": "2026-07-03", "nav": 1.02},
                {"observation_date": "2026-07-10", "nav": 1.03},
            ], source_file_id=source.id, source_fragment_id=fragment.id,
        )

        result = store.bind_file_to_product(session, source.id, target.id)

        assert result is not None
        assert result["bound_nav"] == 1
        assert result["dropped_duplicate_nav"] == 1
        series = store.get_nav_series(session, target.id)
        assert [(point.observation_date.isoformat(), point.nav) for point in series] == [
            ("2026-07-03", 1.10),
            ("2026-07-10", 1.03),
        ]


# ---------------------------------------------------------------------------
# Product lifecycle
# ---------------------------------------------------------------------------


class TestProductLifecycle:
    def test_create_and_confirm(self, session: Session):
        product = store.create_product(
            session, standard_name="CTA 稳健一号", manager_name="某私募", strategy="commodity_cta"
        )
        assert product.confirmation_status == ConfirmationStatus.PENDING

        confirmed = store.confirm_product(session, product.id, "analyst_a")
        assert confirmed is not None
        assert confirmed.confirmation_status == ConfirmationStatus.CONFIRMED
        assert confirmed.confirmed_by == "analyst_a"

    def test_reject(self, session: Session):
        product = store.create_product(session, standard_name="低质量产品")
        rejected = store.reject_product(session, product.id)
        assert rejected is not None
        assert rejected.confirmation_status == ConfirmationStatus.REJECTED

    def test_search_and_filter(self, session: Session):
        store.create_product(session, standard_name="CTA A", strategy="commodity_cta")
        store.create_product(session, standard_name="量化 B", strategy="equity_quant")
        store.create_product(session, standard_name="CTA C", strategy="commodity_cta")

        cta_products = store.list_products(session, strategy="commodity_cta")
        assert len(cta_products) == 2

        search_results = store.list_products(session, search="CTA")
        assert len(search_results) == 2

    def test_delete_product_removes_nav_candidates(self, session: Session):
        source = store.register_file(session, filename="candidate.png", file_hash="candidate_delete_hash")
        product = store.create_product(session, standard_name="待删除候选产品")
        store.create_nav_candidate_version(
            session, product.id, source.id,
            [{"observation_date": "2026-01-01", "nav": 1.0}],
        )

        assert store.delete_product(session, product.id) is True
        assert session.get(ProductEntity, product.id) is None


# ---------------------------------------------------------------------------
# Product merge
# ---------------------------------------------------------------------------


class TestProductMerge:
    def test_merge_relinks_children(self, session: Session):
        source = store.create_product(session, standard_name="产品 A 旧名")
        target = store.create_product(session, standard_name="产品 A 新名")

        # Add NAV and alias to source.
        store.add_nav_observations(session, source.id, [
            {"observation_date": "2025-01-31", "nav": 1.0},
            {"observation_date": "2025-02-28", "nav": 1.02},
        ])
        store.add_alias(session, source.id, "旧别名")

        merged = store.merge_products(session, source.id, target.id)
        assert merged is not None
        assert merged.id == target.id

        # NAV should now belong to target.
        nav_series = store.get_nav_series(session, target.id)
        assert len(nav_series) == 2

        # Source standard_name should be an alias of target.
        target_full = store.get_product(session, target.id)
        alias_texts = [a.alias for a in target_full.aliases]
        assert "产品 A 旧名" in alias_texts
        assert "旧别名" in alias_texts

        # Source should be marked as merged.
        source_refreshed = session.get(ProductEntity, source.id)
        assert source_refreshed.merged_into_id == target.id


# ---------------------------------------------------------------------------
# NAV traceability
# ---------------------------------------------------------------------------


class TestNavTraceability:
    def test_nav_linked_to_source(self, session: Session):
        file = store.register_file(session, filename="weekly.png", file_hash="nav_hash")
        product = store.create_product(session, standard_name="CTA X")

        inserted = store.add_nav_observations(
            session, product.id,
            [{"observation_date": "2025-03-31", "nav": 1.05}],
            source_file_id=file.id,
            frequency="monthly",
        )
        assert inserted == 1

        series = store.get_nav_series(session, product.id)
        assert len(series) == 1
        assert series[0].source_file_id == file.id
        assert series[0].review_status == ReviewStatus.PENDING

    def test_duplicate_nav_skipped(self, session: Session):
        product = store.create_product(session, standard_name="CTA Y")
        points = [{"observation_date": "2025-01-31", "nav": 1.0}]
        store.add_nav_observations(session, product.id, points)
        inserted_again = store.add_nav_observations(session, product.id, points)
        assert inserted_again == 0

    def test_review_workflow(self, session: Session):
        product = store.create_product(session, standard_name="CTA Z")
        store.add_nav_observations(session, product.id, [
            {"observation_date": "2025-01-31", "nav": 1.0},
            {"observation_date": "2025-02-28", "nav": 1.01},
        ])
        series = store.get_nav_series(session, product.id)
        ids = [o.id for o in series]

        updated = store.review_nav_observations(session, ids, "reviewed", "analyst_b")
        assert updated == 2

        reviewed = store.get_nav_series(session, product.id, reviewed_only=True)
        assert len(reviewed) == 2

    def test_evidence_snapshot_binds_nav_fact_fragment_and_file_version(self, session: Session):
        source = store.register_file(session, filename="周报.png", file_hash="snapshot-hash")
        product = store.create_product(session, standard_name="可追溯产品")
        fragment = store.add_fragment(
            session, file_id=source.id, fragment_type="chart", page_number=3,
            content_text="产品净值曲线", product_id=product.id,
        )
        store.add_nav_observations(
            session, product.id, [{"observation_date": "2026-01-02", "nav": 1.0}],
            source_file_id=source.id, source_fragment_id=fragment.id,
        )
        fact = store.add_fact(
            session, product_id=product.id, field_name="strategy", field_value="commodity_cta",
            source_file_id=source.id, source_fragment_id=fragment.id,
        )

        evidence, citations = store.build_research_evidence_snapshot(session, [product.id])

        record = evidence[product.id][0]
        assert record["nav_observation_ids"]
        assert record["fact_ids"] == [fact.id]
        assert record["fragment_ids"] == [fragment.id]
        assert record["files"] == [{
            "file_id": source.id,
            "filename": "周报.png",
            "file_hash": "snapshot-hash",
            "version": 1,
            "report_period": None,
        }]
        assert citations[0]["page_number"] == 3
        assert citations[0]["fragment_id"] == fragment.id

    def test_evidence_snapshot_cites_source_file_without_fragment(self, session: Session):
        source = store.register_file(session, filename="净值.csv", file_hash="source-only")
        product = store.create_product(session, standard_name="来源文件产品")
        store.add_nav_observations(
            session, product.id, [{"observation_date": "2026-01-02", "nav": 1.0}],
            source_file_id=source.id,
        )

        _evidence, citations = store.build_research_evidence_snapshot(session, [product.id])

        assert citations == [{
            "file_id": source.id,
            "filename": "净值.csv",
            "page_number": None,
            "fragment_id": None,
            "fragment_type": "source_file",
            "snippet": "净值来源文件（尚无可定位的材料片段）",
            "confidence": None,
        }]


# ---------------------------------------------------------------------------
# Structured facts
# ---------------------------------------------------------------------------


class TestStructuredFacts:
    def test_fact_with_evidence(self, session: Session):
        file = store.register_file(session, filename="facts.pdf", file_hash="fact_hash")
        product = store.create_product(session, standard_name="CTA F")
        fragment = store.add_fragment(
            session, file_id=file.id, fragment_type="text",
            content_text="管理费 1.5%", ocr_confidence=0.92, product_id=product.id,
        )

        fact = store.add_fact(
            session, product_id=product.id, field_name="management_fee",
            field_value="1.5%", source_file_id=file.id,
            source_fragment_id=fragment.id, confidence=0.92,
        )
        assert fact.extraction_version == 1

        facts = store.get_facts_for_product(session, product.id)
        assert len(facts) == 1
        assert facts[0].source_fragment_id == fragment.id


# ---------------------------------------------------------------------------
# Agent run audit
# ---------------------------------------------------------------------------


class TestAgentRunAudit:
    def test_full_run_lifecycle(self, session: Session):
        run = store.create_agent_run(session, user_query="筛选回撤低于10%的CTA", session_id="s1")
        assert run.phase == RunPhase.PLAN

        store.add_tool_invocation(
            session, run.id, tool_name="screen_products",
            input_summary={"max_drawdown": 0.10}, output_summary={"candidates": 3},
            duration_ms=120.5,
        )

        completed = store.complete_agent_run(
            session, run.id, phase="complete",
            tools_used=["screen_products"], answer="已筛选出3只产品",
            duration_ms=250.0,
        )
        assert completed is not None
        assert completed.phase == "complete"

        runs = store.list_agent_runs(session, session_id="s1")
        assert len(runs) == 1


# ---------------------------------------------------------------------------
# Snapshot & Decision
# ---------------------------------------------------------------------------


class TestDecisionWorkflow:
    def test_snapshot_and_decision(self, session: Session):
        snapshot = store.create_snapshot(session, label="2026-07 筛选", content={"products": ["p1", "p2"]})
        decision = store.create_decision(
            session, decision_type="recommendation",
            title="CTA FOF 初始配置", content={"weights": {"p1": 0.6, "p2": 0.4}},
            data_snapshot_id=snapshot.id, status=DecisionStatus.PENDING_REVIEW,
        )
        assert decision.status == DecisionStatus.PENDING_REVIEW

        reviewed = store.review_decision(
            session, decision.id, status=DecisionStatus.APPROVED, reviewer="ic_chair"
        )
        assert reviewed is not None
        assert reviewed.status == DecisionStatus.APPROVED
        assert reviewed.reviewer == "ic_chair"

    def test_veto_with_reason(self, session: Session):
        decision = store.create_decision(
            session, decision_type="allocation", content={"weights": {}},
        )
        vetoed = store.review_decision(
            session, decision.id, status=DecisionStatus.VETOED,
            reviewer="risk_officer", veto_reason="产品流动性不满足要求",
        )
        assert vetoed is not None
        assert vetoed.veto_reason == "产品流动性不满足要求"
