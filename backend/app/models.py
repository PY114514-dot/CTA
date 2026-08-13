"""P0 data models: products, materials, evidence, NAV, agent runs and decisions.

Design principles from the master design document:
- Every NAV point and material fact is traceable to its source file and version.
- Products have a human-confirmation workflow (pending → confirmed → merged).
- Files are versioned; old recommendations can reference historical snapshots.
- Agent runs and decisions are immutable audit records.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ParsingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_NO_NAV = "completed_no_nav"
    FAILED = "failed"


class ConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ProductStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"
    UNKNOWN = "unknown"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    MACHINE_REVIEWED = "machine_reviewed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class RunPhase(StrEnum):
    PLAN = "plan"
    RETRIEVE = "retrieve"
    ACT = "act"
    VERIFY = "verify"
    REFLECT = "reflect"
    REPLAN = "replan"
    COMPLETE = "complete"
    FAILED = "failed"


class DecisionType(StrEnum):
    RECOMMENDATION = "recommendation"
    ALLOCATION = "allocation"
    SCREENING = "screening"
    COMPARISON = "comparison"


class DecisionStatus(StrEnum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    VETOED = "vetoed"
    SUPERSEDED = "superseded"


class TrackingStatus(StrEnum):
    """Health of a recommendation at a post-recommendation review point."""

    NORMAL = "normal"
    BREACH = "breach"  # deviation exceeded threshold, review triggered
    INSUFFICIENT_DATA = "insufficient_data"  # not enough post-decision NAV to evaluate


class ReviewTaskStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ReviewTaskType(StrEnum):
    METHODOLOGY = "methodology"  # methodology review triggered by outer loop
    MANUAL = "manual"  # manually created review task


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return uuid.uuid4().hex[:20]


# ---------------------------------------------------------------------------
# 1. Raw File (uploaded material)
# ---------------------------------------------------------------------------


class RawFile(Base):
    """An uploaded material file and its extraction audit."""

    __tablename__ = "raw_files"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    file_hash: Mapped[str] = mapped_column(String(64), index=True, doc="SHA-256 hex digest")
    filename: Mapped[str] = mapped_column(String(512), doc="Original filename")
    mime_type: Mapped[str | None] = mapped_column(String(128), default=None)
    size_bytes: Mapped[int | None] = mapped_column(Integer, default=None)
    source: Mapped[str | None] = mapped_column(String(256), default=None, doc="Upload origin or batch label")
    report_period: Mapped[str | None] = mapped_column(String(64), default=None, doc="E.g. 2026-06 or 2026Q2")
    version: Mapped[int] = mapped_column(Integer, default=1, doc="Monotonic version for same logical file")
    parsing_status: Mapped[str] = mapped_column(String(20), default=ParsingStatus.PENDING)
    parsing_error: Mapped[str | None] = mapped_column(Text, default=None)
    extraction_audit: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=None, doc="Persisted OCR/VLM/CV provenance for this parse"
    )
    ingestion_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=None, doc="Stable manifest/user hints used by every retry stage"
    )
    storage_path: Mapped[str | None] = mapped_column(String(1024), default=None, doc="Local or object-store path")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    fragments: Mapped[list[DocumentFragment]] = relationship(back_populates="file", cascade="all, delete-orphan")
    nav_observations: Mapped[list[NavObservation]] = relationship(back_populates="source_file")


# ---------------------------------------------------------------------------
# 2. Document Fragment (extracted evidence)
# ---------------------------------------------------------------------------


class DocumentFragment(Base):
    """A located piece of evidence extracted from a raw file."""

    __tablename__ = "document_fragments"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    file_id: Mapped[str] = mapped_column(ForeignKey("raw_files.id"), index=True)
    fragment_type: Mapped[str] = mapped_column(String(32), doc="text | table | chart | image | nav_curve")
    page_number: Mapped[int | None] = mapped_column(Integer, default=None)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None, doc="{x, y, w, h} in page coordinates")
    content_text: Mapped[str | None] = mapped_column(Text, default=None)
    content_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None, doc="Structured payload (table cells, curve points)")
    ocr_confidence: Mapped[float | None] = mapped_column(Float, default=None)
    product_id: Mapped[str | None] = mapped_column(ForeignKey("product_entities.id"), default=None, index=True)
    extraction_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    file: Mapped[RawFile] = relationship(back_populates="fragments")
    product: Mapped[ProductEntity | None] = relationship(back_populates="fragments")


# ---------------------------------------------------------------------------
# 3. Product Entity
# ---------------------------------------------------------------------------


class ProductEntity(Base):
    """A private fund product with confirmation workflow."""

    __tablename__ = "product_entities"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    standard_name: Mapped[str] = mapped_column(String(256), index=True)
    manager_name: Mapped[str | None] = mapped_column(String(256), default=None, index=True)
    strategy: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    inception_date: Mapped[date | None] = mapped_column(Date, default=None)
    close_date: Mapped[date | None] = mapped_column(Date, default=None)
    nav_frequency: Mapped[str | None] = mapped_column(String(20), default=None, doc="daily | weekly | monthly")
    status: Mapped[str] = mapped_column(String(20), default=ProductStatus.UNKNOWN)
    confirmation_status: Mapped[str] = mapped_column(String(20), default=ConfirmationStatus.PENDING, index=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(128), default=None)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    merged_into_id: Mapped[str | None] = mapped_column(ForeignKey("product_entities.id"), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    aliases: Mapped[list[ProductAlias]] = relationship(back_populates="product", cascade="all, delete-orphan")
    fragments: Mapped[list[DocumentFragment]] = relationship(back_populates="product")
    nav_observations: Mapped[list[NavObservation]] = relationship(back_populates="product", cascade="all, delete-orphan")
    facts: Mapped[list[StructuredFact]] = relationship(back_populates="product", cascade="all, delete-orphan")


class ProductAlias(Base):
    """Alternative names or codes for entity resolution."""

    __tablename__ = "product_aliases"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    product_id: Mapped[str] = mapped_column(ForeignKey("product_entities.id"), index=True)
    alias: Mapped[str] = mapped_column(String(256), index=True)
    alias_type: Mapped[str] = mapped_column(String(32), default="name", doc="name | code | backup_name")
    source_file_id: Mapped[str | None] = mapped_column(ForeignKey("raw_files.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    product: Mapped[ProductEntity] = relationship(back_populates="aliases")


# ---------------------------------------------------------------------------
# 4. NAV Observation
# ---------------------------------------------------------------------------


class NavObservation(Base):
    """A single dated NAV point, traceable to source material."""

    __tablename__ = "nav_observations"
    __table_args__ = (
        UniqueConstraint("product_id", "observation_date", name="uq_nav_product_date"),
    )

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    product_id: Mapped[str] = mapped_column(ForeignKey("product_entities.id"), index=True)
    observation_date: Mapped[date] = mapped_column(Date, index=True)
    nav: Mapped[float] = mapped_column(Float, doc="Unit NAV (positive)")
    acc_nav: Mapped[float | None] = mapped_column(Float, default=None, doc="Accumulated NAV")
    frequency: Mapped[str | None] = mapped_column(String(20), default=None)
    source_file_id: Mapped[str | None] = mapped_column(ForeignKey("raw_files.id"), default=None)
    source_fragment_id: Mapped[str | None] = mapped_column(ForeignKey("document_fragments.id"), default=None)
    review_status: Mapped[str] = mapped_column(String(20), default=ReviewStatus.PENDING, index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    product: Mapped[ProductEntity] = relationship(back_populates="nav_observations")
    source_file: Mapped[RawFile | None] = relationship(back_populates="nav_observations")


class NavCandidateVersion(Base):
    """One unapproved NAV extraction, isolated from the official series.

    Image/PDF extraction is evidence, not a data mutation.  A reviewer must
    explicitly publish one of these versions before it becomes available to
    screening or reports.
    """

    __tablename__ = "nav_candidate_versions"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    product_id: Mapped[str] = mapped_column(ForeignKey("product_entities.id"), index=True)
    source_file_id: Mapped[str] = mapped_column(ForeignKey("raw_files.id"), index=True)
    source_fragment_id: Mapped[str | None] = mapped_column(ForeignKey("document_fragments.id"), default=None)
    frequency: Mapped[str | None] = mapped_column(String(20), default=None)
    points: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    published_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)


# ---------------------------------------------------------------------------
# 5. Structured Fact
# ---------------------------------------------------------------------------


class StructuredFact(Base):
    """An extracted attribute about a product, with evidence and confidence."""

    __tablename__ = "structured_facts"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    product_id: Mapped[str] = mapped_column(ForeignKey("product_entities.id"), index=True)
    field_name: Mapped[str] = mapped_column(String(128), doc="E.g. management_fee, lock_period, aum")
    field_value: Mapped[str] = mapped_column(Text)
    source_file_id: Mapped[str | None] = mapped_column(ForeignKey("raw_files.id"), default=None)
    source_fragment_id: Mapped[str | None] = mapped_column(ForeignKey("document_fragments.id"), default=None)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    extraction_version: Mapped[int] = mapped_column(Integer, default=1)
    superseded_by: Mapped[str | None] = mapped_column(ForeignKey("structured_facts.id"), default=None)
    confirmation_status: Mapped[str] = mapped_column(String(20), default=ConfirmationStatus.PENDING)
    confirmed_by: Mapped[str | None] = mapped_column(String(128), default=None)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    product: Mapped[ProductEntity] = relationship(back_populates="facts")


# ---------------------------------------------------------------------------
# 6. Agent Run (immutable audit)
# ---------------------------------------------------------------------------


class AgentRun(Base):
    """One execution of the FOF Agent, capturing the full audit trail."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    user_query: Mapped[str] = mapped_column(Text)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    phase: Mapped[str] = mapped_column(String(20), default=RunPhase.PLAN)
    tools_used: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=None, doc="[{file_id, page, fragment_id}]")
    reflection: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    answer: Mapped[str | None] = mapped_column(Text, default=None)
    data_snapshot_id: Mapped[str | None] = mapped_column(ForeignKey("data_snapshots.id"), default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    tool_invocations: Mapped[list[ToolInvocation]] = relationship(back_populates="run", cascade="all, delete-orphan")
    decisions: Mapped[list[DecisionRecord]] = relationship(back_populates="run")


class ToolInvocation(Base):
    """A single tool call within an agent run."""

    __tablename__ = "tool_invocations"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    input_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    output_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    invoked_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    run: Mapped[AgentRun] = relationship(back_populates="tool_invocations")


# ---------------------------------------------------------------------------
# 7. Data Snapshot & Decision Record
# ---------------------------------------------------------------------------


class DataSnapshot(Base):
    """An immutable point-in-time capture of input data for reproducibility."""

    __tablename__ = "data_snapshots"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    label: Mapped[str | None] = mapped_column(String(256), default=None)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, doc="Frozen input data (NAV series, params, product IDs)")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DecisionRecord(Base):
    """A recommendation or screening decision with human review workflow."""

    __tablename__ = "decision_records"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"), default=None, index=True)
    decision_type: Mapped[str] = mapped_column(String(32), default=DecisionType.RECOMMENDATION)
    title: Mapped[str | None] = mapped_column(String(512), default=None)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, doc="Recommendation payload (weights, candidates, rationale)")
    data_snapshot_id: Mapped[str | None] = mapped_column(ForeignKey("data_snapshots.id"), default=None)
    status: Mapped[str] = mapped_column(String(20), default=DecisionStatus.DRAFT, index=True)
    reviewer: Mapped[str | None] = mapped_column(String(128), default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    veto_reason: Mapped[str | None] = mapped_column(Text, default=None)
    adjustment: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None, doc="Human modifications to the recommendation")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    run: Mapped[AgentRun | None] = relationship(back_populates="decisions")
    tracking_records: Mapped[list[RecommendationTracking]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )
    review_tasks: Mapped[list[ReviewTask]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# 8. Post-recommendation tracking & methodology review (P4 outer loop)
# ---------------------------------------------------------------------------


class RecommendationTracking(Base):
    """One post-recommendation performance review point (outer loop).

    Recomputes actual portfolio performance since the decision date and
    compares it against the expectation frozen in the decision/snapshot.
    When deviation exceeds threshold, a methodology ReviewTask is created.
    """

    __tablename__ = "recommendation_tracking"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), index=True)
    review_date: Mapped[date] = mapped_column(Date, doc="As-of date of this tracking review")
    periods_elapsed: Mapped[int] = mapped_column(Integer, default=0)
    expected_return: Mapped[float | None] = mapped_column(Float, default=None, doc="Baseline from decision snapshot")
    actual_return: Mapped[float | None] = mapped_column(Float, default=None, doc="Weighted portfolio return since decision")
    return_deviation: Mapped[float | None] = mapped_column(Float, default=None, doc="actual - expected")
    actual_max_drawdown: Mapped[float | None] = mapped_column(Float, default=None)
    per_product: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None, doc="Per-product actual metrics")
    status: Mapped[str] = mapped_column(String(20), default=TrackingStatus.NORMAL, index=True)
    breach_reasons: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    triggered_task_id: Mapped[str | None] = mapped_column(ForeignKey("review_tasks.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    decision: Mapped[DecisionRecord] = relationship(back_populates="tracking_records")
    triggered_task: Mapped[ReviewTask | None] = relationship(foreign_keys=[triggered_task_id])


class ReviewTask(Base):
    """A methodology or manual review task with auditable human resolution.

    The outer loop may only *propose* changes; resolution (methodology edits,
    weight changes, prompt/skill updates) requires human review recorded here.
    """

    __tablename__ = "review_tasks"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default=_uuid)
    decision_id: Mapped[str | None] = mapped_column(ForeignKey("decision_records.id"), default=None, index=True)
    tracking_id: Mapped[str | None] = mapped_column(ForeignKey("recommendation_tracking.id"), default=None, index=True)
    task_type: Mapped[str] = mapped_column(String(32), default=ReviewTaskType.METHODOLOGY)
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(String(20), default=ReviewTaskStatus.OPEN, index=True)
    priority: Mapped[str] = mapped_column(String(16), default="normal", doc="low | normal | high")
    assignee: Mapped[str | None] = mapped_column(String(128), default=None)
    proposal: Mapped[str | None] = mapped_column(Text, default=None, doc="Outer-loop suggested change (non-binding)")
    resolution: Mapped[str | None] = mapped_column(Text, default=None, doc="Human decision / outcome")
    resolved_by: Mapped[str | None] = mapped_column(String(128), default=None)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    decision: Mapped[DecisionRecord | None] = relationship(back_populates="review_tasks")
