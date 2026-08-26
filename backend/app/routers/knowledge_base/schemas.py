"""Pydantic request / response schemas for the knowledge-base API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# File schemas
# ---------------------------------------------------------------------------


class FileRegisterRequest(BaseModel):
    filename: str
    file_hash: str
    mime_type: str | None = None
    size_bytes: int | None = None
    source: str | None = None
    report_period: str | None = None


class FileStatusUpdate(BaseModel):
    parsing_status: str
    parsing_error: str | None = None


class FileMaterialNatureUpdate(BaseModel):
    material_nature: Literal["CTA策略介绍", "CTA研究/市场资料", "忽略"]


class FileBindRequest(BaseModel):
    """Human choice used to resolve an uploaded file's product identity.

    Existing products are selected by ``product_id``.  When the source file
    contains a new product, the UI sends ``product_name`` and we create a
    pending entity before attaching the file evidence to it.
    """

    product_id: str | None = None
    product_name: str | None = None
    manager_name: str | None = None
    strategy: str | None = None


class CurveBindingChoice(BaseModel):
    """One explicit user confirmation for one chart curve."""

    fragment_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)


class BatchCurveBindRequest(BaseModel):
    bindings: list[CurveBindingChoice] = Field(min_length=1, max_length=32)


class FragmentCreate(BaseModel):
    file_id: str
    fragment_type: str
    page_number: int | None = None
    bbox: dict[str, Any] | None = None
    content_text: str | None = None
    content_data: dict[str, Any] | None = None
    ocr_confidence: float | None = None
    product_id: str | None = None


# ---------------------------------------------------------------------------
# Product schemas
# ---------------------------------------------------------------------------


class ProductCreate(BaseModel):
    standard_name: str
    manager_name: str | None = None
    strategy: str | None = None
    inception_date: date | None = None
    nav_frequency: str | None = None
    status: str = "unknown"
    notes: str | None = None


class ProductConfirm(BaseModel):
    confirmed_by: str = "user"


class StrategyDisclosure(BaseModel):
    arbitrage_type: str | None = None
    markets_or_sectors: list[str] = Field(default_factory=list, max_length=6)
    holding_period: str | None = None
    directional_exposure: str | None = None


class ProductUpdate(BaseModel):
    standard_name: str | None = None
    manager_name: str | None = None
    strategy: str | None = None
    inception_date: date | None = None
    close_date: date | None = None
    nav_frequency: str | None = None
    status: str | None = None
    notes: str | None = None
    strategy_disclosure: StrategyDisclosure | None = None


class AliasCreate(BaseModel):
    alias: str
    alias_type: str = "name"
    source_file_id: str | None = None


class MergeRequest(BaseModel):
    source_id: str
    target_id: str


# ---------------------------------------------------------------------------
# NAV schemas
# ---------------------------------------------------------------------------


class NavPointInput(BaseModel):
    observation_date: date
    nav: float = Field(gt=0)
    acc_nav: float | None = None


class NavBulkCreate(BaseModel):
    product_id: str
    points: list[NavPointInput]
    source_file_id: str | None = None
    source_fragment_id: str | None = None
    frequency: str | None = None


class NavReviewRequest(BaseModel):
    observation_ids: list[str]
    status: str = "reviewed"
    reviewed_by: str = "user"


class NavBulkQuery(BaseModel):
    """批量读取多只产品净值序列的请求体（GET 无法承载 749 个 ID）。"""

    product_ids: list[str] = Field(min_length=1, max_length=1000)
    reviewed_only: bool = False


class MachineNavReviewRequest(BaseModel):
    """Optional product subset for a batch machine review run."""

    product_ids: list[str] | None = None


class NavReplaceRequest(BaseModel):
    points: list[NavPointInput] = Field(min_length=2)
    frequency: str | None = None
    source_file_id: str | None = None
    source_fragment_id: str | None = None
    reviewed_by: str = "user"

    @classmethod
    def _ordered_dates(cls, points: list[NavPointInput]) -> bool:
        dates = [point.observation_date for point in points]
        return dates == sorted(dates) and len(dates) == len(set(dates))

    @model_validator(mode="after")
    def validate_dates(self) -> "NavReplaceRequest":
        if not self._ordered_dates(self.points):
            raise ValueError("净值日期必须严格递增且不重复")
        return self


class NavCandidatePublishRequest(BaseModel):
    """Explicit publication decision for an extracted NAV candidate."""

    mode: str = Field(pattern="^(replace|merge_missing|discard)$")
    reviewed_by: str = "user"


# ---------------------------------------------------------------------------
# Fact schemas
# ---------------------------------------------------------------------------


class FactCreate(BaseModel):
    product_id: str
    field_name: str
    field_value: str
    source_file_id: str | None = None
    source_fragment_id: str | None = None
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Agent run schemas
# ---------------------------------------------------------------------------


class AgentRunCreate(BaseModel):
    user_query: str
    session_id: str | None = None
    plan: dict[str, Any] | None = None


class AgentRunComplete(BaseModel):
    phase: str = "complete"
    tools_used: list[str] | None = None
    citations: list[dict[str, Any]] | None = None
    reflection: dict[str, Any] | None = None
    answer: str | None = None
    data_snapshot_id: str | None = None
    duration_ms: float | None = None


class ToolInvocationCreate(BaseModel):
    run_id: str
    tool_name: str
    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    status: str = "ok"
    error_message: str | None = None
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Snapshot & Decision schemas
# ---------------------------------------------------------------------------


class SnapshotCreate(BaseModel):
    label: str | None = None
    content: dict[str, Any]


class DecisionCreate(BaseModel):
    decision_type: str = "recommendation"
    title: str | None = None
    content: dict[str, Any]
    run_id: str | None = None
    data_snapshot_id: str | None = None
    status: str = "draft"


class DecisionReview(BaseModel):
    status: str
    reviewer: str | None = None
    veto_reason: str | None = None
    adjustment: dict[str, Any] | None = None
