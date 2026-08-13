"""FOF Agent conversation endpoint: retrieval-augmented chat with citations.

Implements a constrained Plan → Retrieve → Act → Reflect loop for the
workbench conversation panel. Quantitative tools are deterministic; when the
user has enabled an LLM it is used only to synthesize those audited results.
"""

from __future__ import annotations

import asyncio
import re
import logging
import os
from pathlib import Path
from statistics import median
from datetime import date
from time import perf_counter
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.dependencies import get_report_config
from app.models import NavObservation, ProductEntity, RawFile
from app.services import product_store as store
from app.services.upload_storage import resolve_upload_path
from app.services import retrieval
from app.services import quant_screen
from app.services.fof_allocation_intent import (
    is_demo_product,
    is_screening_candidate,
)
from app.services.fof_allocation_research import build_allocation_research
from app.services.nav_metrics import calculate_nav_analysis
from app.services.nav_quality import assess_nav_quality as assess_product_nav_quality
from app.services.chart_extractor.vlm_extractor import DashScopeProvider, OpenAICompatibleProvider
from app.schemas import NavAnalysisRequest, NetAssetValuePoint, DataFrequency

router = APIRouter(prefix="/api/kb", tags=["产品知识库"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    product_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None


class ChatCitation(BaseModel):
    file_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    fragment_id: str | None = None
    fragment_type: str | None = None
    snippet: str | None = None
    confidence: float | None = None


class ChatToolCall(BaseModel):
    name: str
    status: str = "ok"
    duration_ms: float | None = None
    summary: str | None = None


class ChatAllocationItem(BaseModel):
    product_id: str
    product_name: str
    weight: float
    score: float
    rationale: str


class ChatAllocationDraft(BaseModel):
    goal: str
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    allocations: list[ChatAllocationItem] = Field(default_factory=list)
    exclusions: list[dict[str, Any]] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    due_diligence_gaps: list[str] = Field(default_factory=list)
    snapshot_id: str | None = None


class ChatMessage(BaseModel):
    role: str = "agent"
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    products_referenced: list[dict[str, Any]] = Field(default_factory=list)
    data_context: list[dict[str, Any]] = Field(default_factory=list)
    method_provenance: dict[str, Any] = Field(default_factory=dict)
    allocation_draft: ChatAllocationDraft | None = None


# ---------------------------------------------------------------------------
# Intent detection (deterministic, replaceable by LLM)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Comparison must win over allocation language such as "配置角色".  The
    # comparison preset intentionally contains both words and must never
    # create a recommendation or a decision record.
    ("compare", re.compile(r"比较|对比|哪个.*好|差异|区别|vs|compare|comparison|versus", re.IGNORECASE)),
    ("recommend", re.compile(r"推荐|配置|权重|组合|FOF|建议|看好.*(?:CTA|cta)|recommend|allocat|weight|portfolio", re.IGNORECASE)),
    ("screen", re.compile(r"筛|找出|选|回撤.*低|波动.*小|夏普.*高|收益.*好|排名|screen|filter|find.*low|find.*best", re.IGNORECASE)),
    ("nav", re.compile(r"净值|收益|回撤|波动|夏普|卡玛|表现|业绩|nav|return|drawdown|volatility|sharpe|performance|metric", re.IGNORECASE)),
    ("describe", re.compile(r"介绍|什么|哪些|信息|概况|详情|策略|管理人|describe|what|which|info|about|strategy|manager", re.IGNORECASE)),
]

def _detect_intent(query: str) -> str:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return "describe"


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatMessage)
def chat(body: ChatRequest, session: Session = Depends(get_session)) -> ChatMessage:
    """Process a natural language query with retrieval and deterministic tools."""
    started = perf_counter()
    intent = _detect_intent(body.query)
    tool_calls: list[ChatToolCall] = []
    citations: list[ChatCitation] = []
    products_referenced: list[dict[str, Any]] = []

    # UI state can outlive a product deletion or a status change.  Never let
    # stale IDs (or records explicitly marked "not a product") masquerade as
    # zero-NAV research candidates.
    selected_ids: list[str] = []
    ignored_ids: list[str] = []
    for product_id in body.product_ids:
        product = session.get(ProductEntity, product_id)
        if product is None or product.confirmation_status == "rejected" or is_demo_product(product):
            ignored_ids.append(product_id)
        else:
            selected_ids.append(product_id)
    if body.product_ids and not selected_ids:
        return ChatMessage(
            content="当前选择的产品已被删除、标记为“不是产品”，或尚未形成有效记录，因此没有可分析的净值数据。请在左侧选择“待确认”或“已确认”的正式产品后重试。",
            tool_calls=[ChatToolCall(
                name="validate_product_context",
                status="error",
                summary=f"已拦截 {len(ignored_ids)} 个无效或已忽略产品上下文",
            )],
        )
    if ignored_ids:
        tool_calls.append(ChatToolCall(
            name="validate_product_context",
            status="ok",
            summary=f"已忽略 {len(ignored_ids)} 个无效或已忽略产品上下文",
        ))

    # A selected product with a curve/disclosure conflict is not a legitimate
    # research input.  Stop here with one actionable next step instead of
    # letting retrieval, an LLM, or a factor template make the conflict look
    # like a formal conclusion.
    blocked_products: list[tuple[ProductEntity, dict[str, Any]]] = []
    for product_id in selected_ids:
        product = session.get(ProductEntity, product_id)
        if product is None:
            continue
        quality = assess_product_nav_quality(product.nav_observations, product.facts)
        if quality["blocking"]:
            blocked_products.append((product, quality))
    if blocked_products:
        product, quality = blocked_products[0]
        reasons = "；".join(quality["reasons"])
        return ChatMessage(
            content=(
                f"「{product.standard_name}」当前不能做正式研究：{reasons}。\n\n"
                "下一步：点击该产品的“复核 / 校准净值”，在原图上核对冲突点并保存人工复核版；"
                "完成前我只可以帮助你查看原始资料和冲突位置，不会输出因子归因或产品推荐。"
            ),
            products_referenced=[{"id": product.id, "name": product.standard_name, "category": "净值待复核"}],
            data_context=[{
                "product_id": product.id,
                "product_name": product.standard_name,
                "research_ready": False,
                "readiness_reason": "净值与材料披露冲突，需人工复核",
                "nav_quality": quality,
            }],
            tool_calls=[ChatToolCall(
                name="validate_nav_quality",
                status="error",
                summary="已拦截存在曲线与披露冲突的产品，等待人工复核",
            )],
            method_provenance={
                "nav_quality": {"method": "本地披露一致性检查", "detail": "比较曲线收益、最大回撤和异常单期跳变"},
            },
        )

    # --- Retrieve phase ---
    retrieval_started = perf_counter()
    retrieval_result = retrieval.retrieve(
        session, body.query,
        product_ids=selected_ids or None,
        limit=12,
    )
    tool_calls.append(ChatToolCall(
        name="search_products",
        status="ok",
        duration_ms=(perf_counter() - retrieval_started) * 1000,
        summary=f"检索到 {retrieval_result.total} 条证据",
    ))

    # Collect citations from retrieval results.
    for result in retrieval_result.results:
        if result.citation:
            citations.append(ChatCitation(
                file_id=result.citation.file_id,
                filename=result.citation.filename,
                page_number=result.citation.page_number,
                fragment_id=result.citation.fragment_id,
                fragment_type=result.citation.fragment_type,
                snippet=result.citation.snippet,
                confidence=result.citation.confidence,
            ))
        if result.product_id and result.product_id not in [p["id"] for p in products_referenced]:
            products_referenced.append({
                "id": result.product_id,
                "name": result.product_name or result.title,
                "category": result.category,
            })

    # --- Act phase (intent-specific tools) ---
    content = ""
    allocation_draft: ChatAllocationDraft | None = None

    if intent == "nav" and selected_ids:
        content = _handle_nav_query(session, selected_ids, tool_calls)
    elif intent == "screen":
        content = _handle_screen_query(session, body.query, retrieval_result, products_referenced, tool_calls)
    elif intent == "compare":
        content = _handle_compare_query(session, selected_ids, tool_calls)
    elif intent == "recommend":
        content, allocation_draft = _handle_recommend_query(
            session, body.query, selected_ids, tool_calls, citations
        )
    elif selected_ids:
        content = _handle_selected_product_overview(session, selected_ids, tool_calls)
    else:
        content = _handle_describe_query(retrieval_result, products_referenced)

    # --- Reflect phase ---
    if not content:
        content = _fallback_answer(retrieval_result, intent)

    # Retrieval and quant tools always run before the LLM.  The model is a
    # constrained research writer, never a substitute for calculation.
    context_ids = selected_ids or [p["id"] for p in products_referenced if p.get("id")]
    data_context = _build_data_context(session, context_ids)
    config = get_report_config()
    if config.enabled:
        image_evidence = ""
        image_started = perf_counter()
        # Source-image evidence may enrich a product description, but it is
        # deliberately excluded from NAV comparison/screening/allocation.
        # Those flows must have exactly one numerical source: reviewed NAV.
        if intent == "describe":
            try:
                image_evidence, image_count = _inspect_source_images(session, context_ids)
                if image_count:
                    tool_calls.append(ChatToolCall(
                        name="inspect_source_images",
                        status="ok",
                        duration_ms=(perf_counter() - image_started) * 1000,
                        summary=f"视觉模型已读取 {image_count} 份原始图片作为补充证据",
                    ))
            except Exception as error:
                logger.warning("VLM source-image inspection failed: %s", error)
                tool_calls.append(ChatToolCall(
                    name="inspect_source_images",
                    status="error",
                    duration_ms=(perf_counter() - image_started) * 1000,
                    summary="原始图片未能由视觉模型读取，继续使用本地工具结果",
                ))
        llm_started = perf_counter()
        try:
            content = _synthesize_with_llm(content, body.query, data_context, config, image_evidence)
            tool_calls.append(ChatToolCall(
                name="llm_research_synthesis",
                status="ok",
                duration_ms=(perf_counter() - llm_started) * 1000,
                summary=f"已使用 {config.model} 基于工具结果生成解读",
            ))
        except Exception as error:
            logger.warning("LLM agent synthesis failed, using deterministic answer: %s", error)
            tool_calls.append(ChatToolCall(
                name="llm_research_synthesis",
                status="error",
                duration_ms=(perf_counter() - llm_started) * 1000,
                summary="LLM 调用失败，已回退到工具计算结果",
            ))

    # Record agent run for audit.
    run = store.create_agent_run(
        session,
        user_query=body.query,
        session_id=body.session_id,
        plan={"intent": intent, "product_ids": selected_ids, "ignored_product_ids": ignored_ids},
    )
    for tc in tool_calls:
        store.add_tool_invocation(
            session, run.id,
            tool_name=tc.name,
            output_summary={"summary": tc.summary},
            status=tc.status,
            duration_ms=tc.duration_ms,
        )
    store.complete_agent_run(
        session, run.id,
        phase="complete",
        tools_used=[tc.name for tc in tool_calls],
        citations=[c.model_dump() for c in citations],
        answer=content[:500],
        data_snapshot_id=allocation_draft.snapshot_id if allocation_draft else None,
        duration_ms=(perf_counter() - started) * 1000,
    )

    return ChatMessage(
        content=content,
        citations=citations,
        tool_calls=tool_calls,
        products_referenced=products_referenced,
        data_context=data_context,
        method_provenance={
            "intent": {"method": "本地规则匹配", "detail": "根据问题关键词选择检索、净值指标、比较或筛选工具"},
            "quant_tools": {"method": "本地确定性统计工具", "detail": "指标、筛选与产品状态校验先于文本生成执行"},
            "vision": {"method": "VLM（可选补充）", "detail": "仅读取原始图片作为证据，不覆盖净值或统计结果"},
            "summary": {
                "method": "LLM" if config.enabled else "本地模板",
                "model": config.model if config.enabled else None,
                "detail": "LLM 只负责解释已审计工具结果" if config.enabled else "未配置 LLM，返回本地工具结果",
            },
        },
        allocation_draft=allocation_draft,
    )


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------


def _synthesize_with_llm(
    raw_answer: str,
    question: str,
    data_context: list[dict[str, Any]],
    config: Any,
    image_evidence: str = "",
) -> str:
    """Use an LLM only to explain audited output produced by local tools."""
    context_lines = []
    for item in data_context:
        interval = f"{item.get('start_date') or '未知'} 至 {item.get('end_date') or '未知'}"
        context_lines.append(
            f"- {item.get('name')}：{item.get('nav_count', 0)} 个点，{item.get('frequency') or '频率未知'}，"
            f"区间 {interval}，{item.get('review_status')}"
        )
    context_text = "\n".join(context_lines) or "本次没有直接使用产品净值序列。"
    prompt = (
        "你是私募 CTA 研究助手。请基于下方工具计算结果回答用户问题。\n"
        "硬性规则：只能使用“工具计算结果”中的业绩数值、筛选规则和配置结论，不得补造或改写净值、"
        "收益、回撤、波动、相关性、权重、持仓、因子暴露、外部数据或引用；"
        "视觉模型补充证据不得用于给出或修正任何业绩数值、筛选结论或配置结论；"
        "如果数据标为待复核，必须简短提示；保留产品名、重要数值和筛选规则；"
        "使用简洁专业的中文，不要说你调用了 LLM。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【本次数据范围】\n{context_text}\n\n"
        f"【视觉模型补充证据】\n{image_evidence or '未使用视觉模型补充证据。'}\n\n"
        f"【工具计算结果】\n{raw_answer}"
    )
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{config.api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            json={
                "model": config.model,
                "messages": [
                    {"role": "system", "content": "你是严谨的量化研究写作助手，绝不虚构或修改已计算数值。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 900,
            },
        )
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("模型返回空内容")
    return content


def _inspect_source_images(session: Session, product_ids: list[str]) -> tuple[str, int]:
    """Read at most two source images with VLM and return text evidence.

    Image output is deliberately treated as supplementary evidence.  The
    deterministic NAV, screening and attribution tools remain authoritative.
    """
    provider_name = os.getenv("VLM_PROVIDER", "dashscope")
    model = os.getenv("VLM_MODEL", "qwen3-vl-flash")
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("VLM_BASE_URL", "")
    if provider_name == "dashscope" and not api_key:
        return "", 0
    if provider_name != "dashscope" and not base_url:
        return "", 0

    source_ids: set[str] = set()
    for product_id in product_ids:
        source_ids.update(
            observation.source_file_id
            for observation in store.get_nav_series(session, product_id, reviewed_only=False)
            if observation.source_file_id
        )
    if not source_ids:
        return "", 0

    files = session.query(RawFile).filter(RawFile.id.in_(source_ids)).all()
    image_files = [
        record for record in files
        if record.storage_path and Path(record.filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ][:2]
    if not image_files:
        return "", 0

    provider = (
        DashScopeProvider(api_key=api_key, model=model)
        if provider_name == "dashscope"
        else OpenAICompatibleProvider(base_url=base_url, api_key=api_key or "not-needed", model=model)
    )
    prompt = (
        "这是一份私募产品原始材料。请仅提取可见的非数值事实：产品名称、图表日期区间、"
        "策略名称、以及是否存在净值曲线。"
        "不要推断持仓或补造不可见数据；用不超过 120 字中文列点。"
    )
    evidence: list[str] = []
    for record in image_files:
        try:
            path = resolve_upload_path(record.storage_path)
        except ValueError:
            continue
        if path is None:
            continue
        if not path.is_file():
            continue
        response = asyncio.run(provider.extract_structure(path.read_bytes(), prompt))
        evidence.append(f"{record.filename}：{response.strip()[:800]}")
    return "\n".join(evidence), len(evidence)


def _build_data_context(session: Session, product_ids: list[str]) -> list[dict[str, Any]]:
    """Return an explicit audit summary of records actually available to Agent."""
    result: list[dict[str, Any]] = []
    for product_id in dict.fromkeys(product_ids):
        product = session.get(ProductEntity, product_id)
        if product is None:
            continue
        observations = store.get_nav_series(session, product_id, reviewed_only=True)
        source_ids = {o.source_file_id for o in observations if o.source_file_id}
        source_files = [
            record.filename for record in session.query(RawFile).filter(RawFile.id.in_(source_ids)).all()
        ] if source_ids else []
        reviewed_count = sum(1 for o in observations if o.review_status == "reviewed")
        result.append({
            "product_id": product.id,
            "name": product.standard_name,
            "nav_count": len(observations),
            "start_date": observations[0].observation_date.isoformat() if observations else None,
            "end_date": observations[-1].observation_date.isoformat() if observations else None,
            "frequency": _infer_frequency(observations, product.nav_frequency),
            "source_files": source_files[:3],
            "fact_count": len(product.facts),
            "review_status": "已复核" if observations and reviewed_count == len(observations) else "待复核" if observations else "无净值",
        })
    return result


def _infer_frequency(observations: list[NavObservation], declared: str | None = None) -> str | None:
    """Prefer declared frequency, otherwise infer it from date spacing."""
    if declared in {"daily", "weekly", "monthly"}:
        return declared
    values = [o.frequency for o in observations if o.frequency in {"daily", "weekly", "monthly"}]
    if values:
        return values[0]
    if len(observations) < 2:
        return None
    gaps = [(b.observation_date - a.observation_date).days for a, b in zip(observations, observations[1:])]
    typical_gap = median(gaps)
    if typical_gap <= 3:
        return "daily"
    if typical_gap <= 10:
        return "weekly"
    return "monthly"


def _handle_nav_query(
    session: Session, product_ids: list[str], tool_calls: list[ChatToolCall]
) -> str:
    """Calculate NAV metrics for selected products."""
    lines: list[str] = []
    for pid in product_ids[:5]:
        started = perf_counter()
        observations = store.get_nav_series(session, pid, reviewed_only=True)
        product = session.get(ProductEntity, pid)
        name = product.standard_name if product else pid

        if len(observations) < 2:
            lines.append(f"**{name}**：净值数据不足（{len(observations)} 个点），无法计算指标。")
            continue

        try:
            nav_points = [
                NetAssetValuePoint(observation_date=o.observation_date, net_asset_value=o.nav)
                for o in observations
            ]
            frequency = _infer_frequency(observations, product.nav_frequency if product else None) or "monthly"
            result = calculate_nav_analysis(NavAnalysisRequest(
                nav_points=nav_points,
                frequency=DataFrequency(frequency),
            ))
            m = result.metrics
            sharpe = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "—"
            lines.append(
                f"**{name}**（{len(observations)} 期，{frequency}）：\n"
                f"  累计收益 {m.cumulative_return:.2%} | 年化 {m.annualized_return:.2%} | "
                f"波动 {m.annualized_volatility:.2%} | 夏普 {sharpe} | "
                f"最大回撤 {m.maximum_drawdown:.2%}"
            )
            tool_calls.append(ChatToolCall(
                name="calculate_nav_metrics",
                status="ok",
                duration_ms=(perf_counter() - started) * 1000,
                summary=f"{name}: 年化 {m.annualized_return:.2%}",
            ))
        except Exception as e:
            lines.append(f"**{name}**：计算失败 — {e}")
            tool_calls.append(ChatToolCall(name="calculate_nav_metrics", status="error", summary=str(e)))

    return "\n\n".join(lines)


def _handle_screen_query(
    session: Session,
    query: str,
    retrieval_result: retrieval.RetrievalResponse,
    products_referenced: list[dict[str, Any]],
    tool_calls: list[ChatToolCall],
) -> str:
    """Screen products using quantitative rules."""
    # Gather candidate product IDs from retrieval + referenced products.
    candidate_ids = list({p["id"] for p in products_referenced if p.get("id")})
    candidate_ids = [
        product_id for product_id in candidate_ids
        if (product := session.get(ProductEntity, product_id)) is not None and is_screening_candidate(product)
    ]
    if not candidate_ids:
        # Never fall back to historical sample series.  This is the normal
        # path when the user asks a broad question with no products selected.
        all_products = session.query(ProductEntity).all()
        candidate_ids = [product.id for product in all_products if is_screening_candidate(product)]

    if not candidate_ids:
        return "当前没有可用于筛选的已确认真实产品。请先核对上传解析结果，并确认至少一个具有净值数据的产品。"

    started = perf_counter()
    # Infer rules from query keywords.
    rules = quant_screen.ScreenRules(min_observations=8, confirmed_only=False)
    if "回撤" in query or "drawdown" in query.lower():
        rules.max_drawdown = 0.15
    if "夏普" in query or "sharpe" in query.lower():
        rules.min_sharpe = 0.5

    results = quant_screen.screen_products(session, candidate_ids, rules)
    passed = [r for r in results if r.passed]
    excluded = [r for r in results if not r.passed]

    tool_calls.append(ChatToolCall(
        name="screen_products",
        status="ok",
        duration_ms=(perf_counter() - started) * 1000,
        summary=f"{len(passed)} 通过 / {len(excluded)} 剔除（共 {len(results)} 个）",
    ))

    lines: list[str] = [f"筛选规则：最低 {rules.min_observations} 期"]
    if rules.max_drawdown:
        lines[0] += f" | 回撤 ≤ {rules.max_drawdown:.0%}"
    if rules.min_sharpe:
        lines[0] += f" | 夏普 ≥ {rules.min_sharpe}"
    lines.append("")

    if passed:
        lines.append(f"**通过（{len(passed)} 个）：**")
        for r in passed:
            m = r.metrics
            if m:
                sharpe = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "—"
                lines.append(f"  {r.product_name}：年化 {m.annualized_return:.2%} | 回撤 {m.maximum_drawdown:.2%} | 夏普 {sharpe}")
            else:
                lines.append(f"  {r.product_name}")
            if r.warnings:
                for w in r.warnings:
                    lines.append(f"    ⚠ {w}")

    if excluded:
        lines.append(f"\n**剔除（{len(excluded)} 个）：**")
        for r in excluded:
            reasons = "；".join(r.exclusion_reasons)
            lines.append(f"  {r.product_name}：{reasons}")

    return "\n".join(lines)


def _handle_compare_query(
    session: Session, product_ids: list[str], tool_calls: list[ChatToolCall]
) -> str:
    """Compare selected products with metrics and correlation."""
    if len(product_ids) < 2:
        return "请选择至少 2 个产品进行比较。"

    started = perf_counter()
    result = quant_screen.compare_products(session, product_ids)
    tool_calls.append(ChatToolCall(
        name="compare_products",
        status="ok",
        duration_ms=(perf_counter() - started) * 1000,
        summary=f"对比 {len(result.products)} 个产品",
    ))

    lines: list[str] = []
    if result.aligned_window:
        lines.append(f"对齐窗口：{result.aligned_window}\n")

    # Metrics table.
    lines.append("| 产品 | 期数 | 年化收益 | 波动率 | 夏普 | 最大回撤 | 卡玛 |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in result.products:
        sharpe = f"{p.sharpe_ratio:.2f}" if p.sharpe_ratio is not None else "—"
        calmar = f"{p.calmar_ratio:.2f}" if p.calmar_ratio is not None else "—"
        lines.append(
            f"| {p.product_name} | {p.observation_count} | {p.annualized_return:.2%} | "
            f"{p.annualized_volatility:.2%} | {sharpe} | {p.maximum_drawdown:.2%} | {calmar} |"
        )

    # Correlation matrix.
    if len(product_ids) >= 2 and result.correlation_matrix:
        lines.append("\n**相关性矩阵：**")
        names = {p.product_id: p.product_name for p in result.products}
        for pid_a in product_ids:
            corrs = []
            for pid_b in product_ids:
                val = result.correlation_matrix.get(pid_a, {}).get(pid_b, 0)
                corrs.append(f"{val:.2f}")
            lines.append(f"  {names.get(pid_a, pid_a)}：{' | '.join(corrs)}")

    return "\n".join(lines)


def _handle_recommend_query(
    session: Session,
    query: str,
    product_ids: list[str],
    tool_calls: list[ChatToolCall],
    citations: list[ChatCitation],
) -> tuple[str, ChatAllocationDraft | None]:
    """Adapt allocation research output to the HTTP chat response contract."""
    research = build_allocation_research(session, query, product_ids)
    if research.tool_summary:
        tool_calls.append(ChatToolCall(
            name="optimize_fof_allocation",
            status="ok",
            duration_ms=research.tool_duration_ms,
            summary=research.tool_summary,
        ))
    existing_keys = {(citation.file_id, citation.fragment_id) for citation in citations}
    for evidence in research.citations or []:
        key = (evidence["file_id"], evidence["fragment_id"])
        if key not in existing_keys:
            citations.append(ChatCitation(**evidence))
            existing_keys.add(key)
    return research.content, ChatAllocationDraft(**research.draft) if research.draft else None


def _handle_describe_query(
    retrieval_result: retrieval.RetrievalResponse,
    products_referenced: list[dict[str, Any]],
) -> str:
    """Describe products/evidence found by retrieval."""
    if not retrieval_result.results:
        return "未在已上传材料中找到相关信息。请上传产品周报、净值图或 PDF 后再试。"

    lines: list[str] = []
    for result in retrieval_result.results[:6]:
        prefix = f"[{result.category}]"
        lines.append(f"{prefix} **{result.title}**\n{result.content}")

    if products_referenced:
        names = [p["name"] for p in products_referenced[:5]]
        lines.append(f"\n涉及产品：{', '.join(names)}")

    return "\n\n".join(lines)


def _handle_selected_product_overview(
    session: Session,
    product_ids: list[str],
    tool_calls: list[ChatToolCall],
) -> str:
    """Describe selected products without discarding their reviewed NAV series.

    Material retrieval can legitimately return no text fragments after a user
    has manually reviewed a chart.  That must not erase the resulting NAV
    series from a request about performance or risk.
    """
    metrics = _handle_nav_query(session, product_ids, tool_calls)
    missing_evidence = []
    for product_id in product_ids[:5]:
        product = session.get(ProductEntity, product_id)
        if product is not None and not product.facts:
            missing_evidence.append(product.standard_name)
    evidence_note = (
        "\n\n策略与管理人资料尚未提取为可引用证据：请补充周报或事实表；"
        "以上收益与风险指标仅来自已复核净值序列。"
        if missing_evidence else ""
    )
    return metrics + evidence_note


def _fallback_answer(retrieval_result: retrieval.RetrievalResponse, intent: str) -> str:
    """When no specific handler produces an answer."""
    if retrieval_result.results:
        return _handle_describe_query(retrieval_result, [])
    return (
        f"已理解您的意图（{intent}），但产品库中暂无匹配数据。\n\n"
        "建议：\n"
        "- 上传私募周报、净值图或 PDF 材料\n"
        "- 在左侧产品库中确认产品实体\n"
        "- 选择产品后再提出具体问题"
    )
