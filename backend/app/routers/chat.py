"""FOF Agent conversation endpoint: retrieval-augmented chat with citations.

Implements a constrained Plan → Retrieve → Act → Reflect loop for the
workbench conversation panel. Quantitative tools are deterministic; when the
user has enabled an LLM it is used only to synthesize those audited results.
"""

from __future__ import annotations

import asyncio
import json
import re
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from datetime import date
from time import perf_counter
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.dependencies import get_report_config
from app.models import AgentRun, DataSnapshot, DecisionRecord, NavObservation, ProductEntity, RawFile
from app.services import product_store as store
from app.services.upload_storage import resolve_upload_path
from app.services import retrieval
from app.services import quant_screen
from app.services.agent_memory import load_preferences, remember_preferences
from app.services.agent_tools import build_plan, validate_requested_tools
from app.services.allocation_agent_graph import request_allocation_approval, run_allocation_agent
from app.services.fof_allocation_intent import (
    is_demo_product,
    is_screening_candidate,
    parse_allocation_intent,
)
from app.services.fof_allocation_research import build_allocation_research
from app.services.nav_metrics import calculate_nav_analysis
from app.services.nav_quality import assess_nav_quality as assess_product_nav_quality
from app.services.chart_extractor.vlm_extractor import DashScopeProvider, OpenAICompatibleProvider
from app.schemas import NavAnalysisRequest, NetAssetValuePoint, DataFrequency
from app.routers.cta_attribution import evaluate_reviewed_product

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
    draft_id: str = ""
    allocations: list[ChatAllocationItem] = Field(default_factory=list)
    exclusions: list[dict[str, Any]] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    portfolio: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: str = "agent"
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    products_referenced: list[dict[str, Any]] = Field(default_factory=list)
    data_context: list[dict[str, Any]] = Field(default_factory=list)
    method_provenance: dict[str, Any] = Field(default_factory=dict)
    allocation_draft: ChatAllocationDraft | None = None


class SaveAllocationDraftRequest(BaseModel):
    draft_id: str


@router.post("/chat/allocation-drafts")
def save_allocation_draft(body: SaveAllocationDraftRequest, session: Session = Depends(get_session)) -> dict[str, str]:
    """Save a calculated configuration only after the user explicitly asks."""
    snapshot = session.get(DataSnapshot, body.draft_id)
    if snapshot is None or snapshot.label != "FOF 配置研究快照":
        raise HTTPException(404, "未找到可保存的配置研究快照")
    draft = (snapshot.content or {}).get("draft")
    if not isinstance(draft, dict) or not draft.get("allocations"):
        raise HTTPException(422, "该研究快照没有可保存的配置")
    decision = store.create_decision(
        session, decision_type="allocation", title="FOF 配置研究草案",
        content={
            **draft,
            "interpreted_constraints": (snapshot.content or {}).get("interpreted_constraints", {}),
            "source_draft_snapshot_id": snapshot.id,
            "workflow": {"stage": "draft", "tracking_enabled": False},
        },
        data_snapshot_id=snapshot.id, status="draft",
    )
    return {"id": decision.id, "status": decision.status}


@router.post("/chat/allocation-drafts/{decision_id}/submit")
def submit_allocation_draft(decision_id: str, session: Session = Depends(get_session)) -> dict[str, str]:
    """Enter the LangGraph pause; this never approves or executes a draft."""
    decision = session.get(DecisionRecord, decision_id)
    if decision is None or decision.decision_type != "allocation":
        raise HTTPException(404, "未找到配置草案")
    if decision.status != "draft":
        raise HTTPException(409, "该配置草案已提交或已完成审批")
    if not request_allocation_approval(decision.id, decision.content or {}).waiting_for_human:
        raise HTTPException(500, "审批工作流未能进入人工确认节点")
    decision.status = "pending_review"
    content = dict(decision.content or {})
    content["workflow"] = {"stage": "awaiting_human_approval", "tracking_enabled": False}
    decision.content = content
    session.commit()
    return {"id": decision.id, "status": decision.status}


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
    # 明确的比较请求始终优先，不能被“最多几只、收益/回撤”等配置词误判。
    if _INTENT_PATTERNS[0][1].search(query):
        return "compare"
    # 用户常直接回复一组配置约束，而不会重复说“配置”或“FOF”。此时不能因
    # “收益、波动”落入单产品净值查询；配置候选池本来就不依赖已选路演材料。
    allocation = parse_allocation_intent(query)
    if allocation.max_products is not None and (
        allocation.min_annualized_return is not None
        or allocation.max_annualized_volatility is not None
        or allocation.max_drawdown is not None
    ):
        return "recommend"
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return "describe"


def _awaits_allocation_preferences(session: Session, session_id: str | None) -> bool:
    if not session_id:
        return False
    previous = session.query(AgentRun).filter(AgentRun.session_id == session_id).order_by(AgentRun.created_at.desc()).first()
    return bool(previous and (previous.plan or {}).get("intent") == "recommend_clarification")


def _has_allocation_preferences(query: str) -> bool:
    return bool(re.search(r"\d+(?:\.\d+)?\s*%|(?:最多|不超过|至多)\s*\d+\s*(?:只|个|款)?", query))


def _needs_allocation_clarification(intent: Any) -> bool:
    """A return target or a risk limit is enough to start a bounded draft."""
    return intent.min_annualized_return is None and intent.max_drawdown is None and intent.max_annualized_volatility is None


def _allocation_clarification() -> str:
    return (
        "开始配置前，请至少给出一个明确目标：目标年化收益，或最大回撤/年化波动上限。\n\n"
        "例如：年化收益至少 10%；或最大回撤不超过 10%。未指定产品数量时，系统默认最多配置 5 只产品。"
    )


def _interpret_allocation_constraints(query: str, config: Any) -> dict[str, object] | None:
    """Ask the LLM for a small constraint contract, then validate locally.

    This is language understanding only.  The optimizer still receives only
    validated numeric fields and never executes prose supplied by the model.
    """
    if not config.enabled:
        return None
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "你只做中文 FOF 配置需求的语义提取，不推荐产品、不计算、不解释。输出 JSON。"},
            {"role": "user", "content": (
                "从用户原话提取配置约束。字段只能是 max_products、max_drawdown、"
                "min_annualized_return、max_annualized_volatility、requested_tools；百分比转为 0 到 1 的小数。"
                "“最多、不超过、至多、最好不超过、尽量不超过”均提取 max_products。"
                "requested_tools 只能从 interpret_allocation_constraints、search_products、optimize_fof_allocation 中选择；"
                "它只是建议，不能执行任何操作。没有明确数值则填 null。只输出 JSON。\n\n用户原话：" + query
            )},
        ],
        "temperature": 0,
        "max_tokens": min(160, config.max_tokens),
        "response_format": {"type": "json_object"},
    }
    if config.model.lower().startswith("deepseek-") and not config.thinking_enabled:
        payload["thinking"] = {"type": "disabled"}
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.post(
                f"{config.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content)
    except Exception as error:
        logger.warning("Allocation constraint interpretation failed: %s", error)
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatMessage)
def chat(body: ChatRequest, session: Session = Depends(get_session)) -> ChatMessage:
    """Process a natural language query with retrieval and deterministic tools."""
    started = perf_counter()
    intent = _detect_intent(body.query)
    if intent != "recommend" and _awaits_allocation_preferences(session, body.session_id):
        intent = "recommend"
    config = get_report_config()
    constraint_started = perf_counter()
    semantic_constraints = _interpret_allocation_constraints(body.query, config) if intent == "recommend" else None
    preliminary_intent = parse_allocation_intent(body.query, semantic_constraints)
    needs_allocation_clarification = intent == "recommend" and _needs_allocation_clarification(preliminary_intent)
    tool_calls: list[ChatToolCall] = []
    citations: list[ChatCitation] = []
    products_referenced: list[dict[str, Any]] = []
    if intent == "recommend" and semantic_constraints is not None:
        accepted, rejected = validate_requested_tools(intent, semantic_constraints.get("requested_tools"))
        tool_calls.append(ChatToolCall(
            name="validate_agent_plan",
            status="ok",
            summary=(f"LLM 建议：{'、'.join(accepted) or '无'}；实际执行仍由固定工作流决定" + (f"；已拒绝：{'、'.join(rejected)}" if rejected else "")),
        ))
    if needs_allocation_clarification:
        tool_calls.append(ChatToolCall(
            name="interpret_allocation_constraints",
            status="ok" if semantic_constraints is not None else "error",
            duration_ms=(perf_counter() - constraint_started) * 1000,
            summary="尚未识别到目标收益或风险上限；先向用户确认",
        ))

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
    if body.product_ids and not selected_ids and intent not in {"screen", "recommend"}:
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
    if blocked_products and intent not in {"screen", "recommend"}:
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

    # Asking for missing allocation constraints is a conversation turn, not a
    # retrieval task. Do not read arbitrary materials before asking the user.
    retrieval_result = retrieval.RetrievalResponse(query=body.query, results=[], total=0)
    if not needs_allocation_clarification:
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
        content = _handle_screen_query(session, body.query, tool_calls)
    elif intent == "compare":
        content = _handle_compare_query(session, selected_ids, tool_calls)
    elif intent == "recommend":
        if needs_allocation_clarification:
            content = _allocation_clarification()
        else:
            content, allocation_draft = _handle_recommend_query(
                session, body.query, tool_calls, citations, session_id=body.session_id,
                semantic_constraints=semantic_constraints,
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
    if config.enabled and not needs_allocation_clarification and intent != "recommend":
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
            if intent == "compare":
                content = _interpret_comparison_with_llm(content, body.query, config)
                tool_name = "llm_comparison_interpretation"
                tool_summary = "已基于确定性对比结果生成差异化解读"
            else:
                scope_note = (
                    "本次为产品库筛选或 FOF 配置：候选范围由工具结果中的“产品库”说明确定，"
                    "不受当前单品研究对象限制。"
                    if intent in {"screen", "recommend"}
                    else "本次只分析当前单品研究对象。"
                )
                content = _synthesize_with_llm(content, body.query, data_context, config, image_evidence, scope_note)
                tool_name = "llm_research_synthesis"
                tool_summary = f"已使用 {config.model} 基于工具结果生成解读"
            tool_calls.append(ChatToolCall(
                name=tool_name,
                status="ok",
                duration_ms=(perf_counter() - llm_started) * 1000,
                summary=tool_summary,
            ))
        except Exception as error:
            logger.warning("LLM agent synthesis failed, using deterministic answer: %s", error)
            tool_calls.append(ChatToolCall(
                name="llm_comparison_interpretation" if intent == "compare" else "llm_research_synthesis",
                status="error",
                duration_ms=(perf_counter() - llm_started) * 1000,
                summary="LLM 调用失败，已回退到工具计算结果",
            ))

    # Record agent run for audit.
    run = store.create_agent_run(
        session,
        user_query=body.query,
        session_id=body.session_id,
        plan={
            "intent": "recommend_clarification" if needs_allocation_clarification else intent,
            "product_ids": selected_ids,
            "ignored_product_ids": ignored_ids,
            "allowed_tools": build_plan(intent),
        },
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
        reflection={
            "outcome": "needs_clarification" if needs_allocation_clarification else "completed",
            "tool_failures": [tc.name for tc in tool_calls if tc.status != "ok"],
            "next_step": "补充配置取舍" if needs_allocation_clarification else "查看结论或继续追问",
        },
        data_snapshot_id=allocation_draft.draft_id if allocation_draft else None,
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
    scope_note: str = "",
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
        f"【候选范围规则】\n{scope_note}\n\n"
        f"【本次数据范围】\n{context_text}\n\n"
        f"【视觉模型补充证据】\n{image_evidence or '未使用视觉模型补充证据。'}\n\n"
        f"【工具计算结果】\n{raw_answer}"
    )
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "你是严谨的量化研究写作助手，绝不虚构或修改已计算数值。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": min(1800 if config.thinking_enabled else 900, config.max_tokens),
    }
    if config.model.lower().startswith("deepseek-") and not config.thinking_enabled:
        payload["thinking"] = {"type": "disabled"}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{config.api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("模型返回空内容")
    return content


def _interpret_comparison_with_llm(raw_answer: str, question: str, config: Any) -> str:
    """Turn audited comparison facts into a decision-specific explanation.

    This path deliberately differs from report synthesis: it asks for the
    material trade-off implied by the user's question, not a sectioned replay
    of every metric.  All numerical facts still come from ``raw_answer``.
    """
    prompt = (
        "你是量化产品比较助手。只基于下方已计算结果回答，不能补造或改写任何数值、"
        "策略归因、管理人信息或风险判断。\n"
        "你的任务不是逐项复述指标，也不要使用“配置角色、收益比较、回撤比较、分散化价值、提示”"
        "之类固定标题。先针对用户问题给一句明确结论；再用不超过三条短句解释真正影响选择的差异；"
        "最后只在数据确实不足或相关性不支持判断时补一句限制。\n"
        "若用户问二选一，说明不同目标下各自更适合的条件；若用户问是否组合，围绕相关性和风险收益取舍回答；"
        "若问题没有给出决策场景，说明最关键的取舍并建议用户补充一个场景。保持简洁、专业的中文，不要说你调用了 LLM。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【已计算对比结果】\n{raw_answer}"
    )
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "你只能解释已计算的产品差异，不得把固定模板伪装成研究结论。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": min(700, config.max_tokens),
    }
    if config.model.lower().startswith("deepseek-") and not config.thinking_enabled:
        payload["thinking"] = {"type": "disabled"}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{config.api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("模型返回空内容")
    return content


def _run_async(coro):
    """Run an async coroutine from sync code without conflicting with the running event loop."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-runner") as pool:
        return pool.submit(asyncio.run, coro).result()


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
        response = _run_async(provider.extract_structure(path.read_bytes(), prompt))
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
    tool_calls: list[ChatToolCall],
) -> str:
    """Screen every confirmed, reviewed product in the product library."""
    candidate_ids = [
        product.id
        for product in session.query(ProductEntity).all()
        if is_screening_candidate(product)
    ]

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
        summary=f"产品库：{len(passed)} 通过 / {len(excluded)} 剔除（共 {len(results)} 个）",
    ))

    lines: list[str] = [f"产品库筛选：共检查 {len(candidate_ids)} 个已确认、净值已复核产品。", f"筛选规则：最低 {rules.min_observations} 期"]
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

    lines: list[str] = ["已按已复核净值完成直接对比。"]
    if result.aligned_window:
        lines.append(f"共同区间：{result.aligned_window}")

    for product in result.products:
        sharpe = f"{product.sharpe_ratio:.2f}" if product.sharpe_ratio is not None else "—"
        calmar = f"{product.calmar_ratio:.2f}" if product.calmar_ratio is not None else "—"
        lines.append(
            f"{product.product_name}：年化收益 {product.annualized_return:.2%} · 年化波动 "
            f"{product.annualized_volatility:.2%} · 最大回撤 {product.maximum_drawdown:.2%} · "
            f"夏普 {sharpe} · 卡玛 {calmar}"
        )

    if len(result.products) == 2:
        left, right = result.products
        return_gap = left.annualized_return - right.annualized_return
        if abs(return_gap) >= 0.001:
            higher = left if return_gap > 0 else right
            lines.append(f"收益：{higher.product_name}年化收益高 {abs(return_gap):.2%}。")

        drawdown_gap = left.maximum_drawdown - right.maximum_drawdown
        if abs(drawdown_gap) >= 0.001:
            shallower = left if drawdown_gap > 0 else right
            lines.append(f"回撤：{shallower.product_name}最大回撤更浅 {abs(drawdown_gap):.2%}。")

        correlation = result.correlation_matrix.get(left.product_id, {}).get(right.product_id)
        if correlation is not None:
            correlation_readout = "低相关，具备分散价值" if correlation <= 0.2 else "高度相关，分散价值有限" if correlation >= 0.8 else "相关性中等"
            lines.append(f"相关性：{correlation:.2f}，{correlation_readout}。")

    lines.append("需要查看净值叠加、指标差异或因子并排时，请使用“产品对比”面板。")

    return "\n".join(lines)


def _handle_recommend_query(
    session: Session,
    query: str,
    tool_calls: list[ChatToolCall],
    citations: list[ChatCitation],
    config: Any | None = None,
    session_id: str | None = None,
    semantic_constraints: dict[str, object] | None = None,
) -> tuple[str, ChatAllocationDraft | None]:
    """Adapt allocation research output to the HTTP chat response contract."""
    started = perf_counter()
    if semantic_constraints is None:
        semantic_constraints = _interpret_allocation_constraints(query, config or get_report_config())
    remembered = load_preferences(session, session_id)
    interpreted = {**remembered, **{key: value for key, value in (semantic_constraints or {}).items() if value is not None}}
    intent = parse_allocation_intent(query, interpreted)
    interpreted_parts = []
    if intent.min_annualized_return is not None:
        interpreted_parts.append(f"年化收益≥{intent.min_annualized_return:.0%}")
    if intent.max_annualized_volatility is not None:
        interpreted_parts.append(f"年化波动≤{intent.max_annualized_volatility:.0%}")
    if intent.max_drawdown is not None:
        interpreted_parts.append(f"最大回撤≤{intent.max_drawdown:.0%}")
    if intent.max_products is not None:
        interpreted_parts.append(f"产品数≤{intent.max_products}只")
    if semantic_constraints is not None:
        tool_calls.append(ChatToolCall(
            name="interpret_allocation_constraints",
            status="ok",
            duration_ms=(perf_counter() - started) * 1000,
            summary="；".join(interpreted_parts) or "未识别到带数值的配置约束",
        ))
    else:
        tool_calls.append(ChatToolCall(
            name="interpret_allocation_constraints",
            status="error",
            duration_ms=(perf_counter() - started) * 1000,
            summary="语义约束解析不可用，仅采用明确的本地数值约束",
        ))
    agent_result = run_allocation_agent(session, query, intent)
    research = agent_result.research
    tool_calls.append(ChatToolCall(
        name="run_allocation_agent",
        status="ok",
        summary=f"LangGraph 已完成 {len(agent_result.plan)} 个受控步骤；配置校验通过",
    ))
    remember_preferences(
        session,
        session_id,
        {key: value for key, value in (semantic_constraints or {}).items() if value is not None},
    )
    if research.tool_summary:
        tool_calls.append(ChatToolCall(
            name="optimize_fof_allocation",
            status="ok",
            duration_ms=research.tool_duration_ms,
            summary=research.tool_summary,
        ))
    # The allocation snapshot retains every linked source for audit.  They are
    # not all evidence used by this answer, so do not append them as chat
    # citations.  The query retrieval above is the only citation set shown to
    # the user.
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
    attribution = [
        _handle_attribution_summary(session, product_id, tool_calls)
        for product_id in product_ids[:3]
    ]
    return "\n\n".join([metrics, *filter(None, attribution)])


def _handle_attribution_summary(
    session: Session, product_id: str, tool_calls: list[ChatToolCall]
) -> str:
    """Add a compact deterministic attribution and risk readout to Agent research.

    The Phase-A result is the same reviewed-NAV attribution shown in the CTA
    panel.  It is deliberately summarized here rather than delegated to the
    LLM, so the research conclusion remains traceable and does not fall back
    to manager-material boilerplate when no disclosure is available.
    """
    started = perf_counter()
    try:
        result = evaluate_reviewed_product(product_id, session)
    except Exception as error:
        tool_calls.append(ChatToolCall(
            name="analyze_attribution_risk",
            status="error",
            duration_ms=(perf_counter() - started) * 1000,
            summary="归因暂不可用，保留净值风险评价",
        ))
        return ""

    performance = result["performance_path"]
    baseline = result["baseline"]
    exposures = result["attribution"]["factor_exposure"]
    significant = [item for item in exposures if item.get("hac_p_value", 1.0) < 0.05]
    main_exposure = max(significant, key=lambda item: abs(item["beta"])) if significant else None
    oos = baseline.get("out_of_sample", {})
    applicability = result["model_applicability"]

    risk = (
        f"风险：最大回撤 {performance['maximum_drawdown']:.2%}，"
        f"最差 5% 平均亏损 {performance['expected_shortfall_5pct']:.2%}，"
        f"卡玛 {performance['calmar']:.3g}"
    )
    if performance.get("recovery_completed") is False:
        risk += f"，当前回撤已持续 {performance['current_drawdown_duration_periods']} 期"

    attribution = f"归因：样本内 R² {baseline['r_squared']:.2f}"
    if oos.get("r_squared") is not None:
        attribution += f"，样本外 R² {oos['r_squared']:.2f}"
    if main_exposure is None:
        attribution += "；当前公开因子中未发现 p<0.05 的显著关联。"
    else:
        label = main_exposure.get("display_name") or main_exposure["factor_name"]
        p_value = main_exposure["hac_p_value"]
        p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
        attribution += f"；主要统计关联为{label}（系数 {main_exposure['beta']:.3f}，{p_text}）。"

    tool_calls.append(ChatToolCall(
        name="analyze_attribution_risk",
        status="ok",
        duration_ms=(perf_counter() - started) * 1000,
        summary=f"{result['product_name']}：R² {baseline['r_squared']:.2f}，{applicability['status']}",
    ))
    return (
        f"**{result['product_name']} · 归因与风险评价：**\n"
        f"- {risk}。\n"
        f"- {attribution}\n"
        f"- 归因适用性：{applicability['reason']}"
    )


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
