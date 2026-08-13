"""Public FOF agent facade backed by the independent state-machine runtime."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.config import DATA_DIRECTORY
from app.schemas import FofRecommendationRequest, FofRecommendationResponse

from .memory import FofSessionMemory
from .runtime import FofAgentRuntime
from .tools import (
    ResilientToolRegistry as FofToolRegistry,
    build_fof_allocation,
    calculate_fund_nav_metrics,
    review_recommendation,
    score_fund_candidate,
)


class FofRecommendationAgent:
    """Create an auditable FOF run with working and persistent memory layers."""

    def __init__(
        self,
        memory_directory: Path | None = None,
        enable_cache: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.memory = FofSessionMemory(memory_directory or DATA_DIRECTORY / "fof_agent")
        self._enable_cache = enable_cache
        self._max_retries = max_retries

    def recommend(self, request: FofRecommendationRequest) -> FofRecommendationResponse:
        session_id = request.session_id or f"fof-{uuid4().hex[:12]}"
        prior = self.memory.recall(session_id)
        registry = self._create_registry()
        state = FofAgentRuntime(registry).run(request)
        reflection = state.memory.reflection
        if reflection is None:
            raise RuntimeError("FOF runtime finished without executing its reflection gate")
        memory_context = self._memory_context(prior, request.risk_profile.value)
        self.memory.remember(session_id, {
            "risk_profile": request.risk_profile.value,
            "recommended_funds": [item.fund_id for item in state.memory.recommendations],
            "reflection_passed": reflection.passed,
            "plan_revisions": state.plan_revisions,
        })
        return FofRecommendationResponse(
            session_id=session_id,
            evaluations=state.memory.evaluations,
            recommendations=state.memory.recommendations,
            tool_trace=registry.trace,
            agent_trace=state.events,
            plan_revisions=state.plan_revisions,
            reflection=reflection,
            memory_context=memory_context,
            method_provenance={
                "agent": {
                    "method": "规则状态机 + 本地统计工具",
                    "detail": "计划、工具调用、反思与重规划均可追踪；本次评分不使用 LLM 自主下单或持仓推断",
                },
            },
        )

    def _create_registry(self) -> FofToolRegistry:
        registry = FofToolRegistry(
            enable_cache=self._enable_cache,
            max_retries=self._max_retries,
        )
        registry.register("calculate_fund_nav_metrics", calculate_fund_nav_metrics)
        registry.register("score_fund_candidate", score_fund_candidate)
        registry.register("build_fof_allocation", build_fof_allocation)
        registry.register("review_recommendation", review_recommendation)
        return registry

    @staticmethod
    def _memory_context(prior: list[dict], current_risk_profile: str) -> list[str]:
        if not prior:
            return ["当前会话尚无历史运行记录；本次仅使用本请求提供的净值数据。"]
        previous = prior[-1]
        context = [f"已读取本地会话的最近一次运行摘要（{previous.get('recorded_at', '未知时间')}）。"]
        if previous.get("risk_profile") != current_risk_profile:
            context.append("本次风险偏好不同于上次；历史结果仅作审计上下文，未参与本次评分。")
        return context
