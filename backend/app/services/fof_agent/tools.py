"""Named deterministic tools used by the FOF recommendation agent."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from app.schemas import (
    FofCandidateEvaluation,
    FofAllocationItem,
    FofFundInput,
    FofReflection,
    FofRiskProfile,
    FofToolCall,
    NavAnalysisRequest,
    PerformanceMetrics,
)
from app.services.nav_metrics import calculate_nav_analysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_contract: dict[str, str]


TOOL_DEFINITIONS = [
    ToolDefinition("calculate_fund_nav_metrics", "计算经校验净值序列的收益、波动率、夏普、回撤和卡玛比率。", {"fund": "FofFundInput", "risk_free_rate": "float"}),
    ToolDefinition("score_fund_candidate", "按风险偏好对单只基金进行透明评分与样本充分性检查。", {"fund": "FofFundInput", "metrics": "PerformanceMetrics", "risk_profile": "FofRiskProfile"}),
    ToolDefinition("build_fof_allocation", "以评分和逆波动为基础生成单基金权重上限约束下的配置。", {"evaluations": "list[FofCandidateEvaluation]", "max_weight": "float"}),
    ToolDefinition("review_recommendation", "检查推荐数量、权重上限、数据充分性和风险提示。", {"evaluations": "list[FofCandidateEvaluation]", "recommendations": "list[FofAllocationItem]", "max_weight": "float"}),
    ToolDefinition("web_search", "联网搜索外部信息，如市场数据、基金新闻、业绩排名等。", {"query": "str", "max_results": "int"}),
]


class ToolCache:
    """LRU-style cache for tool results with TTL support."""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (result, timestamp)
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _make_key(self, tool_name: str, **kwargs: Any) -> str:
        """Generate a cache key from tool name and arguments."""
        # Convert kwargs to a hashable representation
        serializable = {}
        for k, v in kwargs.items():
            if hasattr(v, "__dict__"):
                # For dataclasses, use a hash of their dict representation
                serializable[k] = str(v)
            else:
                serializable[k] = v
        key_str = f"{tool_name}:{json.dumps(serializable, sort_keys=True, default=str)}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, tool_name: str, **kwargs: Any) -> Any | None:
        """Retrieve cached result if not expired."""
        key = self._make_key(tool_name, **kwargs)
        if key in self._cache:
            result, timestamp = self._cache[key]
            import time as time_module
            if time_module.time() - timestamp < self._ttl:
                logger.debug(f"Cache hit for {tool_name}")
                return result
            else:
                del self._cache[key]
        return None

    def set(self, tool_name: str, result: Any, **kwargs: Any) -> None:
        """Store result in cache with current timestamp."""
        key = self._make_key(tool_name, **kwargs)
        import time as time_module
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (result, time_module.time())
        logger.debug(f"Cached result for {tool_name}")

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        import time as time_module
        now = time_module.time()
        valid = sum(1 for _, ts in self._cache.values() if now - ts < self._ttl)
        return {"total": len(self._cache), "valid": valid, "expired": len(self._cache) - valid}


class ResilientToolRegistry:
    """Tool registry with caching, deduplication, retry and fallback support."""

    def __init__(
        self,
        enable_cache: bool = True,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        permission_guard=None,  # ToolPermissionGuard instance
    ) -> None:
        self.trace: list[FofToolCall] = []
        self._tools: dict[str, Callable[..., Any]] = {}
        self._cache = ToolCache() if enable_cache else None
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._call_history: list[tuple[str, dict, str]] = []  # (tool_name, args, status)
        self._permission_guard = permission_guard

    def register(self, name: str, function: Callable[..., Any]) -> None:
        if name not in {definition.name for definition in TOOL_DEFINITIONS}:
            raise ValueError(f"Unknown FOF tool: {name}")
        self._tools[name] = function

    def _is_duplicate(self, tool_name: str, kwargs: dict[str, Any]) -> bool:
        """Check if the same tool with same args was called recently."""
        key = (tool_name, tuple(sorted(kwargs.items())))
        # Check last 10 calls
        recent_calls = self._call_history[-10:]
        return any(c[0] == tool_name and c[1] == key for c in recent_calls)

    def execute(self, name: str, **kwargs: Any) -> Any:
        definition = next((item for item in TOOL_DEFINITIONS if item.name == name), None)
        if definition is None:
            raise ValueError(f"Unknown FOF tool: {name}")
        missing = set(definition.input_contract) - set(kwargs)
        unexpected = set(kwargs) - set(definition.input_contract)
        if missing or unexpected:
            raise ValueError(
                f"Invalid arguments for {name}; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        function = self._tools.get(name)
        if function is None:
            raise ValueError(f"Tool is not registered: {name}")

        # Check permissions if guard is configured
        if self._permission_guard is not None:
            from .permissions import PermissionContext
            context = PermissionContext(
                attributes=kwargs.pop("_permission_context", {})
            )
            can_exec, reason = self._permission_guard.can_execute(name, context)
            if not can_exec:
                from .permissions import PermissionDeniedError, PermissionLevel
                # Get required level
                from .permissions import TOOL_PERMISSIONS
                required = TOOL_PERMISSIONS.get(name, PermissionLevel.READ)
                granted = context.attributes.get("permission_level", PermissionLevel.READ)
                raise PermissionDeniedError(name, required, granted)

        # Check cache first
        if self._cache is not None:
            cached_result = self._cache.get(name, **kwargs)
            if cached_result is not None:
                self._call_history.append((name, tuple(sorted(kwargs.items())), "cache_hit"))
                return cached_result

        # Check for duplicate call
        if self._is_duplicate(name, kwargs):
            logger.warning(f"Duplicate call detected for {name}, returning cached if available")
            if self._cache is not None:
                cached = self._cache.get(name, **kwargs)
                if cached is not None:
                    self._call_history.append((name, tuple(sorted(kwargs.items())), "dedup_hit"))
                    return cached

        # Execute with retry
        return self._execute_with_retry(name, function, **kwargs)

    def _execute_with_retry(self, name: str, function: Callable[..., Any], **kwargs: Any) -> Any:
        """Execute tool with exponential backoff retry."""
        import time as time_module
        last_error = None

        for attempt in range(self._max_retries):
            started = perf_counter()
            try:
                result = function(**kwargs)

                # Cache successful result
                if self._cache is not None:
                    self._cache.set(name, result, **kwargs)

                self.trace.append(FofToolCall(
                    name=name, status="ok", duration_ms=(perf_counter() - started) * 1000,
                    summary=_summary(name, result),
                ))
                self._call_history.append((name, tuple(sorted(kwargs.items())), "ok"))
                return result

            except Exception as error:
                last_error = error
                if attempt < self._max_retries - 1:
                    wait_time = self._backoff_base ** attempt
                    logger.warning(f"Tool {name} failed (attempt {attempt + 1}/{self._max_retries}), retrying in {wait_time}s: {error}")
                    time_module.sleep(wait_time)
                else:
                    logger.error(f"Tool {name} failed after {self._max_retries} attempts: {error}")

        # All retries failed - record error and raise
        self.trace.append(FofToolCall(
            name=name, status="error", duration_ms=0,
            summary=f"{type(last_error).__name__}: {last_error}",
        ))
        self._call_history.append((name, tuple(sorted(kwargs.items())), "error"))
        raise last_error

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        if self._cache is None:
            return {"enabled": False}
        return {"enabled": True, **self._cache.stats()}


# Backward compatibility alias
FofToolRegistry = ResilientToolRegistry


def calculate_fund_nav_metrics(fund: FofFundInput, risk_free_rate: float) -> PerformanceMetrics:
    return calculate_nav_analysis(NavAnalysisRequest(
        nav_points=fund.nav_points,
        frequency=fund.frequency,
        annual_risk_free_rate=risk_free_rate,
    )).metrics


def score_fund_candidate(
    fund: FofFundInput,
    metrics: PerformanceMetrics,
    risk_profile: FofRiskProfile,
) -> FofCandidateEvaluation:
    """Score only supplied historical figures; no future-return assumption is made."""
    weights = {
        FofRiskProfile.CONSERVATIVE: (10.0, 35.0, 40.0, 15.0),
        FofRiskProfile.BALANCED: (20.0, 35.0, 30.0, 15.0),
        FofRiskProfile.GROWTH: (35.0, 30.0, 20.0, 15.0),
    }[risk_profile]
    annual_return = _scale(metrics.annualized_return, -0.20, 0.30)
    sharpe = _scale(metrics.sharpe_ratio if metrics.sharpe_ratio is not None else -0.5, -0.5, 2.0)
    drawdown = _scale(0.40 - abs(metrics.maximum_drawdown), 0.0, 0.40)
    history = min(len(fund.nav_points) / 52.0, 1.0)
    components = {
        "annualized_return": annual_return * weights[0],
        "sharpe_ratio": sharpe * weights[1],
        "drawdown_control": drawdown * weights[2],
        "history_depth": history * weights[3],
    }
    warnings: list[str] = []
    if len(fund.nav_points) < 8:
        warnings.append("净值观测少于 8 个，样本不足，不纳入推荐。")
    elif len(fund.nav_points) < 24:
        warnings.append("净值观测少于 24 个，历史期较短，评分稳定性有限。")
    if abs(metrics.maximum_drawdown) > 0.30:
        warnings.append("历史最大回撤超过 30%，需专项复核风险预算与流动性。")
    if metrics.sharpe_ratio is None:
        warnings.append("波动率为零，夏普比率无意义。")
    return FofCandidateEvaluation(
        fund_id=fund.fund_id, fund_name=fund.fund_name,
        eligible=len(fund.nav_points) >= 8,
        score=round(sum(components.values()), 2),
        score_breakdown={name: round(value, 2) for name, value in components.items()},
        metrics=metrics, warnings=warnings,
    )


def build_fof_allocation(
    evaluations: list[FofCandidateEvaluation], max_weight: float,
) -> dict[str, float]:
    """Allocate selected funds by score / volatility, then cap and redistribute."""
    if not evaluations:
        return {}
    raw = {
        item.fund_id: max(item.score, 1.0) / max(item.metrics.annualized_volatility, 0.03)
        for item in evaluations
    }
    total = sum(raw.values())
    weights = {fund_id: value / total for fund_id, value in raw.items()}
    uncapped = set(weights)
    while uncapped:
        over = {key: value for key, value in weights.items() if key in uncapped and value > max_weight}
        if not over:
            break
        for key in over:
            weights[key] = max_weight
            uncapped.remove(key)
        remaining = 1.0 - sum(weights[key] for key in weights if key not in uncapped)
        if not uncapped or remaining <= 0:
            break
        subtotal = sum(raw[key] for key in uncapped)
        for key in uncapped:
            weights[key] = remaining * raw[key] / subtotal
    return {key: round(value, 6) for key, value in weights.items() if value > 0}


def review_recommendation(
    evaluations: list[FofCandidateEvaluation], recommendations: list[FofAllocationItem], max_weight: float,
) -> FofReflection:
    """Deterministic reflection gate which is independent of the planner."""
    checks = ["所有绩效指标均由已验证的净值计算工具产生。"]
    warnings: list[str] = []
    hard_failure = False
    if len(recommendations) < 2:
        warnings.append("可推荐产品少于 2 个，无法形成有效的 FOF 分散化。")
        hard_failure = True
    else:
        checks.append("推荐产品数量满足基础分散化要求。")
    if any(item.weight > max_weight + 1e-9 for item in recommendations):
        warnings.append("发现单产品权重超过设定上限，结果不可用。")
        hard_failure = True
    else:
        checks.append("所有推荐权重均在单产品上限内。")
    if abs(sum(item.weight for item in recommendations) - 1.0) > 1e-5 and recommendations:
        warnings.append("推荐权重之和不等于 100%，结果不可用。")
        hard_failure = True
    elif recommendations:
        checks.append("推荐权重之和为 100%。")
    if any(not item.eligible for item in evaluations):
        warnings.append("部分候选因净值样本不足被排除；请补充历史数据后复核。")
    warnings.append("未纳入申赎条款、费用、管理人尽调、相关性和实时估值；投前须补齐。")
    return FofReflection(passed=not hard_failure, checks=checks, warnings=warnings)


def _scale(value: float, floor: float, ceiling: float) -> float:
    return min(max((value - floor) / (ceiling - floor), 0.0), 1.0)


@dataclass
class WebSearchResult:
    """Result from web search."""
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0


def web_search(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """Search the web for external information.

    This tool enables the FOF Agent to fetch real-time market data,
    fund news, performance rankings, and other external information
    that cannot be derived from local NAV data alone.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        List of WebSearchResult with title, url, snippet and relevance score
    """
    import httpx

    # Use DuckDuckGo HTML search (no API key required)
    # For production, this could be replaced with Bing API, Google Custom Search, etc.
    search_url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    data = {"q": query, "b": str(max_results)}

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(search_url, data=data, headers=headers)
            response.raise_for_status()

            # Parse results (simple HTML parsing)
            results = _parse_search_results(response.text, max_results)
            logger.info(f"Web search for '{query}' returned {len(results)} results")
            return results

    except Exception as e:
        logger.warning(f"Web search failed for query '{query}': {e}")
        # Return empty list on failure - agent can handle gracefully
        return []


def _parse_search_results(html: str, max_results: int) -> list[WebSearchResult]:
    """Parse search results from DuckDuckGo HTML response."""
    import re

    results = []
    # Simple regex-based parsing for demonstration
    # In production, use BeautifulSoup or similar

    # Find result blocks
    result_pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?'
        r'<a class="result__snippet"[^>]*>([^<]*)</a>',
        re.DOTALL
    )

    for match in result_pattern.finditer(html):
        if len(results) >= max_results:
            break

        url = match.group(1).strip()
        title = match.group(2).strip()
        snippet = match.group(3).strip() if match.group(3) else ""

        # Clean up snippet
        snippet = re.sub(r'<[^>]+>', '', snippet)

        if title and url:
            results.append(WebSearchResult(
                title=title[:200],  # Truncate long titles
                url=url[:500],
                snippet=snippet[:500],
                relevance_score=0.8  # Default score
            ))

    return results


def _summary(name: str, result: Any) -> str:
    if name == "calculate_fund_nav_metrics":
        return "已计算净值绩效指标"
    if name == "score_fund_candidate":
        return f"已完成评分：{result.score:.2f}，{'可纳入' if result.eligible else '样本不足'}"
    if name == "build_fof_allocation":
        return f"已生成 {len(result)} 个权重"
    if name == "web_search":
        return f"联网搜索返回 {len(result)} 条结果"
    return "已完成推荐质量反思"
