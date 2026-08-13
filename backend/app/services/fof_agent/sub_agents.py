"""Sub-agent system for FOF Agent.

Enables creating isolated sub-agents for specific tasks such as:
- Explore: Search and analyze codebase/data
- Plan: Generate execution plans
- General: Handle arbitrary subtasks
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SubAgentType(StrEnum):
    """Types of sub-agents available."""

    EXPLORE = "explore"  # Search and analyze
    PLAN = "plan"  # Generate plans
    GENERAL = "general"  # General purpose


@dataclass
class SubAgentResult:
    """Result from executing a sub-agent."""

    agent_type: SubAgentType
    success: bool
    output: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class SubAgentContext:
    """Context passed to sub-agents."""

    task: str
    parent_context: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 5
    timeout_seconds: float = 60.0
    # For isolation: sub-agents get a clean context
    isolated_vars: dict[str, Any] = field(default_factory=dict)


class BaseSubAgent(ABC):
    """Base class for sub-agents."""

    def __init__(self, name: str, agent_type: SubAgentType) -> None:
        self.name = name
        self.agent_type = agent_type

    @abstractmethod
    async def execute(self, context: SubAgentContext) -> SubAgentResult:
        """Execute the sub-agent's task."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this sub-agent."""
        pass


class ExploreSubAgent(BaseSubAgent):
    """Sub-agent for exploring and analyzing data/code."""

    def __init__(self) -> None:
        super().__init__("explorer", SubAgentType.EXPLORE)

    def get_system_prompt(self) -> str:
        return """You are an exploration agent. Your task is to:
1. Search for relevant information based on the query
2. Analyze and summarize findings
3. Provide actionable insights

Focus on accuracy and completeness. Report what you found and any gaps."""

    async def execute(self, context: SubAgentContext) -> SubAgentResult:
        import time as time_module

        start = time_module.perf_counter()

        try:
            # Simulate exploration logic
            # In a real implementation, this would:
            # - Search through data/code
            # - Analyze patterns
            # - Return structured findings

            query = context.task
            logger.info(f"Explore agent processing: {query}")

            # For now, return a placeholder result
            output = f"Exploration complete for: {query}"
            artifacts = {
                "query": query,
                "findings": [],
                "recommendations": [],
            }

            return SubAgentResult(
                agent_type=self.agent_type,
                success=True,
                output=output,
                artifacts=artifacts,
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return SubAgentResult(
                agent_type=self.agent_type,
                success=False,
                output="",
                error=str(e),
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )


class PlanSubAgent(BaseSubAgent):
    """Sub-agent for generating execution plans."""

    def __init__(self) -> None:
        super().__init__("planner", SubAgentType.PLAN)

    def get_system_prompt(self) -> str:
        return """You are a planning agent. Your task is to:
1. Understand the goal and constraints
2. Break down into actionable steps
3. Identify dependencies and risks
4. Create a detailed plan

Provide a structured plan with clear milestones."""

    async def execute(self, context: SubAgentContext) -> SubAgentResult:
        import time as time_module

        start = time_module.perf_counter()

        try:
            goal = context.task
            logger.info(f"Plan agent processing: {goal}")

            # Simulate planning logic
            output = f"Plan generated for: {goal}"
            artifacts = {
                "goal": goal,
                "steps": [
                    {"id": 1, "description": "Analyze requirements", "dependencies": []},
                    {"id": 2, "description": "Execute core logic", "dependencies": [1]},
                    {"id": 3, "description": "Validate results", "dependencies": [2]},
                ],
                "estimated_duration": "5 minutes",
            }

            return SubAgentResult(
                agent_type=self.agent_type,
                success=True,
                output=output,
                artifacts=artifacts,
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return SubAgentResult(
                agent_type=self.agent_type,
                success=False,
                output="",
                error=str(e),
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )


class GeneralSubAgent(BaseSubAgent):
    """General purpose sub-agent."""

    def __init__(self) -> None:
        super().__init__("general", SubAgentType.GENERAL)

    def get_system_prompt(self) -> str:
        return """You are a general-purpose agent. Your task is to:
1. Understand the user's request
2. Execute the requested operation
3. Report results clearly

Be thorough and accurate."""

    async def execute(self, context: SubAgentContext) -> SubAgentResult:
        import time as time_module

        start = time_module.perf_counter()

        try:
            task = context.task
            logger.info(f"General agent processing: {task}")

            output = f"Task completed: {task}"
            artifacts = {"task": task, "status": "completed"}

            return SubAgentResult(
                agent_type=self.agent_type,
                success=True,
                output=output,
                artifacts=artifacts,
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )

        except Exception as e:
            return SubAgentResult(
                agent_type=self.agent_type,
                success=False,
                output="",
                error=str(e),
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )


class SubAgentManager:
    """Manager for creating and running sub-agents."""

    def __init__(self) -> None:
        self._agents: dict[SubAgentType, type[BaseSubAgent]] = {
            SubAgentType.EXPLORE: ExploreSubAgent,
            SubAgentType.PLAN: PlanSubAgent,
            SubAgentType.GENERAL: GeneralSubAgent,
        }
        self._execution_history: list[SubAgentResult] = []

    def create_agent(self, agent_type: SubAgentType) -> BaseSubAgent:
        """Create a new sub-agent instance."""
        agent_class = self._agents.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class()

    async def run_agent(
        self,
        agent_type: SubAgentType,
        context: SubAgentContext,
    ) -> SubAgentResult:
        """Run a sub-agent with the given context."""
        agent = self.create_agent(agent_type)
        result = await agent.execute(context)

        # Record execution
        self._execution_history.append(result)

        logger.info(
            f"Sub-agent '{agent_type}' completed: "
            f"success={result.success}, duration={result.duration_ms:.1f}ms"
        )

        return result

    def get_history(self) -> list[SubAgentResult]:
        """Get the execution history of all sub-agents."""
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()


# Integration with main FOF Agent
class SubAgentTool:
    """Tool wrapper for invoking sub-agents from the main agent."""

    def __init__(self, manager: SubAgentManager | None = None) -> None:
        self._manager = manager or SubAgentManager()

    async def execute(
        self,
        agent_type: SubAgentType,
        task: str,
        **kwargs: Any,
    ) -> SubAgentResult:
        """Execute a sub-agent for a specific task."""

        context = SubAgentContext(
            task=task,
            parent_context=kwargs.get("parent_context", {}),
            max_iterations=kwargs.get("max_iterations", 5),
            timeout_seconds=kwargs.get("timeout_seconds", 60.0),
            isolated_vars=kwargs.get("isolated_vars", {}),
        )

        return await self._manager.run_agent(agent_type, context)

    def explore(self, task: str, **kwargs: Any) -> SubAgentResult:
        """Shortcut for running an explore agent."""
        return asyncio.run(self.execute(SubAgentType.EXPLORE, task, **kwargs))

    def plan(self, task: str, **kwargs: Any) -> SubAgentResult:
        """Shortcut for running a plan agent."""
        return asyncio.run(self.execute(SubAgentType.PLAN, task, **kwargs))

    def general(self, task: str, **kwargs: Any) -> SubAgentResult:
        """Shortcut for running a general agent."""
        return asyncio.run(self.execute(SubAgentType.GENERAL, task, **kwargs))
