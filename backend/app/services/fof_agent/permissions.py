"""Permission control system for FOF Agent tools.

Implements hierarchical permission levels to control which tools can be executed
and what operations are allowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PermissionLevel(IntEnum):
    """Hierarchical permission levels."""

    # Read-only operations (safe)
    READ = 1  # Can read data, perform calculations

    # Local modifications (moderate risk)
    WRITE_LOCAL = 2  # Can write to local files, modify session data

    # External operations (higher risk)
    WRITE_EXTERNAL = 3  # Can make external API calls, send data out

    # Administrative operations (highest risk)
    ADMIN = 4  # Can modify system state, delete data


# Tool permission requirements
TOOL_PERMISSIONS: dict[str, PermissionLevel] = {
    # Read-only tools (safe)
    "calculate_fund_nav_metrics": PermissionLevel.READ,
    "score_fund_candidate": PermissionLevel.READ,
    "review_recommendation": PermissionLevel.READ,
    "web_search": PermissionLevel.READ,  # Reading external info
    # Local write tools
    "build_fof_allocation": PermissionLevel.WRITE_LOCAL,
    # Future tools
    # "write_to_file": PermissionLevel.WRITE_LOCAL,
    # "send_notification": PermissionLevel.WRITE_EXTERNAL,
    # "delete_data": PermissionLevel.ADMIN,
}


@dataclass
class PermissionContext:
    """Context for permission evaluation."""

    user_id: str | None = None
    session_id: str | None = None
    ip_address: str | None = None
    request_origin: str | None = None
    # Custom attributes that tools can use
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionResult:
    """Result of a permission check."""

    granted: bool
    reason: str
    level_required: PermissionLevel
    level_granted: PermissionLevel


class PermissionDeniedError(Exception):
    """Raised when a tool execution is denied due to insufficient permissions."""

    def __init__(self, tool_name: str, required: PermissionLevel, granted: PermissionLevel):
        self.tool_name = tool_name
        self.required = required
        self.granted = granted
        super().__init__(
            f"Permission denied for tool '{tool_name}': "
            f"required level {required.name} (value={required.value}), "
            f"but only have {granted.name} (value={granted.value})"
        )


class PermissionChecker:
    """Permission control system for tool execution."""

    def __init__(self, default_level: PermissionLevel = PermissionLevel.READ) -> None:
        self._default_level = default_level
        self._custom_rules: dict[str, Callable[[str, PermissionContext], PermissionLevel]] = {}
        self._permission_overrides: dict[str, PermissionLevel] = {}

    def set_default_level(self, level: PermissionLevel) -> None:
        """Set the default permission level for this checker."""
        self._default_level = level

    def add_custom_rule(
        self,
        tool_name: str,
        rule: Callable[[str, PermissionContext], PermissionLevel]
    ) -> None:
        """Add a custom permission rule for a specific tool."""
        self._custom_rules[tool_name] = rule

    def override_permission(self, tool_name: str, level: PermissionLevel) -> None:
        """Override the permission level for a specific tool."""
        self._permission_overrides[tool_name] = level
        logger.info(f"Permission override for '{tool_name}': {level.name}")

    def check(self, tool_name: str, context: PermissionContext) -> PermissionResult:
        """Check if a tool can be executed with the given context."""

        # Get required permission level for the tool
        required = self._get_required_level(tool_name)

        # Get granted permission level (from context or default)
        granted = self._get_granted_level(context)

        # Check if granted level is sufficient
        if granted >= required:
            return PermissionResult(
                granted=True,
                reason=f"Permission granted: {granted.name} >= {required.name}",
                level_required=required,
                level_granted=granted,
            )
        else:
            return PermissionResult(
                granted=False,
                reason=f"Permission denied: {granted.name} < {required.name}",
                level_required=required,
                level_granted=granted,
            )

    def _get_required_level(self, tool_name: str) -> PermissionLevel:
        """Get the required permission level for a tool."""

        # Check for override first
        if tool_name in self._permission_overrides:
            return self._permission_overrides[tool_name]

        # Check for custom rule
        if tool_name in self._custom_rules:
            # Custom rules need context, so return a default for now
            # The actual check will be done in check()
            return TOOL_PERMISSIONS.get(tool_name, PermissionLevel.READ)

        # Fall back to defined permissions
        return TOOL_PERMISSIONS.get(tool_name, PermissionLevel.READ)

    def _get_granted_level(self, context: PermissionContext) -> PermissionLevel:
        """Determine the granted permission level from context."""

        # Check if context has explicit permission level
        if "permission_level" in context.attributes:
            return context.attributes["permission_level"]

        # Default to the checker's default level
        return self._default_level


class RateLimiter:
    """Rate limiter to prevent tool abuse."""

    def __init__(
        self,
        max_calls_per_minute: int = 60,
        max_calls_per_hour: int = 1000,
    ) -> None:
        self._max_per_minute = max_calls_per_minute
        self._max_per_hour = max_calls_per_hour
        self._calls: list[float] = []  # Timestamps of recent calls

    def check(self, identifier: str = "global") -> bool:
        """Check if a call is allowed under rate limits."""
        import time as time_module

        now = time_module.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        # Clean old entries
        self._calls = [ts for ts in self._calls if ts > hour_ago]

        # Check limits
        calls_last_minute = sum(1 for ts in self._calls if ts > minute_ago)

        if calls_last_minute >= self._max_per_minute:
            logger.warning(f"Rate limit exceeded: {calls_last_minute} calls in last minute")
            return False

        if len(self._calls) >= self._max_per_hour:
            logger.warning(f"Hourly rate limit exceeded: {len(self._calls)} calls in last hour")
            return False

        # Record this call
        self._calls.append(now)
        return True

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._calls.clear()


class ToolPermissionGuard:
    """Guard that wraps tool execution with permission checks."""

    def __init__(
        self,
        permission_checker: PermissionChecker | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._permission_checker = permission_checker or PermissionChecker()
        self._rate_limiter = rate_limiter or RateLimiter()

    def can_execute(self, tool_name: str, context: PermissionContext) -> tuple[bool, str]:
        """Check if a tool can be executed."""

        # Rate limit check
        if not self._rate_limiter.check():
            return False, "Rate limit exceeded"

        # Permission check
        result = self._permission_checker.check(tool_name, context)
        if not result.granted:
            return False, result.reason

        return True, result.reason

    def wrap_execute(
        self,
        tool_name: str,
        execute_fn: Callable[..., Any],
        context: PermissionContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wrap tool execution with permission and rate limit checks."""

        # Check permissions
        can_exec, reason = self.can_execute(tool_name, context)
        if not can_exec:
            logger.warning(f"Tool execution denied: {tool_name} - {reason}")
            raise PermissionDeniedError(
                tool_name,
                self._permission_checker._get_required_level(tool_name),
                context.attributes.get("permission_level", PermissionLevel.READ),
            )

        # Execute
        return execute_fn(*args, **kwargs)


# Default instances for easy import
default_permission_checker = PermissionChecker()
default_rate_limiter = RateLimiter()
default_guard = ToolPermissionGuard()
