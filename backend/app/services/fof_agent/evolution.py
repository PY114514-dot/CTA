"""Agent self-evolution system for FOF Agent.

Enables the agent to learn from user feedback and沉淀 reusable rules as Skills.
This is inspired by Claude Code's skill system.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SkillCategory(StrEnum):
    """Categories of skills."""

    VALIDATION = "validation"  # Input/output validation rules
    TRANSFORMATION = "transformation"  # Data transformation rules
    ROUTING = "routing"  # Decision routing rules
    FORMATTING = "formatting"  # Output formatting rules
    CUSTOM = "custom"  # Custom domain rules


class SkillDecision(StrEnum):
    """Decision on how to handle a new skill."""

    ADD = "add"  # Add as new skill
    MERGE = "merge"  # Merge with existing skill
    DISCARD = "discard"  # Discard (duplicate/invalid)


@dataclass
class Skill:
    """A reusable skill that the agent has learned."""

    name: str
    description: str
    category: SkillCategory
    trigger_pattern: str  # Regex or keyword that triggers this skill
    action: dict[str, Any]  # What to do when triggered
    provenance: list[str] = field(default_factory=list)  # Sources that led to this skill
    version: int = 1
    usage_count: int = 0
    success_rate: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class UserFeedback:
    """User feedback that can be used to create or update skills."""

    feedback_id: str
    original_input: str
    agent_output: str
    user_correction: str | None  # None if positive feedback
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SkillExtractionResult:
    """Result of extracting potential skills from feedback."""

    decision: SkillDecision
    reason: str
    skill: Skill | None = None
    merge_target: str | None = None
    confidence: float = 0.0


class SkillStore:
    """Persistent storage for skills."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.skills_file = storage_path / "skills.jsonl"
        self._skills: dict[str, Skill] = {}
        self._load()

    def _load(self) -> None:
        """Load skills from disk."""
        if not self.skills_file.exists():
            return

        for line in self.skills_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    skill = Skill(**data)
                    self._skills[skill.name] = skill
                except Exception as e:
                    logger.warning(f"Failed to load skill: {e}")

    def save(self, skill: Skill) -> None:
        """Save a skill to disk."""
        self._skills[skill.name] = skill
        with self.skills_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(skill.__dict__, ensure_ascii=False) + "\n")

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_by_category(self, category: SkillCategory) -> list[Skill]:
        """Get all skills in a category."""
        return [s for s in self._skills.values() if s.category == category]

    def find_matching(self, input_text: str) -> list[Skill]:
        """Find skills that match the input text."""
        import re

        matching = []
        for skill in self._skills.values():
            try:
                if re.search(skill.trigger_pattern, input_text, re.IGNORECASE):
                    matching.append(skill)
            except re.error:
                # Invalid regex, skip
                continue
        return matching


class FeedbackAnalyzer:
    """Analyzes user feedback to extract potential skills."""

    def __init__(self, skill_store: SkillStore) -> None:
        self.skill_store = skill_store

    def analyze(self, feedback: UserFeedback) -> SkillExtractionResult:
        """Analyze feedback and decide if a new skill should be created."""

        # Positive feedback - may want to formalize the pattern
        if feedback.user_correction is None:
            return SkillExtractionResult(
                decision=SkillDecision.DISCARD,
                reason="Positive feedback doesn't require new skill",
                confidence=0.5,
            )

        # Negative feedback - extract what went wrong
        correction = feedback.user_correction

        # Simple heuristic: if correction contains specific patterns, create a skill
        skill_name = self._extract_skill_name(feedback.original_input, correction)
        trigger_pattern = self._extract_trigger_pattern(feedback.original_input)
        category = self._infer_category(feedback.original_input, correction)

        if not trigger_pattern:
            return SkillExtractionResult(
                decision=SkillDecision.DISCARD,
                reason="Could not extract trigger pattern",
                confidence=0.3,
            )

        # Check for existing similar skills
        existing = self.skill_store.find_matching(feedback.original_input)
        if existing:
            # Check if we should merge
            best_match = max(existing, key=lambda s: s.success_rate)
            if best_match.success_rate > 0.8:
                return SkillExtractionResult(
                    decision=SkillDecision.MERGE,
                    skill=None,
                    merge_target=best_match.name,
                    reason=f"Similar skill '{best_match.name}' exists with high success rate",
                    confidence=0.7,
                )

        # Create new skill
        new_skill = Skill(
            name=skill_name,
            description=f"Auto-generated skill from user feedback: {correction[:100]}",
            category=category,
            trigger_pattern=trigger_pattern,
            action={
                "type": "transform",
                "description": feedback.user_correction,
            },
            provenance=[f"feedback:{feedback.feedback_id}"],
        )

        return SkillExtractionResult(
            decision=SkillDecision.ADD,
            skill=new_skill,
            reason="New skill extracted from negative feedback",
            confidence=0.6,
        )

    def _extract_skill_name(self, original: str, correction: str) -> str:
        """Extract a name for the new skill."""
        # Simple heuristic: use keywords from correction
        words = correction.split()[:3]
        return "_".join(words).lower().replace(",", "").replace(".", "")[:50]

    def _extract_trigger_pattern(self, original: str) -> str | None:
        """Extract a trigger pattern from the original input."""
        # Simple: use the whole input as pattern
        # In production, would use NLP to extract key phrases
        if len(original) < 3:
            return None
        import re
        # Escape special regex chars
        return re.escape(original[:100])

    def _infer_category(self, original: str, correction: str) -> SkillCategory:
        """Infer the category of the skill."""
        text = (original + correction).lower()

        if any(w in text for w in ["validate", "check", "verify", "error"]):
            return SkillCategory.VALIDATION
        elif any(w in text for w in ["convert", "transform", "format"]):
            return SkillCategory.TRANSFORMATION
        elif any(w in text for w in ["if", "when", "condition", "route"]):
            return SkillCategory.ROUTING
        elif any(w in text for w in ["format", "output", "display", "style"]):
            return SkillCategory.FORMATTING
        else:
            return SkillCategory.CUSTOM


class AgentEvolution:
    """Main class for agent self-evolution."""

    def __init__(self, storage_path: Path) -> None:
        self.skill_store = SkillStore(storage_path)
        self.feedback_analyzer = FeedbackAnalyzer(self.skill_store)
        self.feedback_history: list[UserFeedback] = []

    def add_feedback(self, feedback: UserFeedback) -> None:
        """Add user feedback to the system."""
        self.feedback_history.append(feedback)
        self._process_feedback(feedback)

    def _process_feedback(self, feedback: UserFeedback) -> None:
        """Process feedback and potentially create new skills."""
        result = self.feedback_analyzer.analyze(feedback)

        if result.decision == SkillDecision.ADD and result.skill:
            self.skill_store.save(result.skill)
            logger.info(f"New skill added: {result.skill.name}")

        elif result.decision == SkillDecision.MERGE and result.merge_target:
            existing = self.skill_store.get(result.merge_target)
            if existing:
                existing.version += 1
                existing.updated_at = datetime.now(timezone.utc).isoformat()
                existing.provenance.append(f"feedback:{feedback.feedback_id}")
                self.skill_store.save(existing)
                logger.info(f"Skill merged: {existing.name}")

    def get_applicable_skills(self, context: str) -> list[Skill]:
        """Get all skills applicable to the current context."""
        return self.skill_store.find_matching(context)

    def record_skill_usage(self, skill_name: str, success: bool) -> None:
        """Record that a skill was used."""
        skill = self.skill_store.get(skill_name)
        if skill:
            skill.usage_count += 1
            # Update success rate
            total = skill.usage_count
            if success:
                skill.success_rate = (skill.success_rate * (total - 1) + 1) / total
            else:
                skill.success_rate = skill.success_rate * (total - 1) / total
            skill.updated_at = datetime.now(timezone.utc).isoformat()
            self.skill_store.save(skill)

    def get_skill_stats(self) -> dict[str, Any]:
        """Get statistics about the skill system."""
        all_skills = list(self.skill_store._skills.values())
        return {
            "total_skills": len(all_skills),
            "by_category": {
                c.value: len(self.skill_store.get_by_category(c))
                for c in SkillCategory
            },
            "total_feedback": len(self.feedback_history),
            "total_usages": sum(s.usage_count for s in all_skills),
            "average_success_rate": (
                sum(s.success_rate for s in all_skills) / len(all_skills)
                if all_skills else 0
            ),
        }


# Example skill definitions for FOF Agent
DEFAULT_SKILLS = [
    Skill(
        name="validate_nav_values",
        description="Validate NAV values are positive and chronological",
        category=SkillCategory.VALIDATION,
        trigger_pattern=r"NAV|nav|净值",
        action={"type": "validate", "rules": ["positive", "chronological"]},
        provenance=["builtin"],
    ),
    Skill(
        name="format_percentage",
        description="Format values as percentages",
        category=SkillCategory.FORMATTING,
        trigger_pattern=r"%|percent",
        action={"type": "format", "style": "percentage"},
        provenance=["builtin"],
    ),
]
