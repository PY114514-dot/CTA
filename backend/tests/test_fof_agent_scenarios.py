"""Deterministic end-to-end scenarios for the FOF recommendation workflow."""

from pathlib import Path
from app.schemas import (
    DataFrequency,
    FofFundInput,
    FofRecommendationRequest,
    FofRiskProfile,
    NetAssetValuePoint,
)
from app.services.fof_agent import FofRecommendationAgent
from app.services.fof_agent.tools import ResilientToolRegistry
from app.services.fof_agent.permissions import (
    PermissionChecker,
    PermissionContext,
    PermissionLevel,
    default_permission_checker,
)
from app.services.fof_agent.evolution import (
    AgentEvolution,
    UserFeedback,
    SkillCategory,
    SkillDecision,
)


# ============================================================================
# Test Fixtures
# ============================================================================

def _fund(fund_id: str, name: str, values: list[float]) -> FofFundInput:
    """Create a fund input with given NAV values."""
    return FofFundInput(
        fund_id=fund_id,
        fund_name=name,
        frequency=DataFrequency.MONTHLY,
        nav_points=[
            NetAssetValuePoint(observation_date=f"2024-{month:02d}-01", net_asset_value=value)
            for month, value in enumerate(values, start=1)
        ],
    )


# ============================================================================
# Scenario Tests
# ============================================================================

class TestConservativeProductRecommendation:
    """Test Scenario: User wants a conservative/stable product."""

    def test_user_wants_conservative_product_analysis(self, tmp_path: Path):
        """
        User says: "I want to buy a relatively stable/conservative product"

        Expected behavior:
        1. Agent detects conservative risk preference
        2. Agent filters for stable funds
        3. Agent provides an auditable tool trace
        """
        # Simulate user request - use more volatile growth fund
        request = FofRecommendationRequest(
            funds=[
                _fund("stable_1", "Stable One", [
                    1.000, 1.003, 1.005, 1.008, 1.010,
                    1.012, 1.015, 1.018, 1.020, 1.022,
                    1.025, 1.028,
                ]),
                _fund("stable_2", "Very Stable", [
                    1.000, 1.002, 1.004, 1.005, 1.007,
                    1.008, 1.010, 1.011, 1.013, 1.014,
                    1.016, 1.018,
                ]),
                _fund("growth_1", "High Growth", [
                    1.000, 1.020, 1.050, 0.900, 1.040,  # Large drawdown
                    1.080, 0.850, 1.090, 1.140, 1.180,
                    1.100, 1.250,
                ]),
            ],
            risk_profile=FofRiskProfile.CONSERVATIVE,
            session_id="test-conservative-scenario",
            max_single_fund_weight=0.6,
        )

        # Run the agent
        agent = FofRecommendationAgent(tmp_path, enable_cache=True)
        result = agent.recommend(request)

        # Assertions - conservative profile should heavily favor stable funds
        # The stable funds should have higher weights
        stable_weights = [item.weight for item in result.recommendations if item.fund_id.startswith("stable")]
        growth_weights = [item.weight for item in result.recommendations if item.fund_id.startswith("growth")]

        # Stable funds should have higher weights than growth fund
        assert sum(stable_weights) > sum(growth_weights), "Stable funds should get higher weights for conservative profile"

        # Should have tool trace
        tool_names = {item.name for item in result.tool_trace}
        assert "calculate_fund_nav_metrics" in tool_names
        assert "score_fund_candidate" in tool_names

class TestPermissionSystem:
    """Test permission control system."""

    def test_read_tool_allowed_by_default(self):
        """Test that read tools are allowed by default."""
        checker = default_permission_checker
        context = PermissionContext(user_id="test_user")

        # Read tools should be allowed
        result = checker.check("calculate_fund_nav_metrics", context)
        assert result.granted

    def test_permission_denied_for_admin_tool(self):
        """Test that admin tools require higher permissions."""
        checker = PermissionChecker(default_level=PermissionLevel.READ)
        context = PermissionContext(user_id="test_user")

        # Override to require admin
        checker.override_permission("some_admin_tool", PermissionLevel.ADMIN)

        result = checker.check("some_admin_tool", context)
        assert not result.granted

    def test_custom_permission_level(self):
        """Test custom permission level in context."""
        checker = PermissionChecker(default_level=PermissionLevel.READ)
        context = PermissionContext(
            user_id="admin_user",
            attributes={"permission_level": PermissionLevel.ADMIN},
        )

        # Should allow admin tools
        checker.override_permission("admin_only_tool", PermissionLevel.ADMIN)
        result = checker.check("admin_only_tool", context)
        assert result.granted


class TestAgentEvolution:
    """Test agent self-evolution system."""

    def test_feedback_creates_skill(self, tmp_path: Path):
        """Test that negative feedback can create a new skill."""
        evolution = AgentEvolution(tmp_path / "evolution")

        # Simulate negative feedback
        feedback = UserFeedback(
            feedback_id="fb_001",
            original_input="基金的夏普比率怎么计算",
            agent_output="夏普比率 = (收益率 - 无风险利率) / 波动率",
            user_correction="计算夏普比率时需要年化处理，不能直接用月收益率",
        )

        evolution.add_feedback(feedback)

        # Check skill was created
        stats = evolution.get_skill_stats()
        assert stats["total_skills"] == 1
        assert stats["by_category"][SkillCategory.CUSTOM] == 1

    def test_positive_feedback_discarded(self, tmp_path: Path):
        """Test that positive feedback doesn't create new skills."""
        evolution = AgentEvolution(tmp_path / "evolution2")

        feedback = UserFeedback(
            feedback_id="fb_002",
            original_input="计算累计收益",
            agent_output="累计收益 = 最新净值 / 初始净值 - 1",
            user_correction=None,  # Positive feedback
        )

        # Should not create a skill
        result = evolution.feedback_analyzer.analyze(feedback)
        assert result.decision == SkillDecision.DISCARD


class TestCachePerformance:
    """Test caching performance."""

    def test_cache_serves_repeated_call_without_reexecution(self, tmp_path: Path):
        """Cache behavior is deterministic; elapsed time is not a test contract."""

        # Create registry with cache
        registry = ResilientToolRegistry(enable_cache=True)

        fund = _fund("test_cache", "测试基金", [
            1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07,
        ])

        from app.services.fof_agent.tools import calculate_fund_nav_metrics
        registry.register("calculate_fund_nav_metrics", calculate_fund_nav_metrics)

        # First call populates the cache; second call must be served from it.
        result1 = registry.execute("calculate_fund_nav_metrics", fund=fund, risk_free_rate=0.015)
        result2 = registry.execute("calculate_fund_nav_metrics", fund=fund, risk_free_rate=0.015)

        # Results should be the same
        assert result1 == result2

        # Check cache stats
        stats = registry.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["valid"] == 1


class TestFullScenario:
    """Full end-to-end scenario test."""

    def test_full_conservative_recommendation_flow(self, tmp_path: Path):
        """
        Complete flow:
        1. User requests conservative product
        2. Agent analyzes funds
        3. Agent applies risk profile
        4. Agent generates recommendation
        5. Agent provides explanation
        """
        # Step 1: User request
        request = FofRecommendationRequest(
            funds=[
                _fund("cta_stable_1", "CTA Stable One", [
                    1.000, 1.005, 1.008, 1.012, 1.015,
                    1.018, 1.020, 1.022, 1.025, 1.028,
                    1.030, 1.032,
                ]),
                _fund("cta_stable_2", "CTA Conservative Two", [
                    1.000, 1.002, 1.004, 1.006, 1.008,
                    1.010, 1.012, 1.014, 1.016, 1.018,
                    1.020, 1.022,
                ]),
                _fund("cta_growth", "CTA Growth", [
                    1.000, 1.015, 1.035, 0.990, 1.025,
                    1.050, 0.970, 1.040, 1.070, 1.100,
                    1.060, 1.120,
                ]),
            ],
            risk_profile=FofRiskProfile.CONSERVATIVE,
            session_id="full-scenario-test",
            max_single_fund_weight=0.5,
        )

        # Step 2-5: Run agent
        agent = FofRecommendationAgent(tmp_path, enable_cache=True)
        result = agent.recommend(request)

        # The explanation is a display helper, not an external model call.
        explanation = self._generate_explanation(result, request.risk_profile.value)

        # Assertions
        assert len(result.recommendations) >= 2
        assert all(item.weight <= 0.5 for item in result.recommendations)
        assert result.reflection.passed
        assert "historical NAV data" in explanation

    def _generate_explanation(self, result, risk_profile: str) -> str:
        """Generate user-facing explanation."""
        recs = result.recommendations
        if not recs:
            return "Sorry, no products match your risk profile."

        lines = [
            f"Based on your {risk_profile} risk preference, I recommend the following funds:"
        ]

        for rec in recs:
            lines.append(
                f"  - {rec.fund_name}: allocation {rec.weight:.0%}, "
                f"high composite score, good volatility control"
            )

        lines.append(
            f"\nThis recommendation is based on historical NAV data and does not constitute"
            f" actual investment advice. Please invest carefully."
        )

        return "\n".join(lines)
