"""Example: Running Conversational FOF Agent with DeepSeek LLM.

Usage:
    # Set environment variables first:
    export LLM_API_BASE="https://api.deepseek.com/v1"
    export LLM_API_KEY="your-deepseek-api-key"
    export LLM_MODEL="deepseek-v4-flash"

    # Run the example:
    python -m app.services.fof_agent.run_conversational_example
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.fof_agent.conversational import ConversationalFofAgent, get_llm_config_from_env
from app.schemas import DataFrequency, FofFundInput, NetAssetValuePoint


def create_sample_funds():
    """Create sample funds for testing."""
    return [
        FofFundInput(
            fund_id="cta001",
            fund_name="CTA稳健一号",
            frequency=DataFrequency.MONTHLY,
            nav_points=[
                NetAssetValuePoint(observation_date=f"2024-{m:02d}-01", net_asset_value=1.0 + m * 0.003)
                for m in range(1, 13)
            ],
        ),
        FofFundInput(
            fund_id="cta002",
            fund_name="CTA进取成长",
            frequency=DataFrequency.MONTHLY,
            nav_points=[
                NetAssetValuePoint(observation_date=f"2024-{m:02d}-01", net_asset_value=1.0 + m * 0.008 if m % 2 == 0 else 1.0 + m * 0.005)
                for m in range(1, 13)
            ],
        ),
        FofFundInput(
            fund_id="cta003",
            fund_name="CTA平衡配置",
            frequency=DataFrequency.MONTHLY,
            nav_points=[
                NetAssetValuePoint(observation_date=f"2024-{m:02d}-01", net_asset_value=1.0 + m * 0.005)
                for m in range(1, 13)
            ],
        ),
    ]


async def main():
    """Main example function."""

    print("="*60)
    print("FOF Agent with DeepSeek LLM Example")
    print("="*60)

    # Check environment variables
    config = get_llm_config_from_env()

    if config:
        print(f"\n[OK] LLM Config Found:")
        print(f"  API Base: {config.api_base}")
        print(f"  Model: {config.model}")
        print(f"  API Key: {config.api_key[:10]}..." if config.api_key else "  API Key: (not set)")
    else:
        print("\n[WARN] No LLM config found in environment variables")
        print("Please set:")
        print("  export LLM_API_BASE=\"https://api.deepseek.com/v1\"")
        print("  export LLM_API_KEY=\"your-api-key\"")
        print("  export LLM_MODEL=\"deepseek-v4-flash\"")

    # Create agent (will auto-detect config from env)
    agent = ConversationalFofAgent()

    print(f"\n[INFO] Agent initialized with LLM: {agent.use_llm}")

    # Sample funds
    funds = create_sample_funds()

    # Test scenarios
    scenarios = [
        "I want a stable product with low risk",
        "I'm looking for growth opportunities",
        "I want a balanced portfolio",
    ]

    for user_message in scenarios:
        print("\n" + "-"*60)
        print(f"User: {user_message}")
        print("-"*60)

        try:
            response = await agent.chat(
                user_message=user_message,
                funds=funds,
            )

            print(f"\n[Intent Detected]: {response['intent']['detected_risk_profile']}")
            print(f"\n[Recommendation]:")
            for rec in response['recommendation'].recommendations:
                print(f"  - {rec.fund_name}: {rec.weight:.1%}")

            print(f"\n[Explanation]:")
            print(response['explanation'])

        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n" + "="*60)
    print("Example completed")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
