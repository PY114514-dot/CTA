"""Offline configuration tests for the optional DeepSeek-compatible client."""

from app.services.fof_agent.conversational import ConversationalFofAgent, get_llm_config_from_env


def test_llm_config_requires_both_endpoint_and_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert get_llm_config_from_env() is None

    monkeypatch.setenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    assert get_llm_config_from_env() is None


def test_agent_uses_supplied_config_without_contacting_external_service(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")

    config = get_llm_config_from_env()
    assert config is not None
    assert config.model == "deepseek-chat"
    agent = ConversationalFofAgent(memory_directory=tmp_path, llm_config=config)
    assert agent.use_llm is True
    assert agent.llm_config == config
