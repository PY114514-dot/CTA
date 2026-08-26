"""Unit tests for local environment persistence and secret handling.

The .env file holds LLM/VLM API keys.  These tests pin down the guarantees
the rest of the app relies on: injection rejection, atomic updates, no
overriding of real environment variables, and permission tightening for
credential keys.
"""

import os

import app.config as config


def test_persist_rejects_newline_injection() -> None:
    try:
        config.persist_local_environment({"VLM_MODEL": "evil\nDASHSCOPE_API_KEY=abc"})
    except ValueError as error:
        assert "无效" in str(error)
    else:
        raise AssertionError("newline injection must be rejected")


def test_persist_rejects_equals_in_key() -> None:
    try:
        config.persist_local_environment({"VLM=MODEL": "qwen3-vl-flash"})
    except ValueError as error:
        assert "无效" in str(error)
    else:
        raise AssertionError("'=' in a key must be rejected")


def test_persist_updates_existing_keys_and_preserves_others(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    env_file = tmp_path / "backend" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("VLM_PROVIDER=dashscope\nKEEP_ME=1\n", encoding="utf-8")

    config.persist_local_environment({"VLM_MODEL": "qwen3-vl-flash"})

    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert "VLM_PROVIDER=dashscope" in lines
    assert "KEEP_ME=1" in lines
    assert "VLM_MODEL=qwen3-vl-flash" in lines


def test_persist_roundtrip_preserves_values_with_equals(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("LLM_API_BASE", raising=False)

    config.persist_local_environment({"LLM_API_BASE": "https://api.example.com/v1?x=1"})

    config.load_local_environment()
    assert os.environ["LLM_API_BASE"] == "https://api.example.com/v1?x=1"


def test_load_local_environment_does_not_override_real_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    env_file = tmp_path / "backend" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("DASHSCOPE_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from_process")

    config.load_local_environment()

    assert os.environ["DASHSCOPE_API_KEY"] == "from_process"


def test_persist_secret_keys_trigger_permission_tightening(monkeypatch, tmp_path) -> None:
    calls: list[object] = []
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "_restrict_env_file_permissions", lambda path: calls.append(path))

    config.persist_local_environment({"DASHSCOPE_API_KEY": "sk-test"})

    assert len(calls) == 1
    assert calls[0].name == ".env"


def test_persist_non_secret_keys_skip_permission_tightening(monkeypatch, tmp_path) -> None:
    calls: list[object] = []
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "_restrict_env_file_permissions", lambda path: calls.append(path))

    config.persist_local_environment({"VLM_MODEL": "qwen3-vl-flash"})

    assert calls == []
