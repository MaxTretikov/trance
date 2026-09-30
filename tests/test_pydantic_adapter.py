from pathlib import Path
from types import SimpleNamespace

import pytest

from trance.adapters import pydantic_ai
from trance.types import Candidate


def test_builds_api_key_model_through_lazy_model_factory(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "trance.models":
            return SimpleNamespace(build_model=lambda candidate: ("model", candidate))
        raise AssertionError(name)

    monkeypatch.setattr(pydantic_ai.importlib, "import_module", fake_import)
    candidate = Candidate("openai", "api_key", "env:OPENAI_API_KEY", "gpt-test", "secret")

    result = pydantic_ai.build(candidate)

    assert result[0] == "model"
    assert result[1] == candidate
    assert calls == ["trance.models"]


def test_routes_cli_candidates_to_cli_adapter(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "trance.cli_model":
            return SimpleNamespace(build_cli_model=lambda candidate: ("cli", candidate))
        raise AssertionError(name)

    monkeypatch.setattr(pydantic_ai.importlib, "import_module", fake_import)
    candidate = Candidate("gemini-cli", "subscription", "gemini", "gemini-test")

    result = pydantic_ai.build(candidate)

    assert result[0] == "cli"
    assert result[1] == candidate
    assert calls == ["trance.cli_model"]


def test_copilot_resolver_removes_ambient_github_tokens(monkeypatch, tmp_path: Path):
    observed = {}

    class Result:
        returncode = 0
        stdout = "resolved-token\n"

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        assert command == ["gh", "auth", "token"]
        return Result()

    monkeypatch.setattr(pydantic_ai.subprocess, "run", fake_run)
    monkeypatch.setattr(
        pydantic_ai.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            build_model=lambda candidate: candidate,
        ),
    )
    candidate = Candidate(
        "github-copilot",
        "subscription",
        "github-cli",
        "gpt-test",
        config={"resolver": "github-cli"},
    )

    resolved = pydantic_ai.build(
        candidate,
        environ={"HOME": "/ambient", "GH_TOKEN": "ambient-token", "PATH": "/bin"},
        home=tmp_path,
    )

    assert resolved.secret == "resolved-token"
    assert resolved.source == "resolved:github-cli"
    assert resolved.config == {}
    assert observed["env"]["HOME"] == str(tmp_path)
    assert "GH_TOKEN" not in observed["env"]


def test_missing_pydantic_ai_error_has_direct_install_advice(monkeypatch):
    class DependencyError(RuntimeError):
        pass

    def fake_import(name):
        if name == "trance.models":
            return SimpleNamespace(
                ModelDependencyError=DependencyError,
                build_model=lambda candidate: (_ for _ in ()).throw(
                    DependencyError("optional dependency missing")
                ),
            )
        raise AssertionError(name)

    monkeypatch.setattr(pydantic_ai.importlib, "import_module", fake_import)
    candidate = Candidate("openai", "api_key", "manual", "gpt-test", "secret")

    with pytest.raises(
        pydantic_ai.MissingPydanticAIDependencyError,
        match=r"uv add pydantic-ai-slim\[openai\]",
    ):
        pydantic_ai.build(candidate)
