from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from trance.adapters import litellm
from trance.types import Candidate


def _candidate(provider: str = "openai", **kwargs: object) -> Candidate:
    return Candidate(provider, "api_key", "env:TEST_KEY", "test-model", "super-secret", kwargs)


def _fake_module(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def completion(**kwargs: object) -> object:
        calls.append(kwargs)
        return "sync-result"

    async def acompletion(**kwargs: object) -> object:
        calls.append(kwargs)
        return "async-result"

    monkeypatch.setattr(litellm, "import_module", lambda name: SimpleNamespace(
        completion=completion, acompletion=acompletion
    ))
    return calls


def test_build_is_lazy_and_passes_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_module(monkeypatch)
    adapter = litellm.build(_candidate())

    assert adapter.model == "test-model"
    assert adapter.completion([{"role": "user", "content": "hi"}], temperature=0) == "sync-result"
    assert asyncio.run(adapter.acompletion([])) == "async-result"
    assert calls[0] == {
        "model": "test-model", "messages": [{"role": "user", "content": "hi"}],
        "api_key": "super-secret", "api_base": None, "temperature": 0,
    }
    assert calls[1]["api_key"] == "super-secret"


def test_fixed_openai_endpoint_uses_openai_model_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_module(monkeypatch)
    adapter = litellm.build(_candidate("moonshot"))
    assert adapter.model == "openai/test-model"
    assert adapter.api_base == "https://api.moonshot.ai/v1"


def test_dynamic_endpoint_must_be_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_module(monkeypatch)
    candidate = _candidate(
        "azure-openai", api_style="openai", base_url="https://evil.example/openai/v1"
    )
    with pytest.raises(litellm.UnsupportedLiteLLMCandidateError, match="approved API endpoint"):
        litellm.build(candidate)


@pytest.mark.parametrize("provider", ["openai-codex", "claude-code", "github-copilot"])
def test_delegated_candidates_are_rejected(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    _fake_module(monkeypatch)
    with pytest.raises(litellm.UnsupportedLiteLLMCandidateError, match="delegated"):
        litellm.build(_candidate(provider))


def test_missing_litellm_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise ModuleNotFoundError("litellm")

    monkeypatch.setattr(litellm, "import_module", missing)
    with pytest.raises(litellm.LiteLLMDependencyError, match="install it directly"):
        litellm.build(_candidate())


def test_repr_redacts_key_and_endpoint_override_cannot_leak_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_module(monkeypatch)
    adapter = litellm.build(_candidate())
    rendered = repr(adapter)
    assert "super-secret" not in rendered
    assert "redacted" in rendered
