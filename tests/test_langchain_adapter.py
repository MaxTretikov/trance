from types import SimpleNamespace

import pytest

from trance.adapters import langchain
from trance.types import Candidate


def _candidate(provider: str = "openai", **config: str) -> Candidate:
    return Candidate(provider, "api_key", "env:TEST_KEY", "test-model", "super-secret", config)


def test_openai_compatible_build_is_lazy_and_passes_resolved_values(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "trance.adapters._openai_compatible":
            return SimpleNamespace(resolve=lambda candidate: ("resolved-model", "secret", "https://api.example/v1"))
        if name == "langchain_openai":
            return SimpleNamespace(ChatOpenAI=lambda **kwargs: kwargs)
        raise AssertionError(name)

    monkeypatch.setattr(langchain, "import_module", fake_import)
    result = langchain.build(_candidate())

    assert result == {
        "model": "resolved-model",
        "api_key": "secret",
        "base_url": "https://api.example/v1",
    }
    assert calls == ["trance.adapters._openai_compatible", "langchain_openai"]


def test_anthropic_uses_native_integration(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        assert name == "langchain_anthropic"
        return SimpleNamespace(ChatAnthropic=lambda **kwargs: kwargs)

    monkeypatch.setattr(langchain, "import_module", fake_import)
    result = langchain.build(_candidate("anthropic"))

    assert result == {"model": "test-model", "api_key": "super-secret"}
    assert calls == ["langchain_anthropic"]


@pytest.mark.parametrize("provider", ["claude-code", "github-copilot", "openai-codex"])
def test_delegated_candidates_are_rejected_without_importing_dependencies(monkeypatch, provider):
    monkeypatch.setattr(langchain, "import_module", lambda _: pytest.fail("unexpected import"))
    candidate = Candidate(
        provider, "subscription", "saved", "test-model", config={"resolver": "cli"}
    )

    with pytest.raises(langchain.UnsupportedLangChainCandidateError, match="explicit API key"):
        langchain.build(candidate)


def test_missing_dependency_has_direct_install_advice(monkeypatch):
    def missing(name):
        if name == "trance.adapters._openai_compatible":
            return SimpleNamespace(resolve=lambda candidate: ("test-model", "secret", None))
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(langchain, "import_module", missing)
    with pytest.raises(langchain.LangChainDependencyError, match="uv add langchain-openai"):
        langchain.build(_candidate())


def test_unsupported_route_does_not_import_langchain(monkeypatch):
    def fake_import(name):
        assert name == "trance.adapters._openai_compatible"
        raise ValueError("unsupported")

    monkeypatch.setattr(langchain, "import_module", fake_import)
    candidate = _candidate("unknown-provider")

    with pytest.raises(
        langchain.UnsupportedLangChainCandidateError, match="approved LangChain route"
    ):
        langchain.build(candidate)
