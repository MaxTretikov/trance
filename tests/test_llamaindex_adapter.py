from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from trance.adapters import llamaindex
from trance.types import Candidate


def _candidate(provider: str = "openai", **config: str) -> Candidate:
    return Candidate(provider, "api_key", "env:TEST_KEY", "test-model", "super-secret", config)


def _patch_resolver(monkeypatch: pytest.MonkeyPatch, resolver: object) -> None:
    module = ModuleType("trance.adapters._openai_compatible")
    module.resolve = resolver  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)


def test_builds_official_openai_llm_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class OpenAI:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("openai", kwargs))

    monkeypatch.setattr(
        llamaindex,
        "import_module",
        lambda name: SimpleNamespace(OpenAI=OpenAI) if name == "llama_index.llms.openai" else None,
    )
    _patch_resolver(monkeypatch, lambda candidate: (candidate.model_name, candidate.secret, None))

    result = llamaindex.build(_candidate())

    assert isinstance(result, OpenAI)
    assert calls == [("openai", {"model": "test-model", "api_key": "super-secret"})]


def test_builds_openai_compatible_llm_with_chat_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class OpenAILike:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        llamaindex,
        "import_module",
        lambda name: SimpleNamespace(OpenAILike=OpenAILike)
        if name == "llama_index.llms.openai_like"
        else None,
    )
    _patch_resolver(
        monkeypatch,
        lambda candidate: (candidate.model_name, candidate.secret, "https://api.example/v1"),
    )

    result = llamaindex.build(_candidate("compatible"))

    assert isinstance(result, OpenAILike)
    assert calls == [
        {
            "model": "test-model",
            "api_key": "super-secret",
            "api_base": "https://api.example/v1",
            "is_chat_model": True,
        }
    ]


def test_missing_integration_has_direct_install_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llamaindex,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    _patch_resolver(monkeypatch, lambda candidate: (candidate.model_name, candidate.secret, None))

    with pytest.raises(
        llamaindex.LlamaIndexDependencyError, match="uv add llama-index-llms-openai"
    ):
        llamaindex.build(_candidate())


def test_resolver_failure_is_safe_and_does_not_leak_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_resolver(
        monkeypatch,
        lambda candidate: (_ for _ in ()).throw(ValueError("secret should not appear")),
    )

    with pytest.raises(llamaindex.UnsupportedLlamaIndexCandidateError) as error:
        llamaindex.build(_candidate("delegated"))

    assert "super-secret" not in str(error.value)
    assert "secret should not appear" not in str(error.value)


def test_missing_key_is_rejected_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llamaindex,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError("must remain lazy")),
    )
    candidate = Candidate("openai", "api_key", "env:TEST_KEY", "test-model")

    with pytest.raises(llamaindex.MissingLlamaIndexCredentialError, match="explicit API key"):
        llamaindex.build(candidate)
