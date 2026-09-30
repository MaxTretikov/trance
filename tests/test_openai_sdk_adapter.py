from types import SimpleNamespace

import pytest

from trance.adapters import openai_sdk
from trance.types import Candidate


def _candidate(
    provider: str = "openai", secret: str | None = "super-secret", **config: str
) -> Candidate:
    return Candidate(provider, "api_key", "env:TEST_KEY", "gpt-test", secret, config)


def _fake_modules(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("sync", kwargs))

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("async", kwargs))

    def fake_import(name: str) -> object:
        if name == "trance.adapters._openai_compatible":
            return SimpleNamespace(
                resolve=lambda candidate: (candidate.model_name, candidate.secret, None)
            )
        if name == "openai":
            return SimpleNamespace(OpenAI=FakeClient, AsyncOpenAI=FakeAsyncClient)
        raise AssertionError(name)

    monkeypatch.setattr(openai_sdk, "import_module", fake_import)
    return calls


def test_builds_sync_client_lazily_with_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_modules(monkeypatch)

    client = openai_sdk.build(_candidate())

    assert client.__class__.__name__ == "FakeClient"
    assert calls == [("sync", {"api_key": "super-secret", "base_url": None})]


def test_builds_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_modules(monkeypatch)

    client = openai_sdk.build(_candidate(), asynchronous=True)

    assert client.__class__.__name__ == "FakeAsyncClient"
    assert calls == [("async", {"api_key": "super-secret", "base_url": None})]


def test_resolves_endpoint_before_importing_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_import(name: str) -> object:
        calls.append(name)
        if name == "trance.adapters._openai_compatible":
            return SimpleNamespace(
                resolve=lambda candidate: (candidate.model_name, candidate.secret, "https://api.example/v1")
            )
        raise AssertionError(name)

    monkeypatch.setattr(openai_sdk, "import_module", fake_import)
    with pytest.raises(AssertionError):
        openai_sdk.build(_candidate("delegated"))
    assert calls == ["trance.adapters._openai_compatible", "openai"]


def test_delegated_candidates_are_rejected_by_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        if name == "trance.adapters._openai_compatible":
            def reject(candidate: Candidate) -> tuple[str, str, str | None]:
                raise RuntimeError("delegated credential is unsupported")

            return SimpleNamespace(resolve=reject)
        raise AssertionError(name)

    monkeypatch.setattr(openai_sdk, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="delegated"):
        openai_sdk.build(_candidate("claude-code"))


def test_missing_dependency_has_install_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        if name == "trance.adapters._openai_compatible":
            return SimpleNamespace(
                resolve=lambda candidate: (candidate.model_name, candidate.secret, None)
            )
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(openai_sdk, "import_module", missing)
    with pytest.raises(openai_sdk.OpenAIDependencyError, match="uv add openai"):
        openai_sdk.build(_candidate())


def test_errors_do_not_include_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "super-secret"

    def fake_import(name: str) -> object:
        if name == "trance.adapters._openai_compatible":
            return SimpleNamespace(resolve=lambda candidate: (candidate.model_name, None, None))
        raise AssertionError(name)

    monkeypatch.setattr(openai_sdk, "import_module", fake_import)
    with pytest.raises(openai_sdk.MissingOpenAICredentialError) as caught:
        openai_sdk.build(_candidate(secret=secret))
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
