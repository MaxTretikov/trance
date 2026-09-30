"""Tests for optional, lazily imported candidate conversion adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trance import Candidate


@pytest.mark.parametrize(
    ("method", "module_name", "kwargs"),
    [
        ("to_openai", "trance.adapters.openai_sdk", {"asynchronous": True}),
        ("to_langchain", "trance.adapters.langchain", {}),
        ("to_llama_index", "trance.adapters.llamaindex", {}),
    ],
)
def test_candidate_conversion_delegates_to_lazy_adapter(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    module_name: str,
    kwargs: dict[str, object],
) -> None:
    candidate = Candidate("openai", "api_key", "test", "gpt-test", "secret")
    calls: list[tuple[Candidate, dict[str, object]]] = []
    result = object()

    def build(received: Candidate, **received_kwargs: object) -> object:
        calls.append((received, received_kwargs))
        return result

    def import_module(name: str) -> SimpleNamespace:
        assert name == module_name
        return SimpleNamespace(build=build)

    monkeypatch.setattr("trance.types.import_module", import_module)

    assert getattr(candidate, method)(**kwargs) is result
    assert calls == [(candidate, kwargs)]
