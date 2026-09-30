from __future__ import annotations

import pytest

from trance.adapters._openai_compatible import (
    OpenAICompatibleResolutionError,
    resolve,
)
from trance.types import Candidate


def candidate(provider: str = "openai", **kwargs: str) -> Candidate:
    return Candidate(provider, "api_key", "env:TEST_KEY", "test-model", "secret", kwargs)


def test_resolves_openai_without_importing_an_sdk() -> None:
    assert resolve(candidate()) == ("test-model", "secret", None)


def test_resolves_fixed_compatible_endpoint() -> None:
    model, key, base_url = resolve(candidate("perplexity"))
    assert (model, key) == ("test-model", "secret")
    assert base_url == "https://api.perplexity.ai"


def test_resolves_validated_dynamic_endpoint() -> None:
    item = candidate(
        "azure-openai",
        api_style="openai",
        base_url="https://resource.openai.azure.com/openai/v1",
    )
    assert resolve(item)[2] == item.config["base_url"]


def test_rejects_arbitrary_endpoint_override() -> None:
    with pytest.raises(OpenAICompatibleResolutionError, match="Arbitrary"):
        resolve(candidate(base_url="https://attacker.example/v1"))


@pytest.mark.parametrize(
    "provider",
    [
        "openai-codex",
        "github-copilot",
        "grok-consumer",
        "gemini-cli",
        "sourcegraph-cody",
        "opencode:local",
    ],
)
def test_rejects_delegated_credentials_even_when_secret_is_present(provider: str) -> None:
    with pytest.raises(OpenAICompatibleResolutionError, match="credentials"):
        resolve(candidate(provider))


def test_rejects_native_only_provider() -> None:
    with pytest.raises(OpenAICompatibleResolutionError, match="endpoint"):
        resolve(candidate("anthropic"))


def test_rejects_non_api_key_auth() -> None:
    item = Candidate("openai", "subscription", "manual", "model", "token")
    with pytest.raises(OpenAICompatibleResolutionError, match="API key"):
        resolve(item)


def test_errors_never_include_secret() -> None:
    secret = "do-not-leak-this-secret"
    item = Candidate("unknown", "api_key", "manual", "model", secret)
    with pytest.raises(OpenAICompatibleResolutionError) as exc:
        resolve(item)
    assert secret not in str(exc.value)
