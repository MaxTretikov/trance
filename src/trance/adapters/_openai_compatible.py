"""Resolve candidates for clients that speak the OpenAI API protocol.

This module intentionally has no SDK dependency.  It only translates a
provider-neutral :class:`~trance.types.Candidate` into the three values an
OpenAI-compatible client needs; the client itself is imported by a public
adapter when requested.
"""

from __future__ import annotations

from trance.types import Candidate


class OpenAICompatibleResolutionError(ValueError):
    """The candidate cannot be represented by an OpenAI-compatible client."""


_DELEGATED_PROVIDERS = frozenset(
    {
        "codex",
        "openai-codex",
        "github-copilot",
        "claude-code",
        "grok-consumer",
        "gemini-cli",
        "sourcegraph-cody",
    }
)


def _error(message: str) -> OpenAICompatibleResolutionError:
    # Keep this helper as a single choke point so future validation errors do
    # not accidentally include a secret or a credential-bearing config value.
    return OpenAICompatibleResolutionError(message)


def resolve(candidate: Candidate) -> tuple[str, str, str | None]:
    """Return ``(model_name, api_key, approved_base_url)`` for ``candidate``.

    Only ordinary API-key candidates are accepted.  Provider endpoints come
    from trance's audited registry; caller supplied endpoints are accepted
    only for providers whose source and registry explicitly validate them.
    """
    provider = candidate.provider.lower().replace("_", "-")
    if provider in _DELEGATED_PROVIDERS or provider.startswith("opencode:"):
        raise _error("Delegated CLI or account credentials are unsupported.")
    if candidate.config.get("resolver") or candidate.config.get("integration"):
        raise _error("Delegated credentials are unsupported.")
    if candidate.auth_kind != "api_key" or not candidate.secret:
        raise _error("An explicit API key is required.")
    if not isinstance(candidate.model_name, str) or not candidate.model_name.strip():
        raise _error("A model name is required.")

    # These imports are deliberately local: importing this adapter remains
    # cheap and does not import Pydantic AI or an optional provider SDK.
    from trance.models import (
        _API_PROVIDERS,
        _CONFIGURED_BASE_URLS,
        _DYNAMIC_BASE_URL_PROVIDERS,
        _OPENAI_COMPATIBLE_FALLBACKS,
        _validated_dynamic_base_url,
    )

    spec = _API_PROVIDERS.get(provider)
    if spec is None:
        raise _error("The provider is unsupported by the OpenAI-compatible adapter.")

    config = candidate.config
    if provider in _DYNAMIC_BASE_URL_PROVIDERS:
        try:
            base_url = _validated_dynamic_base_url(provider, candidate)
        except Exception as exc:
            raise _error("The provider has no approved API endpoint.") from exc
    elif provider in _CONFIGURED_BASE_URLS:
        if config.get("api_style") != "openai":
            raise _error("The provider requires an OpenAI-compatible API style.")
        configured = config.get("base_url", "").rstrip("/")
        if configured not in _CONFIGURED_BASE_URLS[provider]:
            raise _error("The provider has no approved API endpoint.")
        base_url = configured
    elif provider in _OPENAI_COMPATIBLE_FALLBACKS:
        base_url = _OPENAI_COMPATIBLE_FALLBACKS[provider]
        if "base_url" in config:
            raise _error("Arbitrary API endpoints are unsupported.")
    elif provider == "openai":
        base_url = None
        if "base_url" in config:
            raise _error("Arbitrary API endpoints are unsupported.")
    elif spec.model_module == "openai" and spec.provider_module == "openai" and spec.base_url:
        base_url = spec.base_url
        if "base_url" in config:
            raise _error("Arbitrary API endpoints are unsupported.")
    else:
        raise _error("The provider has no approved OpenAI-compatible endpoint.")

    return candidate.model_name, candidate.secret, base_url


__all__ = ["OpenAICompatibleResolutionError", "resolve"]
