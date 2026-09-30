"""Optional LiteLLM adapter for provider-neutral candidates.

LiteLLM is deliberately imported only by :func:`build`.  Importing trance, or
scanning credentials, therefore does not require LiteLLM to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from trance.types import Candidate


class LiteLLMAdapterError(RuntimeError):
    """Base class for LiteLLM adapter errors."""


class LiteLLMDependencyError(LiteLLMAdapterError):
    """LiteLLM is not installed or does not expose the required API."""


class UnsupportedLiteLLMCandidateError(LiteLLMAdapterError):
    """A candidate cannot be represented safely by LiteLLM."""


class MissingLiteLLMCredentialError(LiteLLMAdapterError):
    """A LiteLLM candidate has no usable API key."""


# LiteLLM's documented model prefixes for ordinary API-key providers.  The
# values are intentionally explicit: an arbitrary candidate provider must not
# become an arbitrary LiteLLM provider by string concatenation.
_MODEL_PREFIXES = {
    "anthropic": "anthropic",
    "cerebras": "cerebras",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "fireworks": "fireworks_ai",
    "gemini": "gemini",
    "google": "gemini",
    "groq": "groq",
    "huggingface": "huggingface",
    "mistral": "mistral",
    "openrouter": "openrouter",
    "perplexity": "perplexity",
    "sambanova": "sambanova",
    "together": "together_ai",
    "xai": "xai",
}

_CLI_PROVIDERS = frozenset(
    {
        "claude-code",
        "codex",
        "gemini-cli",
        "github-copilot",
        "grok-consumer",
        "openai-codex",
        "sourcegraph-cody",
    }
)


def _model_and_base_url(candidate: Candidate) -> tuple[str, str | None]:
    """Resolve a safe LiteLLM model name and endpoint for ``candidate``."""
    provider = candidate.provider.lower().replace("_", "-")
    config = candidate.config

    if provider in _CLI_PROVIDERS or provider.startswith("opencode:"):
        raise UnsupportedLiteLLMCandidateError(
            f"Provider {candidate.provider!r} uses delegated CLI or account credentials; "
            "LiteLLM requires an explicit API key."
        )
    if config.get("resolver") or config.get("integration"):
        raise UnsupportedLiteLLMCandidateError(
            f"Provider {candidate.provider!r} uses delegated credentials; "
            "LiteLLM requires an explicit API key."
        )
    if candidate.auth_kind != "api_key":
        raise UnsupportedLiteLLMCandidateError(
            f"Provider {candidate.provider!r} has auth kind {candidate.auth_kind!r}; "
            "only ordinary API-key candidates are supported by LiteLLM."
        )

    # Reuse trance's allowlist and endpoint validation.  Importing this module
    # is cheap and does not import Pydantic AI or any provider SDK.
    from trance.models import (
        _API_PROVIDERS,
        _CONFIGURED_BASE_URLS,
        _DYNAMIC_BASE_URL_PROVIDERS,
        _validated_dynamic_base_url,
    )

    spec = _API_PROVIDERS.get(provider)
    if spec is None:
        raise UnsupportedLiteLLMCandidateError(
            f"Provider {candidate.provider!r} is not supported by the LiteLLM adapter."
        )

    base_url = spec.base_url
    if provider in _DYNAMIC_BASE_URL_PROVIDERS:
        try:
            base_url = _validated_dynamic_base_url(provider, candidate)
        except Exception as exc:
            raise UnsupportedLiteLLMCandidateError(
                f"Provider {candidate.provider!r} has no approved API endpoint."
            ) from exc
    elif provider in _CONFIGURED_BASE_URLS:
        if config.get("api_style") != "openai":
            raise UnsupportedLiteLLMCandidateError(
                f"Provider {candidate.provider!r} requires an OpenAI-compatible API style."
            )
        configured = config.get("base_url", "").rstrip("/")
        if configured not in _CONFIGURED_BASE_URLS[provider]:
            raise UnsupportedLiteLLMCandidateError(
                f"Provider {candidate.provider!r} has no approved API endpoint."
            )
        base_url = configured
    elif "base_url" in config:
        # A base URL is meaningful only for the provider-specific allowlists
        # above.  Do not let a user supplied URL silently change routing.
        raise UnsupportedLiteLLMCandidateError(
            f"Provider {candidate.provider!r} has an unsupported API endpoint."
        )

    if base_url is not None:
        # LiteLLM's OpenAI-compatible interface uses the openai provider when a
        # fixed endpoint is supplied, while preserving the selected model name.
        return f"openai/{candidate.model_name}", base_url
    if provider == "openai":
        return candidate.model_name, None
    prefix = _MODEL_PREFIXES.get(provider)
    if prefix is None:
        raise UnsupportedLiteLLMCandidateError(
            f"Provider {candidate.provider!r} has no confirmed LiteLLM model mapping."
        )
    return f"{prefix}/{candidate.model_name}", None


@dataclass(frozen=True, slots=True, repr=False)
class LiteLLMAdapter:
    """Secret-safe wrapper around the optional module-level LiteLLM client."""

    _litellm: Any
    model: str
    api_key: str
    api_base: str | None = None

    def __repr__(self) -> str:
        return (
            "LiteLLMAdapter("
            f"model={self.model!r}, api_key='<redacted>', api_base={self.api_base!r})"
        )

    def completion(self, messages: Any, **kwargs: Any) -> Any:
        call = dict(kwargs)
        call.update(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return self._litellm.completion(**call)

    async def acompletion(self, messages: Any, **kwargs: Any) -> Any:
        call = dict(kwargs)
        call.update(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return await self._litellm.acompletion(**call)


def build(candidate: Candidate) -> LiteLLMAdapter:
    """Build a LiteLLM adapter, importing LiteLLM only when explicitly asked."""
    if not candidate.secret:
        raise MissingLiteLLMCredentialError(
            f"Provider {candidate.provider!r} requires an explicit API key for LiteLLM."
        )
    model, api_base = _model_and_base_url(candidate)
    try:
        litellm = import_module("litellm")
    except (ImportError, ModuleNotFoundError) as exc:
        raise LiteLLMDependencyError(
            "LiteLLM support is unavailable; install it directly with `uv add litellm` "
            "(or `pip install litellm`)."
        ) from exc
    if not callable(getattr(litellm, "completion", None)) or not callable(
        getattr(litellm, "acompletion", None)
    ):
        raise LiteLLMDependencyError(
            "The installed LiteLLM package does not expose completion and acompletion."
        )
    return LiteLLMAdapter(litellm, model, candidate.secret, api_base)
