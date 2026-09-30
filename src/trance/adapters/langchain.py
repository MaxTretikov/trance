"""Build LangChain chat models from provider-neutral candidates.

LangChain integrations are optional and are imported only by :func:`build`.
Install the integration you need directly, for example ``uv add
langchain-openai`` or ``uv add langchain-anthropic``.  OpenAI-compatible
endpoints use ``ChatOpenAI``; LangChain documents that non-standard provider
features may be lost when routed through that compatibility interface.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from trance.types import Candidate


class LangChainAdapterError(RuntimeError):
    """Base error raised by the optional LangChain adapter."""


class LangChainDependencyError(LangChainAdapterError):
    """The requested LangChain integration is not installed."""


class UnsupportedLangChainCandidateError(LangChainAdapterError):
    """A candidate cannot be represented safely by a LangChain integration."""


class MissingLangChainCredentialError(LangChainAdapterError):
    """A candidate has no explicit credential suitable for LangChain."""


def _reject_delegated(candidate: Candidate) -> None:
    """Reject credentials that require a CLI, account, or other resolver."""
    provider = candidate.provider.lower().replace("_", "-")
    if (
        candidate.auth_kind != "api_key"
        or not candidate.secret
        or candidate.config.get("resolver")
        or candidate.config.get("integration")
        or provider in {"openai-codex", "codex", "github-copilot"}
        or provider.endswith("-cli")
        or provider.startswith("opencode:")
    ):
        raise UnsupportedLangChainCandidateError(
            f"Provider {candidate.provider!r} requires an explicit API key for LangChain."
        )


def _dependency_error(package: str, cause: BaseException) -> LangChainDependencyError:
    return LangChainDependencyError(
        f"LangChain support is unavailable; install {package!r} directly "
        f"with `uv add {package}`."
    )


def _build_anthropic(candidate: Candidate) -> Any:
    try:
        module = import_module("langchain_anthropic")
    except (ImportError, ModuleNotFoundError) as exc:
        raise _dependency_error("langchain-anthropic", exc) from exc
    chat_anthropic = getattr(module, "ChatAnthropic", None)
    if not callable(chat_anthropic):
        raise LangChainDependencyError(
            "The installed langchain-anthropic package does not expose ChatAnthropic."
        )
    return chat_anthropic(model=candidate.model_name, api_key=candidate.secret)


def _build_openai_compatible(candidate: Candidate) -> Any:
    try:
        resolver = import_module("trance.adapters._openai_compatible")
        model_name, api_key, base_url = resolver.resolve(candidate)
    except UnsupportedLangChainCandidateError:
        raise
    except Exception as exc:
        raise UnsupportedLangChainCandidateError(
            f"Provider {candidate.provider!r} has no approved LangChain route."
        ) from exc

    try:
        module = import_module("langchain_openai")
    except (ImportError, ModuleNotFoundError) as exc:
        raise _dependency_error("langchain-openai", exc) from exc
    chat_openai = getattr(module, "ChatOpenAI", None)
    if not callable(chat_openai):
        raise LangChainDependencyError(
            "The installed langchain-openai package does not expose ChatOpenAI."
        )
    return chat_openai(model=model_name, api_key=api_key, base_url=base_url)


def build(candidate: Candidate) -> Any:
    """Build a LangChain chat model, importing its integration on demand."""
    _reject_delegated(candidate)
    if candidate.provider.lower().replace("_", "-") == "anthropic":
        return _build_anthropic(candidate)
    return _build_openai_compatible(candidate)


__all__ = [
    "LangChainAdapterError",
    "LangChainDependencyError",
    "MissingLangChainCredentialError",
    "UnsupportedLangChainCandidateError",
    "build",
]
