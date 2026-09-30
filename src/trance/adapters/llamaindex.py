"""Optional LlamaIndex adapter for provider-neutral candidates.

LlamaIndex is imported only by :func:`build`.  Discovery and importing
``trance`` therefore do not require a LlamaIndex installation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from trance.types import Candidate


class LlamaIndexAdapterError(RuntimeError):
    """Base class for LlamaIndex adapter errors."""


class LlamaIndexDependencyError(LlamaIndexAdapterError):
    """LlamaIndex is not installed or does not expose the required API."""


class UnsupportedLlamaIndexCandidateError(LlamaIndexAdapterError):
    """A candidate cannot be represented safely by LlamaIndex."""


class MissingLlamaIndexCredentialError(LlamaIndexAdapterError):
    """A LlamaIndex candidate has no usable API key."""


def _dependency_error(package: str, cause: BaseException) -> LlamaIndexDependencyError:
    return LlamaIndexDependencyError(
        "LlamaIndex support is unavailable; install the required integration directly "
        f"with `uv add {package}` (or `pip install {package}`)."
    )


def build(candidate: Candidate) -> Any:
    """Build a LlamaIndex LLM from ``candidate`` on explicit request.

    The shared OpenAI-compatible resolver validates the endpoint and rejects
    delegated credentials.  This adapter never places credential values in an
    exception message.
    """
    if not candidate.secret:
        raise MissingLlamaIndexCredentialError(
            f"Provider {candidate.provider!r} requires an explicit API key for LlamaIndex."
        )

    try:
        from trance.adapters._openai_compatible import resolve

        model_name, api_key, base_url = resolve(candidate)
    except UnsupportedLlamaIndexCandidateError:
        raise
    except Exception as exc:
        raise UnsupportedLlamaIndexCandidateError(
            f"Provider {candidate.provider!r} is not supported by the LlamaIndex adapter."
        ) from exc

    if not api_key:
        raise MissingLlamaIndexCredentialError(
            f"Provider {candidate.provider!r} requires an explicit API key for LlamaIndex."
        )

    if base_url is None:
        module_name = "llama_index.llms.openai"
        package_name = "llama-index-llms-openai"
        class_name = "OpenAI"
        kwargs = {"model": model_name, "api_key": api_key}
    else:
        module_name = "llama_index.llms.openai_like"
        package_name = "llama-index-llms-openai-like"
        class_name = "OpenAILike"
        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "api_base": base_url,
            "is_chat_model": True,
        }

    try:
        module = import_module(module_name)
        llm_class = getattr(module, class_name)
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise _dependency_error(package_name, exc) from exc

    return llm_class(**kwargs)


__all__ = [
    "LlamaIndexAdapterError",
    "LlamaIndexDependencyError",
    "MissingLlamaIndexCredentialError",
    "UnsupportedLlamaIndexCandidateError",
    "build",
]
