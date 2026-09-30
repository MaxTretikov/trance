"""Optional adapter for the official OpenAI Python SDK.

The SDK is intentionally imported only when :func:`build` is called.  A
provider-neutral scan can therefore be used without installing ``openai``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trance.types import Candidate


class OpenAISDKAdapterError(RuntimeError):
    """Base class for errors raised by the OpenAI SDK adapter."""


class OpenAIDependencyError(OpenAISDKAdapterError):
    """The official OpenAI SDK is unavailable or incomplete."""


class MissingOpenAICredentialError(OpenAISDKAdapterError):
    """The candidate has no credential usable by the OpenAI SDK."""


def build(candidate: Candidate, *, asynchronous: bool = False) -> Any:
    """Build an official OpenAI client for ``candidate`` on explicit request.

    Endpoint and credential resolution is delegated to the allowlisted
    OpenAI-compatible adapter.  The SDK is loaded only after that validation,
    and receives the resolved values explicitly instead of reading ambient
    environment variables.
    """
    resolver = import_module("trance.adapters._openai_compatible")
    _model_name, api_key, base_url = resolver.resolve(candidate)
    if not api_key:
        raise MissingOpenAICredentialError(
            f"Provider {candidate.provider!r} has no API key for the OpenAI SDK."
        )

    try:
        openai = import_module("openai")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OpenAIDependencyError(
            "OpenAI SDK support is unavailable; install it directly with `uv add openai`."
        ) from exc

    client_name = "AsyncOpenAI" if asynchronous else "OpenAI"
    client_type = getattr(openai, client_name, None)
    if not callable(client_type):
        raise OpenAIDependencyError(
            "The installed OpenAI package does not expose the required client; "
            "install or upgrade it with `uv add openai`."
        )
    return client_type(api_key=api_key, base_url=base_url)


__all__ = [
    "MissingOpenAICredentialError",
    "OpenAIDependencyError",
    "OpenAISDKAdapterError",
    "build",
]
