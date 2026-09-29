"""Discover an existing Hugging Face Hub login through its official resolver.

``huggingface_hub.get_token`` resolves the Hub's configured token sources,
including its saved login. It consults process-global environment and cache
configuration, so this adapter only invokes it for the real process context.
Synthetic ``environ``/``home`` inputs are never allowed to fall back to the
current user's credentials.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def _is_process_context(environ: Mapping[str, str], home: Path) -> bool:
    """Whether the given scan context is exactly the running user's context."""
    return environ is os.environ and home == Path.home()


def _resolve_token() -> str | None:
    """Ask huggingface_hub to resolve its configured login token lazily."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return the saved Hub login as an account-authenticated model candidate.

    The optional ``huggingface_hub`` dependency is imported only when scanning
    the actual process context. No network request is made, and token values
    are retained only on the returned redacted :class:`Candidate`.
    """
    if not _is_process_context(environ, Path(home)):
        return []

    token = _resolve_token()
    if not token or not token.strip():
        return []

    model = environ.get("TRANCE_MODEL_HUGGINGFACE", "").strip() or _DEFAULT_MODEL
    return [
        Candidate(
            provider="huggingface",
            auth_kind="account",
            source="huggingface_hub",
            model_name=model,
            secret=token.strip(),
        )
    ]
