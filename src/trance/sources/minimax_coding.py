"""Discover explicitly configured MiniMax Token Plan subscription keys.

This adapter reads the documented ``MINIMAX_API_KEY`` environment variable
only. It recognizes MiniMax Token Plan keys (``sk-cp-``) and never inspects
browser sessions, CLI login state, or unrelated credential stores.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_KEY_ENV = "MINIMAX_API_KEY"
_HOST_ENV = "MINIMAX_API_HOST"
_DEFAULT_HOST = "https://api.minimax.io"
_ALLOWED_HOSTS = {
    "https://api.minimax.io",
    "https://api.minimaxi.com",
}


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a candidate for a configured MiniMax Token Plan key.

    ``home`` is accepted for the shared source protocol. No home-directory
    files are documented as a reusable credential source for this API.
    ``MINIMAX_API_HOST`` may select either documented regional host.
    """
    del home
    key = environ.get(_KEY_ENV)
    if not isinstance(key, str) or not key.startswith("sk-cp-"):
        return []

    host = (environ.get(_HOST_ENV) or _DEFAULT_HOST).rstrip("/")
    if host not in _ALLOWED_HOSTS:
        return []

    return [
        Candidate(
            provider="minimax-coding-plan",
            auth_kind="subscription",
            source=f"env:{_KEY_ENV}",
            model_name=environ.get("MINIMAX_MODEL") or "MiniMax-M2.7",
            secret=key,
            config={"base_url": f"{host}/v1", "api_style": "openai"},
        )
    ]
