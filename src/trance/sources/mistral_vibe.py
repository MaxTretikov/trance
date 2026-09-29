"""Discover explicitly configured Mistral API keys used by Vibe.

Vibe's documented headless credential interface is ``MISTRAL_API_KEY`` or
``~/.vibe/.env``. Browser sign-in may provision credentials internally; this
source does not inspect Vibe's keyring or private token stores.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_ENV_KEY = "MISTRAL_API_KEY"
_DEFAULT_MODEL = "mistral-small-latest"


def _dotenv_value(path: Path, key: str) -> str | None:
    """Read one simple dotenv assignment without loading unrelated variables."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, separator, value = stripped.partition("=")
        if not separator or name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        return value or None
    return None


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a candidate for a Mistral key configured for Vibe.

    Environment values take precedence over the documented ``~/.vibe/.env``
    value, matching Vibe's configuration behavior. No network checks occur.
    """
    secret = environ.get(_ENV_KEY, "").strip()
    source = f"env:{_ENV_KEY}"
    if not secret:
        secret = _dotenv_value(home / ".vibe" / ".env", _ENV_KEY) or ""
        source = "~/.vibe/.env:MISTRAL_API_KEY"
    if not secret:
        return []

    model = environ.get("TRANCE_MODEL_MISTRAL", "").strip() or _DEFAULT_MODEL
    return [
        Candidate(
            provider="mistral",
            auth_kind="api_key",
            source=source,
            model_name=model,
            secret=secret,
        )
    ]
