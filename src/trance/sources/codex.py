"""Discovery of an existing Codex CLI ChatGPT subscription login."""

import json
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_AUTH_FILE = "auth.json"
_MODEL_NAME = "gpt-5.6-luna"
_MODEL_ENV = "TRANCE_MODEL_OPENAI_CODEX"
_MAX_AUTH_BYTES = 128 * 1024


def _has_oauth_tokens(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(value.get(name), str) and bool(value[name].strip())
        for name in ("access_token", "refresh_token", "id_token")
    )


def _is_chatgpt_auth(auth_file: Path) -> bool:
    """Check auth metadata without retaining or exposing credential values."""
    try:
        with auth_file.open("rb") as stream:
            raw = stream.read(_MAX_AUTH_BYTES + 1)
        if len(raw) > _MAX_AUTH_BYTES:
            return False
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    if not isinstance(payload, Mapping) or "OPENAI_API_KEY" in payload:
        return False
    auth_mode = payload.get("auth_mode")
    if auth_mode == "apikey":
        return False
    # Older Codex files omitted auth_mode; their tokens-only shape is safe to
    # recognize because it cannot be an API-key login.
    if auth_mode not in (None, "chatgpt"):
        return False
    return _has_oauth_tokens(payload.get("tokens"))


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a descriptor for the Codex CLI auth cache, when present.

    Pydantic AI's ``OpenAICodexProvider`` reads this file itself (including
    token refresh behavior). Keeping the candidate secret-free avoids copying
    the refresh token into another credential object or loggable structure.
    """
    configured_home = environ.get("CODEX_HOME", "").strip()
    codex_home = Path(configured_home) if configured_home else home / ".codex"
    auth_file = codex_home / _AUTH_FILE
    if not auth_file.is_file() or not _is_chatgpt_auth(auth_file):
        return []

    return [
        Candidate(
            provider="openai-codex",
            auth_kind="subscription",
            source=str(auth_file),
            model_name=environ.get(_MODEL_ENV, "").strip() or _MODEL_NAME,
            config={"codex_home": str(codex_home)},
        )
    ]
