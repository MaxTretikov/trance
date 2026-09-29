"""Discover explicitly configured Alibaba Cloud Coding Plan credentials.

Qwen Code's former Qwen OAuth sign-in has been discontinued.  This source
supports the current subscription-backed Coding Plan API key only; it never
reads browser profiles or legacy OAuth credential caches.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trance.types import Candidate

_ENV_KEY = "BAILIAN_CODING_PLAN_API_KEY"
_DEFAULT_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
_KNOWN_BASE_URLS = {
    "https://coding.dashscope.aliyuncs.com/v1",
    "https://coding-intl.dashscope.aliyuncs.com/v1",
}


def _read_settings(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _configured_entries(settings: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    providers = settings.get("modelProviders")
    if not isinstance(providers, Mapping):
        return []
    entries = providers.get("openai", ())
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, Mapping)]


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a candidate for a configured Qwen Code Coding Plan key.

    Reads only ``~/.qwen/settings.json`` and the caller-supplied environment.
    Coding Plan keys have the documented ``sk-sp-`` prefix. A key in
    settings.json is considered only when declared through ``env`` and the
    selected provider entry points at an official Coding Plan endpoint.
    """
    settings = _read_settings(home / ".qwen" / "settings.json")
    configured_env = settings.get("env", {})
    if not isinstance(configured_env, Mapping):
        configured_env = {}

    entries = _configured_entries(settings)
    matching = [
        entry
        for entry in entries
        if str(entry.get("baseUrl", "")).rstrip("/") in _KNOWN_BASE_URLS
        and (entry.get("envKey") == _ENV_KEY or entry.get("envKey") in (None, ""))
    ]

    secret = environ.get(_ENV_KEY)
    source = "env:BAILIAN_CODING_PLAN_API_KEY"
    if not secret:
        configured_secret = configured_env.get(_ENV_KEY)
        if isinstance(configured_secret, str) and configured_secret:
            secret = configured_secret
            source = "~/.qwen/settings.json:env.BAILIAN_CODING_PLAN_API_KEY"

    # An environment credential alone follows Qwen's documented headless
    # setup, whose default Coding Plan endpoint is the Beijing endpoint.
    if not secret or not isinstance(secret, str) or not secret.startswith("sk-sp-"):
        return []

    selected = matching[0] if matching else {}
    base_url = str(selected.get("baseUrl") or _DEFAULT_BASE_URL).rstrip("/")
    model_settings = settings.get("model", {})
    configured_model = model_settings.get("name") if isinstance(model_settings, Mapping) else None
    model = (
        selected.get("id")
        or configured_model
        or environ.get("OPENAI_MODEL")
        or "qwen3-coder-plus"
    )
    return [
        Candidate(
            provider="qwen-code",
            auth_kind="subscription",
            source=source,
            model_name=str(model) if model else None,
            secret=secret,
            config={"base_url": base_url, "api_style": "openai"},
        )
    ]
