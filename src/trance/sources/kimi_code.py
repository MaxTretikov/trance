"""Discover an authenticated Kimi Code CLI for programmatic delegation.

Kimi Code's managed subscription uses OAuth. The CLI owns and refreshes that
credential, so this adapter only checks the documented config reference and
credential file's presence; it never reads token contents.
"""

from __future__ import annotations

import shutil
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trance.types import Candidate

_OAUTH_KEY = "oauth/kimi-code"
_DEFAULT_SHARE_DIR = ".kimi"


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _find_managed_provider(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    providers = config.get("providers")
    if not isinstance(providers, Mapping):
        return None
    for name, provider in providers.items():
        if not isinstance(provider, Mapping):
            continue
        oauth = provider.get("oauth")
        if isinstance(oauth, Mapping) and oauth.get("key") == _OAUTH_KEY:
            return str(name), provider
    return None


def _selected_model(config: Mapping[str, Any], provider_name: str) -> str:
    models = config.get("models")
    if not isinstance(models, Mapping):
        return "kimi-for-coding"
    default_name = config.get("default_model")
    if isinstance(default_name, str) and default_name:
        model = models.get(default_name)
        if isinstance(model, Mapping) and model.get("provider") == provider_name:
            selected = model.get("model")
            if isinstance(selected, str) and selected:
                return selected
    for model in models.values():
        if isinstance(model, Mapping) and model.get("provider") == provider_name:
            selected = model.get("model")
            if isinstance(selected, str) and selected:
                return selected
    return "kimi-for-coding"


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a secretless descriptor when Kimi Code OAuth is configured.

    Kimi Code documents ``KIMI_SHARE_DIR`` and otherwise uses ``~/.kimi``.
    The JSON credential is tested for presence only. Requests must be delegated
    to the official ``kimi`` CLI, which performs its own token refresh.
    """
    configured_share_dir = environ.get("KIMI_SHARE_DIR")
    if configured_share_dir:
        # Path.expanduser() consults the process account database, which can
        # escape the caller's injected home during synthetic scans.
        if configured_share_dir == "~":
            share_dir = home
        elif configured_share_dir.startswith("~/"):
            share_dir = home / configured_share_dir[2:]
        else:
            share_dir = Path(configured_share_dir)
    else:
        share_dir = home / _DEFAULT_SHARE_DIR
    config = _read_config(share_dir / "config.toml")
    configured = _find_managed_provider(config)
    if configured is None:
        return []

    provider_name, _provider = configured
    credential_file = share_dir / "credentials" / "kimi-code.json"
    if not credential_file.is_file():
        return []

    cli = shutil.which("kimi", path=environ.get("PATH"))
    if cli is None:
        return []

    return [
        Candidate(
            provider="kimi-code",
            auth_kind="subscription",
            source="kimi CLI OAuth",
            model_name=_selected_model(config, provider_name),
            secret=None,
            config={
                "integration": "kimi-cli",
                "command": cli,
                "protocol": "print",
                "auth_method": "kimi_code_oauth",
            },
        )
    ]
