"""Discover explicit setup tokens and saved Claude Code subscription logins.

Saved credentials remain inside the official Claude CLI. This module asks the
CLI for its documented auth status and checks only for the documented OAuth
credential file; it never reads credential files.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_MODEL_ENV = "TRANCE_MODEL_CLAUDE_CODE"
_MAX_CREDENTIAL_BYTES = 256 * 1024


def _is_regular_path(path: Path) -> bool:
    """Reject symlinked path components and require a regular file."""
    resolved = path.absolute()
    current = Path(resolved.anchor)
    for component in resolved.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                return False
        except OSError:
            return False
    try:
        return stat.S_ISREG(resolved.stat().st_mode)
    except OSError:
        return False


def extract_credentials(candidate: Candidate) -> Mapping[str, str]:
    """Extract a known Claude OAuth access token from a saved login.

    Claude's credentials file contains unrelated account and refresh data, so
    only recognized OAuth objects and their access token fields are accepted.
    Unknown JSON shapes are treated as unresolved rather than copied into the
    returned mapping.
    """
    if candidate.provider != "claude-code":
        return {}
    if candidate.secret:
        return {"token": candidate.secret}
    config = candidate.config
    auth_home = config.get("auth_home")
    if not auth_home:
        return {}
    config_dir = config.get("config_dir")
    credentials = Path(config_dir) if config_dir else Path(auth_home) / ".claude"
    credentials = credentials / ".credentials.json"
    if not _is_regular_path(credentials):
        return {}
    try:
        with credentials.open("rb") as stream:
            raw = stream.read(_MAX_CREDENTIAL_BYTES + 1)
        if len(raw) > _MAX_CREDENTIAL_BYTES:
            return {}
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    for section_name in ("claudeAiOauth", "oauthAccount"):
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            continue
        token = section.get("accessToken")
        if isinstance(token, str) and token.strip():
            return {"token": token.strip()}
    return {}


def _status_explicitly_rejects_login(stdout: str) -> bool:
    """Reject status payloads that explicitly identify an API-key login.

    The status schema is not required for discovery, but current Claude CLIs
    may expose these fields. Unknown or non-JSON output remains eligible when
    the documented credentials marker and successful status command agree.
    """
    try:
        status = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(status, dict):
        return False
    if status.get("loggedIn") is False:
        return True
    api_key_values = {"api_key", "api-key", "apikey", "api key", "console"}
    for key in ("authMethod", "apiProvider"):
        value = status.get(key)
        if isinstance(value, str) and value.strip().casefold() in api_key_values:
            return True
    return False


def _saved_login(environ: Mapping[str, str], home: Path) -> Candidate | None:
    auth_home = home.expanduser().resolve()
    configured_dir = environ.get("CLAUDE_CONFIG_DIR")
    config_path = Path(configured_dir).expanduser().resolve() if configured_dir else None
    credentials_dir = config_path if config_path is not None else auth_home / ".claude"
    credentials = credentials_dir / ".credentials.json"
    if not credentials.is_file():
        return None

    cli = shutil.which("claude", path=environ.get("PATH", ""))
    if cli is None:
        return None

    child_env = {"PATH": environ.get("PATH", ""), "HOME": str(auth_home)}
    if config_path is not None:
        child_env["CLAUDE_CONFIG_DIR"] = str(config_path)
    try:
        result = subprocess.run(
            [cli, "auth", "status"],
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    if _status_explicitly_rejects_login(result.stdout):
        return None
    if not credentials.is_file():
        return None

    config = {
        "command": cli,
        "auth_home": str(auth_home),
        "saved_login": "true",
    }
    if config_path is not None:
        config["config_dir"] = str(config_path)

    return Candidate(
        provider="claude-code",
        auth_kind="subscription",
        source="claude-cli-login",
        model_name=environ.get(_MODEL_ENV, "").strip() or "sonnet",
        secret=None,
        config=config,
    )


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return candidates for explicit setup tokens and a saved CLI login.

    The status subprocess receives only the supplied ``PATH``, ``HOME``, and
    optional ``CLAUDE_CONFIG_DIR``. API keys, bearer tokens, and other process
    environment values are not forwarded to the CLI.
    """
    token = environ.get(_TOKEN_ENV, "").strip()
    found: list[Candidate] = []
    if token:
        found.append(Candidate(
            provider="claude-code",
            auth_kind="subscription",
            source=f"env:{_TOKEN_ENV}",
            model_name=environ.get(_MODEL_ENV, "").strip() or "sonnet",
            secret=token,
            config={"credential_env": _TOKEN_ENV},
        ))
    saved = _saved_login(environ, home)
    if saved is not None:
        found.append(saved)
    return found
