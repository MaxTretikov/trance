"""Discover an authenticated official Grok Build consumer CLI session.

The official ``grok`` CLI supports browser OIDC and device-code login and owns
the resulting refreshable session.  This source deliberately does not parse
the CLI's private credential store.  It only confirms that the documented Grok
home exists and leaves authentication, refresh, subscription checks, and
request transport to ``grok`` itself.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_DEFAULT_HOME = ".grok"
_DEFAULT_MODEL = "grok-build"
_MAX_AUTH_BYTES = 1 * 1024 * 1024


def _grok_home(environ: Mapping[str, str], home: Path) -> Path:
    """Resolve GROK_HOME against the supplied home, never the real home."""
    supplied_home = home.resolve()
    configured = environ.get("GROK_HOME", "").strip()
    if not configured:
        return supplied_home / _DEFAULT_HOME
    if configured == "~":
        return supplied_home
    if configured.startswith("~/"):
        return supplied_home / configured[2:]
    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = supplied_home / configured_path
    return configured_path.absolute()


def _auth_file(environ: Mapping[str, str], grok_home: Path) -> Path:
    """Resolve the official auth-file override without following symlinks."""
    configured = environ.get("GROK_AUTH_PATH", "").strip()
    if not configured:
        return (grok_home / "auth.json").absolute()
    path = Path(configured)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.absolute()


def _has_symlink_component(path: Path) -> bool:
    """Return whether any component of an absolute path is a symlink."""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


def _has_oidc_record(path: Path) -> bool:
    """Inspect only bounded auth metadata, without retaining credential data."""
    try:
        with path.open("rb") as stream:
            contents = stream.read(_MAX_AUTH_BYTES + 1)
        if len(contents) > _MAX_AUTH_BYTES:
            return False
        payload = json.loads(contents.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    return any(
        isinstance(record, Mapping) and record.get("auth_mode") == "oidc"
        for record in payload.values()
    )


def extract_credentials(candidate: Candidate) -> Mapping[str, str]:
    """Extract documented explicit OIDC fields, if the auth record has any."""
    if candidate.provider != "grok-consumer":
        return {}
    configured = candidate.config.get("auth_file", "")
    if not configured:
        return {}
    path = Path(configured).absolute()
    if _has_symlink_component(path):
        return {}
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(_MAX_AUTH_BYTES + 1)
            file_stat = os.fstat(stream.fileno())
            if len(raw) > _MAX_AUTH_BYTES or not stat.S_ISREG(file_stat.st_mode):
                return {}
            payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, str] = {}
    for record in payload.values():
        if not isinstance(record, Mapping) or record.get("auth_mode") != "oidc":
            continue
        for name in ("access_token", "refresh_token", "id_token"):
            value = record.get(name)
            if isinstance(value, str) and value.strip():
                result[name] = value
        break
    return result


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a secretless candidate for an installed, logged-in Grok CLI.

    xAI documents ``~/.grok`` (or ``GROK_HOME``) as Grok's user state/config
    home with an ``auth.json`` login marker. The marker is checked for a
    non-symlink regular file only; its contents are never opened or parsed. The model adapter
    must invoke the official CLI with this home so the CLI can validate and
    refresh the session without exposing its token to trance.
    """
    path = environ.get("PATH", "")
    if not path:
        return []
    command = shutil.which("grok", path=path)
    if command is None:
        return []
    command_path = Path(command).resolve()

    grok_home = _grok_home(environ, home)
    auth_file = _auth_file(environ, grok_home)
    auth_directory = auth_file.parent
    if (
        _has_symlink_component(grok_home)
        or not grok_home.is_dir()
        or _has_symlink_component(auth_directory)
        or not auth_directory.is_dir()
        or _has_symlink_component(auth_file)
        or not auth_file.is_file()
        or not _has_oidc_record(auth_file)
    ):
        return []

    model = environ.get("TRANCE_MODEL_GROK_CONSUMER", "").strip() or _DEFAULT_MODEL
    return [
        Candidate(
            provider="grok-consumer",
            auth_kind="account",
            source="grok-cli-login",
            model_name=model,
            secret=None,
            config={
                "integration": "grok-cli",
                "command": str(command_path),
                "auth_home": str(grok_home.absolute()),
                "auth_file": str(auth_file),
                "saved_login": "true",
            },
        )
    ]
