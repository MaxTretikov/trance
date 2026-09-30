"""Discover a saved Google Gemini CLI account login.

The Gemini CLI owns the OAuth credential cache.  This adapter checks only the
small amount of metadata needed to establish that a refreshable login exists;
it never places credential values in a :class:`~trance.types.Candidate`.

Google's terms restrict third-party use of this login.  A successful discovery
therefore emits :class:`GeminiCLITermsWarning` so callers see that risk without
having to opt in through a separate setting.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import warnings
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_DEFAULT_HOME = ".gemini"
_DEFAULT_AUTH_FILE = "oauth_creds.json"
_DEFAULT_MODEL = "auto"
_MODEL_ENV = "TRANCE_MODEL_GEMINI_CLI"
_MAX_METADATA_BYTES = 64 * 1024
_TERMS_URL = "https://github.com/google-gemini/gemini-cli/blob/main/docs/resources/tos-privacy.md"
GEMINI_CLI_TERMS_WARNING = (
    "Using Gemini CLI OAuth through third-party software violates Google's published "
    "terms and may result in account suspension or termination. "
    f"See {_TERMS_URL}"
)

__all__ = ["GEMINI_CLI_TERMS_WARNING", "GeminiCLITermsWarning", "scan"]


class GeminiCLITermsWarning(UserWarning):
    """Warning emitted when a Gemini CLI OAuth login is discovered."""


def _gemini_home(environ: Mapping[str, str], home: Path) -> Path:
    """Resolve the CLI base home and return its ``.gemini`` state directory."""
    supplied_home = home.absolute()
    configured = environ.get("GEMINI_CLI_HOME", "").strip()
    if not configured:
        return supplied_home / _DEFAULT_HOME
    if configured == "~":
        base_home = supplied_home
    elif configured.startswith("~/"):
        base_home = supplied_home / configured[2:]
    else:
        base_home = Path(configured)
        if not base_home.is_absolute():
            base_home = supplied_home / base_home
    return base_home.absolute() / _DEFAULT_HOME


def _has_symlink_component(path: Path) -> bool:
    """Return whether any component of an absolute path is a symlink."""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


def _has_oauth_metadata(auth_file: Path) -> bool:
    """Check a bounded OAuth cache payload without retaining credential values."""
    try:
        with auth_file.open("rb") as stream:
            payload = stream.read(_MAX_METADATA_BYTES + 1)
    except (OSError, UnicodeError):
        return False
    if len(payload) > _MAX_METADATA_BYTES:
        return False
    try:
        metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, Mapping):
        return False
    refresh_token = metadata.get("refresh_token")
    return isinstance(refresh_token, str) and bool(refresh_token.strip())


def extract_credentials(candidate: Candidate) -> Mapping[str, str]:
    """Extract valid OAuth fields from the candidate's saved Gemini login."""
    if candidate.provider != "gemini-cli":
        return {}
    configured = candidate.config.get("auth_file", "")
    if not configured:
        return {}
    path = Path(configured).absolute()
    current = Path(path.anchor)
    descriptor = -1
    try:
        for component in path.parts[1:]:
            current /= component
            if current.is_symlink():
                return {}
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(_MAX_METADATA_BYTES + 1)
            file_stat = os.fstat(stream.fileno())
            if len(raw) > _MAX_METADATA_BYTES or not stat.S_ISREG(file_stat.st_mode):
                return {}
            payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, Mapping):
        return {}
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        return {}
    result: dict[str, str] = {}
    for name in ("refresh_token", "access_token"):
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            result[name] = value
    return result


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a secretless candidate for a valid Gemini CLI OAuth cache."""
    path = environ.get("PATH", "")
    if not path:
        return []
    command = shutil.which("gemini", path=path)
    if command is None:
        return []
    command_path = Path(command).absolute()

    auth_home = _gemini_home(environ, home)
    auth_file = auth_home / _DEFAULT_AUTH_FILE
    if (
        _has_symlink_component(auth_home)
        or not auth_home.is_dir()
        or _has_symlink_component(auth_file)
        or not auth_file.is_file()
        or not _has_oauth_metadata(auth_file)
    ):
        return []

    warnings.warn(
        GEMINI_CLI_TERMS_WARNING,
        GeminiCLITermsWarning,
        stacklevel=2,
    )
    model = environ.get(_MODEL_ENV, "").strip() or _DEFAULT_MODEL
    return [
        Candidate(
            provider="gemini-cli",
            auth_kind="account",
            source="gemini-cli-login",
            model_name=model,
            secret=None,
            config={
                "command": str(command_path),
                "auth_home": str(auth_home),
                "auth_file": str(auth_file),
                "saved_login": "true",
            },
        )
    ]
