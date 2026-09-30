"""Discover credentials saved by the OpenCode CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_MAX_AUTH_BYTES = 256 * 1024
_VENDOR_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash",
    "xai": "grok-4.7",
    "groq": "openai/gpt-oss-120b",
    "deepseek": "deepseek-chat",
    "mistral": "mistral-small-latest",
    "openrouter": "openai/gpt-4o-mini",
}


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


def _read_auth(path: Path) -> Mapping[str, object] | None:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_AUTH_BYTES + 1)
        if len(raw) > _MAX_AUTH_BYTES:
            return None
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def extract_credentials(candidate: Candidate) -> Mapping[str, str]:
    """Extract only the selected vendor's allowlisted OpenCode fields."""
    if not candidate.provider.startswith("opencode:"):
        return {}
    configured = candidate.config.get("auth_file", "")
    vendor = candidate.config.get("vendor_provider", "").strip().lower()
    if not configured or not _VENDOR_PATTERN.fullmatch(vendor):
        return {}
    if candidate.provider != f"opencode:{vendor}":
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
            auth = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(auth, Mapping):
        return {}
    entry = auth.get(vendor)
    if not isinstance(entry, Mapping):
        return {}
    auth_type = entry.get("type")
    if auth_type == "api":
        value = entry.get("key")
        return {"key": value} if isinstance(value, str) and value.strip() else {}
    if auth_type == "oauth":
        access, refresh = entry.get("access"), entry.get("refresh")
        expires = entry.get("expires")
        if (
            isinstance(access, str) and access.strip()
            and isinstance(refresh, str) and refresh.strip()
            and isinstance(expires, (int, float)) and not isinstance(expires, bool)
        ):
            return {"access": access, "refresh": refresh}
        return {}
    if auth_type == "wellknown":
        key, token = entry.get("key"), entry.get("token")
        if isinstance(key, str) and key.strip() and isinstance(token, str) and token.strip():
            return {"key": key, "token": token}
    return {}


def _valid_auth(entry: object) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    auth_type = entry.get("type")
    if auth_type == "oauth":
        if not all(
            isinstance(entry.get(key), str) and entry[key].strip()
            for key in ("refresh", "access")
        ):
            return None
        expires = entry.get("expires")
        return (
            "account"
            if isinstance(expires, (int, float)) and not isinstance(expires, bool)
            else None
        )
    if auth_type == "api":
        value = entry.get("key")
        return "api_key" if isinstance(value, str) and value.strip() else None
    if auth_type == "wellknown":
        key, token = entry.get("key"), entry.get("token")
        return (
            "account"
            if all(isinstance(value, str) and value.strip() for value in (key, token))
            else None
        )
    return None


def _model_for(environ: Mapping[str, str], vendor: str) -> str | None:
    override = environ.get(f"TRANCE_MODEL_OPENCODE_{vendor.upper()}", "").strip()
    if override:
        return override if override.startswith(f"{vendor}/") else None
    default = _DEFAULT_MODELS.get(vendor)
    return f"{vendor}/{default}" if default else None


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return secretless candidates for valid saved OpenCode credentials."""
    command = shutil.which("opencode", path=environ.get("PATH", ""))
    if command is None:
        return []
    configured_data_home = environ.get("XDG_DATA_HOME", "").strip()
    data_home = (
        Path(configured_data_home)
        if configured_data_home
        else home / ".local" / "share"
    )
    auth_file = (data_home / "opencode" / "auth.json").absolute()
    if _has_symlink_component(auth_file) or not auth_file.is_file():
        return []
    auth = _read_auth(auth_file)
    if auth is None:
        return []
    candidates: list[Candidate] = []
    for raw_vendor, entry in auth.items():
        if not isinstance(raw_vendor, str):
            continue
        vendor = raw_vendor.strip().lower()
        if not _VENDOR_PATTERN.fullmatch(vendor):
            continue
        auth_kind = _valid_auth(entry)
        model = _model_for(environ, vendor)
        if auth_kind is None or model is None:
            continue
        candidates.append(
            Candidate(
                provider=f"opencode:{vendor}",
                auth_kind=auth_kind,
                source="opencode-auth",
                model_name=model,
                secret=None,
                config={
                    "integration": "opencode-cli",
                    "command": str(Path(command).absolute()),
                    "auth_file": str(auth_file),
                    "vendor_provider": vendor,
                    "saved_login": "true",
                },
            )
        )
    return candidates
