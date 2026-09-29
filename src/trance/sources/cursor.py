"""Discover a signed-in Cursor CLI for ACP delegation.

Cursor's documented account login is held by its CLI. This source deliberately
does not read or export that credential; callers delegate requests to ACP.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate


def _cli_path(environ: Mapping[str, str], home: Path) -> str | None:
    """Find Cursor's documented CLI entry point without searching auth files."""
    # `agent` is the current documented command; older CLI docs use
    # `cursor-agent`. Prefer Cursor's own install directory before PATH's
    # generic `agent` name to avoid mistaking another vendor's CLI.
    for name in ("agent", "cursor-agent"):
        bundled = home / ".cursor" / "bin" / name
        if bundled.is_file() and os.access(bundled, os.X_OK):
            return str(bundled)
    for name in ("cursor-agent", "agent"):
        path = shutil.which(name, path=environ.get("PATH"))
        if path:
            return path
    return None


def _authenticated(cli: str, environ: Mapping[str, str]) -> bool:
    """Use the CLI's local status command; never parse or persist credentials."""
    try:
        result = subprocess.run(
            [cli, "status"],
            env=dict(environ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    # Status is human-readable in the documented CLI. Fail closed if the
    # output does not explicitly say authenticated, or indicates otherwise.
    output = f"{result.stdout}\n{result.stderr}".casefold()
    return (
        result.returncode == 0
        and "authenticat" in output
        and "not authenticated" not in output
        and "unauthenticated" not in output
    )


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a secretless Cursor ACP candidate when its CLI reports login.

    The resulting candidate represents a delegated CLI session, not a raw
    token usable by arbitrary OpenAI-compatible clients.
    """
    child_env = dict(environ)
    child_env.setdefault("HOME", str(home))
    cli = _cli_path(environ, home)
    if cli is None or not _authenticated(cli, child_env):
        return []

    return [
        Candidate(
            provider="cursor",
            auth_kind="subscription",
            source="cursor-agent status",
            model_name="cursor/agent",
            secret=None,
            config={
                "command": cli,
                "protocol": "acp",
                "transport": "stdio",
                "auth_method": "cursor_login",
            },
        )
    ]
