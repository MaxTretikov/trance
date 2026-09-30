"""Discover an authenticated Sourcegraph Cody CLI session.

Cody owns credentials in operating-system secure storage. This adapter checks
the documented ``cody auth whoami`` command and returns a secretless delegated
candidate; it does not inspect keychain files or export access tokens.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_MODEL_ENV = "TRANCE_MODEL_SOURCEGRAPH_CODY"
_DEFAULT_MODEL = "default"
_STATUS_TIMEOUT_SECONDS = 2.0
_PAT_ENV_VARS = ("SRC_ACCESS_TOKEN", "SRC_ENDPOINT")


def extract_credentials(candidate: Candidate) -> Mapping[str, str]:
    """Return no credentials; Cody stores them in OS secure storage."""
    return {}


def _cli_path(environ: Mapping[str, str]) -> str | None:
    """Find Cody's documented CLI entry point on the supplied PATH."""
    path = environ.get("PATH", "")
    if not path:
        return None
    command = shutil.which("cody", path=path)
    return str(Path(command).resolve()) if command is not None else None


def _authenticated(cli: str, environ: Mapping[str, str]) -> bool:
    """Ask Cody for login status without accepting explicit PAT environment auth."""
    child_env = dict(environ)
    # Cody documents these variables as an alternative API-token login. Keep
    # this source scoped to the CLI's stored account login instead.
    for variable in _PAT_ENV_VARS:
        child_env.pop(variable, None)
    try:
        result = subprocess.run(
            [cli, "auth", "whoami"],
            env=child_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_STATUS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    output = f"{result.stdout}\n{result.stderr}".casefold()
    return (
        result.returncode == 0
        and "authenticated as" in output
        and "not authenticated" not in output
        and "unauthenticated" not in output
    )


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a secretless candidate if Cody reports a stored signed-in session.

    ``home`` scopes the CLI's configuration lookup to the caller's selected
    home directory. The CLI manages its secure-storage lookup and login flow.
    """
    auth_home = home.expanduser().resolve()
    child_env = dict(environ)
    child_env["HOME"] = str(auth_home)
    cli = _cli_path(environ)
    if cli is None or not _authenticated(cli, child_env):
        return []

    model = environ.get(_MODEL_ENV, "").strip() or _DEFAULT_MODEL
    return [
        Candidate(
            provider="sourcegraph-cody",
            auth_kind="account",
            source="cody-cli-login",
            model_name=model,
            secret=None,
            config={
                "command": cli,
                "auth_home": str(auth_home),
                "saved_login": "true",
            },
        )
    ]
