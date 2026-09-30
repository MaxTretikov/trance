"""GitHub Copilot subscription credential discovery.

Only explicitly supported token environment variables are read.  The GitHub
CLI fallback is returned as a resolver descriptor so callers can decide when
to invoke the documented ``gh auth token`` command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_TOKEN_ENV_VARS = (
    "GITHUB_COPILOT_API_KEY",
    "GITHUB_COPILOT_API_TOKEN",
    "COPILOT_GITHUB_TOKEN",
)
_GITHUB_CLI_AUTH_ENV_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_HOST",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)
_MODEL_ENV = "TRANCE_MODEL_GITHUB_COPILOT"
_STATUS_TIMEOUT_SECONDS = 2.0
_TOKEN_TIMEOUT_SECONDS = 3.0


def extract_credentials(
    candidate: Candidate,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Mapping[str, str]:
    """Resolve a Copilot candidate's token only when explicitly requested.

    The GitHub CLI is the authority for saved Copilot login state.  Keep its
    environment free of ambient GitHub token and host selectors so a generic
    GitHub login cannot be silently substituted.  The returned mapping is
    intentionally small and contains no command output other than the token.
    """
    if candidate.provider != "github-copilot":
        return {}
    if candidate.secret:
        return {"token": candidate.secret}
    config = candidate.config
    if config.get("resolver") not in {"command", "github-cli"}:
        return {}
    executable = config.get("executable", "gh")
    child_env = dict(os.environ if environ is None else environ)
    for variable in _GITHUB_CLI_AUTH_ENV_VARS:
        child_env.pop(variable, None)
    auth_home = str(home) if home is not None else config.get("auth_home")
    if auth_home:
        child_env["HOME"] = auth_home
    try:
        result = subprocess.run(
            [executable, "auth", "token"],
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_TOKEN_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("GitHub CLI token resolution failed") from exc
    token = result.stdout.strip()
    if result.returncode != 0 or not token:
        raise RuntimeError("GitHub CLI did not return an authenticated token")
    return {"token": token}


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return Copilot credentials available through supported local sources.

    Token values remain in memory on the returned candidates and are never
    printed. Duplicate values exported under multiple supported names are
    represented once. ``home`` supplies a consistent HOME for the quiet GitHub
    CLI authentication check. Copilot CLI keychain data is left to its own
    credential resolver because it is stored in the OS keychain.
    """
    found: list[Candidate] = []
    seen: set[str] = set()
    model_name = environ.get(_MODEL_ENV, "").strip() or "gpt-5.4"

    for variable in _TOKEN_ENV_VARS:
        token = environ.get(variable, "").strip()
        if token and token not in seen:
            seen.add(token)
            found.append(
                Candidate(
                    provider="github-copilot",
                    auth_kind="subscription",
                    source=f"env:{variable}",
                    model_name=model_name,
                    secret=token,
                )
            )

    # Copilot CLI documents gh as its last-resort auth source. Verify a local
    # login without retrieving or printing its token, then leave it behind a
    # resolver for the caller. Generic GitHub token variables, enterprise
    # selectors, and enterprise tokens are omitted from the child environment
    # so they cannot be mistaken for the consumer Copilot login.
    gh = shutil.which("gh", path=environ.get("PATH"))
    if gh:
        gh_environ = dict(environ)
        for variable in _GITHUB_CLI_AUTH_ENV_VARS:
            gh_environ.pop(variable, None)
        gh_environ["HOME"] = str(home)
        try:
            authenticated = subprocess.run(
                [gh, "auth", "status"],
                env=gh_environ,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_STATUS_TIMEOUT_SECONDS,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            authenticated = False
    else:
        authenticated = False

    if authenticated:
        found.append(
            Candidate(
                provider="github-copilot",
                auth_kind="subscription",
                source="github-cli",
                model_name=model_name,
                config={
                    "resolver": "command",
                    "executable": str(Path(gh).resolve()),
                    "args": "auth token",
                    "auth_home": str(home.expanduser().resolve()),
                },
            )
        )

    return found
