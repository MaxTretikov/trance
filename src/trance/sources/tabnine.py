"""Discover an explicitly configured Tabnine PAT for its supported CLI.

Tabnine documents Personal Access Tokens for headless CLI use. This adapter
only accepts the documented ``TABNINE_TOKEN`` environment variable and only
returns a candidate when the supported ``tabnine`` command is available. It
does not inspect IDE or CLI credential stores.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a Tabnine CLI candidate for an explicit PAT in the environment."""
    del home
    token = environ.get("TABNINE_TOKEN", "").strip()
    if not token:
        return []

    command = shutil.which("tabnine", path=environ.get("PATH"))
    if command is None:
        return []

    return [
        Candidate(
            provider="tabnine",
            auth_kind="subscription",
            source="env:TABNINE_TOKEN",
            model_name="tabnine/default",
            secret=token,
            config={
                "command": command,
                "credential_env": "TABNINE_TOKEN",
                "integration": "tabnine_cli",
            },
        )
    ]
