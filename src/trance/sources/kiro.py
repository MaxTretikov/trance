"""Kiro subscription API-key discovery.

Kiro documents ``KIRO_API_KEY`` as its supported credential for headless CLI
usage. Kiro's browser/device-flow session is intentionally not read here: its
credentials are managed by the CLI and are not documented as a third-party API
credential.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate


def scan(
    environ: Mapping[str, str], home: Path
) -> list[Candidate]:
    """Return Kiro API-key credentials explicitly present in the environment.

    ``home`` is accepted to satisfy the common source interface. Kiro's
    supported subscription API-key flow uses an environment variable; local
    CLI login state is not a documented credential source for external clients.
    """
    del home
    key = environ.get("KIRO_API_KEY", "").strip()
    if not key:
        return []
    return [
        Candidate(
            provider="kiro",
            auth_kind="api_key",
            source="env:KIRO_API_KEY",
            model_name="kiro",
            secret=key,
        )
    ]
