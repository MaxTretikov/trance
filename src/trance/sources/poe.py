"""Poe account API key discovery.

Poe documents ``POE_API_KEY`` for external applications. This source reads
that explicit setting only; it does not inspect browser sessions or cookies.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a Poe subscription candidate for a configured API key."""
    del home  # Included for the common source scanner contract.
    key = environ.get("POE_API_KEY", "").strip()
    if not key:
        return []
    return [
        Candidate(
            provider="poe",
            auth_kind="subscription",
            source="env:POE_API_KEY",
            model_name=environ.get("TRANCE_MODEL_POE", "").strip() or "GPT-5.4",
            secret=key,
        )
    ]
