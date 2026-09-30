"""Provider neutral credential material resolution.

This module intentionally imports no provider SDK.  Source adapters may expose
an ``extract_credentials(candidate)`` function for saved CLI/OAuth sessions;
that function is imported only when the caller accesses ``candidate.credentials``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING

from trance.types import CredentialMaterial

if TYPE_CHECKING:
    from trance.types import Candidate


_EXTRACTOR_MODULES: tuple[tuple[str, str], ...] = (
    ("codex", "openai-codex"),
    ("codex", "codex"),
    ("copilot", "github-copilot"),
    ("claude_code", "claude-code"),
    ("grok_consumer", "grok-consumer"),
    ("gemini_cli", "gemini-cli"),
    ("cody", "sourcegraph-cody"),
    ("opencode", "opencode:"),
)


def _extractor_module(provider: str) -> str | None:
    for module, selector in _EXTRACTOR_MODULES:
        if selector.endswith(":") and provider.startswith(selector):
            return module
        if provider == selector:
            return module
    return None


def _mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str) and item
    }


def resolve_credentials(candidate: Candidate) -> CredentialMaterial:
    """Resolve direct values without importing an SDK or validating a token."""
    module_name = _extractor_module(candidate.provider)
    if module_name is not None:
        try:
            extractor = getattr(
                importlib.import_module(f"trance.sources.{module_name}"),
                "extract_credentials",
            )
        except (ImportError, AttributeError):
            extractor = None
        if extractor is not None:
            try:
                values = _mapping(extractor(candidate))
            except Exception:
                values = {}
            if candidate.provider == "claude-code" and "token" in values:
                values = {
                    ("access_token" if key == "token" else key): value
                    for key, value in values.items()
                }
            if values:
                return CredentialMaterial(
                    kind=candidate.auth_kind,
                    values=values,
                    reference=candidate.source,
                )

    if candidate.secret:
        key = "access_token" if candidate.provider == "claude-code" else "api_key"
        return CredentialMaterial(
            kind=candidate.auth_kind,
            values={key: candidate.secret},
            reference=candidate.source,
        )

    if candidate.provider == "aws-bedrock":
        values = {
            key: value
            for key, value in candidate.config.items()
            if key in {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"}
            and value
        }
        return CredentialMaterial(
            kind=candidate.auth_kind,
            values=values,
            reference=candidate.source,
        )

    return CredentialMaterial(
        kind=candidate.auth_kind,
        values={},
        reference=candidate.source,
    )


__all__ = ["resolve_credentials"]
