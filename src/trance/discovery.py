"""Discover configured provider credentials without choosing an SDK.

Discovery is intentionally explicit: each source module knows the small set of
locations it is allowed to inspect. This module only coordinates those sources.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trance.types import Candidate, FoundModel

logger = logging.getLogger(__name__)

# Keep this list static and cheap to inspect. Imports happen only when scanning.
_SOURCES: tuple[tuple[str, str], ...] = (
    # Put subscription/account sources first. The generic API-key scanner runs
    # later so a provider-specific subscription integration wins on duplicates.
    ("codex", "openai-codex"),
    ("copilot", "github-copilot"),
    ("claude_code", "claude-code"),
    ("poe", "poe"),
    ("minimax_coding", "minimax-coding-plan"),
    ("mistral_vibe", "mistral"),
    ("qwen_code", "qwen-code"),
    ("zai", "*"),
    ("huggingface_login", "huggingface"),
    ("grok_consumer", "grok-consumer"),
    ("gemini_cli", "gemini-cli"),
    ("cody", "sourcegraph-cody"),
    ("opencode", "opencode:*"),
    ("aws_bedrock", "aws-bedrock"),
    ("google_vertex", "google-vertex"),
    ("compatible_api", "*"),
    ("cloud_api", "*"),
    ("env_api", "*"),
)


def _fingerprint(candidate: Candidate) -> tuple[str, str]:
    secret = getattr(candidate, "secret", None)
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest() if isinstance(secret, str) else ""
    return (candidate.provider, digest)


def _provider_matches(provider: str, selector: str) -> bool:
    """Return whether a provider ID matches an exact or prefix selector."""
    return selector == "*" or provider == selector or (
        selector.endswith("*") and provider.startswith(selector[:-1])
    )


def scan(
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    *,
    providers: Iterable[str] | None = None,
    strict: bool = False,
) -> list[Candidate]:
    """Find credentials from supported local sources.

    ``providers`` filters by the provider IDs exposed on returned candidates.
    No model inference requests are made; credential status is not validated.
    In non-strict mode a broken optional source is skipped, allowing other
    providers through. Scanning constructs no model client and does not import
    Pydantic AI or LiteLLM.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    user_home = Path.home() if home is None else Path(home)
    selected = None if providers is None else frozenset(providers)
    found: list[Candidate] = []
    seen: set[tuple[str, str]] = set()

    for module_name, provider_id in _SOURCES:
        if selected is not None and provider_id != "*" and not any(
            _provider_matches(provider_id, selector) or _provider_matches(selector, provider_id)
            for selector in selected
        ):
            continue
        try:
            scanner = importlib.import_module(f"trance.sources.{module_name}").scan
            candidates = scanner(env, user_home)
        except Exception as exc:
            if strict:
                raise
            logger.debug("Credential source %s failed (%s)", provider_id, type(exc).__name__)
            continue

        for candidate in candidates:
            if selected is not None and not any(
                _provider_matches(candidate.provider, selector) for selector in selected
            ):
                continue
            key = _fingerprint(candidate)
            if candidate.secret and key in seen:
                continue
            seen.add(key)
            found.append(candidate)
    return found


def scan_models(
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    *,
    providers: Iterable[str] | None = None,
    strict: bool = False,
) -> list[FoundModel]:
    """Discover candidates and explicitly build Pydantic AI models."""
    found: list[FoundModel] = []
    for candidate in scan(environ, home, providers=providers, strict=strict):
        try:
            model = candidate.to_pydantic_ai(environ=environ, home=home)
        except Exception:
            if strict:
                raise
            logger.debug("Could not construct model for provider %s", candidate.provider)
            continue
        found.append(
            FoundModel(
                provider=candidate.provider,
                auth_kind=candidate.auth_kind,
                source=candidate.source,
                model_name=candidate.model_name,
                model=model,
            )
        )
    return found


def clients(*args: Any, **kwargs: Any) -> list[Any]:
    """Return constructed Pydantic AI models through the explicit adapter."""
    return [item.model for item in scan_models(*args, **kwargs)]


def models(*args: Any, **kwargs: Any) -> list[Any]:
    """Alias for :func:`clients`."""
    return clients(*args, **kwargs)
