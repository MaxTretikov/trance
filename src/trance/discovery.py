"""Discover configured provider credentials and materialize model clients.

Discovery is intentionally explicit: each source module knows the small set of
locations it is allowed to inspect. This module only coordinates those sources.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from trance.types import Candidate, FoundModel

logger = logging.getLogger(__name__)

_GITHUB_CLI_AUTH_ENV_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_HOST",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)

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


def _is_cli_candidate(candidate: Candidate) -> bool:
    # These integrations delegate auth to a local CLI and cannot be represented
    # by the ordinary API-key client factory.
    if candidate.provider in {"openai-codex", "codex"}:
        return False
    return (
        candidate.provider in {"claude-code", "grok-consumer", "gemini-cli", "sourcegraph-cody"}
        or candidate.provider.startswith("opencode:")
        or bool((getattr(candidate, "config", None) or {}).get("resolver"))
    )


def _provider_matches(provider: str, selector: str) -> bool:
    """Return whether a provider ID matches an exact or prefix selector."""
    return selector == "*" or provider == selector or (
        selector.endswith("*") and provider.startswith(selector[:-1])
    )


def _resolve_copilot(candidate: Candidate, environ: Mapping[str, str], home: Path) -> Candidate:
    """Resolve the documented GitHub CLI token without exposing command output."""
    if candidate.provider != "github-copilot" or not (candidate.config or {}).get("resolver"):
        return candidate
    try:
        child_env = dict(environ)
        child_env["HOME"] = str(home)
        # `gh auth token` otherwise prefers these variables or targets an
        # enterprise host, which could silently turn a generic PAT into Copilot.
        for variable in _GITHUB_CLI_AUTH_ENV_VARS:
            child_env.pop(variable, None)
        result = subprocess.run(
            ["gh", "auth", "token"],
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("GitHub CLI token resolution failed") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("GitHub CLI did not return an authenticated token")
    # models.build_model intentionally rejects the unresolved source marker;
    # use an in-memory marker that allows it to accept this explicit token.
    return replace(candidate, source="resolved:github-cli", secret=result.stdout.strip(), config={})


def _claude_cli_candidate(candidate: Candidate, environ: Mapping[str, str]) -> Candidate:
    """Resolve Claude Code's executable using only the supplied environment."""
    command = (candidate.config or {}).get("command", "claude")
    if os.path.dirname(command):
        candidates = [Path(command)]
    else:
        candidates = [
            Path(directory) / command
            for directory in environ.get("PATH", "").split(os.pathsep)
            if directory
        ]
    executable = next(
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    if executable is None:
        raise ValueError("Claude Code CLI command was not found on the supplied PATH")
    config = dict(candidate.config)
    config["command"] = str(executable)
    return replace(candidate, config=config)


def _materialize(candidate: Candidate, environ: Mapping[str, str] | None = None) -> Any:
    if candidate.provider in {"openai-codex", "codex", "aws-bedrock", "google-vertex"}:
        return importlib.import_module("trance.models").build_model(candidate)
    if _is_cli_candidate(candidate):
        if candidate.provider == "claude-code":
            candidate = _claude_cli_candidate(candidate, os.environ if environ is None else environ)
        try:
            factory = importlib.import_module("trance.cli_model").build_cli_model
        except (ImportError, AttributeError) as exc:
            raise ValueError("No safe text-only CLI model adapter is installed") from exc
        # The adapter itself owns the allowlist and raises for unsupported CLI
        # providers. Do not fall back to an API model for delegated credentials.
        return factory(candidate)
    if (candidate.config or {}).get("integration") or (candidate.config or {}).get("resolver"):
        raise ValueError("No safe model adapter for this delegated credential")
    factory = importlib.import_module("trance.models").build_model
    return factory(candidate)


def scan(
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    *,
    providers: Iterable[str] | None = None,
    strict: bool = False,
) -> list[FoundModel]:
    """Find credentials from supported local sources and build model objects.

    ``providers`` filters by the provider IDs exposed on returned candidates.
    No model inference requests are made; credential status is not validated.
    In non-strict mode a broken optional source or unsupported model factory is
    skipped, allowing other providers through.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    user_home = Path.home() if home is None else Path(home)
    selected = None if providers is None else frozenset(providers)
    found: list[FoundModel] = []
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
            try:
                materialized_candidate = _resolve_copilot(candidate, env, user_home)
                key = _fingerprint(materialized_candidate)
                if key in seen:
                    continue
                model = _materialize(materialized_candidate, env)
                seen.add(key)
                found.append(
                    FoundModel(
                        provider=candidate.provider,
                        auth_kind=candidate.auth_kind,
                        source=candidate.source,
                        model_name=candidate.model_name,
                        model=model,
                    )
                )
            except Exception as exc:
                if strict:
                    raise
                logger.debug(
                    "Could not construct model for provider %s (%s)",
                    candidate.provider,
                    type(exc).__name__,
                )
    return found


def clients(*args: Any, **kwargs: Any) -> list[Any]:
    """Return only the constructed model objects from :func:`scan`."""
    return [item.model for item in scan(*args, **kwargs)]


def models(*args: Any, **kwargs: Any) -> list[Any]:
    """Alias for :func:`clients` for callers who prefer the model terminology."""
    return clients(*args, **kwargs)
