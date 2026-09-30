"""Build Pydantic AI models from provider-neutral candidates.

This module is an optional adapter: importing it is cheap, and Pydantic AI is
loaded only when :func:`build` is called.  The package itself does not declare
Pydantic AI as a required dependency for the provider-neutral discovery API.
"""

from __future__ import annotations

import importlib
import os
import subprocess  # noqa: F401  # Legacy patch point; source resolver uses same module.
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from trance.sources import copilot as copilot_source
from trance.types import Candidate


class PydanticAIAdapterError(RuntimeError):
    """Base error raised by the optional Pydantic AI adapter."""


class MissingPydanticAIDependencyError(PydanticAIAdapterError):
    """Pydantic AI is not installed in the active environment."""


def _is_cli_candidate(candidate: Candidate) -> bool:
    if candidate.provider in {"openai-codex", "codex"}:
        return False
    return (
        candidate.provider in {"claude-code", "grok-consumer", "gemini-cli", "sourcegraph-cody"}
        or candidate.provider.startswith("opencode:")
        or bool((candidate.config or {}).get("resolver"))
    )


def _resolve_copilot(candidate: Candidate, environ: Mapping[str, str], home: Path) -> Candidate:
    """Resolve GitHub CLI auth without allowing ambient GitHub variables."""
    if candidate.provider != "github-copilot" or not (candidate.config or {}).get("resolver"):
        return candidate
    # The source resolver uses the candidate's recorded executable and home.
    # ``environ`` and ``home`` remain parameters for API compatibility; the
    # resolver deliberately sanitizes the process environment itself.
    credentials = copilot_source.extract_credentials(candidate, environ=environ, home=home)
    token = credentials.get("token")
    if not token:
        raise RuntimeError("GitHub CLI did not return an authenticated token")
    return replace(candidate, source="resolved:github-cli", secret=token, config={})


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
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None
    )
    if executable is None:
        raise ValueError("Claude Code CLI command was not found on the supplied PATH")
    config = dict(candidate.config)
    config["command"] = str(executable)
    return replace(candidate, config=config)


def _missing_dependency(provider: str, cause: BaseException) -> MissingPydanticAIDependencyError:
    # Keep the installation advice independent of trance's packaging extras.
    return MissingPydanticAIDependencyError(
        f"Pydantic AI support for {provider!r} is unavailable. Install it with "
        "`uv add pydantic-ai-slim[openai]` (and the provider's Pydantic AI "
        "dependency when required)."
    )


def build(
    candidate: Candidate,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Any:
    """Build a Pydantic AI model for ``candidate`` on demand.

    Provider SDK imports occur inside this function through the existing model
    and CLI factories.  ``environ`` and ``home`` are used for deterministic,
    safe credential resolution and never default to values embedded in a
    candidate.
    """
    env = os.environ if environ is None else environ
    user_home = Path.home() if home is None else Path(home)
    candidate = _resolve_copilot(candidate, env, user_home)

    try:
        if _is_cli_candidate(candidate):
            if candidate.provider == "claude-code":
                candidate = _claude_cli_candidate(candidate, env)
            cli_module = importlib.import_module("trance.cli_model")
            return cli_module.build_cli_model(candidate)

        if (candidate.config or {}).get("integration") or (candidate.config or {}).get("resolver"):
            raise ValueError("No safe model adapter for this delegated credential")

        models_module = importlib.import_module("trance.models")
        return models_module.build_model(candidate)
    except MissingPydanticAIDependencyError:
        raise
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "pydantic_ai" or exc.name.startswith("pydantic_ai.")):
            raise _missing_dependency(candidate.provider, exc) from exc
        raise
    except Exception as exc:
        models_module = importlib.import_module("trance.models")
        dependency_error = getattr(models_module, "ModelDependencyError", ())
        if dependency_error and isinstance(exc, dependency_error):
            raise _missing_dependency(candidate.provider, exc) from exc
        raise


__all__ = [
    "MissingPydanticAIDependencyError",
    "PydanticAIAdapterError",
    "build",
]
