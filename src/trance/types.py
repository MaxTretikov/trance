"""Small public data types shared by credential scanners and resolvers."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.models import Model


class _RedactedMapping(Mapping[str, str]):
    """Read-only mapping whose representation never includes its values."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr({key: "<redacted>" for key in self._values})


@dataclass(frozen=True, slots=True, repr=False)
class CredentialMaterial:
    """Credential values resolved from a discovered candidate.

    Values are kept immutable and are deliberately omitted from ``repr`` so a
    credential object can be inspected safely in a debugger or log message.
    """

    kind: str
    values: Mapping[str, str] = field(default_factory=dict)
    reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _RedactedMapping(self.values))

    @property
    def value(self) -> str | None:
        """Return the sole credential value, when there is exactly one."""
        return next(iter(self.values.values())) if len(self.values) == 1 else None

    def __repr__(self) -> str:
        safe_values = {key: "<redacted>" for key in self.values}
        return (
            "CredentialMaterial("
            f"kind={self.kind!r}, values={safe_values!r}, "
            f"reference={self.reference!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class Candidate:
    """A credential discovered by a source scanner.

    ``secret`` is deliberately hidden from representations. ``config`` is
    copied and made read-only, and its values are hidden because resolvers may
    place credential material there.
    """

    provider: str
    auth_kind: str
    source: str
    model_name: str
    secret: str | None = None
    config: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _RedactedMapping(self.config))

    @property
    def credentials(self) -> CredentialMaterial:
        """Resolve this candidate's credential material on access."""
        from trance.credentials import resolve_credentials

        return resolve_credentials(self)

    def __repr__(self) -> str:
        safe_config = {key: "<redacted>" for key in self.config}
        return (
            "Candidate("
            f"provider={self.provider!r}, "
            f"auth_kind={self.auth_kind!r}, "
            f"source={self.source!r}, "
            f"model_name={self.model_name!r}, "
            f"secret={'<redacted>' if self.secret is not None else None!r}, "
            f"config={safe_config!r})"
        )

    def to_pydantic_ai(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        home: Path | None = None,
    ) -> Any:
        """Build this candidate as a Pydantic AI model on explicit request.

        The optional dependency is imported only when this method is called.
        ``environ`` and ``home`` are forwarded so adapters remain deterministic
        in tests and can resolve delegated credentials safely.
        """
        adapter = import_module("trance.adapters.pydantic_ai")
        return adapter.build(
            self,
            environ=os.environ if environ is None else environ,
            home=Path.home() if home is None else Path(home),
        )

    def to_litellm(self) -> Any:
        """Build this candidate through the optional LiteLLM adapter."""
        adapter = import_module("trance.adapters.litellm")
        return adapter.build(self)

    def to_openai(self, *, asynchronous: bool = False) -> Any:
        """Build this candidate through the optional OpenAI SDK adapter.

        The OpenAI SDK is imported only when this method is called.  Set
        ``asynchronous`` to receive an async client instead of the synchronous
        client.
        """
        adapter = import_module("trance.adapters.openai_sdk")
        return adapter.build(self, asynchronous=asynchronous)

    def to_langchain(self) -> Any:
        """Build this candidate through the optional LangChain adapter."""
        adapter = import_module("trance.adapters.langchain")
        return adapter.build(self)

    def to_llama_index(self) -> Any:
        """Build this candidate through the optional LlamaIndex adapter."""
        adapter = import_module("trance.adapters.llamaindex")
        return adapter.build(self)


@dataclass(frozen=True, slots=True, repr=False)
class FoundModel:
    """A discovered credential and its constructed Pydantic AI model."""

    provider: str
    auth_kind: str
    source: str
    model_name: str
    model: Model

    def __repr__(self) -> str:
        return (
            "FoundModel("
            f"provider={self.provider!r}, "
            f"auth_kind={self.auth_kind!r}, "
            f"source={self.source!r}, "
            f"model_name={self.model_name!r}, model=<redacted>)"
        )
