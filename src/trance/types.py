"""Small public data types shared by credential scanners and resolvers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import Model


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
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

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
