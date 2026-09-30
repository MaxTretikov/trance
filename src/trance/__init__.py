"""Discover local model sessions and expose them through a stable API.

The package root intentionally avoids importing discovery and model adapters
until a caller uses them. This keeps a plain ``import trance`` inexpensive.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_EXPORTS = {
    "scan": ("trance.discovery", "scan"),
    "clients": ("trance.discovery", "clients"),
    "Candidate": ("trance.types", "Candidate"),
    "FoundModel": ("trance.types", "FoundModel"),
    "TransientCLIModelError": ("trance.cli_model", "TransientCLIModelError"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load public API objects only when requested."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
