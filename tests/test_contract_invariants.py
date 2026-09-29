"""Cross-cutting contracts shared by every discovery source.

These tests deliberately use only synthetic homes and environments.  They are
kept separate from provider-specific tests so a new source cannot accidentally
skip the basic safety and result-shape guarantees.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from trance import discovery
from trance.types import Candidate, FoundModel

SOURCE_MODULES = tuple(module for module, _provider in discovery._SOURCES)


def test_candidate_is_a_redacted_immutable_snapshot() -> None:
    config = {"api_key": "config-secret", "nested": "also-secret"}
    candidate = Candidate("provider", "api_key", "synthetic", "model", "top-secret", config)

    config["api_key"] = "changed-after-construction"
    rendered = repr(candidate)

    assert candidate.config["api_key"] == "config-secret"
    assert all(secret not in rendered for secret in ("top-secret", "config-secret", "also-secret"))
    assert "<redacted>" in rendered
    with pytest.raises(TypeError):
        candidate.config["api_key"] = "mutated"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        candidate.provider = "changed"  # type: ignore[misc]
    assert not hasattr(candidate, "__dict__")


def test_found_model_is_a_redacted_immutable_snapshot() -> None:
    marker = "synthetic-model-secret"
    model = SimpleNamespace(repr_marker=marker)
    found = FoundModel("provider", "subscription", "synthetic", "model", model)

    rendered = repr(found)
    assert found.model is model
    assert marker not in rendered
    assert "model=<redacted>" in rendered
    with pytest.raises(FrozenInstanceError):
        found.model = object()  # type: ignore[misc]
    assert not hasattr(found, "__dict__")


@pytest.mark.parametrize("module_name", SOURCE_MODULES)
def test_registered_scanner_empty_environment_is_offline_and_side_effect_free(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every registered source accepts the common empty scan safely.

    An empty PATH prevents optional CLI probes.  The subprocess guard catches
    accidental command execution, while the filesystem snapshot catches
    credential-cache creation or other writes.  The fake home also ensures a
    scanner cannot silently consult the process user's home directory.
    """
    module = importlib.import_module(f"trance.sources.{module_name}")
    fake_home = tmp_path / "synthetic-home"
    fake_home.mkdir()
    before = _tree_snapshot(fake_home)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: pytest.fail("consulted real home")))

    def forbidden_subprocess(*args: object, **kwargs: object) -> None:
        pytest.fail(f"{module_name} executed a subprocess during an empty scan")

    monkeypatch.setattr("subprocess.run", forbidden_subprocess)

    candidates = module.scan({"PATH": ""}, fake_home)

    assert candidates == []
    assert _tree_snapshot(fake_home) == before


@pytest.mark.parametrize("module_name", SOURCE_MODULES)
def test_registered_scanner_results_obey_candidate_contract(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any candidates returned by an adapter have the public safe shape."""
    module = importlib.import_module(f"trance.sources.{module_name}")

    def forbidden_subprocess(*args: object, **kwargs: object) -> None:
        pytest.fail(f"{module_name} executed a subprocess during a contract scan")

    monkeypatch.setattr("subprocess.run", forbidden_subprocess)
    candidates = module.scan({"PATH": ""}, tmp_path)

    for candidate in candidates:
        assert isinstance(candidate, Candidate)
        assert all(isinstance(value, str) for value in (
            candidate.provider,
            candidate.auth_kind,
            candidate.source,
            candidate.model_name,
        ))
        assert isinstance(candidate.config, Mapping)
        if candidate.secret is not None:
            assert candidate.secret not in repr(candidate)
        for value in candidate.config.values():
            if value:
                assert value not in repr(candidate)


def _tree_snapshot(root: Path) -> tuple[tuple[str, bool, int], ...]:
    """Return a stable, platform-independent snapshot of a test tree."""
    entries: list[tuple[str, bool, int]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            entries.append((relative, False, path.stat().st_size))
        else:
            entries.append((relative, True, 0))
    return tuple(entries)
