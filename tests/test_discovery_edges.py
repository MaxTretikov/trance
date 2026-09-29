"""Boundary tests for discovery orchestration.

These tests replace source scanners and model factories with small in-memory
fakes.  They exercise the coordinator without credentials, subprocesses, or
provider SDKs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trance import discovery
from trance.types import Candidate


def candidate(
    provider: str = "demo",
    *,
    source: str = "env:DEMO_KEY",
    secret: str | None = "secret",
    config: dict[str, str] | None = None,
) -> Candidate:
    return Candidate(
        provider=provider,
        auth_kind="api_key",
        source=source,
        model_name="demo-model",
        secret=secret,
        config=config or {},
    )


def install_source(
    monkeypatch: pytest.MonkeyPatch,
    sources: tuple[tuple[str, str], ...],
    scanners: dict[str, Any],
    *,
    model_factory: Any | None = None,
    cli_factory: Any | None = None,
) -> None:
    """Install import fakes for a small, explicit source registry."""
    monkeypatch.setattr(discovery, "_SOURCES", sources)

    def fake_import(name: str) -> Any:
        if name.startswith("trance.sources."):
            return SimpleNamespace(scan=scanners[name.rsplit(".", 1)[1]])
        if name == "trance.models":
            return SimpleNamespace(
                build_model=model_factory or (lambda item: ("api", item.provider))
            )
        if name == "trance.cli_model":
            return SimpleNamespace(
                build_cli_model=cli_factory or (lambda item: ("cli", item.provider))
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("demo", ["demo"]),
        ("opencode:*", ["opencode:openai", "opencode:anthropic"]),
        ("*", ["demo", "opencode:openai", "opencode:anthropic"]),
    ],
)
def test_provider_selectors_support_exact_prefix_and_wildcard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selector: str,
    expected: list[str],
) -> None:
    items = [candidate(), candidate("opencode:openai"), candidate("opencode:anthropic")]
    install_source(
        monkeypatch,
        (("fake", "*"),),
        {"fake": lambda env, home: items},
    )

    result = discovery.scan({}, tmp_path, providers=[selector])

    assert [item.provider for item in result] == expected


def test_selector_matches_wildcard_source_registration_before_scanning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scanned: list[str] = []
    install_source(
        monkeypatch,
        (("opencode", "opencode:*"), ("other", "other")),
        {
            "opencode": lambda env, home: scanned.append("opencode")
            or [candidate("opencode:openai")],
            "other": lambda env, home: scanned.append("other") or [candidate("other")],
        },
    )

    result = discovery.scan({}, tmp_path, providers=["opencode:openai"])

    assert [item.provider for item in result] == ["opencode:openai"]
    assert scanned == ["opencode"]


def test_source_order_is_preserved_and_same_secret_deduplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = candidate("same", source="first")
    duplicate = candidate("same", source="second")
    later = candidate("later", source="third")
    install_source(
        monkeypatch,
        (("first", "first"), ("second", "second")),
        {
            "first": lambda env, home: [first, later],
            "second": lambda env, home: [duplicate],
        },
    )

    result = discovery.scan({}, tmp_path)

    assert [(item.provider, item.source) for item in result] == [
        ("same", "first"),
        ("later", "third"),
    ]


def test_secretless_candidates_deduplicate_at_materialization_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    first = candidate("account", source="saved-a", secret=None)
    second = candidate("account", source="saved-b", secret=None)
    install_source(
        monkeypatch,
        (("fake", "fake"),),
        {"fake": lambda env, home: [first, second]},
        model_factory=lambda item: calls.append(item.source) or item.source,
    )

    result = discovery.scan({}, tmp_path)

    assert [item.source for item in result] == ["saved-a"]
    assert calls == ["saved-a"]


def test_non_strict_scan_skips_source_and_factory_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_source(
        monkeypatch,
        (("broken_source", "broken-source"), ("working", "working")),
        {
            "broken_source": lambda env, home: (_ for _ in ()).throw(RuntimeError("offline")),
            "working": lambda env, home: [candidate("working")],
        },
        model_factory=lambda item: (_ for _ in ()).throw(ValueError("unsupported"))
        if item.provider == "working"
        else object(),
    )

    assert discovery.scan({}, tmp_path) == []


@pytest.mark.parametrize("failure", ["source", "factory"])
def test_strict_scan_reraises_source_or_materialization_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    if failure == "source":
        install_source(
            monkeypatch,
            (("fake", "fake"),),
            {"fake": lambda env, home: (_ for _ in ()).throw(RuntimeError("source failed"))},
        )
        expected = "source failed"
    else:
        install_source(
            monkeypatch,
            (("fake", "fake"),),
            {"fake": lambda env, home: [candidate()]},
            model_factory=lambda item: (_ for _ in ()).throw(ValueError("factory failed")),
        )
        expected = "factory failed"

    with pytest.raises((RuntimeError, ValueError), match=expected):
        discovery.scan({}, tmp_path, strict=True)


def test_copilot_resolution_sanitizes_environment_and_candidate_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = candidate(
        "github-copilot",
        source="github-cli",
        secret=None,
        config={"resolver": "gh auth token", "private": "should-not-leak"},
    )
    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        captured.update(args=args, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout="resolved-secret\n")

    monkeypatch.setattr(discovery.subprocess, "run", fake_run)
    resolved = discovery._resolve_copilot(
        original,
        {
            "HOME": "/wrong",
            "GH_TOKEN": "generic",
            "GITHUB_TOKEN": "generic-two",
            "GH_HOST": "enterprise.example",
        },
        tmp_path,
    )

    assert resolved.secret == "resolved-secret"
    assert resolved.source == "resolved:github-cli"
    assert resolved.config == {}
    assert captured["kwargs"]["env"]["HOME"] == str(tmp_path)
    assert all(
        name not in captured["kwargs"]["env"]
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_HOST")
    )
    assert original.config["private"] == "should-not-leak"


def test_materialization_routes_cli_and_standard_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            build_model=lambda item: calls.append(("api", item.provider)) or "api-model",
            build_cli_model=lambda item: calls.append(("cli", item.provider)) or "cli-model",
        ),
    )
    standard = candidate("openai-codex")
    cli = candidate("gemini-cli", secret=None)
    delegated = candidate("unknown", config={"integration": "private-cli"})

    assert discovery._materialize(standard) == "api-model"
    assert discovery._materialize(cli) == "cli-model"
    with pytest.raises(ValueError, match="No safe model adapter"):
        discovery._materialize(delegated)
    assert calls == [("api", "openai-codex"), ("cli", "gemini-cli")]


def test_copilot_resolution_failures_are_strictly_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = candidate("github-copilot", secret=None, config={"resolver": "command"})
    monkeypatch.setattr(
        discovery.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )

    with pytest.raises(RuntimeError, match="did not return"):
        discovery._resolve_copilot(original, {}, Path("/tmp/home"))


def test_clients_and_models_are_thin_scan_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    expected = [SimpleNamespace(model="one"), SimpleNamespace(model="two")]

    def fake_scan(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(discovery, "scan", fake_scan)

    assert discovery.clients("env", providers=["demo"]) == ["one", "two"]
    assert discovery.models("env", providers=["demo"]) == ["one", "two"]
    assert calls == [
        (("env",), {"providers": ["demo"]}),
        (("env",), {"providers": ["demo"]}),
    ]
