"""Regression checks for credential discovery's local security boundaries."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from trance import cli_model, discovery
from trance.adapters import pydantic_ai
from trance.sources import (
    claude_code,
    codex,
    copilot,
    cursor,
    gemini_cli,
    grok_consumer,
    kiro,
    opencode,
    qwen_code,
)
from trance.types import Candidate, FoundModel


@pytest.mark.parametrize("kind", ["candidate", "found-model"])
def test_public_object_reprs_never_reveal_synthetic_secrets(kind: str) -> None:
    marker = "TRANCE_TEST_SECRET_7b93e2"
    candidate = Candidate(
        "sample", "subscription", "synthetic", "model", secret=marker,
        config={"access_token": marker},
    )
    value = candidate if kind == "candidate" else FoundModel(
        "sample", "subscription", "synthetic", "model", candidate
    )

    assert marker not in repr(value)


def test_scanners_stay_inside_the_supplied_fake_home(monkeypatch, tmp_path: Path) -> None:
    """An isolated scan must never consult the process user's home directory."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".qwen").mkdir()
    (fake_home / ".qwen" / "settings.json").write_text("{}", encoding="utf-8")
    read_paths: list[Path] = []
    checked_paths: list[Path] = []
    original_read_text = Path.read_text
    original_is_file = Path.is_file

    def tracked_read_text(path: Path, *args, **kwargs):
        read_paths.append(path.resolve())
        return original_read_text(path, *args, **kwargs)

    def tracked_is_file(path: Path, *args, **kwargs):
        checked_paths.append(path.resolve())
        return original_is_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: pytest.fail("consulted real home")))
    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    monkeypatch.setattr(Path, "is_file", tracked_is_file)

    env = {"PATH": "", "HOME": str(fake_home)}
    scanners = (codex.scan, copilot.scan, claude_code.scan, cursor.scan,
                opencode.scan, kiro.scan, qwen_code.scan)
    for scanner in scanners:
        scanner(env, fake_home)

    assert read_paths
    root = fake_home.resolve()
    assert all(path == root or root in path.parents for path in read_paths)
    assert all(path == root or root in path.parents for path in checked_paths)


def test_empty_path_prevents_implicit_shell_probes(monkeypatch, tmp_path: Path) -> None:
    """With no CLI on PATH, discovery is local and makes no subprocess calls."""
    import subprocess

    def forbidden_run(*args, **kwargs):
        pytest.fail("scanner executed a command without a discoverable documented CLI")

    monkeypatch.setattr(subprocess, "run", forbidden_run)
    monkeypatch.setattr(copilot.shutil, "which", lambda *args, **kwargs: None)
    monkeypatch.setattr(cursor.shutil, "which", lambda *args, **kwargs: None)
    env = {"PATH": "", "HOME": str(tmp_path)}
    assert copilot.scan(env, tmp_path) == []
    assert cursor.scan(env, tmp_path) == []


def test_default_discovery_with_stubbed_factories_is_offline_and_ignores_dotenv_browser_data(
    monkeypatch, tmp_path: Path
) -> None:
    marker = "TRANCE_TEST_SECRET_1af4d8"
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={marker}\n", encoding="utf-8")
    browser = tmp_path / ".config" / "chromium" / "Default"
    browser.mkdir(parents=True)
    (browser / "Login Data").write_text(marker, encoding="utf-8")
    import urllib.request

    def forbidden_network(*args, **kwargs):
        pytest.fail("discovery attempted network access")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    assert discovery.scan({"PATH": "", "HOME": str(tmp_path)}, tmp_path) == []


def test_duplicate_suppression_does_not_log_or_emit_secret(
    caplog, monkeypatch, tmp_path: Path
) -> None:
    marker = "TRANCE_TEST_SECRET_95c1a6"
    candidate = Candidate("sample", "api_key", "synthetic", "model", secret=marker)
    calls: list[Candidate] = []

    def fake_import(name: str):
        if name == "trance.sources.fake":
            return SimpleNamespace(scan=lambda env, home: [candidate, candidate])
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("fake", "sample"),))
    caplog.set_level(logging.DEBUG, logger="trance.discovery")

    result = discovery.scan({}, tmp_path)

    assert len(result) == 1
    assert calls == []
    assert marker not in caplog.text
    assert marker not in repr(result)


def test_copilot_resolution_does_not_forward_generic_github_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    """A generic GitHub PAT must not override gh's logged-in Copilot identity."""
    marker = "TRANCE_TEST_SECRET_GENERIC_GH_TOKEN"
    observed: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = "copilot-session-token\n"

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return Result()

    monkeypatch.setattr(copilot.subprocess, "run", fake_run)
    candidate = Candidate(
        "github-copilot", "subscription", "github-cli", "model",
        config={"resolver": "command"},
    )

    resolved = pydantic_ai._resolve_copilot(
        candidate,
        {"PATH": "/synthetic/bin", "GH_TOKEN": marker, "GITHUB_TOKEN": marker},
        tmp_path,
    )

    child_env = observed["env"]
    assert "GH_TOKEN" not in child_env
    assert "GITHUB_TOKEN" not in child_env
    assert child_env["HOME"] == str(tmp_path)
    assert marker not in repr(resolved)


def test_copilot_resolution_does_not_forward_enterprise_github_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    """A Copilot resolver must not turn enterprise GitHub auth into a token source."""
    observed: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = "copilot-session-token\n"

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return Result()

    monkeypatch.setattr(copilot.subprocess, "run", fake_run)
    candidate = Candidate(
        "github-copilot", "subscription", "github-cli", "model",
        config={"resolver": "command"},
    )

    pydantic_ai._resolve_copilot(
        candidate,
        {
            "PATH": "/synthetic/bin",
            "GH_HOST": "ghe.example.test",
            "GH_ENTERPRISE_TOKEN": "enterprise-token",
            "GITHUB_ENTERPRISE_TOKEN": "enterprise-token-2",
        },
        tmp_path,
    )

    child_env = observed["env"]
    assert "GH_HOST" not in child_env
    assert "GH_ENTERPRISE_TOKEN" not in child_env
    assert "GITHUB_ENTERPRISE_TOKEN" not in child_env


def test_grok_candidate_redacts_auth_metadata_and_rejects_symlink_paths(
    monkeypatch, tmp_path: Path
) -> None:
    """Grok discovery must not traverse symlinked credential locations."""
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    (real_home / "auth.json").write_text('{"refresh_token":"synthetic-secret"}')
    link_parent = tmp_path / "linked-parent"
    try:
        link_parent.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")

    monkeypatch.setattr(grok_consumer.shutil, "which", lambda *args, **kwargs: "/usr/bin/grok")
    assert grok_consumer.scan(
        {"PATH": "/usr/bin", "GROK_HOME": "linked-parent/real-home"}, tmp_path
    ) == []

    safe_home = tmp_path / "safe-home"
    safe_home.mkdir()
    try:
        (safe_home / "auth.json").symlink_to(real_home / "auth.json")
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this runner")
    assert grok_consumer.scan(
        {"PATH": "/usr/bin", "GROK_HOME": str(safe_home)}, tmp_path
    ) == []


def test_grok_candidate_repr_never_contains_auth_bytes(monkeypatch, tmp_path: Path) -> None:
    marker = "TRANCE_GROK_SYNTHETIC_REFRESH_SECRET"
    grok_home = tmp_path / "grok-home"
    grok_home.mkdir()
    (grok_home / "auth.json").write_text(
        '{"account":{"auth_mode":"oidc","refresh_token":"' + marker + '"}}'
    )
    monkeypatch.setattr(grok_consumer.shutil, "which", lambda *args, **kwargs: "/usr/bin/grok")

    [candidate] = grok_consumer.scan({"PATH": "/usr/bin", "GROK_HOME": str(grok_home)}, tmp_path)

    assert candidate.secret is None
    assert marker not in repr(candidate)


def test_gemini_cli_redacts_oauth_bytes_and_warns_with_terms_risk(
    monkeypatch, tmp_path: Path
) -> None:
    marker = "TRANCE_GEMINI_SYNTHETIC_REFRESH_SECRET"
    gemini_home = tmp_path / ".gemini"
    gemini_home.mkdir()
    (gemini_home / "oauth_creds.json").write_text(
        '{"refresh_token":"' + marker + '"}', encoding="utf-8"
    )
    monkeypatch.setattr(gemini_cli.shutil, "which", lambda *args, **kwargs: "/usr/bin/gemini")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        [candidate] = gemini_cli.scan({"PATH": "/usr/bin"}, tmp_path)

    assert candidate.secret is None
    assert marker not in repr(candidate)
    messages = [str(item.message) for item in caught if issubclass(item.category, UserWarning)]
    assert len(messages) == 1
    assert "terms" in messages[0].casefold()
    assert "suspension" in messages[0].casefold()


def test_gemini_cli_rejects_symlinked_oauth_file_and_ancestor(
    monkeypatch, tmp_path: Path
) -> None:
    real_home = tmp_path / "real-gemini"
    real_home.mkdir()
    auth_file = real_home / "oauth_creds.json"
    auth_file.write_text('{"refresh_token":"synthetic"}', encoding="utf-8")
    monkeypatch.setattr(gemini_cli.shutil, "which", lambda *args, **kwargs: "/usr/bin/gemini")

    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")
    assert gemini_cli.scan(
        {"PATH": "/usr/bin", "GEMINI_CLI_HOME": "linked-parent/real-gemini"}, tmp_path
    ) == []

    safe_home = tmp_path / "safe-gemini"
    safe_home.mkdir()
    try:
        (safe_home / "oauth_creds.json").symlink_to(auth_file)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this runner")
    assert gemini_cli.scan(
        {"PATH": "/usr/bin", "GEMINI_CLI_HOME": str(safe_home)}, tmp_path
    ) == []


def test_gemini_cli_does_not_false_positive_on_api_key_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gemini_cli.shutil, "which", lambda *args, **kwargs: "/usr/bin/gemini")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert gemini_cli.scan(
            {"PATH": "/usr/bin", "GEMINI_API_KEY": "synthetic-api-key"}, tmp_path
        ) == []
    assert caught == []


def test_cli_bridge_rejects_image_capability_in_text_only_adapter() -> None:
    class UserPromptPart:
        content = "describe this"

    class ImagePart:
        content = b"synthetic-image-bytes"

    message = SimpleNamespace(parts=[UserPromptPart(), ImagePart()])
    parameters = SimpleNamespace(
        function_tools=[],
        native_tools=[],
        output_tools=[],
        output_object=None,
        allow_image_output=True,
        instruction_parts=[],
    )

    with pytest.raises(cli_model.UnsupportedCLIModelError, match="text content only"):
        cli_model._text_from_messages([message], parameters)
