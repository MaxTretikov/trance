import json
import subprocess
from pathlib import Path

import pytest

from trance.sources import claude_code
from trance.sources.claude_code import scan


def test_scan_finds_explicit_setup_token_without_exposing_it_in_metadata(
    tmp_path: Path,
) -> None:
    token = "subscription-token-value"

    candidates = scan({"CLAUDE_CODE_OAUTH_TOKEN": token}, tmp_path)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "claude-code"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == "env:CLAUDE_CODE_OAUTH_TOKEN"
    assert candidate.model_name == "sonnet"
    assert candidate.secret == token
    assert candidate.config == {"credential_env": "CLAUDE_CODE_OAUTH_TOKEN"}
    assert token not in repr(candidate.config)


def test_scan_ignores_empty_token(tmp_path: Path) -> None:
    assert scan({"CLAUDE_CODE_OAUTH_TOKEN": "  "}, tmp_path) == []


def test_scan_uses_trimmed_model_override_and_keeps_default(tmp_path: Path) -> None:
    [candidate] = scan(
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token-value",
            "TRANCE_MODEL_CLAUDE_CODE": "  claude-opus  ",
        },
        tmp_path,
    )
    assert candidate.model_name == "claude-opus"

    [default_candidate] = scan(
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token-value",
            "TRANCE_MODEL_CLAUDE_CODE": "  ",
        },
        tmp_path,
    )
    assert default_candidate.model_name == "sonnet"


def test_scan_does_not_probe_private_cli_login_files(tmp_path: Path) -> None:
    # The standard CLI credential-store location can exist without an
    # environment token; this scanner intentionally does not parse it.
    credentials = tmp_path / ".claude" / ".credentials.json"
    credentials.parent.mkdir()
    credentials.write_text('{"oauthAccount": {"accessToken": "private"}}')

    assert scan({}, tmp_path) == []


def _status_result(status: object, *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["/fake/bin/claude", "auth", "status"],
        returncode=returncode,
        stdout=json.dumps(status) if isinstance(status, dict) else str(status),
        stderr="",
    )


def _write_oauth_marker(home: Path, config_dir: Path | None = None) -> None:
    credentials = (config_dir or home / ".claude") / ".credentials.json"
    credentials.parent.mkdir(parents=True, exist_ok=True)
    credentials.write_text("opaque OAuth credentials", encoding="utf-8")


def _patch_cli_status(monkeypatch: pytest.MonkeyPatch, result: object) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(claude_code.shutil, "which", lambda name, path: "/fake/bin/claude")

    def fake_run(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, **kwargs})
        return result

    monkeypatch.setattr(claude_code.subprocess, "run", fake_run)
    return calls


def test_scan_finds_saved_subscription_login_from_cli_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_cli_status(
        monkeypatch,
        _status_result(
            {
                "loggedIn": True,
                "apiProvider": "firstParty",
                "authMethod": "oauth_token",
                "subscriptionType": " Pro ",
            }
        ),
    )
    _write_oauth_marker(tmp_path)

    [candidate] = scan({"PATH": "/fake/bin", "TRANCE_MODEL_CLAUDE_CODE": "claude-opus"}, tmp_path)

    assert candidate.provider == "claude-code"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == "claude-cli-login"
    assert candidate.model_name == "claude-opus"
    assert candidate.secret is None
    assert candidate.config == {
        "command": "/fake/bin/claude",
        "auth_home": str(tmp_path),
        "saved_login": "true",
    }
    assert calls[0]["args"] == (["/fake/bin/claude", "auth", "status"],)


def test_saved_login_passes_only_minimal_child_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_cli_status(
        monkeypatch,
        _status_result(
            {
                "loggedIn": True,
                "apiProvider": "firstParty",
                "authMethod": "claude.ai",
            }
        ),
    )
    environ = {
        "PATH": "/custom/bin",
        "HOME": "/untrusted/home",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "custom-config"),
        "CLAUDE_CODE_OAUTH_TOKEN": "private-token",
        "ANTHROPIC_API_KEY": "private-api-key",
        "TRANCE_MODEL_CLAUDE_CODE": "claude-haiku",
    }
    _write_oauth_marker(tmp_path, Path(environ["CLAUDE_CONFIG_DIR"]))

    candidates = scan(environ, tmp_path)
    candidate = candidates[-1]

    assert candidate.config["config_dir"] == str(tmp_path / "custom-config")

    assert calls[0]["env"] == {
        "PATH": "/custom/bin",
        "HOME": str(tmp_path),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "custom-config"),
    }
    assert calls[0]["stdin"] is subprocess.DEVNULL
    assert calls[0]["stderr"] is subprocess.DEVNULL
    assert calls[0]["timeout"] == 2.0
    assert calls[0]["check"] is False


def test_saved_login_normalizes_relative_home_and_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_cli_status(monkeypatch, _status_result("Logged in"))
    monkeypatch.chdir(tmp_path)
    relative_home = Path("relative-home")
    relative_config = Path("relative-config")
    _write_oauth_marker(tmp_path, tmp_path / relative_config)

    [candidate] = scan(
        {"PATH": "/fake/bin", "CLAUDE_CONFIG_DIR": str(relative_config)},
        relative_home,
    )

    expected_home = str((tmp_path / relative_home).resolve())
    expected_config = str((tmp_path / relative_config).resolve())
    assert candidate.config["auth_home"] == expected_home
    assert candidate.config["config_dir"] == expected_config
    assert calls[0]["env"]["HOME"] == expected_home  # type: ignore[index]
    assert calls[0]["env"]["CLAUDE_CONFIG_DIR"] == expected_config  # type: ignore[index]


def test_scan_rejects_successful_cli_status_without_oauth_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_status(monkeypatch, _status_result("Logged in with an API key"))

    assert scan({"PATH": "/fake/bin"}, tmp_path) == []


def test_scan_does_not_probe_cli_without_oauth_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(claude_code.shutil, "which", lambda name, path: "/fake/bin/claude")

    def fail_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("CLI status was probed without a credential marker")

    monkeypatch.setattr(claude_code.subprocess, "run", fail_run)

    assert scan({"PATH": "/fake/bin"}, tmp_path) == []


def test_scan_rejects_explicit_api_key_status_with_stale_oauth_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_status(
        monkeypatch,
        _status_result(
            {
                "loggedIn": True,
                "apiProvider": "firstParty",
                "authMethod": "api_key",
                "subscriptionType": "max",
            }
        ),
    )
    _write_oauth_marker(tmp_path)

    assert scan({"PATH": "/fake/bin"}, tmp_path) == []


def test_scan_ignores_cli_status_output_and_requires_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_status(
        monkeypatch,
        _status_result("not JSON; documented command is text output"),
    )
    _write_oauth_marker(tmp_path)

    [candidate] = scan({"PATH": "/fake/bin"}, tmp_path)
    assert candidate.auth_kind == "subscription"


def test_scan_ignores_cli_status_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(claude_code.shutil, "which", lambda name, path: "/fake/bin/claude")
    _write_oauth_marker(tmp_path)

    def fake_run(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, **kwargs})
        raise subprocess.TimeoutExpired(cmd="claude", timeout=2.0)

    monkeypatch.setattr(claude_code.subprocess, "run", fake_run)

    assert scan({"PATH": "/fake/bin"}, tmp_path) == []
    assert calls[0]["timeout"] == 2.0


def test_scan_does_not_read_private_credentials_when_cli_reports_saved_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    credentials = tmp_path / ".claude" / ".credentials.json"
    credentials.parent.mkdir()
    credentials.write_text('{"oauthAccount": {"accessToken": "private"}}', encoding="utf-8")
    def fail_private_read(path: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError("scanner read the private Claude credentials file")

    monkeypatch.setattr(Path, "read_text", fail_private_read)
    _patch_cli_status(
        monkeypatch,
        _status_result(
            {
                "loggedIn": True,
                "apiProvider": "firstParty",
                "authMethod": "oauth_token",
                "subscriptionType": "max",
            }
        ),
    )

    [candidate] = scan({"PATH": "/fake/bin"}, tmp_path)

    assert candidate.secret is None
    assert "private" not in repr(candidate)


def test_scan_keeps_explicit_token_and_saved_login_as_distinct_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_status(
        monkeypatch,
        _status_result(
            {
                "loggedIn": True,
                "apiProvider": "firstParty",
                "authMethod": "oauth_token",
                "subscriptionType": "enterprise",
            }
        ),
    )
    _write_oauth_marker(tmp_path)

    candidates = scan(
        {"PATH": "/fake/bin", "CLAUDE_CODE_OAUTH_TOKEN": "explicit-token"},
        tmp_path,
    )

    assert [candidate.source for candidate in candidates] == [
        "env:CLAUDE_CODE_OAUTH_TOKEN",
        "claude-cli-login",
    ]
    assert candidates[0].secret == "explicit-token"
    assert candidates[1].secret is None
