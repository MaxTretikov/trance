import json
from pathlib import Path
from unittest.mock import patch

import pytest

from trance.sources.gemini_cli import GEMINI_CLI_TERMS_WARNING, GeminiCLITermsWarning, scan


def _write_cache(home: Path, value: object) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "oauth_creds.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_scan_discovers_refreshable_login_and_warns(tmp_path: Path) -> None:
    gemini_home = tmp_path / ".gemini"
    _write_cache(gemini_home, {"refresh_token": "opaque-refresh-token"})

    with patch("trance.sources.gemini_cli.shutil.which", return_value="/usr/bin/gemini"):
        with pytest.warns(GeminiCLITermsWarning, match="suspension or termination") as caught:
            [candidate] = scan({"PATH": "/usr/bin"}, tmp_path)

    terms_url = "https://github.com/google-gemini/gemini-cli/blob/main/docs/resources/tos-privacy.md"
    assert str(caught[0].message) == GEMINI_CLI_TERMS_WARNING
    assert "using gemini cli oauth through third-party software" in str(caught[0].message).lower()
    assert "Google's published terms" in str(caught[0].message)
    assert "account suspension or termination" in str(caught[0].message)
    assert terms_url in str(caught[0].message)
    assert candidate.provider == "gemini-cli"
    assert candidate.auth_kind == "account"
    assert candidate.source == "gemini-cli-login"
    assert candidate.model_name == "auto"
    assert candidate.secret is None
    assert candidate.config == {
        "command": "/usr/bin/gemini",
        "auth_home": str(gemini_home),
        "auth_file": str(gemini_home / "oauth_creds.json"),
        "saved_login": "true",
    }
    assert "opaque-refresh-token" not in repr(candidate)


def test_scan_honors_home_and_model_overrides(tmp_path: Path) -> None:
    base_home = tmp_path / "state"
    gemini_home = base_home / ".gemini"
    _write_cache(gemini_home, {"refresh_token": "refresh"})

    with patch("trance.sources.gemini_cli.shutil.which", return_value="/bin/gemini"):
        [candidate] = scan(
            {
                "PATH": "/bin",
                "GEMINI_CLI_HOME": str(base_home),
                "TRANCE_MODEL_GEMINI_CLI": " gemini-2.5-pro ",
            },
            tmp_path,
        )

    assert candidate.model_name == "gemini-2.5-pro"
    assert candidate.config["auth_home"] == str(gemini_home)


@pytest.mark.parametrize("override", ["state", "~/state"])
def test_scan_appends_gemini_directory_to_relative_and_tilde_base_overrides(
    tmp_path: Path, override: str
) -> None:
    gemini_home = tmp_path / "state" / ".gemini"
    _write_cache(gemini_home, {"refresh_token": "refresh"})

    with patch("trance.sources.gemini_cli.shutil.which", return_value="/bin/gemini"):
        [candidate] = scan({"PATH": "/bin", "GEMINI_CLI_HOME": override}, tmp_path)

    assert candidate.config["auth_home"] == str(gemini_home)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"access_token": "api-key-only"},
        {"refresh_token": "   "},
        [],
        "not-json-object",
    ],
)
def test_scan_rejects_non_refreshable_or_api_key_only_metadata(
    tmp_path: Path, payload: object
) -> None:
    _write_cache(tmp_path / ".gemini", payload)

    with patch("trance.sources.gemini_cli.shutil.which", return_value="/bin/gemini"):
        assert scan({"PATH": "/bin"}, tmp_path) == []


def test_scan_rejects_oversized_metadata(tmp_path: Path) -> None:
    gemini_home = tmp_path / ".gemini"
    gemini_home.mkdir()
    (gemini_home / "oauth_creds.json").write_bytes(
        b'{"refresh_token":"valid", "padding":"' + b"x" * (64 * 1024) + b'"}'
    )

    with patch("trance.sources.gemini_cli.shutil.which", return_value="/bin/gemini"):
        assert scan({"PATH": "/bin"}, tmp_path) == []


def test_scan_rejects_symlinked_cache_and_ancestors(tmp_path: Path) -> None:
    real_home = tmp_path / "real-gemini"
    _write_cache(real_home, {"refresh_token": "refresh"})
    linked_home = tmp_path / ".gemini"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")

    with patch("trance.sources.gemini_cli.shutil.which", return_value="/bin/gemini"):
        assert scan({"PATH": "/bin"}, tmp_path) == []

    linked_home.unlink()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")
    with patch("trance.sources.gemini_cli.shutil.which", return_value="/bin/gemini"):
        assert scan(
            {"PATH": "/bin", "GEMINI_CLI_HOME": str(linked_parent / "real-gemini")},
            tmp_path,
        ) == []


def test_scan_requires_gemini_on_supplied_path(tmp_path: Path) -> None:
    _write_cache(tmp_path / ".gemini", {"refresh_token": "refresh"})

    with patch("trance.sources.gemini_cli.shutil.which", return_value=None):
        assert scan({"PATH": "/empty"}, tmp_path) == []
