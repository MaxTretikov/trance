from pathlib import Path
from unittest.mock import patch

import pytest

from trance.sources.grok_consumer import scan


def test_scan_returns_secretless_candidate_for_grok_home(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    (grok_home / "auth.json").write_text(
        '{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8"
    )

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        [candidate] = scan({"PATH": "/usr/bin"}, tmp_path)

    assert candidate.provider == "grok-consumer"
    assert candidate.auth_kind == "account"
    assert candidate.source == "grok-cli-login"
    assert candidate.model_name == "grok-build"
    assert candidate.secret is None
    assert candidate.config == {
        "integration": "grok-cli",
        "command": "/usr/bin/grok",
        "auth_home": str(grok_home),
        "auth_file": str(grok_home / "auth.json"),
        "saved_login": "true",
    }


def test_scan_honors_grok_home_override_and_model(tmp_path: Path) -> None:
    grok_home = tmp_path / "grok-state"
    grok_home.mkdir()
    (grok_home / "auth.json").write_text(
        '{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8"
    )

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        [candidate] = scan(
            {
                "PATH": "/usr/bin",
                "GROK_HOME": str(grok_home.relative_to(tmp_path)),
                "TRANCE_MODEL_GROK_CONSUMER": "grok-build-0.1",
            },
            tmp_path,
        )

    assert candidate.model_name == "grok-build-0.1"
    assert candidate.config["auth_home"] == str(grok_home)
    assert candidate.config["auth_file"] == str(grok_home / "auth.json")


def test_scan_honors_auth_path_override(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    auth_file = tmp_path / "credentials" / "grok-auth.json"
    auth_file.parent.mkdir()
    auth_file.write_text('{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8")

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        [candidate] = scan(
            {"PATH": "/usr/bin", "GROK_AUTH_PATH": str(auth_file)},
            tmp_path,
        )

    assert candidate.config["auth_file"] == str(auth_file)


def test_scan_requires_safe_auth_path_directory(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    auth_file = tmp_path / "missing" / "grok-auth.json"

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan(
            {"PATH": "/usr/bin", "GROK_AUTH_PATH": str(auth_file)},
            tmp_path,
        ) == []

    target = tmp_path / "real-credentials"
    target.mkdir()
    try:
        auth_file.parent.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")
    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan(
            {"PATH": "/usr/bin", "GROK_AUTH_PATH": str(auth_file)},
            tmp_path,
        ) == []


def test_tilde_grok_home_uses_injected_home(tmp_path: Path) -> None:
    (tmp_path / ".grok").mkdir()
    (tmp_path / ".grok" / "auth.json").write_text(
        '{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8"
    )

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        result = scan({"PATH": "/usr/bin", "GROK_HOME": "~/.grok"}, tmp_path)

    assert len(result) == 1


def test_scan_requires_grok_cli_and_state_home(tmp_path: Path) -> None:
    with patch("trance.sources.grok_consumer.shutil.which", return_value=None):
        assert scan({}, tmp_path) == []

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan({}, tmp_path) == []


def test_scan_does_not_fall_back_to_process_path(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    (grok_home / "auth.json").write_text(
        '{"xai::api_key":{"auth_mode":"api_key"}}', encoding="utf-8"
    )

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan({}, tmp_path) == []


def test_scan_rejects_symlinked_grok_home(tmp_path: Path) -> None:
    target = tmp_path / "real-grok-home"
    target.mkdir()
    (target / "auth.json").write_text(
        '{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8"
    )
    try:
        (tmp_path / ".grok").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan({"PATH": "/usr/bin"}, tmp_path) == []


def test_scan_rejects_symlinked_grok_home_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "real-grok-home"
    target.mkdir()
    (target / "auth.json").write_text(
        '{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8"
    )
    link = tmp_path / "linked-parent"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan(
            {"PATH": "/usr/bin", "GROK_HOME": str(link / "real-grok-home")},
            tmp_path,
        ) == []


def test_scan_rejects_symlinked_auth_marker(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    target = tmp_path / "auth.json"
    target.write_text('{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8")
    try:
        (grok_home / "auth.json").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this runner")

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan({"PATH": "/usr/bin"}, tmp_path) == []


def test_scan_rejects_symlinked_auth_path_ancestor(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    target = tmp_path / "real-credentials"
    target.mkdir()
    auth_file = target / "grok-auth.json"
    auth_file.write_text('{"session":{"auth_mode":"oidc","key":"opaque"}}', encoding="utf-8")
    link = tmp_path / "linked-credentials"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan(
            {"PATH": "/usr/bin", "GROK_AUTH_PATH": str(link / "grok-auth.json")},
            tmp_path,
        ) == []

def test_scan_does_not_read_auth_marker_and_redacts_repr(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    (grok_home / "auth.json").write_text(
        '{"session":{"auth_mode":"oidc","key":"secret-token"}}', encoding="utf-8"
    )

    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        [candidate] = scan({"PATH": "/usr/bin"}, tmp_path)

    assert candidate.secret is None
    assert "secret-token" not in repr(candidate)
    assert "auth.json" not in repr(candidate)


def test_scan_skips_malformed_and_oversized_auth_files(tmp_path: Path) -> None:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    auth_file = grok_home / "auth.json"

    auth_file.write_text("not-json", encoding="utf-8")
    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan({"PATH": "/usr/bin"}, tmp_path) == []

    auth_file.write_bytes(b"{" + b"a" * (1024 * 1024) + b"}")
    with patch("trance.sources.grok_consumer.shutil.which", return_value="/usr/bin/grok"):
        assert scan({"PATH": "/usr/bin"}, tmp_path) == []
