from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trance.sources import cursor


def test_scan_emits_secretless_acp_candidate_for_signed_in_cli(tmp_path: Path):
    observed = {}

    def authenticated(_cli, environ):
        observed.update(environ)
        return True

    with (
        patch.object(cursor, "_cli_path", return_value="/usr/bin/cursor-agent"),
        patch.object(cursor, "_authenticated", side_effect=authenticated),
    ):
        result = cursor.scan({"PATH": "/usr/bin"}, tmp_path)

    assert observed["HOME"] == str(tmp_path)
    assert len(result) == 1
    candidate = result[0]
    assert candidate.provider == "cursor"
    assert candidate.auth_kind == "subscription"
    assert candidate.secret is None
    assert candidate.config["protocol"] == "acp"


def test_scan_returns_empty_without_cli(tmp_path: Path):
    with patch.object(cursor, "_cli_path", return_value=None):
        assert cursor.scan({}, tmp_path) == []


def test_status_requires_explicit_success_and_authentication():
    success = SimpleNamespace(returncode=0, stdout="Authenticated as user", stderr="")
    rejected = SimpleNamespace(returncode=0, stdout="Not authenticated", stderr="")
    with patch.object(cursor.subprocess, "run", return_value=success):
        assert cursor._authenticated("cursor-agent", {})
    with patch.object(cursor.subprocess, "run", return_value=rejected):
        assert not cursor._authenticated("cursor-agent", {})


def test_status_timeout_fails_closed():
    timeout = cursor.subprocess.TimeoutExpired("cursor-agent", 2)
    with patch.object(cursor.subprocess, "run", side_effect=timeout):
        assert not cursor._authenticated("cursor-agent", {})
