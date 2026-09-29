from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from trance.sources import cody


def test_scan_returns_secretless_candidate_for_authenticated_cli(tmp_path: Path):
    observed = {}

    def authenticated(_cli, environ):
        observed.update(environ)
        return True

    with (
        patch.object(cody, "_cli_path", return_value="/usr/bin/cody"),
        patch.object(cody, "_authenticated", side_effect=authenticated),
    ):
        result = cody.scan({"PATH": "/usr/bin"}, tmp_path)

    assert observed["HOME"] == str(tmp_path)
    assert len(result) == 1
    candidate = result[0]
    assert candidate.provider == "sourcegraph-cody"
    assert candidate.auth_kind == "account"
    assert candidate.source == "cody-cli-login"
    assert candidate.model_name == "default"
    assert candidate.secret is None
    assert candidate.config == {
        "command": "/usr/bin/cody",
        "auth_home": str(tmp_path),
        "saved_login": "true",
    }


def test_scan_returns_empty_without_cli(tmp_path: Path):
    with patch.object(cody, "_cli_path", return_value=None):
        assert cody.scan({}, tmp_path) == []


def test_authenticated_checks_documented_whoami_and_omits_env_token():
    observed = {}

    def run(args, **kwargs):
        observed["args"] = args
        observed["env"] = kwargs["env"]
        observed["capture_output"] = kwargs["capture_output"]
        return CompletedProcess(args, 0, "✔ Authenticated as alice on https://sg.example", "")

    with patch.object(cody.subprocess, "run", side_effect=run):
        assert cody._authenticated(
            "/usr/bin/cody",
            {"SRC_ENDPOINT": "https://sg.example", "SRC_ACCESS_TOKEN": "pat-secret"},
        )

    assert observed["args"] == ["/usr/bin/cody", "auth", "whoami"]
    assert "SRC_ENDPOINT" not in observed["env"]
    assert "SRC_ACCESS_TOKEN" not in observed["env"]
    assert observed["capture_output"] is True


def test_authenticated_rejects_exit_zero_without_identity():
    result = CompletedProcess(["cody"], 0, "Already logged out", "")
    with patch.object(cody.subprocess, "run", return_value=result):
        assert not cody._authenticated("cody", {})


def test_authenticated_requires_explicit_identity():
    result = CompletedProcess(["cody"], 1, "Not authenticated", "")
    with patch.object(cody.subprocess, "run", return_value=result):
        assert not cody._authenticated("cody", {})


def test_authenticated_fails_closed_on_timeout():
    timeout = cody.subprocess.TimeoutExpired("cody", 2)
    with patch.object(cody.subprocess, "run", side_effect=timeout):
        assert not cody._authenticated("cody", {})


def test_scan_uses_supplied_home_and_model_override(tmp_path: Path):
    observed = {}

    def authenticated(_cli, environ):
        observed.update(environ)
        return True

    with (
        patch.object(cody, "_cli_path", return_value="/usr/bin/cody"),
        patch.object(cody, "_authenticated", side_effect=authenticated),
    ):
        result = cody.scan(
            {
                "PATH": "/usr/bin",
                "HOME": "/untrusted",
                "TRANCE_MODEL_SOURCEGRAPH_CODY": "enterprise-model",
            },
            tmp_path,
        )

    assert observed["HOME"] == str(tmp_path)
    assert result[0].model_name == "enterprise-model"


def test_cli_path_does_not_fall_back_to_process_path():
    with patch.object(cody.shutil, "which") as which:
        assert cody._cli_path({}) is None
    which.assert_not_called()
