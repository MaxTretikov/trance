from pathlib import Path
from unittest.mock import patch

from trance.sources import tabnine


def test_scan_returns_candidate_for_explicit_pat_and_cli(tmp_path: Path):
    with patch.object(tabnine.shutil, "which", return_value="/usr/bin/tabnine"):
        candidates = tabnine.scan(
            {"TABNINE_TOKEN": "pat-value", "PATH": "/usr/bin"}, tmp_path
        )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "tabnine"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == "env:TABNINE_TOKEN"
    assert candidate.model_name == "tabnine/default"
    assert candidate.secret == "pat-value"
    assert candidate.config["command"] == "/usr/bin/tabnine"
    assert candidate.config["credential_env"] == "TABNINE_TOKEN"
    assert candidate.config["integration"] == "tabnine_cli"
    assert "pat-value" not in repr(candidate)


def test_scan_ignores_blank_pat(tmp_path: Path):
    with patch.object(tabnine.shutil, "which") as which:
        assert tabnine.scan({"TABNINE_TOKEN": "  "}, tmp_path) == []
    which.assert_not_called()


def test_scan_requires_cli(tmp_path: Path):
    with patch.object(tabnine.shutil, "which", return_value=None):
        assert tabnine.scan({"TABNINE_TOKEN": "pat-value"}, tmp_path) == []


def test_scan_does_not_read_home_credentials(tmp_path: Path):
    with patch.object(tabnine.shutil, "which", return_value="/usr/bin/tabnine"):
        assert tabnine.scan({"PATH": "/usr/bin"}, tmp_path) == []
