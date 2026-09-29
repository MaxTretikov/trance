import json
from pathlib import Path

import pytest

from trance.sources import opencode


def _auth(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / ".local" / "share" / "opencode" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_scan_emits_valid_provider_candidates_without_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    path = _auth(tmp_path, {
        "openai": {"type": "api", "key": "secret-openai"},
        "anthropic": {"type": "wellknown", "key": "key", "token": "secret-token"},
        "google": {"type": "oauth", "refresh": "refresh", "access": "access", "expires": 1},
        "groq": {"type": "api", "key": "secret-groq"},
        "xai": {"type": "api", "key": "secret-xai"},
        "unknown": {"type": "api", "key": "secret"},
    })
    monkeypatch.setattr(opencode.shutil, "which", lambda name, path: "/bin/opencode")
    candidates = opencode.scan({"PATH": "/bin"}, tmp_path)
    assert [(item.provider, item.auth_kind, item.model_name) for item in candidates] == [
        ("opencode:openai", "api_key", "openai/gpt-4o-mini"),
        ("opencode:anthropic", "account", "anthropic/claude-haiku-4-5-20251001"),
        ("opencode:google", "account", "google/gemini-2.5-flash"),
        ("opencode:groq", "api_key", "groq/openai/gpt-oss-120b"),
        ("opencode:xai", "api_key", "xai/grok-4.7"),
    ]
    assert all(item.secret is None for item in candidates)
    assert candidates[0].source == "opencode-auth"
    assert candidates[0].config == {
        "integration": "opencode-cli", "command": str(Path("/bin/opencode").absolute()),
        "auth_file": str(path.absolute()), "vendor_provider": "openai", "saved_login": "true",
    }
    assert "secret-openai" not in repr(candidates)


def test_model_override_must_be_vendor_qualified(monkeypatch, tmp_path: Path) -> None:
    _auth(tmp_path, {"openai": {"type": "api", "key": "secret"}})
    monkeypatch.setattr(opencode.shutil, "which", lambda name, path: "/bin/opencode")
    assert opencode.scan(
        {"PATH": "/bin", "TRANCE_MODEL_OPENCODE_OPENAI": "gpt-custom"}, tmp_path
    ) == []
    [candidate] = opencode.scan(
        {"PATH": "/bin", "TRANCE_MODEL_OPENCODE_OPENAI": "openai/custom"}, tmp_path
    )
    assert candidate.model_name == "openai/custom"


def test_scan_requires_cli_and_rejects_symlinked_paths(monkeypatch, tmp_path: Path) -> None:
    path = _auth(tmp_path, {"openai": {"type": "api", "key": "secret"}})
    monkeypatch.setattr(opencode.shutil, "which", lambda name, path: None)
    assert opencode.scan({"PATH": "/bin"}, tmp_path) == []
    monkeypatch.setattr(opencode.shutil, "which", lambda name, path: "/bin/opencode")
    target = tmp_path / "real-data"
    target.mkdir()
    path.unlink()
    path.parent.rmdir()
    (tmp_path / ".local" / "share").rmdir()
    (tmp_path / ".local").rmdir()
    try:
        (tmp_path / ".local").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this runner")
    assert opencode.scan({"PATH": "/bin"}, tmp_path) == []


def test_scan_uses_xdg_data_home_and_ignores_invalid_entries(monkeypatch, tmp_path: Path) -> None:
    data_home = tmp_path / "data"
    path = data_home / "opencode" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "openai": {"type": "api", "key": ""},
        "bad/vendor": {"type": "api", "key": "key"},
        "openrouter": {"type": "api", "key": "key"},
    }), encoding="utf-8")
    monkeypatch.setattr(opencode.shutil, "which", lambda name, path: "/bin/opencode")
    [candidate] = opencode.scan({"PATH": "/bin", "XDG_DATA_HOME": str(data_home)}, tmp_path)
    assert candidate.provider == "opencode:openrouter"
    assert candidate.model_name == "openrouter/openai/gpt-4o-mini"
