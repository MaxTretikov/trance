import json
from pathlib import Path
from subprocess import CompletedProcess

from trance.adapters import pydantic_ai
from trance.sources import claude_code, cody, copilot
from trance.types import Candidate


def test_copilot_extracts_saved_cli_token_with_sanitized_environment(monkeypatch, tmp_path):
    observed = {}

    def run(args, **kwargs):
        observed.update(args=args, **kwargs)
        return CompletedProcess(args, 0, stdout="saved-token\n", stderr="")

    monkeypatch.setattr(copilot.subprocess, "run", run)
    monkeypatch.setenv("GH_TOKEN", "ambient-token")
    candidate = Candidate(
        "github-copilot",
        "subscription",
        "github-cli",
        "gpt-5.4",
        config={
            "resolver": "command",
            "executable": "/custom/bin/gh",
            "auth_home": str(tmp_path),
        },
    )

    assert copilot.extract_credentials(candidate) == {"token": "saved-token"}
    assert observed["args"] == ["/custom/bin/gh", "auth", "token"]
    assert observed["env"]["HOME"] == str(tmp_path)
    assert "GH_TOKEN" not in observed["env"]
    assert "GITHUB_TOKEN" not in observed["env"]


def test_copilot_extractor_uses_explicit_environment_and_home(monkeypatch, tmp_path):
    observed = {}

    def run(args, **kwargs):
        observed.update(kwargs)
        return CompletedProcess(args, 0, stdout="controlled-token\n", stderr="")

    monkeypatch.setattr(copilot.subprocess, "run", run)
    candidate = Candidate(
        "github-copilot",
        "subscription",
        "github-cli",
        "gpt-5.4",
        config={"resolver": "command", "executable": "/custom/bin/gh"},
    )

    assert copilot.extract_credentials(
        candidate,
        environ={"PATH": "/controlled", "GH_TOKEN": "ambient-token"},
        home=tmp_path,
    ) == {"token": "controlled-token"}
    assert observed["env"] == {"PATH": "/controlled", "HOME": str(tmp_path)}


def test_claude_extracts_only_known_oauth_access_token(tmp_path):
    credentials = tmp_path / ".claude" / ".credentials.json"
    credentials.parent.mkdir()
    credentials.write_text(
        json.dumps({
            "claudeAiOauth": {
                "accessToken": "claude-access-token",
                "refreshToken": "refresh-token",
            },
            "unrelated": "do not return",
        }),
        encoding="utf-8",
    )
    candidate = Candidate(
        "claude-code",
        "subscription",
        "claude-cli-login",
        "sonnet",
        config={"auth_home": str(tmp_path)},
    )

    assert claude_code.extract_credentials(candidate) == {"token": "claude-access-token"}


def test_claude_rejects_unknown_credentials_shape(tmp_path):
    credentials = tmp_path / ".claude" / ".credentials.json"
    credentials.parent.mkdir()
    credentials.write_text('{"token": "do-not-return"}', encoding="utf-8")
    candidate = Candidate(
        "claude-code", "subscription", "claude-cli-login", "sonnet",
        config={"auth_home": str(tmp_path)},
    )

    assert claude_code.extract_credentials(candidate) == {}


def test_claude_rejects_symlinked_credentials_file(tmp_path):
    real = tmp_path / "real-credentials.json"
    real.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "do-not-follow"}}),
        encoding="utf-8",
    )
    credentials = tmp_path / ".claude" / ".credentials.json"
    credentials.parent.mkdir()
    credentials.symlink_to(real)
    candidate = Candidate(
        "claude-code", "subscription", "claude-cli-login", "sonnet",
        config={"auth_home": str(tmp_path)},
    )

    assert claude_code.extract_credentials(candidate) == {}


def test_cody_does_not_export_secure_storage_credentials():
    candidate = Candidate("sourcegraph-cody", "account", "cody-cli-login", "default")
    assert cody.extract_credentials(candidate) == {}


def test_pydantic_adapter_reuses_copilot_source_resolver(monkeypatch):
    candidate = Candidate(
        "github-copilot",
        "subscription",
        "github-cli",
        "gpt-5.4",
        config={"resolver": "command"},
    )
    calls = []

    monkeypatch.setattr(
        copilot,
        "extract_credentials",
        lambda value, **kwargs: calls.append((value, kwargs)) or {"token": "resolved-token"},
    )
    resolved = pydantic_ai._resolve_copilot(candidate, {}, Path("/tmp"))

    assert calls == [(candidate, {"environ": {}, "home": Path("/tmp")})]
    assert resolved.secret == "resolved-token"
    assert resolved.source == "resolved:github-cli"
    assert resolved.config == {}
