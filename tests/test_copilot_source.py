from subprocess import CompletedProcess

from trance.sources import copilot


def test_scan_reads_supported_environment_tokens_without_duplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(copilot.shutil, "which", lambda *args, **kwargs: None)
    candidates = copilot.scan(
        {
            "PATH": "",
            "GITHUB_COPILOT_API_KEY": "copilot-secret",
            "COPILOT_GITHUB_TOKEN": "copilot-secret",
            "GITHUB_COPILOT_API_TOKEN": "copilot-api-secret",
            "GH_TOKEN": "generic-gh-secret",
            "GITHUB_TOKEN": " ",
        },
        home=tmp_path,
    )

    assert [(item.source, item.secret) for item in candidates] == [
        ("env:GITHUB_COPILOT_API_KEY", "copilot-secret"),
        ("env:GITHUB_COPILOT_API_TOKEN", "copilot-api-secret"),
    ]
    assert all(item.provider == "github-copilot" for item in candidates)
    assert all(item.auth_kind == "subscription" for item in candidates)


def test_scan_applies_trimmed_model_override(monkeypatch, tmp_path):
    monkeypatch.setattr(copilot.shutil, "which", lambda *args, **kwargs: None)
    [candidate] = copilot.scan(
        {
            "PATH": "",
            "GITHUB_COPILOT_API_KEY": "copilot-secret",
            "TRANCE_MODEL_GITHUB_COPILOT": "  claude-sonnet  ",
        },
        home=tmp_path,
    )
    assert candidate.model_name == "claude-sonnet"

    [default_candidate] = copilot.scan(
        {
            "PATH": "",
            "GITHUB_COPILOT_API_KEY": "copilot-secret",
            "TRANCE_MODEL_GITHUB_COPILOT": "  ",
        },
        home=tmp_path,
    )
    assert default_candidate.model_name == "gpt-5.4"


def test_scan_returns_lazy_github_cli_resolver(monkeypatch, tmp_path):
    monkeypatch.setattr(copilot.shutil, "which", lambda *args, **kwargs: "/usr/bin/gh")
    observed = {}

    def run(args, **kwargs):
        observed["args"] = args
        observed["env"] = kwargs["env"]
        return CompletedProcess(args, 0)

    monkeypatch.setattr(copilot.subprocess, "run", run)
    candidates = copilot.scan(
        {
            "PATH": "/usr/bin",
            "HOME": "/different-home",
            "GH_TOKEN": "generic",
            "GITHUB_TOKEN": "generic",
            "GH_HOST": "ghe.example.test",
            "GH_ENTERPRISE_TOKEN": "enterprise",
            "GITHUB_ENTERPRISE_TOKEN": "enterprise-2",
        },
        home=tmp_path,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "github-cli"
    assert candidate.secret is None
    assert candidate.config == {
        "resolver": "command",
        "executable": "gh",
        "args": "auth token",
    }
    assert observed["args"] == ["/usr/bin/gh", "auth", "status"]
    assert observed["env"]["HOME"] == str(tmp_path)
    assert "GH_TOKEN" not in observed["env"]
    assert "GITHUB_TOKEN" not in observed["env"]
    assert "GH_HOST" not in observed["env"]
    assert "GH_ENTERPRISE_TOKEN" not in observed["env"]
    assert "GITHUB_ENTERPRISE_TOKEN" not in observed["env"]


def test_scan_ignores_missing_tokens_and_unauthenticated_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(copilot.shutil, "which", lambda *args, **kwargs: None)
    assert copilot.scan({"PATH": ""}, home=tmp_path) == []


def test_scan_does_not_return_unauthed_cli_resolver(monkeypatch, tmp_path):
    monkeypatch.setattr(copilot.shutil, "which", lambda *args, **kwargs: "/usr/bin/gh")
    monkeypatch.setattr(
        copilot.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args, 1),
    )
    assert copilot.scan({"PATH": "/usr/bin"}, home=tmp_path) == []
