"""Offline end-to-end checks using synthetic credentials and an isolated home."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models import Model

from trance import scan


def _forbid_subprocess(*_args, **_kwargs):
    raise AssertionError("local integration scan attempted a CLI probe")


def _fake_cli_path(tmp_path: Path, name: str) -> Path:
    """Create a PATH-discoverable placeholder on each runner OS."""
    suffix = ".cmd" if os.name == "nt" else ""
    path = tmp_path / "bin" / f"{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o700)
    return path


def _assert_cli_command(actual: object, cli_path: Path, *arguments: str) -> None:
    """Compare a CLI invocation while allowing Windows PATH casing differences."""
    assert isinstance(actual, (list, tuple))
    assert actual
    assert os.path.normcase(os.path.normpath(actual[0])) == os.path.normcase(
        os.path.normpath(str(cli_path))
    )
    assert list(actual[1:]) == list(arguments)


def test_env_and_poe_credentials_build_pydantic_models_and_agents(
    monkeypatch, tmp_path: Path
) -> None:
    openai_key = "synthetic-openai-token-do-not-use"
    poe_key = "synthetic-poe-token-do-not-use"
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    found = scan(
        {
            "HOME": str(tmp_path),
            "PATH": "",
            "OPENAI_API_KEY": openai_key,
            "POE_API_KEY": poe_key,
        },
        tmp_path,
    )

    assert [(item.provider, item.model_name) for item in found] == [
        ("poe", "GPT-5.4"),
        ("openai", "gpt-4o-mini"),
    ]
    assert all(isinstance(item.model, Model) for item in found)
    agents = [Agent(item.model) for item in found]
    assert len(agents) == 2
    assert openai_key not in repr(found)
    assert poe_key not in repr(found)


def test_codex_chatgpt_oauth_cache_builds_model_and_agent(
    monkeypatch, tmp_path: Path
) -> None:
    """The Codex resolver reads the complete OAuth cache from CODEX_HOME."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "synthetic-codex-access-token",
                    "refresh_token": "synthetic-codex-refresh-token",
                    "id_token": "synthetic-codex-id-token",
                    "account_id": "synthetic-codex-account",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    found = scan(
        {"HOME": str(tmp_path), "PATH": "", "CODEX_HOME": str(codex_home)},
        tmp_path,
        providers={"openai-codex"},
    )

    assert [(item.provider, item.source, item.model_name) for item in found] == [
        ("openai-codex", str(codex_home / "auth.json"), "gpt-5.6-luna")
    ]
    assert isinstance(found[0].model, Model)
    assert isinstance(Agent(found[0].model), Agent)
    assert "synthetic-codex" not in repr(found)


def test_grok_and_gemini_environment_keys_build_models_and_agents(
    monkeypatch, tmp_path: Path
) -> None:
    """Grok and both documented Gemini key names remain offline and secret-free."""
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)
    found = scan(
        {
            "HOME": str(tmp_path),
            "PATH": "",
            "XAI_API_KEY": "synthetic-xai-token-do-not-use",
            "GEMINI_API_KEY": "synthetic-gemini-token-do-not-use",
            "GOOGLE_API_KEY": "synthetic-google-token-do-not-use",
        },
        tmp_path,
        providers={"xai", "google"},
    )

    assert [(item.provider, item.source, item.model_name) for item in found] == [
        ("google", "env:GOOGLE_API_KEY", "gemini-2.5-flash"),
        ("google", "env:GEMINI_API_KEY", "gemini-2.5-flash"),
        ("xai", "env:XAI_API_KEY", "grok-4.7"),
    ]
    assert all(isinstance(item.model, Model) for item in found)
    assert all(isinstance(Agent(item.model), Agent) for item in found)
    assert "synthetic-xai-token" not in repr(found)
    assert "synthetic-gemini-token" not in repr(found)
    assert "synthetic-google-token" not in repr(found)


def test_saved_grok_login_builds_agent_with_isolated_cli_environment(
    monkeypatch, tmp_path: Path
) -> None:
    """A saved Grok session is passed to the CLI without exposing its store."""
    grok_home = tmp_path / "grok-state"
    grok_home.mkdir()
    auth_file = grok_home / "auth.json"
    auth_contents = json.dumps(
        {
            "https://auth.x.ai::test": {
                "auth_mode": "oidc",
                "refresh_token": "synthetic-refresh-token",
            }
        }
    )
    auth_file.write_text(auth_contents, encoding="utf-8")
    cli_path = _fake_cli_path(tmp_path, "grok")
    bridge_calls: dict[str, object] = {}

    async def fake_cli(
        command: list[str], environment: dict[str, str], cwd: str, provider: str
    ) -> str:
        bridge_calls.update(command=command, environment=environment, cwd=cwd, provider=provider)
        staged_auth = Path(environment["GROK_AUTH_PATH"])
        bridge_calls["staged_auth_is_file"] = staged_auth.is_file()
        bridge_calls["staged_auth_is_symlink"] = staged_auth.is_symlink()
        bridge_calls["staged_auth_mode"] = (
            staged_auth.stat().st_mode & 0o777 if os.name != "nt" else None
        )
        bridge_calls["staged_auth_contents"] = staged_auth.read_text(encoding="utf-8")
        return '{"text":"hello from fake Grok"}'

    monkeypatch.setenv("HOME", str(tmp_path / "untrusted-home"))
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "untrusted-grok-home"))
    monkeypatch.setattr("trance.cli_model._run_bounded", fake_cli)

    found = scan(
        {"HOME": str(tmp_path), "PATH": str(cli_path.parent), "GROK_HOME": str(grok_home)},
        tmp_path,
        providers={"grok-consumer"},
    )

    assert len(found) == 1
    assert found[0].source == "grok-cli-login"
    result = asyncio.run(Agent(found[0].model).run("Say hello"))

    assert result.output == "hello from fake Grok"
    command = bridge_calls["command"]
    _assert_cli_command(
        command,
        cli_path,
        "--tools",
        "",
        "--no-subagents",
        "--no-memory",
        "--disable-web-search",
        "--max-turns",
        "1",
        "--no-auto-update",
        "--sandbox",
        "read-only",
        "--cwd",
        bridge_calls["cwd"],
        "-p",
        "<user>\nSay hello",
        "--output-format",
        "json",
        "-m",
        found[0].model_name,
    )
    assert all(isinstance(argument, str) for argument in command)  # type: ignore[union-attr]
    environment = bridge_calls["environment"]
    assert environment["GROK_HOME"] != str(grok_home)  # type: ignore[index]
    assert environment["GROK_HOME"].startswith(bridge_calls["cwd"])  # type: ignore[index]
    staged_auth = Path(environment["GROK_AUTH_PATH"])  # type: ignore[arg-type]
    assert staged_auth.parent == Path(environment["GROK_HOME"])  # type: ignore[arg-type]
    assert staged_auth != auth_file
    assert bridge_calls["staged_auth_is_file"] is True
    assert bridge_calls["staged_auth_is_symlink"] is False
    if os.name != "nt":
        assert bridge_calls["staged_auth_mode"] == 0o600
    assert bridge_calls["staged_auth_contents"] == auth_contents
    assert environment["HOME"].startswith(bridge_calls["cwd"])  # type: ignore[index]
    assert environment["HOME"] != str(tmp_path / "untrusted-home")  # type: ignore[index]
    assert bridge_calls["cwd"] != str(Path.cwd())
    assert bridge_calls["provider"] == "grok-consumer"
    assert auth_file.read_text(encoding="utf-8") == auth_contents
    assert auth_contents not in repr(found)


def test_saved_gemini_login_builds_agent_with_terms_warning_and_staged_auth(
    monkeypatch, tmp_path: Path
) -> None:
    gemini_home = tmp_path / ".gemini"
    gemini_home.mkdir()
    auth_file = gemini_home / "oauth_creds.json"
    auth_contents = '{"refresh_token":"synthetic-gemini-refresh"}'
    auth_file.write_text(auth_contents, encoding="utf-8")
    cli_path = _fake_cli_path(tmp_path, "gemini")
    bridge_calls: dict[str, object] = {}

    async def fake_cli(
        command: list[str], environment: dict[str, str], cwd: str, provider: str
    ) -> str:
        bridge_calls.update(command=command, environment=environment, cwd=cwd, provider=provider)
        staged = Path(environment["GEMINI_CLI_HOME"]) / ".gemini" / "oauth_creds.json"
        settings = staged.parent / "settings.json"
        bridge_calls.update(
            staged_is_file=staged.is_file(),
            staged_mode=staged.stat().st_mode & 0o777 if os.name != "nt" else None,
            staged_contents=staged.read_text(encoding="utf-8"),
            settings=json.loads(settings.read_text(encoding="utf-8")),
        )
        return '{"response":"hello from fake Gemini"}'

    monkeypatch.setattr("trance.cli_model._run_bounded", fake_cli)
    with pytest.warns(UserWarning, match="Google's published terms"):
        found = scan(
            {"HOME": str(tmp_path), "PATH": str(cli_path.parent)},
            tmp_path,
            providers={"gemini-cli"},
        )

    assert len(found) == 1
    with pytest.warns(UserWarning, match="Google's published terms"):
        result = asyncio.run(Agent(found[0].model).run("Say hello"))

    assert result.output == "hello from fake Gemini"
    _assert_cli_command(
        bridge_calls["command"],
        cli_path,
        "-e",
        "none",
        "-p",
        "<user>\nSay hello",
        "--output-format",
        "json",
        "--model",
        "auto",
    )
    environment = bridge_calls["environment"]
    assert environment["GEMINI_CLI_HOME"].startswith(bridge_calls["cwd"])  # type: ignore[index]
    assert environment["HOME"].startswith(bridge_calls["cwd"])  # type: ignore[index]
    assert bridge_calls["staged_is_file"] is True
    if os.name != "nt":
        assert bridge_calls["staged_mode"] == 0o600
    assert bridge_calls["staged_contents"] == auth_contents
    assert bridge_calls["settings"] == {
        "security": {"auth": {"selectedType": "oauth-personal"}},
        "tools": {"core": []},
        "mcp": {"allowed": []},
    }
    assert auth_file.read_text(encoding="utf-8") == auth_contents


def test_saved_cody_login_builds_agent_with_keychain_home_and_stdin(
    monkeypatch, tmp_path: Path
) -> None:
    cli_path = _fake_cli_path(tmp_path, "cody")
    bridge_calls: dict[str, object] = {}
    status_calls: list[dict[str, object]] = []

    def fake_status(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        status_calls.append(kwargs)
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="Authenticated as user", stderr=""
        )

    async def fake_cli(
        command: list[str],
        environment: dict[str, str],
        cwd: str,
        provider: str,
        *,
        stdin_text: str | None = None,
    ) -> str:
        bridge_calls.update(
            command=command,
            environment=environment,
            cwd=cwd,
            provider=provider,
            stdin_text=stdin_text,
        )
        return "hello from fake Cody"

    monkeypatch.setattr(subprocess, "run", fake_status)
    monkeypatch.setattr("trance.cli_model._run_bounded", fake_cli)
    found = scan(
        {
            "HOME": str(tmp_path),
            "PATH": str(cli_path.parent),
            "SRC_ACCESS_TOKEN": "must-not-forward",
        },
        tmp_path,
        providers={"sourcegraph-cody"},
    )

    assert len(found) == 1
    result = asyncio.run(Agent(found[0].model).run("Say hello"))
    assert result.output == "hello from fake Cody"
    _assert_cli_command(bridge_calls["command"], cli_path, "chat", "--stdin")
    assert bridge_calls["stdin_text"] == "<user>\nSay hello"
    assert bridge_calls["environment"]["HOME"] == str(tmp_path)  # type: ignore[index]
    assert bridge_calls["environment"]["USERPROFILE"] == str(tmp_path)  # type: ignore[index]
    assert bridge_calls["cwd"] != str(Path.cwd())
    assert status_calls[0]["env"]["HOME"] == str(tmp_path)  # type: ignore[index]
    assert "SRC_ACCESS_TOKEN" not in status_calls[0]["env"]  # type: ignore[operator]


def test_saved_opencode_provider_builds_agent_with_deny_all_config(
    monkeypatch, tmp_path: Path
) -> None:
    data_home = tmp_path / "data"
    auth_file = data_home / "opencode" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_contents = json.dumps(
        {
            "anthropic": {
                "type": "oauth",
                "refresh": "synthetic-refresh",
                "access": "synthetic-access",
                "expires": 4102444800000,
            }
        }
    )
    auth_file.write_text(auth_contents, encoding="utf-8")
    cli_path = _fake_cli_path(tmp_path, "opencode")
    bridge_calls: dict[str, object] = {}

    async def fake_cli(
        command: list[str], environment: dict[str, str], cwd: str, provider: str
    ) -> str:
        bridge_calls.update(command=command, environment=environment, cwd=cwd, provider=provider)
        config = Path(environment["OPENCODE_CONFIG"])
        bridge_calls["config"] = json.loads(config.read_text(encoding="utf-8"))
        staged = Path(environment["XDG_DATA_HOME"]) / "opencode" / "auth.json"
        bridge_calls.update(
            staged_mode=staged.stat().st_mode & 0o777 if os.name != "nt" else None,
            staged_contents=staged.read_text(),
        )
        return '{"type":"text","part":{"type":"text","text":"hello from fake OpenCode"}}'

    monkeypatch.setattr("trance.cli_model._run_bounded", fake_cli)
    found = scan(
        {"HOME": str(tmp_path), "PATH": str(cli_path.parent), "XDG_DATA_HOME": str(data_home)},
        tmp_path,
    )

    assert len(found) == 1
    assert found[0].provider == "opencode:anthropic"
    assert found[0].model_name == "anthropic/claude-haiku-4-5-20251001"
    result = asyncio.run(Agent(found[0].model).run("Say hello"))
    assert result.output == "hello from fake OpenCode"
    _assert_cli_command(
        bridge_calls["command"],
        cli_path,
        "--pure",
        "run",
        "--format",
        "json",
        "--model",
        "anthropic/claude-haiku-4-5-20251001",
        "<user>\nSay hello",
    )
    assert bridge_calls["config"] == {"permission": {"*": "deny"}, "plugin": [], "mcp": {}}
    if os.name != "nt":
        assert bridge_calls["staged_mode"] == 0o600
    assert bridge_calls["staged_contents"] == auth_contents
    assert auth_file.read_text(encoding="utf-8") == auth_contents


def test_qwen_and_minimax_plans_use_documented_endpoints(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)
    found = scan(
        {
            "HOME": str(tmp_path),
            "PATH": "",
            "BAILIAN_CODING_PLAN_API_KEY": "sk-sp-synthetic-qwen-key",
            "MINIMAX_API_KEY": "sk-cp-synthetic-minimax-key",
        },
        tmp_path,
        providers={"qwen-code", "minimax-coding-plan"},
    )

    models = {item.provider: item.model for item in found}
    assert set(models) == {"qwen-code", "minimax-coding-plan"}
    assert str(models["qwen-code"].provider.client.base_url).rstrip("/") == (
        "https://coding.dashscope.aliyuncs.com/v1"
    )
    assert str(models["minimax-coding-plan"].provider.client.base_url).rstrip("/") == (
        "https://api.minimax.io/v1"
    )


def test_compatible_and_cloud_api_credentials_build_pydantic_models(
    monkeypatch, tmp_path: Path
) -> None:
    """Exercise both wildcard discovery sources through the public scan API."""
    moonshot_key = "synthetic-moonshot-token-do-not-use"
    nebius_key = "synthetic-nebius-token-do-not-use"
    azure_key = "synthetic-azure-token-do-not-use"
    cloudflare_key = "synthetic-cloudflare-token-do-not-use"
    account_id = "a" * 32
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    found = scan(
        {
            "HOME": str(tmp_path),
            "PATH": "",
            "MOONSHOT_API_KEY": moonshot_key,
            "NEBIUS_API_KEY": nebius_key,
            "AZURE_OPENAI_API_KEY": azure_key,
            "AZURE_OPENAI_ENDPOINT": "https://synthetic.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT": "synthetic-deployment",
            "CLOUDFLARE_ACCOUNT_ID": account_id,
            "CLOUDFLARE_API_KEY": cloudflare_key,
        },
        tmp_path,
        providers={"moonshot", "nebius", "azure-openai", "cloudflare"},
    )

    assert [(item.provider, item.model_name) for item in found] == [
        ("moonshot", "kimi-k2.6"),
        ("nebius", "Qwen/Qwen3.5-397B-A17B"),
        ("azure-openai", "synthetic-deployment"),
        ("cloudflare", "@cf/meta/llama-3.1-8b-instruct"),
    ]
    assert all(isinstance(item.model, Model) for item in found)
    assert all(isinstance(Agent(item.model), Agent) for item in found)
    assert moonshot_key not in repr(found)
    assert nebius_key not in repr(found)
    assert azure_key not in repr(found)
    assert cloudflare_key not in repr(found)


def test_provider_filter_selects_one_compatible_and_one_cloud_provider(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)
    environment = {
        "HOME": str(tmp_path),
        "PATH": "",
        "MOONSHOT_API_KEY": "synthetic-moonshot-token",
        "NEBIUS_API_KEY": "synthetic-nebius-token",
        "DASHSCOPE_API_KEY": "synthetic-dashscope-token",
        "CLOUDFLARE_ACCOUNT_ID": "b" * 32,
        "CLOUDFLARE_API_KEY": "synthetic-cloudflare-token",
    }

    found = scan(
        environment,
        tmp_path,
        providers={"nebius", "dashscope"},
    )

    assert [(item.provider, item.source) for item in found] == [
        ("nebius", "env:NEBIUS_API_KEY"),
        ("dashscope", "env:DASHSCOPE_API_KEY"),
    ]
    assert all(isinstance(item.model, Model) for item in found)


def test_saved_claude_login_builds_agent_and_uses_text_only_cli_bridge(
    monkeypatch, tmp_path: Path
) -> None:
    credentials_dir = tmp_path / ".claude"
    credentials_dir.mkdir()
    (credentials_dir / ".credentials.json").write_text("")
    cli_path = _fake_cli_path(tmp_path, "claude")
    status_calls: list[tuple[object, ...]] = []
    bridge_calls: dict[str, object] = {}

    def fake_status(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        status_calls.append(args)
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout=(
                '{"loggedIn":true,"apiProvider":"firstParty",'
                '"authMethod":"claude.ai"}'
            ),
            stderr="",
        )

    async def fake_cli(
        command: list[str], environment: dict[str, str], cwd: str, provider: str
    ) -> str:
        bridge_calls.update(command=command, environment=environment, cwd=cwd, provider=provider)
        return '{"result":"hello from fake Claude"}'

    monkeypatch.setattr(subprocess, "run", fake_status)
    monkeypatch.setattr("trance.sources.claude_code.shutil.which", lambda name, path: str(cli_path))
    monkeypatch.setattr("trance.cli_model._run_bounded", fake_cli)

    found = scan(
        {"HOME": str(tmp_path), "PATH": str(cli_path.parent)},
        tmp_path,
        providers={"claude-code"},
    )

    assert len(found) == 1
    assert found[0].source == "claude-cli-login"
    assert found[0].model_name == "sonnet"
    result = asyncio.run(Agent(found[0].model).run("Say hello"))

    assert result.output == "hello from fake Claude"
    assert len(status_calls) == 1
    assert len(status_calls[0]) == 1
    _assert_cli_command(status_calls[0][0], cli_path, "auth", "status")
    command = bridge_calls["command"]
    assert isinstance(command, list)
    _assert_cli_command(command, cli_path, *command[1:])
    assert "--safe-mode" in command  # type: ignore[operator]
    assert "--tools" in command  # type: ignore[operator]
    assert command[command.index("--tools") + 1] == ""  # type: ignore[union-attr]
    assert "--disallowedTools" in command  # type: ignore[operator]
    assert command[command.index("--disallowedTools") + 1] == "mcp__*"  # type: ignore[union-attr]
    assert bridge_calls["provider"] == "claude-code"


def test_provider_filter_excludes_other_sources_and_cli_probes(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)
    found = scan(
        {
            "HOME": str(tmp_path),
            "PATH": "",
            "OPENAI_API_KEY": "synthetic-openai-token",
            "POE_API_KEY": "synthetic-poe-token",
            "ANTHROPIC_API_KEY": "synthetic-anthropic-token",
            "GH_TOKEN": "synthetic-github-token",
            "CLAUDE_CODE_OAUTH_TOKEN": "synthetic-claude-token",
        },
        tmp_path,
        providers={"openai"},
    )

    assert [(item.provider, item.source) for item in found] == [
        ("openai", "env:OPENAI_API_KEY")
    ]
