from __future__ import annotations

import asyncio
import os
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from trance.cli_model import (
    _MAX_AUTH_BYTES,
    CLIModelError,
    UnsupportedCLIModelError,
    _parse_output,
    _promote_gemini_refresh,
    _promote_grok_refresh,
    _text_from_messages,
    build_cli_model,
)
from trance.types import Candidate


class FakeModel:
    def __init__(self) -> None:
        pass


class FakeTextPart:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModelResponse:
    def __init__(self, *, parts: list[object], model_name: str) -> None:
        self.parts = parts
        self.model_name = model_name


class UserPromptPart:
    def __init__(self, content: str) -> None:
        self.content = content


class ModelRequest:
    def __init__(self, content: str) -> None:
        self.parts = [UserPromptPart(content)]


def fake_pydantic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("pydantic_ai")
    models = types.ModuleType("pydantic_ai.models")
    models.Model = FakeModel  # type: ignore[attr-defined]
    messages = types.ModuleType("pydantic_ai.messages")
    messages.ModelResponse = FakeModelResponse  # type: ignore[attr-defined]
    messages.TextPart = FakeTextPart  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pydantic_ai", package)
    monkeypatch.setitem(sys.modules, "pydantic_ai.models", models)
    monkeypatch.setitem(sys.modules, "pydantic_ai.messages", messages)


@pytest.mark.parametrize(
    ("provider", "stdout", "expected"),
    [
        ("claude-code", '{"result":"hello"}', "hello"),
        ("grok-consumer", '{"text":"hello"}', "hello"),
        ("gemini-cli", '{"response":"hello"}', "hello"),
        ("sourcegraph-cody", "hello\n", "hello"),
    ],
)
def test_parse_vendor_output(provider: str, stdout: str, expected: str) -> None:
    assert _parse_output(provider, stdout) == expected


def test_parse_opencode_jsonl_text_events_and_rejects_non_text() -> None:
    output = (
        '{"type":"text","part":{"type":"text","text":"hello"}}\n'
        '{"type":"text","part":{"type":"text","text":" world"}}\n'
    )
    assert _parse_output("opencode:openai", output) == "hello world"
    with pytest.raises(CLIModelError, match="unsupported 'tool' event"):
        _parse_output("opencode:openai", '{"type":"tool","part":{}}')
    with pytest.raises(CLIModelError, match="malformed JSONL"):
        _parse_output("opencode:openai", "not-json")


def test_malformed_structured_output_fails_closed() -> None:
    with pytest.raises(CLIModelError, match="malformed JSON"):
        _parse_output("claude-code", "not json")


def test_unknown_provider_is_unsupported() -> None:
    with pytest.raises(UnsupportedCLIModelError):
        build_cli_model(Candidate("unknown", "subscription", "test", "m"))
    with pytest.raises(UnsupportedCLIModelError):
        build_cli_model(Candidate("kimi-code", "subscription", "test", "kimi-for-coding"))


def test_claude_adapter_calls_cli_without_shell_and_keeps_explicit_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pydantic_ai(monkeypatch)
    captured: dict[str, object] = {}

    async def run(command: list[str], environment: dict[str, str], cwd: str, provider: str) -> str:
        captured["command"] = command
        captured["environment"] = environment
        captured["cwd"] = cwd
        captured["provider"] = provider
        return '{"result":"answer"}'

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    candidate = Candidate(
        "claude-code",
        "subscription",
        "env:CLAUDE_CODE_OAUTH_TOKEN",
        "claude-sonnet",
        secret="test-token",
        config={"credential_env": "CLAUDE_CODE_OAUTH_TOKEN", "command": "/fake/claude"},
    )
    model = build_cli_model(candidate)
    request = ModelRequest("hello")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "higher-precedence-api-key")
    monkeypatch.setenv("HOME", "/untrusted/home")
    monkeypatch.setenv("CLAUDE_CODE_SHELL_PREFIX", "untrusted hook")
    response = asyncio.run(
        model.request(
            [request],
            None,
            SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
        )
    )

    assert response.parts[0].content == "answer"
    assert captured["command"] == [
        "/fake/claude",
        "--safe-mode",
        "--tools",
        "",
        "--no-session-persistence",
        "--disallowedTools",
        "mcp__*",
        "-p",
        "--output-format",
        "json",
        "--model",
        "claude-sonnet",
        "<user>\nhello",
    ]
    assert captured["environment"]["CLAUDE_CODE_OAUTH_TOKEN"] == "test-token"  # type: ignore[index]
    assert set(captured["environment"]) <= {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "HOME",
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "CLAUDE_CODE_SIMPLE",
        "CLAUDE_CODE_OAUTH_TOKEN",
    }
    assert captured["environment"]["HOME"] == captured["cwd"]  # type: ignore[index]
    assert captured["environment"]["CLAUDE_CODE_SIMPLE"] == "1"  # type: ignore[index]
    assert "--tools" in captured["command"]  # type: ignore[operator]
    assert captured["command"][captured["command"].index("--tools") + 1] == ""  # type: ignore[union-attr]
    assert captured["cwd"] != "/home/maxtretikov/Projects/Code/Academic/trance"


def test_claude_saved_login_preserves_auth_home_and_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pydantic_ai(monkeypatch)
    captured: dict[str, object] = {}

    async def run(command: list[str], environment: dict[str, str], cwd: str, provider: str) -> str:
        captured["command"] = command
        captured["environment"] = environment
        captured["cwd"] = cwd
        captured["provider"] = provider
        return '{"result":"answer"}'

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/untrusted/changed-config")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-forward")
    command = str(tmp_path / "claude")
    auth_home = tmp_path / "claude-home"
    config_dir = tmp_path / "claude-config"
    candidate = Candidate(
        "claude-code",
        "subscription",
        "claude-cli-login",
        "claude-sonnet",
        config={
            "command": command,
            "auth_home": str(auth_home),
            "config_dir": str(config_dir),
            "saved_login": "true",
        },
    )

    response = asyncio.run(
        build_cli_model(candidate).request(
            [ModelRequest("hello")],
            None,
            SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
        )
    )

    assert response.parts[0].content == "answer"
    assert captured["command"] == [
        command,
        "--safe-mode",
        "--tools",
        "",
        "--no-session-persistence",
        "--disallowedTools",
        "mcp__*",
        "-p",
        "--output-format",
        "json",
        "--model",
        "claude-sonnet",
        "<user>\nhello",
    ]
    environment = captured["environment"]
    assert environment["HOME"] == str(auth_home)  # type: ignore[index]
    assert environment["USERPROFILE"] == str(auth_home)  # type: ignore[index]
    assert environment["CLAUDE_CONFIG_DIR"] == str(config_dir)  # type: ignore[index]
    assert "ANTHROPIC_API_KEY" not in environment  # type: ignore[operator]
    assert "CLAUDE_CODE_SIMPLE" not in environment  # type: ignore[operator]
    assert environment["HOME"] != captured["cwd"]  # type: ignore[index]


def test_cody_saved_login_uses_stdin_and_isolates_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pydantic_ai(monkeypatch)
    captured: dict[str, object] = {}

    async def run(
        command: list[str],
        environment: dict[str, str],
        cwd: str,
        provider: str,
        *,
        stdin_text: str | None = None,
    ) -> str:
        captured["command"] = command
        captured["environment"] = environment
        captured["cwd"] = cwd
        captured["provider"] = provider
        captured["stdin_text"] = stdin_text
        return "answer from Cody\n"

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    command = str(tmp_path / "cody")
    auth_home = tmp_path / "cody-home"
    candidate = Candidate(
        "sourcegraph-cody",
        "account",
        "cody-cli auth whoami",
        "default",
        config={
            "command": command,
            "auth_home": str(auth_home),
            "saved_login": "true",
        },
    )
    for name in (
        "SRC_ACCESS_TOKEN",
        "SRC_ENDPOINT",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR",
    ):
        monkeypatch.setenv(name, f"untrusted-{name}")

    response = asyncio.run(
        build_cli_model(candidate).request(
            [ModelRequest("hello")],
            None,
            SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
        )
    )

    assert response.parts[0].content == "answer from Cody"
    assert captured["command"] == [command, "chat", "--stdin"]
    assert captured["stdin_text"] == "<user>\nhello"  # type: ignore[comparison-overlap]
    environment = captured["environment"]
    assert environment["HOME"] == str(auth_home)  # type: ignore[index]
    assert environment["USERPROFILE"] == str(auth_home)  # type: ignore[index]
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == "untrusted-DBUS_SESSION_BUS_ADDRESS"  # type: ignore[index]
    assert environment["XDG_RUNTIME_DIR"] == "untrusted-XDG_RUNTIME_DIR"  # type: ignore[index]
    assert "SRC_ACCESS_TOKEN" not in environment  # type: ignore[operator]
    assert "SRC_ENDPOINT" not in environment  # type: ignore[operator]
    assert "OPENAI_API_KEY" not in environment  # type: ignore[operator]
    assert captured["cwd"] != str(auth_home)


def test_cody_explicit_model_is_passed_to_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pydantic_ai(monkeypatch)
    captured: dict[str, object] = {}

    async def run(
        command: list[str],
        environment: dict[str, str],
        cwd: str,
        provider: str,
        *,
        stdin_text: str | None = None,
    ) -> str:
        captured["command"] = command
        return "answer"

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    command = str(tmp_path / "cody")
    candidate = Candidate(
        "sourcegraph-cody",
        "account",
        "cody-cli auth whoami",
        "claude-sonnet",
        config={
            "command": command,
            "auth_home": str(tmp_path / "cody-home"),
            "saved_login": "true",
        },
    )
    asyncio.run(
        build_cli_model(candidate).request(
            [ModelRequest("hello")],
            None,
            SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
        )
    )
    assert captured["command"] == [command, "chat", "--stdin", "--model", "claude-sonnet"]


def test_grok_saved_login_isolated_and_promotes_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pydantic_ai(monkeypatch)
    auth_home = tmp_path / "grok-home"
    auth_home.mkdir()
    auth_file = tmp_path / "auth.json"
    auth_file.write_bytes(b'{"refresh_token":"old-secret"}')
    captured: dict[str, object] = {}

    async def run(command: list[str], environment: dict[str, str], cwd: str, provider: str) -> str:
        captured["command"] = command
        captured["environment"] = environment
        captured["cwd"] = cwd
        staged = Path(environment["GROK_AUTH_PATH"])
        assert staged.is_file()
        assert not staged.is_symlink()
        staged.write_bytes(b'{"refresh_token":"new-secret"}')
        return '{"text":"answer"}'

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    candidate = Candidate(
        "grok-consumer",
        "subscription",
        "grok-cli-login",
        "grok-build",
        config={
            "integration": "grok-cli",
            "command": "/fake/grok",
            "auth_home": str(auth_home),
            "auth_file": str(auth_file),
            "saved_login": "true",
        },
    )
    monkeypatch.setenv("GROK_API_KEY", "must-not-forward")
    monkeypatch.setenv("GROK_HOME", "/untrusted/grok-home")

    response = asyncio.run(
        build_cli_model(candidate).request(
            [ModelRequest("hello")],
            None,
            SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
        )
    )

    assert response.parts[0].content == "answer"
    command = captured["command"]
    assert command[:12] == [
        "/fake/grok",
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
    ]
    assert command[12] == captured["cwd"]  # type: ignore[index]
    assert command[13:] == ["-p", "<user>\nhello", "--output-format", "json", "-m", "grok-build"]
    environment = captured["environment"]
    assert environment["GROK_HOME"] != str(auth_home)  # type: ignore[index]
    assert environment["HOME"] != str(auth_home)  # type: ignore[index]
    assert environment["GROK_AUTH_PATH"] != str(auth_file)  # type: ignore[index]
    assert "GROK_API_KEY" not in environment  # type: ignore[operator]
    assert "GROK_HOME" in environment  # type: ignore[operator]
    assert auth_file.read_bytes() == b'{"refresh_token":"new-secret"}'
    assert b"new-secret" not in repr(captured).encode()


def test_grok_unchanged_staged_auth_is_not_replaced(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    staged_file = tmp_path / "staged-auth.json"
    auth_file.write_bytes(b'{"refresh_token":"same"}')
    staged_file.write_bytes(auth_file.read_bytes())
    original = auth_file.lstat()
    staged_original = staged_file.lstat()

    _promote_grok_refresh(staged_file, auth_file, original, staged_original)

    assert auth_file.lstat().st_ino == original.st_ino


def test_grok_oversized_refresh_is_rejected(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    staged_file = tmp_path / "staged-auth.json"
    auth_file.write_bytes(b"old")
    staged_file.write_bytes(b"x" * (_MAX_AUTH_BYTES + 1))

    with pytest.raises(CLIModelError, match="too large"):
        _promote_grok_refresh(
            staged_file,
            auth_file,
            auth_file.lstat(),
            staged_file.lstat(),
        )


def test_gemini_saved_login_isolated_and_promotes_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pydantic_ai(monkeypatch)
    auth_home = tmp_path / ".gemini"
    auth_home.mkdir()
    auth_file = auth_home / "oauth_creds.json"
    auth_file.write_bytes(b'{"refresh_token":"old-secret"}')
    captured: dict[str, object] = {}

    async def run(command: list[str], environment: dict[str, str], cwd: str, provider: str) -> str:
        captured["command"] = command
        captured["environment"] = environment
        captured["cwd"] = cwd
        staged = Path(environment["GEMINI_CLI_HOME"]) / ".gemini" / "oauth_creds.json"
        assert staged.is_file()
        if os.name != "nt":
            assert staged.stat().st_mode & 0o777 == 0o600
        settings = Path(environment["GEMINI_CLI_HOME"]) / ".gemini" / "settings.json"
        assert settings.read_text(encoding="utf-8") == (
            '{"security": {"auth": {"selectedType": "oauth-personal"}}, '
            '"tools": {"core": []}, "mcp": {"allowed": []}}'
        )
        staged.write_bytes(b'{"refresh_token":"new-secret"}')
        return '{"response":"answer"}'

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    candidate = Candidate(
        "gemini-cli",
        "account",
        "gemini-cli-login",
        "auto",
        config={
            "command": "/fake/gemini",
            "auth_home": str(auth_home),
            "auth_file": str(auth_file),
            "saved_login": "true",
        },
    )
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-forward")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/untrusted/key.json")
    with pytest.warns(UserWarning, match="published terms"):
        response = asyncio.run(
            build_cli_model(candidate).request(
                [ModelRequest("hello")],
                None,
                SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
            )
        )

    assert response.parts[0].content == "answer"
    assert captured["command"] == [
        "/fake/gemini",
        "-e",
        "none",
        "-p",
        "<user>\nhello",
        "--output-format",
        "json",
        "--model",
        "auto",
    ]
    environment = captured["environment"]
    assert environment["GEMINI_CLI_HOME"] != str(auth_home)  # type: ignore[index]
    assert "GEMINI_API_KEY" not in environment  # type: ignore[operator]
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment  # type: ignore[operator]
    assert auth_file.read_bytes() == b'{"refresh_token":"new-secret"}'
    assert b"new-secret" not in repr(captured).encode()


def test_opencode_cli_isolates_environment_and_stages_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pydantic_ai(monkeypatch)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"openai":{"key":"secret"}}', encoding="utf-8")
    captured: dict[str, object] = {}

    async def run(command: list[str], environment: dict[str, str], cwd: str, provider: str) -> str:
        captured.update(command=command, environment=environment, cwd=cwd, provider=provider)
        staged = Path(environment["XDG_DATA_HOME"]) / "opencode" / "auth.json"
        assert staged.read_text(encoding="utf-8") == auth_file.read_text(encoding="utf-8")
        if os.name != "nt":
            assert staged.stat().st_mode & 0o777 == 0o600
        config = Path(environment["OPENCODE_CONFIG"])
        assert config.read_text(encoding="utf-8") == (
            '{"permission":{"*":"deny"},"plugin":[],"mcp":{}}'
        )
        return '{"type":"text","part":{"type":"text","text":"answer"}}\n'

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XDG_DATA_HOME", "OPENCODE_CONFIG"):
        monkeypatch.setenv(name, "untrusted-value")
    command = str(tmp_path / "opencode")
    candidate = Candidate(
        "opencode:openai",
        "api_key",
        "opencode-login",
        "openai/gpt-5",
        config={
            "integration": "opencode-cli",
            "command": command,
            "auth_file": str(auth_file),
            "vendor_provider": "openai",
            "saved_login": "true",
        },
    )
    response = asyncio.run(
        build_cli_model(candidate).request(
            [ModelRequest("hello")],
            None,
            SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
        )
    )
    assert response.parts[0].content == "answer"
    assert captured["command"] == [
        command,
        "--pure",
        "run",
        "--format",
        "json",
        "--model",
        "openai/gpt-5",
        "<user>\nhello",
    ]
    environment = captured["environment"]
    assert environment["XDG_DATA_HOME"] != str(auth_file.parent)  # type: ignore[index]
    assert environment["HOME"] != str(auth_file.parent)  # type: ignore[index]
    assert "OPENAI_API_KEY" not in environment  # type: ignore[operator]
    assert "ANTHROPIC_API_KEY" not in environment  # type: ignore[operator]


def test_opencode_google_account_does_not_warn_as_gemini_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pydantic_ai(monkeypatch)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")

    async def run(command: list[str], environment: dict[str, str], cwd: str, provider: str) -> str:
        return '{"type":"text","part":{"type":"text","text":"answer"}}\n'

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    command = str(tmp_path / "opencode")
    candidate = Candidate(
        "opencode:google",
        "account",
        "opencode-login",
        "google/gemini-2.5-pro",
        config={"command": command, "auth_file": str(auth_file), "saved_login": "true"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(
            build_cli_model(candidate).request(
                [ModelRequest("hello")],
                None,
                SimpleNamespace(function_tools=(), output_tools=(), instruction_parts=()),
            )
        )
    assert not any("Gemini CLI" in str(item.message) for item in caught)


def test_gemini_unchanged_staged_auth_is_not_replaced(tmp_path: Path) -> None:
    auth_file = tmp_path / "oauth_creds.json"
    staged_file = tmp_path / "staged-oauth-creds.json"
    auth_file.write_bytes(b'{"refresh_token":"same"}')
    staged_file.write_bytes(auth_file.read_bytes())
    original = auth_file.lstat()
    staged_original = staged_file.lstat()

    _promote_gemini_refresh(staged_file, auth_file, original, staged_original)

    assert auth_file.lstat().st_ino == original.st_ino


def test_tool_calls_are_rejected_before_starting_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pydantic_ai(monkeypatch)
    model = build_cli_model(Candidate("claude-code", "subscription", "test", "claude-sonnet"))
    request = ModelRequest("hello")
    with pytest.raises(UnsupportedCLIModelError, match="do not support"):
        asyncio.run(model.request([request], None, SimpleNamespace(function_tools=(object(),))))


def test_non_text_parts_and_image_output_capability_are_rejected() -> None:
    class UserPromptPart:
        content = "hello"

    class ImagePart:
        content = b"image"

    message = SimpleNamespace(parts=[UserPromptPart(), ImagePart()])
    base = dict(
        function_tools=(),
        native_tools=(),
        output_tools=(),
        output_object=None,
        instruction_parts=(),
    )
    with pytest.raises(UnsupportedCLIModelError, match="text content only"):
        _text_from_messages([message], SimpleNamespace(**base))
    with pytest.raises(UnsupportedCLIModelError, match="text content only"):
        _text_from_messages(
            [SimpleNamespace(parts=[UserPromptPart()])],
            SimpleNamespace(**base, allow_image_output=True),
        )


def test_cli_runner_stops_at_output_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(b"four")
            self.stdout.feed_eof()
            self.returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    async def spawn(*args: object, **kwargs: object) -> FakeProcess:
        return FakeProcess()

    monkeypatch.setattr("trance.cli_model._MAX_OUTPUT_BYTES", 3)
    monkeypatch.setattr("trance.cli_model.asyncio.create_subprocess_exec", spawn)
    from trance.cli_model import _run_bounded

    with pytest.raises(CLIModelError, match="output limit"):
        asyncio.run(_run_bounded(["fake"], {}, str(tmp_path), "claude-code"))
