from __future__ import annotations

import asyncio
import os
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from trance.cli_model import (
    _MAX_AUTH_BYTES,
    CLIModelError,
    UnsupportedCLIModelError,
    _gemini_auth_file,
    _grok_auth_file,
    _opencode_auth_file,
    _parse_output,
    _promote_grok_refresh,
    _run_bounded,
    _stage_auth,
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


class UserPromptPart:
    def __init__(self, content: object) -> None:
        self.content = content


class ModelRequest:
    def __init__(self, *parts: object, instructions: str | None = None) -> None:
        self.parts = list(parts)
        self.instructions = instructions


def request_parameters(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "function_tools": (),
        "native_tools": (),
        "output_tools": (),
        "output_object": None,
        "allow_image_output": False,
        "instruction_parts": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_text_formatter_preserves_roles_lists_and_instructions() -> None:
    system = SimpleNamespace(parts=[])

    class SystemPromptPart:
        content = "system body"

    class TextPart:
        content = ["first", "second"]

    class ModelResponse:
        parts = [TextPart()]
        instructions = "response instruction"

    system.parts = [SystemPromptPart()]
    result = _text_from_messages(
        [system, ModelResponse()],
        request_parameters(instruction_parts=[SimpleNamespace(content="global instruction")]),
    )
    assert result == (
        "<system>\nglobal instruction\n\n"
        "<system>\nsystem body\n\n"
        "<system>\nresponse instruction\n\n"
        "<assistant>\nfirst\nsecond"
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"native_tools": (object(),)},
        {"output_object": object()},
        {"allow_image_output": True},
    ],
)
def test_text_formatter_rejects_unsupported_request_features(parameters: dict[str, object]) -> None:
    with pytest.raises(UnsupportedCLIModelError):
        _text_from_messages(
            [ModelRequest(UserPromptPart("hello"))], request_parameters(**parameters)
        )


def test_text_formatter_rejects_empty_and_tool_history() -> None:
    with pytest.raises(ValueError, match="no text prompt"):
        _text_from_messages([], request_parameters())

    class ToolCallPart:
        content = "ignored"

    with pytest.raises(UnsupportedCLIModelError, match="tool-call history"):
        _text_from_messages([SimpleNamespace(parts=[ToolCallPart()])], request_parameters())


def test_auth_file_validators_reject_relative_missing_directory_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedCLIModelError, match="must be absolute"):
        _opencode_auth_file("relative.json")
    with pytest.raises(UnsupportedCLIModelError, match="unavailable"):
        _opencode_auth_file(str(tmp_path / "missing.json"))

    home = tmp_path / "home"
    home.mkdir()
    auth = home / "auth.json"
    auth.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(auth)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    with pytest.raises(UnsupportedCLIModelError, match="unsafe"):
        _opencode_auth_file(str(link))
    with pytest.raises(UnsupportedCLIModelError, match="unsafe"):
        _grok_auth_file(str(home), str(link))
    with pytest.raises(UnsupportedCLIModelError, match="unsafe"):
        _gemini_auth_file(str(home), str(link))


def test_stage_auth_enforces_bound_and_private_staging(tmp_path: Path) -> None:
    source = tmp_path / "auth.json"
    staged = tmp_path / "staged.json"
    source.write_bytes(b"secret")
    original = _stage_auth(source, staged, "Test")
    assert original.st_size == 6
    assert staged.read_bytes() == b"secret"
    assert staged.is_file()
    assert not staged.is_symlink()
    if os.name == "posix":
        assert stat.S_IMODE(staged.stat().st_mode) == 0o600
    with pytest.raises(UnsupportedCLIModelError):
        _stage_auth(source, staged, "Test")

    source.write_bytes(b"x" * (_MAX_AUTH_BYTES + 1))
    with pytest.raises(UnsupportedCLIModelError, match="too large"):
        _stage_auth(source, tmp_path / "large-staged.json", "Test")


def test_promotion_refuses_concurrent_source_change_and_removes_temp_files(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    staged = tmp_path / "staged.json"
    auth.write_bytes(b"old")
    staged.write_bytes(b"new")
    original = auth.lstat()
    staged_original = staged.lstat()
    staged.write_bytes(b"refreshed")
    auth.write_bytes(b"changed")
    with pytest.raises(CLIModelError, match="concurrently"):
        _promote_grok_refresh(staged, auth, original, staged_original)
    assert auth.read_bytes() == b"changed"
    assert not list(tmp_path.glob(".trance-auth-*"))


def test_promotion_rejects_staged_symlink(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    staged_target = tmp_path / "target"
    staged = tmp_path / "staged"
    auth.write_bytes(b"old")
    staged_target.write_bytes(b"new")
    try:
        staged.symlink_to(staged_target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    with pytest.raises(CLIModelError, match="unsafe"):
        _promote_grok_refresh(staged, auth, auth.lstat(), staged.lstat())


@pytest.mark.parametrize(
    ("provider", "output", "message"),
    [
        ("claude-code", "", "empty output"),
        ("claude-code", '{"result": 3}', "no textual result"),
        ("grok-consumer", "[]", "no textual result"),
        ("gemini-cli", '{"response": null}', "no textual response"),
        ("opencode:x", '{"type":"text"}', "invalid text event"),
    ],
)
def test_output_parser_rejects_missing_text(provider: str, output: str, message: str) -> None:
    with pytest.raises(CLIModelError, match=message):
        _parse_output(provider, output)


class FakeStdout:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    async def read(self, _: int) -> bytes:
        return next(self._chunks, b"")


class FakeStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeProcess:
    def __init__(self, chunks: list[bytes], returncode: int = 0) -> None:
        self.stdout = FakeStdout(chunks)
        self.stdin = FakeStdin()
        self.returncode: int | None = None
        self.final_returncode = returncode
        self.killed = False

    async def wait(self) -> int:
        self.returncode = self.final_returncode if not self.killed else -9
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def test_run_bounded_drains_stdin_and_decodes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess([b"hel", b"lo"])

    async def spawn(*args: object, **kwargs: object) -> FakeProcess:
        assert kwargs["stdin"] is asyncio.subprocess.PIPE
        assert kwargs["stderr"] is asyncio.subprocess.DEVNULL
        assert kwargs["limit"] == 64 * 1024
        return process

    monkeypatch.setattr("trance.cli_model.asyncio.create_subprocess_exec", spawn)
    result = asyncio.run(_run_bounded(["fake"], {"PATH": "x"}, ".", "provider", "prompt"))
    assert result == "hello"
    assert process.stdin.data == b"prompt"
    assert process.stdin.closed


def test_run_bounded_reports_nonzero_and_non_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess([b"failure"], returncode=7)

    async def spawn(*args: object, **kwargs: object) -> FakeProcess:
        return process

    monkeypatch.setattr("trance.cli_model.asyncio.create_subprocess_exec", spawn)
    with pytest.raises(CLIModelError, match="status 7"):
        asyncio.run(_run_bounded(["fake"], {}, ".", "provider"))

    process = FakeProcess([b"\xff"])
    with pytest.raises(CLIModelError, match="non-UTF-8"):
        asyncio.run(_run_bounded(["fake"], {}, ".", "provider"))


def test_run_bounded_kills_process_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess([b""])

    async def spawn(*args: object, **kwargs: object) -> FakeProcess:
        return process

    async def timeout(awaitable: object, _: float) -> object:
        if hasattr(awaitable, "close"):
            awaitable.close()  # type: ignore[attr-defined]
        raise TimeoutError

    monkeypatch.setattr("trance.cli_model.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("trance.cli_model.asyncio.wait_for", timeout)
    with pytest.raises(TimeoutError):
        asyncio.run(_run_bounded(["fake"], {}, ".", "provider"))
    assert process.killed


def test_request_uses_fresh_cwd_and_cleans_it_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pydantic_ai(monkeypatch)
    seen: dict[str, object] = {}

    async def run(command: list[str], environment: dict[str, str], cwd: str, provider: str) -> str:
        seen.update(command=command, environment=environment, cwd=cwd, provider=provider)
        assert Path(cwd).is_dir()
        assert environment["HOME"] == cwd
        raise CLIModelError("synthetic failure")

    monkeypatch.setattr("trance.cli_model._run_bounded", run)
    model = build_cli_model(
        Candidate(
            "claude-code",
            "subscription",
            "test",
            "claude-sonnet",
            config={"command": "claude"},
        )
    )
    with pytest.raises(CLIModelError, match="synthetic failure"):
        asyncio.run(
            model.request(
                [ModelRequest(UserPromptPart("hello"))], None, request_parameters()
            )
        )
    cwd = seen["cwd"]
    assert isinstance(cwd, str)
    assert not Path(cwd).exists()


def test_opencode_requires_absolute_command() -> None:
    with pytest.raises(UnsupportedCLIModelError, match="absolute"):
        build_cli_model(
            Candidate(
                "opencode:openai",
                "account",
                "test",
                "model",
                config={"command": "opencode"},
            )
        )
