"""Small Pydantic AI bridge to documented vendor CLI interfaces.

CLI credentials stay inside the vendor command's supported auth mechanism. This
adapter intentionally supports text only: coding-agent CLIs do not expose a
portable mapping for arbitrary Pydantic AI tool calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

from trance.types import Candidate

_TIMEOUT_SECONDS = 120.0
_MAX_OUTPUT_BYTES = 1_048_576
_MAX_AUTH_BYTES = 1_048_576


class UnsupportedCLIModelError(ValueError):
    """Raised when a discovered credential has no safe documented adapter."""


class CLIModelError(RuntimeError):
    """Raised when a vendor CLI fails or returns an invalid response."""


class TransientCLIModelError(CLIModelError):
    """Raised when a CLI failure may succeed if retried later.

    ``retry_after_seconds`` is provider supplied timing metadata only.  Raw
    CLI output is deliberately never retained or included in the exception.
    """

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


_RETRY_AFTER = re.compile(r"(?:retry[- _]?after)\s*[:=]\s*(\d+(?:\.\d+)?)", re.I)
_HTTP_STATUS = re.compile(r"\b([45]\d\d)\b")


def _classify_cli_failure(provider: str, returncode: int, output: str) -> CLIModelError:
    """Create a safe terminal or transient error from bounded CLI output."""
    statuses = {int(match.group(1)) for match in _HTTP_STATUS.finditer(output)}
    retry_match = _RETRY_AFTER.search(output)
    retry_after = float(retry_match.group(1)) if retry_match else None
    if 429 in statuses or re.search(r"rate\s*limit|too\s*many\s*requests", output, re.I):
        return TransientCLIModelError(
            f"{provider} CLI rate limited", retry_after_seconds=retry_after
        )
    if any(500 <= status <= 599 for status in statuses):
        return TransientCLIModelError(f"{provider} CLI temporary server failure")
    return CLIModelError(f"{provider} CLI exited with status {returncode}")


def _text_from_messages(messages: list[Any], parameters: Any) -> str:
    """Convert a basic text conversation to a single CLI prompt."""
    if (
        getattr(parameters, "function_tools", None)
        or getattr(parameters, "native_tools", None)
        or getattr(parameters, "output_tools", None)
    ):
        raise UnsupportedCLIModelError("Vendor CLI adapters do not support Pydantic AI tools")
    if getattr(parameters, "output_object", None):
        raise UnsupportedCLIModelError("Vendor CLI adapters do not support structured output")
    if getattr(parameters, "allow_image_output", False):
        raise UnsupportedCLIModelError("CLI adapters support text content only")
    sections: list[str] = []
    for message in messages:
        role = "assistant" if message.__class__.__name__ == "ModelResponse" else "user"
        parts: list[str] = []
        for part in getattr(message, "parts", ()):
            name = part.__class__.__name__
            if name == "SystemPromptPart":
                role = "system"
            elif name not in {"UserPromptPart", "TextPart", "SystemPromptPart"}:
                if name in {"ToolCallPart", "ToolReturnPart", "RetryPromptPart"}:
                    raise UnsupportedCLIModelError("CLI adapters cannot replay tool-call history")
                raise UnsupportedCLIModelError("CLI adapters support text content only")
            content = getattr(part, "content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                if any(not isinstance(item, str) for item in content):
                    raise UnsupportedCLIModelError("CLI adapters support text content only")
                parts.extend(content)
            elif content:
                raise UnsupportedCLIModelError("CLI adapters support text content only")
        message_instructions = getattr(message, "instructions", None)
        if isinstance(message_instructions, str) and message_instructions:
            sections.append(f"<system>\n{message_instructions}")
        if parts:
            sections.append(f"<{role}>\n" + "\n".join(parts))
    instructions = getattr(parameters, "instruction_parts", None) or ()
    instruction_text = "\n".join(
        str(getattr(item, "content", "")) for item in instructions if getattr(item, "content", None)
    )
    if instruction_text:
        sections.insert(0, f"<system>\n{instruction_text}")
    if not sections:
        raise ValueError("CLI model request contains no text prompt")
    return "\n\n".join(sections)


def _parse_output(provider: str, stdout: str) -> str:
    """Extract the final textual answer from documented CLI output formats."""
    output = stdout.strip()
    if not output:
        raise CLIModelError(f"{provider} CLI returned empty output")
    if provider == "claude-code":
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise CLIModelError("Claude Code returned malformed JSON") from exc
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, str):
            raise CLIModelError("Claude Code JSON response has no textual result")
        return result
    if provider == "grok-consumer":
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise CLIModelError("Grok Build returned malformed JSON") from exc
        result = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(result, str):
            raise CLIModelError("Grok Build JSON response has no textual result")
        return result
    if provider == "gemini-cli":
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise CLIModelError("Gemini CLI returned malformed JSON") from exc
        result = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(result, str):
            raise CLIModelError("Gemini CLI JSON response has no textual response")
        return result
    if provider == "sourcegraph-cody":
        # ``cody chat --stdin`` deliberately emits the answer as plain text.
        # Do not attempt JSON extraction here: Cody's text stream can contain
        # arbitrary JSON as part of the answer.
        return output
    if provider.startswith("opencode:"):
        # OpenCode's JSON mode is a JSONL event stream.  Only text events are
        # safe for this text-only adapter; tool and error events are rejected
        # rather than silently returning a partial answer.
        texts: list[str] = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CLIModelError("OpenCode returned malformed JSONL") from exc
            if not isinstance(event, dict):
                raise CLIModelError("OpenCode returned an invalid JSON event")
            event_type = event.get("type")
            if event_type != "text":
                raise CLIModelError(f"OpenCode returned unsupported {event_type!r} event")
            part = event.get("part")
            if not isinstance(part, dict) or part.get("type") != "text":
                raise CLIModelError("OpenCode returned an invalid text event")
            text = part.get("text")
            if not isinstance(text, str):
                raise CLIModelError("OpenCode text event has no textual content")
            texts.append(text)
        if not texts:
            raise CLIModelError("OpenCode returned no text events")
        return "".join(texts)
    raise CLIModelError(f"No safe text-only output parser for {provider}")


def _opencode_auth_file(auth_file: str) -> Path:
    """Validate OpenCode's documented auth store without reading its contents."""
    path = Path(auth_file)
    if not path.is_absolute():
        raise UnsupportedCLIModelError("OpenCode authentication file must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise UnsupportedCLIModelError("OpenCode authentication file is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise UnsupportedCLIModelError("OpenCode authentication file is unsafe")
    return path


def _stage_opencode_auth(auth_file: Path, staged_file: Path) -> os.stat_result:
    return _stage_auth(auth_file, staged_file, "OpenCode")


def _promote_opencode_refresh(
    staged_file: Path,
    auth_file: Path,
    original_stat: os.stat_result,
    staged_original_stat: os.stat_result,
) -> None:
    _promote_refresh(staged_file, auth_file, original_stat, staged_original_stat, "OpenCode")


def _grok_auth_file(auth_home: str, configured_file: str | None = None) -> Path:
    """Validate the documented Grok auth file without reading its contents."""
    home = Path(auth_home)
    if not home.is_absolute():
        raise UnsupportedCLIModelError("Grok authentication home must be absolute")
    try:
        home_stat = home.lstat()
        auth_file = Path(configured_file) if configured_file else home / "auth.json"
        if not auth_file.is_absolute():
            raise UnsupportedCLIModelError("Grok authentication file is unsafe")
        auth_stat = auth_file.lstat()
    except OSError as exc:
        raise UnsupportedCLIModelError("Grok authentication file is unavailable") from exc
    if not stat.S_ISDIR(home_stat.st_mode) or stat.S_ISLNK(home_stat.st_mode):
        raise UnsupportedCLIModelError("Grok authentication home is unsafe")
    if not stat.S_ISREG(auth_stat.st_mode) or stat.S_ISLNK(auth_stat.st_mode):
        raise UnsupportedCLIModelError("Grok authentication file is unsafe")
    return auth_file


def _stage_grok_auth(auth_file: Path, staged_file: Path) -> os.stat_result:
    """Copy auth state into a private regular file without following links."""
    return _stage_auth(auth_file, staged_file, "Grok")


def _promote_grok_refresh(
    staged_file: Path,
    auth_file: Path,
    original_stat: os.stat_result,
    staged_original_stat: os.stat_result,
) -> None:
    """Atomically promote changed auth state when the source is unchanged."""
    _promote_refresh(staged_file, auth_file, original_stat, staged_original_stat, "Grok")


def _gemini_auth_file(auth_home: str, configured_file: str | None = None) -> Path:
    """Validate the documented Gemini OAuth cache without reading its contents."""
    home = Path(auth_home)
    if not home.is_absolute():
        raise UnsupportedCLIModelError("Gemini authentication home must be absolute")
    try:
        home_stat = home.lstat()
        auth_file = Path(configured_file) if configured_file else home / "oauth_creds.json"
        if not auth_file.is_absolute():
            raise UnsupportedCLIModelError("Gemini authentication file is unsafe")
        auth_stat = auth_file.lstat()
    except OSError as exc:
        raise UnsupportedCLIModelError("Gemini authentication file is unavailable") from exc
    if not stat.S_ISDIR(home_stat.st_mode) or stat.S_ISLNK(home_stat.st_mode):
        raise UnsupportedCLIModelError("Gemini authentication home is unsafe")
    if not stat.S_ISREG(auth_stat.st_mode) or stat.S_ISLNK(auth_stat.st_mode):
        raise UnsupportedCLIModelError("Gemini authentication file is unsafe")
    return auth_file


def _stage_gemini_auth(auth_file: Path, staged_file: Path) -> os.stat_result:
    """Copy Gemini OAuth state into a private regular file without following links."""
    return _stage_auth(auth_file, staged_file, "Gemini")


def _stage_auth(auth_file: Path, staged_file: Path, provider: str) -> os.stat_result:
    """Copy bounded auth state into a private regular file without following links."""
    source_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(auth_file, flags)
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size > _MAX_AUTH_BYTES:
            raise UnsupportedCLIModelError(f"{provider} authentication file is unsafe or too large")
        staged_fd = os.open(staged_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with (
            os.fdopen(staged_fd, "wb") as target,
            os.fdopen(source_fd, "rb", closefd=False) as source,
        ):
            remaining = source_stat.st_size
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    raise UnsupportedCLIModelError(
                        f"{provider} authentication file changed while staging"
                    )
                target.write(chunk)
                remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
        return source_stat
    except OSError as exc:
        raise UnsupportedCLIModelError(f"Could not stage {provider} authentication") from exc
    finally:
        if source_fd is not None:
            try:
                os.close(source_fd)
            except OSError:
                pass


def _promote_gemini_refresh(
    staged_file: Path,
    auth_file: Path,
    original_stat: os.stat_result,
    staged_original_stat: os.stat_result,
) -> None:
    """Atomically promote changed Gemini OAuth state when the source is unchanged."""
    _promote_refresh(staged_file, auth_file, original_stat, staged_original_stat, "Gemini")


def _promote_refresh(
    staged_file: Path,
    auth_file: Path,
    original_stat: os.stat_result,
    staged_original_stat: os.stat_result,
    provider: str,
) -> None:
    """Atomically promote changed auth state when its source is unchanged."""
    try:
        staged_stat = staged_file.lstat()
        current_stat = auth_file.lstat()
    except OSError as exc:
        raise CLIModelError(f"Could not inspect refreshed {provider} authentication") from exc
    if not stat.S_ISREG(staged_stat.st_mode) or stat.S_ISLNK(staged_stat.st_mode):
        raise CLIModelError(f"{provider} CLI produced an unsafe authentication file")
    if staged_stat.st_size > _MAX_AUTH_BYTES:
        raise CLIModelError(f"Refreshed {provider} authentication is too large")
    if (
        staged_stat.st_ino == staged_original_stat.st_ino
        and staged_stat.st_size == staged_original_stat.st_size
        and staged_stat.st_mtime_ns == staged_original_stat.st_mtime_ns
    ):
        return
    if (
        current_stat.st_dev != original_stat.st_dev
        or current_stat.st_ino != original_stat.st_ino
        or current_stat.st_size != original_stat.st_size
        or current_stat.st_mtime_ns != original_stat.st_mtime_ns
    ):
        raise CLIModelError(
            f"{provider} authentication changed concurrently; refusing refresh promotion"
        )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=auth_file.parent, prefix=".trance-auth-", delete=False
        ) as target:
            temporary = target.name
            with staged_file.open("rb") as source:
                remaining = staged_stat.st_size
                while remaining:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        raise CLIModelError(
                            f"Refreshed {provider} authentication changed while copying"
                        )
                    target.write(chunk)
                    remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, auth_file)
        temporary = None
    except OSError as exc:
        raise CLIModelError(f"Could not promote refreshed {provider} authentication") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class _VendorCLIModel:
    """Mixin implementing the shared one-shot subprocess request."""

    def __init__(self, candidate: Candidate, command: str, provider: str) -> None:
        from pydantic_ai.models import Model

        # Model is the Pydantic AI integration boundary; use its initializer
        # without coupling module import to an optional runtime installation.
        Model.__init__(self)
        self._candidate = candidate
        self._command = command
        self._provider_id = provider

    @property
    def model_name(self) -> str:
        return self._candidate.model_name

    @property
    def system(self) -> str:
        return self._provider_id

    async def request(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
    ) -> Any:
        del model_settings
        prompt = _text_from_messages(messages, model_request_parameters)
        if self._provider_id == "gemini-cli":
            from trance.sources.gemini_cli import (
                GEMINI_CLI_TERMS_WARNING,
                GeminiCLITermsWarning,
            )

            warnings.warn(GEMINI_CLI_TERMS_WARNING, GeminiCLITermsWarning, stacklevel=2)
        # Give each vendor only a small runtime environment. Explicit Claude
        # tokens get a fresh HOME; saved sessions are staged into a temporary
        # home so hooks and plugins from the user's home cannot run.
        environment = {
            name: value
            for name in ("PATH", "SYSTEMROOT", "WINDIR")
            if (value := os.environ.get(name)) is not None
        }
        saved_login = self._candidate.config.get("saved_login") == "true" or (
            self._provider_id == "grok-consumer"
            or self._provider_id.startswith("opencode:")
        )
        if self._provider_id == "claude-code" and not saved_login:
            environment["CLAUDE_CODE_SIMPLE"] = "1"
        if self._candidate.secret and self._provider_id == "claude-code":
            key = self._candidate.config.get("credential_env")
            if key != "CLAUDE_CODE_OAUTH_TOKEN":
                raise UnsupportedCLIModelError("Unexpected Claude credential environment variable")
            environment[key] = self._candidate.secret
        try:
            with tempfile.TemporaryDirectory(prefix="trance-cli-") as temp_cwd:
                command = self._argv(prompt, temp_cwd)
                auth_home = self._candidate.config.get("auth_home") if saved_login else None
                if saved_login and not auth_home and self._provider_id == "claude-code":
                    raise UnsupportedCLIModelError(
                        "Saved Claude login is missing its authentication home"
                    )
                if self._provider_id == "grok-consumer":
                    if not auth_home:
                        raise UnsupportedCLIModelError(
                            "Saved Grok login is missing its authentication home"
                        )
                    auth_file = _grok_auth_file(
                        auth_home, self._candidate.config.get("auth_file")
                    )
                    grok_home = Path(temp_cwd) / "grok-home"
                    grok_home.mkdir()
                    grok_staged = grok_home / "auth.json"
                    grok_original = _stage_grok_auth(auth_file, grok_staged)
                    grok_staged_original = grok_staged.lstat()
                    isolated_home = Path(temp_cwd) / "home"
                    isolated_home.mkdir()
                    environment["HOME"] = str(isolated_home)
                    environment["USERPROFILE"] = str(isolated_home)
                    environment["GROK_HOME"] = str(grok_home)
                    environment["GROK_AUTH_PATH"] = str(grok_staged)
                elif self._provider_id == "gemini-cli":
                    if not auth_home:
                        raise UnsupportedCLIModelError(
                            "Saved Gemini login is missing its authentication home"
                        )
                    auth_file = _gemini_auth_file(
                        auth_home, self._candidate.config.get("auth_file")
                    )
                    gemini_base = Path(temp_cwd) / "gemini-home"
                    gemini_home = gemini_base / ".gemini"
                    gemini_home.mkdir(parents=True)
                    gemini_staged = gemini_home / "oauth_creds.json"
                    gemini_original = _stage_gemini_auth(auth_file, gemini_staged)
                    gemini_staged_original = gemini_staged.lstat()
                    settings_payload = json.dumps(
                        {
                            "security": {"auth": {"selectedType": "oauth-personal"}},
                            "tools": {"core": []},
                            "mcp": {"allowed": []},
                        }
                    ).encode("utf-8")
                    settings_fd = os.open(
                        gemini_home / "settings.json",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    with os.fdopen(settings_fd, "wb") as settings:
                        settings.write(settings_payload)
                        settings.flush()
                        os.fsync(settings.fileno())
                    isolated_home = Path(temp_cwd) / "home"
                    isolated_home.mkdir()
                    environment["HOME"] = str(isolated_home)
                    environment["USERPROFILE"] = str(isolated_home)
                    environment["GEMINI_CLI_HOME"] = str(gemini_base)
                elif self._provider_id == "sourcegraph-cody":
                    if not auth_home:
                        raise UnsupportedCLIModelError(
                            "Saved Cody login is missing its authentication home"
                        )
                    if not Path(auth_home).is_absolute():
                        raise UnsupportedCLIModelError(
                            "Cody authentication home must be absolute"
                        )
                    # Cody resolves its OS keychain account from HOME. Keep
                    # that home intact while giving the process a clean cwd.
                    environment["HOME"] = auth_home
                    environment["USERPROFILE"] = auth_home
                    # Secret Service based keychains need these session
                    # handles; they contain no provider credentials.
                    for name in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
                        if value := os.environ.get(name):
                            environment[name] = value
                elif self._provider_id.startswith("opencode:"):
                    auth_file = _opencode_auth_file(
                        self._candidate.config.get("auth_file", "")
                    )
                    data_home = Path(temp_cwd) / "xdg-data"
                    opencode_data = data_home / "opencode"
                    opencode_data.mkdir(parents=True)
                    opencode_staged = opencode_data / "auth.json"
                    opencode_original = _stage_opencode_auth(auth_file, opencode_staged)
                    opencode_staged_original = opencode_staged.lstat()
                    config_home = Path(temp_cwd) / "xdg-config"
                    cache_home = Path(temp_cwd) / "xdg-cache"
                    state_home = Path(temp_cwd) / "xdg-state"
                    config_dir = Path(temp_cwd) / "opencode-config"
                    for directory in (config_home, cache_home, state_home, config_dir):
                        directory.mkdir()
                    config_file = Path(temp_cwd) / "opencode.json"
                    config_payload = json.dumps(
                        {"permission": {"*": "deny"}, "plugin": [], "mcp": {}},
                        separators=(",", ":"),
                    ).encode("utf-8")
                    config_fd = os.open(
                        config_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    with os.fdopen(config_fd, "wb") as config_stream:
                        config_stream.write(config_payload)
                        config_stream.flush()
                        os.fsync(config_stream.fileno())
                    isolated_home = Path(temp_cwd) / "home"
                    isolated_home.mkdir()
                    environment["HOME"] = str(isolated_home)
                    environment["USERPROFILE"] = str(isolated_home)
                    environment["XDG_DATA_HOME"] = str(data_home)
                    environment["XDG_CONFIG_HOME"] = str(config_home)
                    environment["XDG_CACHE_HOME"] = str(cache_home)
                    environment["XDG_STATE_HOME"] = str(state_home)
                    environment["OPENCODE_CONFIG_DIR"] = str(config_dir)
                    environment["OPENCODE_CONFIG"] = str(config_file)
                else:
                    home = auth_home or temp_cwd
                    environment["HOME"] = home
                    environment["USERPROFILE"] = home
                if saved_login and self._provider_id == "claude-code":
                    config_dir = self._candidate.config.get("config_dir")
                    if config_dir:
                        environment["CLAUDE_CONFIG_DIR"] = config_dir
                environment["TMPDIR"] = temp_cwd
                environment["TEMP"] = temp_cwd
                try:
                    if self._provider_id == "sourcegraph-cody":
                        stdout = await _run_bounded(
                            command,
                            environment,
                            temp_cwd,
                            self._provider_id,
                            stdin_text=prompt,
                        )
                    else:
                        stdout = await _run_bounded(
                            command, environment, temp_cwd, self._provider_id
                        )
                finally:
                    if self._provider_id == "grok-consumer":
                        _promote_grok_refresh(
                            grok_staged, auth_file, grok_original, grok_staged_original
                        )
                    elif self._provider_id == "gemini-cli":
                        _promote_gemini_refresh(
                            gemini_staged, auth_file, gemini_original, gemini_staged_original
                        )
                    elif self._provider_id.startswith("opencode:"):
                        _promote_opencode_refresh(
                            opencode_staged,
                            auth_file,
                            opencode_original,
                            opencode_staged_original,
                        )
        except TimeoutError as exc:
            raise TransientCLIModelError(f"{self._provider_id} CLI timed out") from exc
        except OSError as exc:
            raise CLIModelError(f"Could not start {self._provider_id} CLI") from exc
        from pydantic_ai.messages import ModelResponse, TextPart

        text = _parse_output(self._provider_id, stdout)
        return ModelResponse(parts=[TextPart(content=text)], model_name=self.model_name)


async def _run_bounded(
    command: list[str],
    environment: dict[str, str],
    cwd: str,
    provider: str,
    stdin_text: str | None = None,
) -> str:
    """Run a CLI while draining bounded stdout and optionally writing stdin."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=environment,
        limit=64 * 1024,
    )
    assert process.stdout is not None
    stderr = getattr(process, "stderr", None)
    chunks: list[bytes] = []
    error_chunks: list[bytes] = []
    size = 0
    error_size = 0

    async def read_output() -> None:
        nonlocal size
        while chunk := await process.stdout.read(64 * 1024):
            size += len(chunk)
            if size > _MAX_OUTPUT_BYTES:
                process.kill()
                raise CLIModelError(f"{_MAX_OUTPUT_BYTES}-byte output limit exceeded")
            chunks.append(chunk)

    async def read_error() -> None:
        nonlocal error_size
        if stderr is None:
            return
        while chunk := await stderr.read(64 * 1024):
            error_size += len(chunk)
            if error_size > _MAX_OUTPUT_BYTES:
                process.kill()
                raise CLIModelError(f"{_MAX_OUTPUT_BYTES}-byte output limit exceeded")
            error_chunks.append(chunk)

    async def write_input() -> None:
        if stdin_text is None:
            return
        assert process.stdin is not None
        process.stdin.write(stdin_text.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()

    try:
        await asyncio.wait_for(
            asyncio.gather(process.wait(), read_output(), read_error(), write_input()),
            _TIMEOUT_SECONDS,
        )
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        bounded_output = b"".join(chunks + error_chunks).decode("utf-8", errors="replace")
        raise _classify_cli_failure(provider, process.returncode, bounded_output)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CLIModelError(f"{provider} CLI returned non-UTF-8 output") from exc


def _model_type() -> type:
    from pydantic_ai.models import Model

    class VendorCLIModel(_VendorCLIModel, Model):
        def _argv(self, prompt: str, cwd: str) -> list[str]:
            command = self._command
            model = self.model_name
            if self._provider_id == "claude-code":
                return [
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
                    model,
                    prompt,
                ]
            if self._provider_id == "grok-consumer":
                return [
                    command,
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
                    cwd,
                    "-p",
                    prompt,
                    "--output-format",
                    "json",
                    "-m",
                    model,
                ]
            if self._provider_id == "gemini-cli":
                return [
                    command,
                    "-e",
                    "none",
                    "-p",
                    prompt,
                    "--output-format",
                    "json",
                    "--model",
                    model,
                ]
            if self._provider_id == "sourcegraph-cody":
                command_line = [command, "chat", "--stdin"]
                if model != "default":
                    command_line.extend(["--model", model])
                return command_line
            if self._provider_id.startswith("opencode:"):
                return [
                    command,
                    "--pure",
                    "run",
                    "--format",
                    "json",
                    "--model",
                    model,
                    prompt,
                ]
            raise UnsupportedCLIModelError(f"No CLI adapter for {self._provider_id!r}")

    return VendorCLIModel


def build_cli_model(candidate: Candidate) -> Any:
    """Build a Pydantic AI Model backed by a supported vendor CLI.

    The CLI must already be authenticated, except for explicitly documented
    environment credentials carried by the candidate. Saved login state is
    copied into an isolated temporary CLI home and never included in errors.
    """
    provider = candidate.provider
    supported = {"claude-code", "grok-consumer", "gemini-cli", "sourcegraph-cody"}
    is_opencode = provider.startswith("opencode:")
    if provider not in supported and not is_opencode:
        raise UnsupportedCLIModelError(f"No documented CLI model adapter for {provider!r}")
    command = candidate.config.get("command") or {
        "claude-code": "claude",
        "grok-consumer": "grok",
        "gemini-cli": "gemini",
        "sourcegraph-cody": "cody",
    }.get(provider, "opencode")
    if is_opencode and not Path(command).is_absolute():
        raise UnsupportedCLIModelError("OpenCode command must be absolute")
    try:
        return _model_type()(candidate, command, provider)
    except ImportError as exc:
        raise UnsupportedCLIModelError("Install trance with its pydantic-ai dependency") from exc
