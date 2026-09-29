"""Cross platform edge cases for file backed source adapters.

These tests use only synthetic homes and environment mappings.  Symlink cases
are skipped on platforms where the test account cannot create symlinks.
"""

import json
from pathlib import Path

import pytest

from trance.sources import (
    aws_bedrock,
    claude_code,
    codex,
    gemini_cli,
    google_vertex,
    grok_consumer,
    kimi_code,
    mistral_vibe,
    opencode,
    qwen_code,
)


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _codex_auth(path: Path) -> None:
    _write_json(path, {"auth_mode": "chatgpt", "tokens": {
        "access_token": "a", "refresh_token": "r", "id_token": "i",
    }})


def test_codex_blank_home_override_falls_back_to_injected_home(tmp_path: Path) -> None:
    _codex_auth(tmp_path / ".codex" / "auth.json")
    [candidate] = codex.scan({"CODEX_HOME": "  "}, tmp_path)
    assert candidate.config["codex_home"] == str(tmp_path / ".codex")


def test_codex_auth_directory_symlink_is_not_required_for_portable_reads(tmp_path: Path) -> None:
    real = tmp_path / "real"
    _codex_auth(real / "auth.json")
    _symlink(tmp_path / "alias", real, directory=True)
    [candidate] = codex.scan({"CODEX_HOME": str(tmp_path / "alias")}, tmp_path)
    assert candidate.provider == "openai-codex"


def test_claude_explicit_token_is_independent_of_unusable_saved_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(claude_code.shutil, "which", lambda name, path: None)
    [candidate] = claude_code.scan(
        {"CLAUDE_CODE_OAUTH_TOKEN": " env-token ", "PATH": ""}, tmp_path
    )
    assert candidate.secret == "env-token"
    assert candidate.source == "env:CLAUDE_CODE_OAUTH_TOKEN"


def test_claude_saved_login_rejects_malformed_status_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    credentials = tmp_path / ".claude" / ".credentials.json"
    credentials.parent.mkdir()
    credentials.write_text("opaque", encoding="utf-8")
    monkeypatch.setattr(claude_code.shutil, "which", lambda name, path: "claude")

    class Result:
        returncode = 1
        stdout = "{broken"

    monkeypatch.setattr(claude_code.subprocess, "run", lambda *args, **kwargs: Result())
    assert claude_code.scan({"PATH": "tools"}, tmp_path) == []


def test_grok_relative_auth_override_is_resolved_from_process_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth = tmp_path / "state" / "auth.json"
    _write_json(auth, {"session": {"auth_mode": "oidc"}})
    (tmp_path / ".grok").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(grok_consumer.shutil, "which", lambda name, path: "grok")
    [candidate] = grok_consumer.scan(
        {"PATH": "tools", "GROK_AUTH_PATH": "state/auth.json"}, tmp_path
    )
    assert candidate.config["auth_file"] == str(auth)


def test_grok_rejects_nested_non_oidc_and_truncated_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".grok"
    home.mkdir()
    auth = home / "auth.json"
    auth.write_text(json.dumps({"session": {"auth_mode": "api_key"}}), encoding="utf-8")
    monkeypatch.setattr(grok_consumer.shutil, "which", lambda name, path: "grok")
    assert grok_consumer.scan({"PATH": "tools"}, tmp_path) == []
    auth.write_bytes(b"{\xff")
    assert grok_consumer.scan({"PATH": "tools"}, tmp_path) == []


def test_gemini_home_override_does_not_consult_process_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "nested" / ".gemini" / "oauth_creds.json"
    _write_json(cache, {"refresh_token": "refresh"})
    monkeypatch.setattr(gemini_cli.shutil, "which", lambda name, path: "gemini")
    with pytest.warns(gemini_cli.GeminiCLITermsWarning):
        [candidate] = gemini_cli.scan(
            {"PATH": "tools", "GEMINI_CLI_HOME": "nested"}, tmp_path
        )
    assert candidate.config["auth_home"] == str(cache.parent)


def test_gemini_invalid_utf8_cache_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / ".gemini" / "oauth_creds.json"
    cache.parent.mkdir()
    cache.write_bytes(b"{\xff")
    monkeypatch.setattr(gemini_cli.shutil, "which", lambda name, path: "gemini")
    assert gemini_cli.scan({"PATH": "tools"}, tmp_path) == []


def test_opencode_vendor_names_are_normalized_and_duplicate_keys_are_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth = tmp_path / "data" / "opencode" / "auth.json"
    _write_json(auth, {" OpenAI ": {"type": "api", "key": "key"}, "OPENAI": {
        "type": "api", "key": "key2"
    }})
    monkeypatch.setattr(opencode.shutil, "which", lambda name, path: "opencode")
    candidates = opencode.scan({"PATH": "tools", "XDG_DATA_HOME": str(tmp_path / "data")}, tmp_path)
    assert [item.provider for item in candidates] == ["opencode:openai", "opencode:openai"]


def test_opencode_invalid_utf8_and_oversized_auth_are_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth = tmp_path / ".local" / "share" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    monkeypatch.setattr(opencode.shutil, "which", lambda name, path: "opencode")
    auth.write_bytes(b"{\xff")
    assert opencode.scan({"PATH": "tools"}, tmp_path) == []
    auth.write_bytes(b"{}" + b"x" * opencode._MAX_AUTH_BYTES)
    assert opencode.scan({"PATH": "tools"}, tmp_path) == []


def test_aws_environment_region_precedence_and_relative_overrides(tmp_path: Path) -> None:
    [candidate] = aws_bedrock.scan(
        {
            "AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_REGION": " us-east-1 ", "AWS_DEFAULT_REGION": "eu-west-1",
            "AWS_CONFIG_FILE": "config/custom", "AWS_SHARED_CREDENTIALS_FILE": "creds/custom",
        },
        tmp_path,
    )
    assert candidate.config["region"] == "us-east-1"
    assert candidate.config["shared_config_file"] == str((tmp_path / "config/custom").resolve())
    assert candidate.config["shared_credentials_file"] == str((tmp_path / "creds/custom").resolve())


def test_aws_malformed_explicit_config_override_fails_closed(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.ini"
    config.write_text("[default\n", encoding="utf-8")
    credentials = tmp_path / "credentials.ini"
    credentials.write_text("[default]\naws_access_key_id=x\n", encoding="utf-8")

    assert aws_bedrock.scan(
        {
            "AWS_CONFIG_FILE": str(config),
            "AWS_SHARED_CREDENTIALS_FILE": str(credentials),
            "AWS_DEFAULT_REGION": "us-east-1",
        },
        tmp_path,
    ) == []


def test_google_vertex_project_and_location_alias_precedence(tmp_path: Path) -> None:
    adc = tmp_path / "adc.json"
    _write_json(
        adc,
        {
            "type": "authorized_user",
            "client_id": "id",
            "refresh_token": "token",
            "quota_project_id": "file",
        },
    )
    [candidate] = google_vertex.scan(
        {
            "GOOGLE_APPLICATION_CREDENTIALS": str(adc), "GCLOUD_PROJECT": "alias-project",
            "VERTEXAI_LOCATION": "europe-west4", "TRANCE_MODEL_GOOGLE_VERTEX": " model ",
        }, tmp_path
    )
    assert candidate.config["project"] == "alias-project"
    assert candidate.config["location"] == "europe-west4"
    assert candidate.model_name == "model"


def test_google_vertex_explicit_relative_adc_uses_injected_home(tmp_path: Path) -> None:
    adc = tmp_path / "relative.json"
    _write_json(
        adc,
        {
            "type": "service_account",
            "project_id": "project",
            "client_email": "a",
            "private_key": "key",
        },
    )
    [candidate] = google_vertex.scan(
        {"GOOGLE_APPLICATION_CREDENTIALS": "relative.json"}, tmp_path
    )
    assert candidate.source == "env:GOOGLE_APPLICATION_CREDENTIALS"
    assert candidate.config["adc_file"] == str(adc)


def test_qwen_environment_key_wins_and_selected_settings_are_used(tmp_path: Path) -> None:
    settings = tmp_path / ".qwen" / "settings.json"
    _write_json(
        settings,
        {
            "modelProviders": {
                "openai": [
                    {
                        "id": "qwen-custom",
                        "baseUrl": "https://coding-intl.dashscope.aliyuncs.com/v1",
                        "envKey": "BAILIAN_CODING_PLAN_API_KEY",
                    }
                ]
            },
            "env": {"BAILIAN_CODING_PLAN_API_KEY": "sk-sp-file"},
        },
    )
    [candidate] = qwen_code.scan({"BAILIAN_CODING_PLAN_API_KEY": "sk-sp-env"}, tmp_path)
    assert candidate.secret == "sk-sp-env"
    assert candidate.model_name == "qwen-custom"
    assert candidate.config["base_url"].endswith("/v1")


def test_qwen_malformed_settings_still_allows_valid_explicit_key(tmp_path: Path) -> None:
    path = tmp_path / ".qwen" / "settings.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")
    [candidate] = qwen_code.scan({"BAILIAN_CODING_PLAN_API_KEY": "sk-sp-explicit"}, tmp_path)
    assert candidate.source == "env:BAILIAN_CODING_PLAN_API_KEY"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("export MISTRAL_API_KEY='quoted'", "quoted"),
        ("MISTRAL_API_KEY=unquoted # comment", "unquoted"),
    ],
)
def test_mistral_dotenv_parsing_handles_export_quotes_and_comments(
    tmp_path: Path, line: str, expected: str
) -> None:
    env_file = tmp_path / ".vibe" / ".env"
    env_file.parent.mkdir()
    env_file.write_text(line, encoding="utf-8")
    [candidate] = mistral_vibe.scan({}, tmp_path)
    assert candidate.secret == expected


def test_mistral_invalid_utf8_dotenv_is_ignored(tmp_path: Path) -> None:
    env_file = tmp_path / ".vibe" / ".env"
    env_file.parent.mkdir()
    env_file.write_bytes(b"MISTRAL_API_KEY=\xff")
    assert mistral_vibe.scan({}, tmp_path) == []


def test_kimi_missing_cli_and_invalid_credentials_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    share = tmp_path / ".kimi"
    (share / "credentials").mkdir(parents=True)
    (share / "credentials" / "kimi-code.json").write_text("{}", encoding="utf-8")
    (share / "config.toml").write_text(
        '[providers."kimi-code".oauth]\nkey="oauth/kimi-code"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(kimi_code.shutil, "which", lambda name, path: None)
    assert kimi_code.scan({"PATH": "tools"}, tmp_path) == []
    (share / "config.toml").write_bytes(b"[providers\xff")
    assert kimi_code.scan({"PATH": "tools"}, tmp_path) == []
