"""Table-driven edge coverage for environment and endpoint source adapters.

These tests deliberately keep all probes offline: CLI status checks are mocked
and endpoint scanners are exercised with synthetic environment mappings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trance.sources import (
    cloud_api,
    cody,
    compatible_api,
    copilot,
    cursor,
    env_api,
    kiro,
    minimax_coding,
    poe,
    tabnine,
    zai,
)


def test_env_api_same_provider_aliases_keep_sources_and_share_override(tmp_path: Path) -> None:
    candidates = env_api.scan(
        {
            "GOOGLE_API_KEY": " primary-token ",
            "GEMINI_API_KEY": "secondary-token",
            "TRANCE_MODEL_GOOGLE": "  gemini-custom  ",
            "ANTHROPIC_API_KEY": "  ",
            "UNSUPPORTED_API_KEY": "ignored-token",
        },
        tmp_path,
    )

    assert [(item.source, item.secret, item.model_name) for item in candidates] == [
        ("env:GOOGLE_API_KEY", "primary-token", "gemini-custom"),
        ("env:GEMINI_API_KEY", "secondary-token", "gemini-custom"),
    ]


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {"DEEPINFRA_TOKEN": "token", "DEEPINFRA_API_KEY": "token"},
            [("env:DEEPINFRA_TOKEN", "token")],
        ),
        (
            {"DEEPINFRA_TOKEN": "token", "DEEPINFRA_API_KEY": "api"},
            [("env:DEEPINFRA_TOKEN", "token"), ("env:DEEPINFRA_API_KEY", "api")],
        ),
        (
            {"DEEPINFRA_TOKEN": "  ", "DEEPINFRA_API_KEY": "api"},
            [("env:DEEPINFRA_API_KEY", "api")],
        ),
    ],
)
def test_compatible_api_alias_precedence_and_blank_values(
    environment: dict[str, str], expected: list[tuple[str, str]], tmp_path: Path
) -> None:
    candidates = compatible_api.scan(environment, tmp_path)

    assert [(item.source, item.secret) for item in candidates] == expected
    assert all(item.config["api_style"] == "openai" for item in candidates)


def test_compatible_api_override_is_provider_specific_and_trimmed(tmp_path: Path) -> None:
    candidates = compatible_api.scan(
        {
            "MOONSHOT_API_KEY": "moon-token",
            "NVIDIA_API_KEY": "nvidia-token",
            "TRANCE_MODEL_MOONSHOT": "  kimi-custom  ",
            "TRANCE_MODEL_NVIDIA": "  ",
        },
        tmp_path,
    )

    assert [item.model_name for item in candidates] == ["kimi-custom", "openai/gpt-oss-20b"]


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("AZURE_OPENAI_ENDPOINT", "http://resource.openai.azure.com"),
        ("AZURE_OPENAI_ENDPOINT", "https://resource.openai.azure.com:443"),
        ("AZURE_OPENAI_ENDPOINT", "https://user@resource.openai.azure.com"),
        ("AZURE_OPENAI_ENDPOINT", "https://resource.openai.azure.com/path"),
        ("DATABRICKS_HOST", "https://workspace.cloud.databricks.com?token=x"),
        ("DATABRICKS_HOST", "https://workspace.azuredatabricks.net#fragment"),
        ("DATABRICKS_HOST", "https://workspace.cloud.databricks.com:443"),
        ("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/other"),
    ],
)
def test_cloud_api_rejects_noncanonical_or_unsafe_urls(
    variable: str, value: str, tmp_path: Path
) -> None:
    environment = {
        "AZURE_OPENAI_API_KEY": "azure-token",
        "AZURE_OPENAI_ENDPOINT": "https://resource.openai.azure.com",
        "AZURE_OPENAI_DEPLOYMENT": "deployment",
        "DATABRICKS_TOKEN": "databricks-token",
        "DATABRICKS_HOST": "https://workspace.cloud.databricks.com",
        "DATABRICKS_MODEL": "serving-model",
        "DASHSCOPE_API_KEY": "dashscope-token",
    }
    environment[variable] = value

    candidates = cloud_api.scan(environment, tmp_path)

    rejected_provider = {
        "AZURE_OPENAI_ENDPOINT": "azure-openai",
        "DATABRICKS_HOST": "databricks",
        "DASHSCOPE_BASE_URL": "dashscope",
    }[variable]
    assert all(item.provider != rejected_provider for item in candidates)


def test_cloud_api_normalizes_case_and_applies_cloudflare_override(tmp_path: Path) -> None:
    candidates = cloud_api.scan(
        {
            "AZURE_OPENAI_API_KEY": "azure-token",
            "AZURE_OPENAI_ENDPOINT": "https://RESOURCE.OPENAI.AZURE.COM/",
            "AZURE_OPENAI_DEPLOYMENT": "deployment",
            "CLOUDFLARE_ACCOUNT_ID": "A" * 32,
            "CLOUDFLARE_API_KEY": "cloudflare-token",
            "TRANCE_MODEL_CLOUDFLARE": "  cf-custom  ",
        },
        tmp_path,
    )

    assert candidates[0].config["base_url"] == "https://resource.openai.azure.com/openai/v1"
    assert candidates[1].model_name == "cf-custom"
    assert "/accounts/" + "a" * 32 + "/" in candidates[1].config["base_url"]


@pytest.mark.parametrize(
    ("environment", "expected_model"),
    [
        ({"POE_API_KEY": "poe-token", "TRANCE_MODEL_POE": "  poe-custom  "}, "poe-custom"),
        ({"POE_API_KEY": "poe-token", "TRANCE_MODEL_POE": "  "}, "GPT-5.4"),
    ],
)
def test_poe_model_override_and_blank_override(
    environment: dict[str, str], expected_model: str, tmp_path: Path
) -> None:
    [candidate] = poe.scan(environment, tmp_path)
    assert candidate.model_name == expected_model
    assert candidate.secret == "poe-token"


def test_minimax_and_zai_model_overrides_remain_scoped_to_each_source(tmp_path: Path) -> None:
    [minimax] = minimax_coding.scan(
        {"MINIMAX_API_KEY": "sk-cp-token", "MINIMAX_MODEL": "MiniMax-custom"}, tmp_path
    )
    candidates = zai.scan(
        {
            "ZAI_API_KEY": "general-token",
            "ZAI_CODING_PLAN_API_KEY": "coding-token",
            "TRANCE_MODEL_ZAI": " general-custom ",
            "TRANCE_MODEL_ZAI_CODING_PLAN": " coding-custom ",
        },
        tmp_path,
    )

    assert minimax.model_name == "MiniMax-custom"
    assert [item.model_name for item in candidates] == ["general-custom", "coding-custom"]
    assert [item.config["base_url"] for item in candidates] == [
        "https://api.z.ai/api/paas/v4",
        "https://api.z.ai/api/coding/paas/v4",
    ]


@pytest.mark.parametrize(
    "error", [OSError("missing executable"), copilot.subprocess.TimeoutExpired("gh", 2)]
)
def test_copilot_cli_probe_failure_is_fail_closed_and_keeps_env_candidates(
    monkeypatch: pytest.MonkeyPatch, error: Exception, tmp_path: Path
) -> None:
    monkeypatch.setattr(copilot.shutil, "which", lambda *args, **kwargs: "/usr/bin/gh")

    def raise_probe_error(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(copilot.subprocess, "run", raise_probe_error)

    candidates = copilot.scan(
        {"PATH": "/usr/bin", "GITHUB_COPILOT_API_KEY": "copilot-token"}, tmp_path
    )

    assert [(item.source, item.secret) for item in candidates] == [
        ("env:GITHUB_COPILOT_API_KEY", "copilot-token")
    ]


def test_cody_and_cursor_probe_failures_return_no_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cody, "_cli_path", lambda environ: "/usr/bin/cody")
    monkeypatch.setattr(
        cody.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    assert cody.scan({"PATH": "/usr/bin"}, tmp_path) == []

    monkeypatch.setattr(cursor, "_cli_path", lambda environ, home: "/usr/bin/agent")
    monkeypatch.setattr(
        cursor.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    assert cursor.scan({"PATH": "/usr/bin"}, tmp_path) == []


def test_tabnine_strips_token_and_requires_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tabnine.shutil, "which", lambda name, path=None: "/usr/bin/tabnine")
    [candidate] = tabnine.scan(
        {"PATH": "/usr/bin", "TABNINE_TOKEN": "  tabnine-token  "}, tmp_path
    )
    assert candidate.secret == "tabnine-token"

    monkeypatch.setattr(tabnine.shutil, "which", lambda name, path=None: None)
    assert tabnine.scan({"PATH": "/usr/bin", "TABNINE_TOKEN": "tabnine-token"}, tmp_path) == []


def test_kiro_and_minimax_ignore_blank_or_invalid_keys(tmp_path: Path) -> None:
    assert kiro.scan({"KIRO_API_KEY": "  "}, tmp_path) == []
    assert minimax_coding.scan({"MINIMAX_API_KEY": "sk-api-token"}, tmp_path) == []
    assert (
        minimax_coding.scan(
            {"MINIMAX_API_KEY": "sk-cp-token", "MINIMAX_API_HOST": "https://evil.example"},
            tmp_path,
        )
        == []
    )
