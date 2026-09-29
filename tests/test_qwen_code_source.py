import json
from pathlib import Path

from trance.sources.qwen_code import scan


def test_scan_finds_explicit_coding_plan_key(tmp_path: Path) -> None:
    token = "sk-sp-test-subscription-key"

    candidates = scan({"BAILIAN_CODING_PLAN_API_KEY": token}, tmp_path)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "qwen-code"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == "env:BAILIAN_CODING_PLAN_API_KEY"
    assert candidate.model_name == "qwen3-coder-plus"
    assert candidate.secret == token
    assert candidate.config == {
        "base_url": "https://coding.dashscope.aliyuncs.com/v1",
        "api_style": "openai",
    }
    assert token not in repr(candidate.config)


def test_scan_reads_documented_settings_env_key(tmp_path: Path) -> None:
    token = "sk-sp-settings-key"
    config_dir = tmp_path / ".qwen"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "modelProviders": {
                    "openai": [
                        {
                            "id": "qwen3.5-plus",
                            "baseUrl": "https://coding-intl.dashscope.aliyuncs.com/v1",
                            "envKey": "BAILIAN_CODING_PLAN_API_KEY",
                        }
                    ]
                },
                "env": {"BAILIAN_CODING_PLAN_API_KEY": token},
            }
        ),
        encoding="utf-8",
    )

    [candidate] = scan({}, tmp_path)

    assert candidate.secret == token
    assert candidate.source == "~/.qwen/settings.json:env.BAILIAN_CODING_PLAN_API_KEY"
    assert candidate.model_name == "qwen3.5-plus"
    assert candidate.config["base_url"] == "https://coding-intl.dashscope.aliyuncs.com/v1"


def test_scan_ignores_non_coding_plan_keys_and_legacy_oauth_cache(
    tmp_path: Path,
) -> None:
    qwen_dir = tmp_path / ".qwen"
    qwen_dir.mkdir()
    (qwen_dir / "oauth_creds.json").write_text(
        '{"access_token":"legacy-oauth-token"}', encoding="utf-8"
    )

    assert scan({"BAILIAN_CODING_PLAN_API_KEY": "ordinary-api-key"}, tmp_path) == []
    assert scan({}, tmp_path) == []


def test_scan_ignores_malformed_settings(tmp_path: Path) -> None:
    qwen_dir = tmp_path / ".qwen"
    qwen_dir.mkdir()
    (qwen_dir / "settings.json").write_text("{broken", encoding="utf-8")

    assert scan({"BAILIAN_CODING_PLAN_API_KEY": "sk-sp-test"}, tmp_path)
