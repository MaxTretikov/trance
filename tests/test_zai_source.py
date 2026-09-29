from pathlib import Path

from trance.sources.zai import scan


def test_scan_discovers_general_and_explicit_coding_plan_keys(tmp_path: Path) -> None:
    candidates = scan(
        {
            "ZAI_API_KEY": "general-key",
            "ZAI_CODING_PLAN_API_KEY": "coding-key",
            "TRANCE_MODEL_ZAI": "glm-custom",
        },
        tmp_path,
    )

    assert [(item.provider, item.auth_kind, item.secret) for item in candidates] == [
        ("zai", "api_key", "general-key"),
        ("zai-coding-plan", "subscription", "coding-key"),
    ]
    assert candidates[0].model_name == "glm-custom"
    assert candidates[0].config == {
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_style": "openai",
    }
    assert candidates[1].model_name == "glm-5.3"
    assert candidates[1].config == {
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_style": "openai",
    }


def test_generic_key_is_never_classified_as_subscription(tmp_path: Path) -> None:
    [candidate] = scan({"ZAI_API_KEY": "ordinary-key"}, tmp_path)

    assert candidate.provider == "zai"
    assert candidate.auth_kind == "api_key"


def test_scan_ignores_blank_values_and_does_not_read_home(tmp_path: Path) -> None:
    (tmp_path / ".zai-auth").write_text("coding-key", encoding="utf-8")

    assert scan({"ZAI_API_KEY": "  ", "ZAI_CODING_PLAN_API_KEY": ""}, tmp_path) == []
