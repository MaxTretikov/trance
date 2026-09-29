from trance.sources import env_api


def test_scan_returns_supported_documented_keys_and_model_overrides(tmp_path):
    candidates = env_api.scan(
        {
            "OPENAI_API_KEY": "  openai-fixture  ",
            "ANTHROPIC_API_KEY": "anthropic-fixture",
            "GOOGLE_API_KEY": "google-fixture",
            "GROQ_API_KEY": "groq-fixture",
            "XAI_API_KEY": "xai-fixture",
            "TRANCE_MODEL_OPENAI": "gpt-test",
        },
        tmp_path,
    )

    assert [(item.provider, item.source) for item in candidates] == [
        ("openai", "env:OPENAI_API_KEY"),
        ("anthropic", "env:ANTHROPIC_API_KEY"),
        ("google", "env:GOOGLE_API_KEY"),
        ("xai", "env:XAI_API_KEY"),
        ("groq", "env:GROQ_API_KEY"),
    ]
    assert [item.secret for item in candidates] == [
        "openai-fixture",
        "anthropic-fixture",
        "google-fixture",
        "xai-fixture",
        "groq-fixture",
    ]
    assert candidates[0].model_name == "gpt-test"
    assert candidates[1].model_name == "claude-haiku-4-5-20251001"
    assert candidates[3].model_name == "grok-4.7"
    assert candidates[4].model_name == "openai/gpt-oss-120b"
    assert candidates[0].auth_kind == "api_key"
    assert "openai-fixture" not in repr(candidates[0])


def test_scan_preserves_distinct_variable_sources_and_ignores_unknown_token_names(tmp_path):
    candidates = env_api.scan(
        {
            "GOOGLE_API_KEY": "google-one",
            "GEMINI_API_KEY": "google-two",
            "CUSTOM_MODEL_TOKEN": "unrecognized",
            "RANDOM_API_KEY": "unrecognized-too",
        },
        tmp_path,
    )

    assert len(candidates) == 2
    assert [candidate.source for candidate in candidates] == [
        "env:GOOGLE_API_KEY",
        "env:GEMINI_API_KEY",
    ]
    assert [candidate.secret for candidate in candidates] == ["google-one", "google-two"]


def test_scan_skips_blank_values_and_blank_model_override(tmp_path):
    assert env_api.scan(
        {"OPENAI_API_KEY": "  ", "TRANCE_MODEL_OPENAI": " custom-model "}, tmp_path
    ) == []

    candidate = env_api.scan(
        {"OPENAI_API_KEY": "fixture", "TRANCE_MODEL_OPENAI": "  "}, tmp_path
    )[0]
    assert candidate.model_name == "gpt-4o-mini"
