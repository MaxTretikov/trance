from pathlib import Path
from unittest.mock import patch

from trance.sources import huggingface_login


def test_scan_uses_official_resolver_and_model_override(tmp_path: Path):
    fake_env = {"TRANCE_MODEL_HUGGINGFACE": "org/model-test"}
    with (
        patch.object(huggingface_login, "_is_process_context", return_value=True),
        patch.object(huggingface_login, "_resolve_token", return_value="hf-fixture"),
    ):
        candidates = huggingface_login.scan(fake_env, tmp_path)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "huggingface"
    assert candidate.auth_kind == "account"
    assert candidate.source == "huggingface_hub"
    assert candidate.model_name == "org/model-test"
    assert candidate.secret == "hf-fixture"
    assert "hf-fixture" not in repr(candidate)


def test_scan_uses_documented_default_model(tmp_path: Path):
    with (
        patch.object(huggingface_login, "_is_process_context", return_value=True),
        patch.object(huggingface_login, "_resolve_token", return_value="hf-fixture"),
    ):
        candidate = huggingface_login.scan({}, tmp_path)[0]

    assert candidate.model_name == "Qwen/Qwen2.5-72B-Instruct"


def test_synthetic_context_never_calls_global_token_resolver(tmp_path: Path):
    with (
        patch.object(huggingface_login, "_is_process_context", return_value=False),
        patch.object(huggingface_login, "_resolve_token") as resolver,
    ):
        assert huggingface_login.scan({"HF_TOKEN": "synthetic"}, tmp_path) == []

    resolver.assert_not_called()


def test_scan_skips_missing_or_blank_token(tmp_path: Path):
    with (
        patch.object(huggingface_login, "_is_process_context", return_value=True),
        patch.object(huggingface_login, "_resolve_token", return_value="  "),
    ):
        assert huggingface_login.scan({}, tmp_path) == []
