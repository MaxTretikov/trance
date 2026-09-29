from pathlib import Path

from trance.sources.minimax_coding import scan


def test_scan_finds_explicit_token_plan_key(tmp_path: Path) -> None:
    key = "sk-cp-synthetic-subscription-key"

    [candidate] = scan({"MINIMAX_API_KEY": key}, tmp_path)

    assert candidate.provider == "minimax-coding-plan"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == "env:MINIMAX_API_KEY"
    assert candidate.model_name == "MiniMax-M2.7"
    assert candidate.secret == key
    assert candidate.config == {
        "base_url": "https://api.minimax.io/v1",
        "api_style": "openai",
    }


def test_scan_uses_documented_mainland_host_and_optional_model(tmp_path: Path) -> None:
    [candidate] = scan(
        {
            "MINIMAX_API_KEY": "sk-cp-synthetic-key",
            "MINIMAX_API_HOST": "https://api.minimaxi.com/",
            "MINIMAX_MODEL": "MiniMax-M3",
        },
        tmp_path,
    )

    assert candidate.model_name == "MiniMax-M3"
    assert candidate.config["base_url"] == "https://api.minimaxi.com/v1"


def test_scan_rejects_non_subscription_or_unknown_endpoint(tmp_path: Path) -> None:
    assert scan({"MINIMAX_API_KEY": "sk-api-ordinary-api-key"}, tmp_path) == []
    assert scan(
        {
            "MINIMAX_API_KEY": "sk-cp-synthetic-key",
            "MINIMAX_API_HOST": "https://attacker.example/v1",
        },
        tmp_path,
    ) == []


def test_scan_does_not_search_files(tmp_path: Path) -> None:
    (tmp_path / ".minimax").mkdir()
    (tmp_path / ".minimax" / "auth.json").write_text(
        '{"api_key":"sk-cp-must-not-be-read"}', encoding="utf-8"
    )

    assert scan({}, tmp_path) == []
