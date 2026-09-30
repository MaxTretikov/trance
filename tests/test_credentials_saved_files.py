"""Credential extraction tests for saved CLI auth caches."""

import json
from pathlib import Path

from trance.sources import codex, gemini_cli, grok_consumer, opencode
from trance.types import Candidate


def _candidate(provider: str, path: Path, **config: str) -> Candidate:
    return Candidate(
        provider=provider,
        auth_kind="account",
        source=str(path),
        model_name="test-model",
        config={"auth_file": str(path), **config},
    )


def test_codex_extracts_only_valid_oauth_tokens(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "access-fixture",
                    "refresh_token": "refresh-fixture",
                    "id_token": "id-fixture",
                    "unexpected": "ignored",
                },
            }
        ),
        encoding="utf-8",
    )
    assert codex.extract_credentials(_candidate("openai-codex", path)) == {
        "access_token": "access-fixture",
        "refresh_token": "refresh-fixture",
        "id_token": "id-fixture",
    }


def test_gemini_extracts_available_nonblank_oauth_fields(tmp_path: Path) -> None:
    path = tmp_path / "oauth_creds.json"
    path.write_text('{"refresh_token":"refresh-fixture", "access_token":"access-fixture"}')
    assert gemini_cli.extract_credentials(_candidate("gemini-cli", path)) == {
        "refresh_token": "refresh-fixture",
        "access_token": "access-fixture",
    }


def test_gemini_requires_refresh_token_before_returning_access_token(tmp_path: Path) -> None:
    path = tmp_path / "oauth_creds.json"
    path.write_text('{"access_token":"access-fixture"}')
    assert gemini_cli.extract_credentials(_candidate("gemini-cli", path)) == {}


def test_opencode_extracts_selected_vendor_shape(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "openai": {"type": "api", "key": "api-fixture"},
                "anthropic": {"type": "api", "key": "other-fixture"},
            }
        )
    )
    assert opencode.extract_credentials(
        _candidate("opencode:openai", path, vendor_provider="openai")
    ) == {"key": "api-fixture"}


def test_grok_opaque_oidc_session_returns_no_credentials(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    path.write_text('{"account":{"auth_mode":"oidc", "key":"opaque-fixture"}}')
    assert grok_consumer.extract_credentials(_candidate("grok-consumer", path)) == {}


def test_extractors_reject_symlinked_auth_files(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_text('{"refresh_token":"fixture"}')
    link = tmp_path / "auth.json"
    link.symlink_to(target)
    candidate = _candidate("gemini-cli", link)
    assert gemini_cli.extract_credentials(candidate) == {}


def test_extractors_reject_candidates_for_another_provider(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "access-fixture",
                    "refresh_token": "refresh-fixture",
                    "id_token": "id-fixture",
                },
            }
        )
    )
    assert codex.extract_credentials(_candidate("gemini-cli", path)) == {}
    assert gemini_cli.extract_credentials(_candidate("openai-codex", path)) == {}
    assert grok_consumer.extract_credentials(_candidate("openai-codex", path)) == {}
    assert opencode.extract_credentials(_candidate("openai-codex", path)) == {}
