from __future__ import annotations

import sys
import types

from trance.types import Candidate, CredentialMaterial


def test_secret_material_and_redaction() -> None:
    marker = "TRANCE_CREDENTIAL_MARKER"
    candidate = Candidate("openai", "api_key", "env:OPENAI_API_KEY", "gpt", marker)

    material = candidate.credentials

    assert material.values == {"api_key": marker}
    assert material.value == marker
    assert marker not in repr(candidate)
    assert marker not in repr(material)


def test_claude_secret_is_access_token() -> None:
    candidate = Candidate("claude-code", "subscription", "env:TOKEN", "sonnet", "secret")

    assert candidate.credentials.values == {"access_token": "secret"}


def test_aws_values_are_limited_to_explicit_key_fields() -> None:
    candidate = Candidate(
        "aws-bedrock",
        "account",
        "env:AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY",
        "model",
        config={
            "aws_access_key_id": "access",
            "aws_secret_access_key": "secret",
            "region": "us-east-1",
        },
    )

    assert candidate.credentials.values == {
        "aws_access_key_id": "access",
        "aws_secret_access_key": "secret",
    }


def test_saved_session_uses_lazy_source_extractor(monkeypatch) -> None:
    source = types.ModuleType("trance.sources.codex")
    source.extract_credentials = lambda candidate: {"access_token": "oauth"}
    monkeypatch.setitem(sys.modules, "trance.sources.codex", source)
    candidate = Candidate("openai-codex", "subscription", "/auth.json", "gpt")

    assert candidate.credentials.values == {"access_token": "oauth"}


def test_multiple_values_have_no_convenience_value() -> None:
    marker = "TRANCE_CREDENTIAL_MARKER"
    material = CredentialMaterial("account", {"access": marker, "refresh": "refresh"})

    assert material.value is None
    assert marker not in repr(material)
