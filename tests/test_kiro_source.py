from pathlib import Path

from trance.sources.kiro import scan


def test_scan_reads_supported_api_key_from_environment() -> None:
    candidates = scan({"KIRO_API_KEY": "ksk_example"}, Path("/unused"))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "kiro"
    assert candidate.auth_kind == "api_key"
    assert candidate.source == "env:KIRO_API_KEY"
    assert candidate.secret == "ksk_example"


def test_scan_ignores_missing_or_blank_api_key() -> None:
    assert scan({}, Path("/unused")) == []
    assert scan({"KIRO_API_KEY": "  "}, Path("/unused")) == []
