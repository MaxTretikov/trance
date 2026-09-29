from pathlib import Path

from trance.sources.mistral_vibe import scan


def test_scan_finds_explicit_environment_key(tmp_path: Path) -> None:
    secret = "mistral-test-key"

    [candidate] = scan({"MISTRAL_API_KEY": secret}, tmp_path)

    assert candidate.provider == "mistral"
    assert candidate.auth_kind == "api_key"
    assert candidate.source == "env:MISTRAL_API_KEY"
    assert candidate.model_name == "mistral-small-latest"
    assert candidate.secret == secret


def test_scan_reads_documented_vibe_dotenv(tmp_path: Path) -> None:
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / ".env").write_text(
        "OTHER=value\nexport MISTRAL_API_KEY='mistral-test-key'\n", encoding="utf-8"
    )

    [candidate] = scan({}, tmp_path)

    assert candidate.secret == "mistral-test-key"
    assert candidate.source == "~/.vibe/.env:MISTRAL_API_KEY"


def test_environment_key_takes_precedence(tmp_path: Path) -> None:
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / ".env").write_text("MISTRAL_API_KEY=file-key\n", encoding="utf-8")

    [candidate] = scan(
        {"MISTRAL_API_KEY": "env-key", "TRANCE_MODEL_MISTRAL": "mistral-large"}, tmp_path
    )

    assert candidate.secret == "env-key"
    assert candidate.source == "env:MISTRAL_API_KEY"
    assert candidate.model_name == "mistral-large"


def test_scan_ignores_missing_or_empty_key(tmp_path: Path) -> None:
    assert scan({}, tmp_path) == []

    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / ".env").write_text("MISTRAL_API_KEY=\n", encoding="utf-8")
    assert scan({}, tmp_path) == []
