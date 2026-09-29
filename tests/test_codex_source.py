import json

from trance.sources import codex


def _fake_auth_cache(path, **overrides):
    """Make an isolated Codex CLI-shaped auth file for scanner tests."""
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "fixture-access-token",
                    "refresh_token": "fixture-refresh-token",
                    "id_token": "fixture-id-token",
                }
            }
            | overrides
        ),
        encoding="utf-8",
    )


def test_scan_finds_codex_home_auth_cache_without_loading_secrets(tmp_path):
    codex_home = tmp_path / "custom-codex-home"
    auth_file = codex_home / "auth.json"
    _fake_auth_cache(auth_file)

    candidates = codex.scan({"CODEX_HOME": str(codex_home)}, tmp_path)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "openai-codex"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == str(auth_file)
    assert candidate.model_name == "gpt-5.6-luna"
    assert candidate.secret is None
    assert candidate.config == {"codex_home": str(codex_home)}


def test_scan_uses_default_codex_directory(tmp_path):
    codex_home = tmp_path / ".codex"
    auth_file = codex_home / "auth.json"
    _fake_auth_cache(auth_file)

    candidates = codex.scan({}, tmp_path)

    assert len(candidates) == 1
    assert candidates[0].source == str(auth_file)
    assert candidates[0].secret is None


def test_scan_uses_nonblank_model_override(tmp_path):
    auth_file = tmp_path / ".codex" / "auth.json"
    _fake_auth_cache(auth_file)

    [candidate] = codex.scan({"TRANCE_MODEL_OPENAI_CODEX": "  o3  "}, tmp_path)

    assert candidate.model_name == "o3"
    [default_candidate] = codex.scan({"TRANCE_MODEL_OPENAI_CODEX": "  "}, tmp_path)
    assert default_candidate.model_name == "gpt-5.6-luna"


def test_scan_ignores_missing_or_non_file_auth_cache(tmp_path):
    assert codex.scan({}, tmp_path) == []

    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    assert codex.scan({}, tmp_path) == []


def test_scan_rejects_api_key_auth_cache_without_exposing_key(tmp_path):
    auth_file = tmp_path / ".codex" / "auth.json"
    _fake_auth_cache(
        auth_file,
        auth_mode="apikey",
        OPENAI_API_KEY="sk-secret-fixture",
        tokens=None,
    )

    assert codex.scan({}, tmp_path) == []
    assert "sk-secret-fixture" not in repr(codex.scan({}, tmp_path))


def test_scan_accepts_legacy_tokens_only_cache(tmp_path):
    auth_file = tmp_path / ".codex" / "auth.json"
    _fake_auth_cache(auth_file)
    payload = json.loads(auth_file.read_text(encoding="utf-8"))
    payload.pop("auth_mode")
    auth_file.write_text(json.dumps(payload), encoding="utf-8")

    assert len(codex.scan({}, tmp_path)) == 1


def test_scan_rejects_malformed_incomplete_and_oversize_cache(tmp_path):
    auth_file = tmp_path / ".codex" / "auth.json"
    _fake_auth_cache(auth_file, tokens={"access_token": "only-one-token"})
    assert codex.scan({}, tmp_path) == []

    auth_file.write_text("{not-json", encoding="utf-8")
    assert codex.scan({}, tmp_path) == []

    auth_file.write_text("x" * (codex._MAX_AUTH_BYTES + 1), encoding="utf-8")
    assert codex.scan({}, tmp_path) == []
