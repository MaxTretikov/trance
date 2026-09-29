from pathlib import Path
from unittest.mock import patch

from trance.sources.kimi_code import scan


def _write_login(share_dir: Path) -> None:
    (share_dir / "credentials").mkdir(parents=True, exist_ok=True)
    # Intentionally token-free fixture: scanner only checks this file exists.
    (share_dir / "credentials" / "kimi-code.json").write_text("{}", encoding="utf-8")
    (share_dir / "config.toml").write_text(
        '''default_model = "kimi-code/k2"

[providers."kimi-code"]
type = "kimi"
base_url = "https://api.kimi.com/coding/v1"
api_key = ""
[providers."kimi-code".oauth]
storage = "file"
key = "oauth/kimi-code"

[models."kimi-code/k2"]
provider = "kimi-code"
model = "kimi-k2"
''',
        encoding="utf-8",
    )


def test_scan_returns_secretless_candidate_for_configured_managed_login(tmp_path: Path) -> None:
    share_dir = tmp_path / ".kimi"
    _write_login(share_dir)

    with patch("trance.sources.kimi_code.shutil.which", return_value="/usr/bin/kimi"):
        [candidate] = scan({"PATH": "/usr/bin"}, tmp_path)

    assert candidate.provider == "kimi-code"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == "kimi CLI OAuth"
    assert candidate.model_name == "kimi-k2"
    assert candidate.secret is None
    assert candidate.config == {
        "integration": "kimi-cli",
        "command": "/usr/bin/kimi",
        "protocol": "print",
        "auth_method": "kimi_code_oauth",
    }


def test_scan_honors_share_dir_override(tmp_path: Path) -> None:
    share_dir = tmp_path / "custom-share"
    _write_login(share_dir)
    with patch("trance.sources.kimi_code.shutil.which", return_value="/usr/bin/kimi"):
        result = scan({"KIMI_SHARE_DIR": str(share_dir)}, tmp_path)
    assert len(result) == 1


def test_tilde_share_dir_uses_injected_home(tmp_path: Path) -> None:
    share_dir = tmp_path / ".kimi"
    _write_login(share_dir)
    with patch("trance.sources.kimi_code.shutil.which", return_value="/usr/bin/kimi"):
        result = scan({"KIMI_SHARE_DIR": "~/.kimi"}, tmp_path)
    assert len(result) == 1


def test_scan_requires_documented_oauth_reference_and_credential_file(tmp_path: Path) -> None:
    share_dir = tmp_path / ".kimi"
    _write_login(share_dir)
    (share_dir / "credentials" / "kimi-code.json").unlink()
    with patch("trance.sources.kimi_code.shutil.which", return_value="/usr/bin/kimi"):
        assert scan({}, tmp_path) == []

    _write_login(share_dir)
    config_path = share_dir / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("oauth/kimi-code", "oauth/other"),
        encoding="utf-8",
    )
    with patch("trance.sources.kimi_code.shutil.which", return_value="/usr/bin/kimi"):
        assert scan({}, tmp_path) == []


def test_scan_requires_kimi_cli(tmp_path: Path) -> None:
    _write_login(tmp_path / ".kimi")
    with patch("trance.sources.kimi_code.shutil.which", return_value=None):
        assert scan({}, tmp_path) == []


def test_scan_ignores_malformed_config(tmp_path: Path) -> None:
    share_dir = tmp_path / ".kimi"
    share_dir.mkdir()
    (share_dir / "config.toml").write_text("[providers", encoding="utf-8")
    with patch("trance.sources.kimi_code.shutil.which", return_value="/usr/bin/kimi"):
        assert scan({}, tmp_path) == []
