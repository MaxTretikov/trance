from trance.sources import poe


def test_scan_reads_documented_poe_key(tmp_path):
    candidates = poe.scan({"POE_API_KEY": " poe-fixture ", "GH_TOKEN": "unrelated"}, tmp_path)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "poe"
    assert candidate.auth_kind == "subscription"
    assert candidate.source == "env:POE_API_KEY"
    assert candidate.model_name == "GPT-5.4"
    assert candidate.secret == "poe-fixture"
    assert "poe-fixture" not in repr(candidate)


def test_scan_supports_model_override_and_ignores_blank_key(tmp_path):
    assert poe.scan({"POE_API_KEY": "  "}, tmp_path) == []

    candidate = poe.scan(
        {"POE_API_KEY": "poe-fixture", "TRANCE_MODEL_POE": " Claude-Sonnet-4.6 "},
        tmp_path,
    )[0]
    assert candidate.model_name == "Claude-Sonnet-4.6"
