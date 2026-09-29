from pathlib import Path
from types import SimpleNamespace

import pytest

from trance import discovery
from trance.types import Candidate, FoundModel


def _candidate(source: str, secret: str = "secret", provider: str = "demo") -> Candidate:
    return Candidate(
        provider=provider,
        auth_kind="api_key",
        source=source,
        model_name="demo-model",
        secret=secret,
    )


def test_scan_materializes_candidates_and_preserves_sources(monkeypatch, tmp_path: Path):
    first = _candidate("env:DEMO_A")
    second = _candidate("env:DEMO_B", secret="another")

    def fake_import(name: str):
        if name.startswith("trance.sources."):
            return SimpleNamespace(scan=lambda env, home: [first, second])
        if name == "trance.models":
            return SimpleNamespace(build_model=lambda candidate: ("client", candidate.source))
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("fake", "demo"),))

    result = discovery.scan({}, tmp_path)

    assert result == [
        FoundModel(
            provider="demo",
            auth_kind="api_key",
            source="env:DEMO_A",
            model_name="demo-model",
            model=("client", "env:DEMO_A"),
        ),
        FoundModel(
            provider="demo",
            auth_kind="api_key",
            source="env:DEMO_B",
            model_name="demo-model",
            model=("client", "env:DEMO_B"),
        ),
    ]


def test_scan_filters_and_deduplicates(monkeypatch, tmp_path: Path):
    candidate = _candidate("env:KEY")
    calls = []

    def fake_import(name: str):
        if name.startswith("trance.sources."):
            return SimpleNamespace(scan=lambda env, home: [candidate, candidate])
        if name == "trance.models":
            return SimpleNamespace(build_model=lambda item: calls.append(item) or object())
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("fake", "demo"),))

    assert discovery.scan({}, tmp_path, providers=["other"]) == []
    result = discovery.scan({}, tmp_path, providers=["demo"])
    assert len(result) == 1
    assert len(calls) == 1


def test_strict_reraises_source_errors(monkeypatch, tmp_path: Path):
    def fake_import(name: str):
        if name.startswith("trance.sources."):
            raise RuntimeError("source unavailable")
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("fake", "demo"),))

    with pytest.raises(RuntimeError, match="source unavailable"):
        discovery.scan({}, tmp_path, strict=True)


def test_secret_fingerprint_is_not_the_plain_secret():
    candidate = _candidate("env:KEY", secret="never-log-this")
    assert discovery._fingerprint(candidate)[-1] != candidate.secret


def test_same_secret_from_multiple_sources_is_built_once(monkeypatch, tmp_path: Path):
    candidates = [_candidate("env:ONE"), _candidate("env:TWO")]
    built = []

    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(scan=lambda env, home: candidates)
            if name.startswith("trance.sources.")
            else SimpleNamespace(build_model=lambda item: built.append(item) or object())
        ),
    )
    monkeypatch.setattr(discovery, "_SOURCES", (("fake", "demo"),))

    result = discovery.scan({}, tmp_path)

    assert len(result) == 1
    assert result[0].source == "env:ONE"
    assert len(built) == 1


def test_codex_uses_standard_model_factory(monkeypatch):
    candidate = Candidate(
        provider="openai-codex",
        auth_kind="subscription",
        source="/home/user/.codex/auth.json",
        model_name="gpt-5.6-luna",
        config={"codex_home": "/home/user/.codex"},
    )
    called = []

    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            build_model=lambda item: called.append((name, item)) or "codex-model"
        ),
    )

    assert discovery._materialize(candidate) == "codex-model"
    assert called[0][0] == "trance.models"


def test_delegated_candidate_without_cli_adapter_is_rejected(monkeypatch):
    candidate = Candidate(
        provider="unsupported-cli",
        auth_kind="subscription",
        source="kimi CLI OAuth",
        model_name="kimi-for-coding",
        config={"integration": "kimi-cli", "command": "kimi"},
    )
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            build_cli_model=lambda item: (_ for _ in ()).throw(ValueError("unsupported"))
        ),
    )

    with pytest.raises(ValueError, match="No safe model adapter"):
        discovery._materialize(candidate)


def test_subscription_candidates_use_supported_model_factories(monkeypatch, tmp_path: Path):
    candidates = {
        "qwen_code": Candidate(
            provider="qwen-code",
            auth_kind="subscription",
            source="env:BAILIAN_CODING_PLAN_API_KEY",
            model_name="qwen-coder",
            secret="sk-sp-qwen",
            config={"base_url": "https://coding.dashscope.aliyuncs.com/v1", "api_style": "openai"},
        ),
        "poe": Candidate(
            provider="poe",
            auth_kind="subscription",
            source="env:POE_API_KEY",
            model_name="GPT-5.4",
            secret="poe-key",
        ),
        "minimax_coding": Candidate(
            provider="minimax-coding-plan",
            auth_kind="subscription",
            source="env:MINIMAX_API_KEY",
            model_name="MiniMax-M2.7",
            secret="sk-cp-minimax",
            config={"base_url": "https://api.minimax.io/v1", "api_style": "openai"},
        ),
        "codex": Candidate(
            provider="openai-codex",
            auth_kind="subscription",
            source="/tmp/codex/auth.json",
            model_name="gpt-5.6-luna",
            config={"codex_home": "/tmp/codex"},
        ),
        "copilot": Candidate(
            provider="github-copilot",
            auth_kind="subscription",
            source="env:GITHUB_COPILOT_API_KEY",
            model_name="gpt-5.4",
            secret="copilot-key",
        ),
        "claude_code": Candidate(
            provider="claude-code",
            auth_kind="subscription",
            source="env:CLAUDE_CODE_OAUTH_TOKEN",
            model_name="claude-sonnet",
            secret="claude-token",
            config={"credential_env": "CLAUDE_CODE_OAUTH_TOKEN"},
        ),
        "huggingface_login": Candidate(
            provider="huggingface",
            auth_kind="account",
            source="huggingface_hub",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            secret="hf-saved-token",
        ),
    }
    built_models = []
    built_cli = []

    def fake_import(name: str):
        if name.startswith("trance.sources."):
            source_name = name.rsplit(".", 1)[1]
            return SimpleNamespace(scan=lambda env, home: [candidates[source_name]])
        if name == "trance.models":
            return SimpleNamespace(
                build_model=lambda item: built_models.append(item) or ("api", item.provider)
            )
        if name == "trance.cli_model":
            return SimpleNamespace(
                build_cli_model=lambda item: built_cli.append(item) or ("cli", item.provider)
            )
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        discovery,
        "_SOURCES",
        tuple((name, candidate.provider) for name, candidate in candidates.items()),
    )
    monkeypatch.setattr(discovery.os, "access", lambda path, mode: True)
    monkeypatch.setattr(discovery.Path, "is_file", lambda self: True)

    results = discovery.scan({"PATH": "/mock/bin"}, tmp_path)

    assert [result.provider for result in results] == [
        candidate.provider for candidate in candidates.values()
    ]
    assert [provider for kind, provider in (result.model for result in results[:5])] == [
        "qwen-code",
        "poe",
        "minimax-coding-plan",
        "openai-codex",
        "github-copilot",
    ]
    assert results[5].model == ("cli", "claude-code")
    assert [item.provider for item in built_cli] == ["claude-code"]
    assert [item.provider for item in built_models] == [
        "qwen-code",
        "poe",
        "minimax-coding-plan",
        "openai-codex",
        "github-copilot",
        "huggingface",
    ]


def test_zai_source_is_discoverable_for_general_and_coding_plan_keys(monkeypatch, tmp_path):
    candidates = []

    def fake_import(name: str):
        if name == "trance.sources.zai":
            from trance.sources.zai import scan

            return SimpleNamespace(scan=scan)
        if name == "trance.models":
            return SimpleNamespace(build_model=lambda item: candidates.append(item) or "model")
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("zai", "*"),))

    result = discovery.scan(
        {"ZAI_API_KEY": "general-key", "ZAI_CODING_PLAN_API_KEY": "coding-key"}, tmp_path
    )

    assert [item.provider for item in result] == ["zai", "zai-coding-plan"]
    assert [item.auth_kind for item in result] == ["api_key", "subscription"]


def test_copilot_cli_token_is_resolved_in_memory_before_model_build(monkeypatch, tmp_path: Path):
    candidate = Candidate(
        provider="github-copilot",
        auth_kind="subscription",
        source="github-cli",
        model_name="gpt-5.4",
        config={"resolver": "command", "executable": "gh", "args": "auth token"},
    )
    built = []
    run_args = []

    monkeypatch.setattr(
        discovery.subprocess,
        "run",
        lambda *args, **kwargs: (
            run_args.append((args, kwargs))
            or SimpleNamespace(returncode=0, stdout="sensitive-token\n", stderr="never-log")
        ),
    )
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(scan=lambda env, home: [candidate])
            if name.startswith("trance.sources.")
            else SimpleNamespace(build_model=lambda item: built.append(item) or "copilot-model")
        ),
    )
    monkeypatch.setattr(discovery, "_SOURCES", (("copilot", "github-copilot"),))

    result = discovery.scan(
        {
            "PATH": "/mock/bin",
            "GH_TOKEN": "generic-pat-one",
            "GITHUB_TOKEN": "generic-pat-two",
            "GH_HOST": "ghe.example.test",
            "GH_ENTERPRISE_TOKEN": "enterprise-pat-one",
            "GITHUB_ENTERPRISE_TOKEN": "enterprise-pat-two",
        },
        tmp_path,
    )

    assert len(result) == 1
    assert result[0].model == "copilot-model"
    assert result[0].source == "github-cli"
    assert built[0].source == "resolved:github-cli"
    assert built[0].secret == "sensitive-token"
    assert run_args[0][0][0] == ["gh", "auth", "token"]
    assert run_args[0][1]["shell"] is False
    assert run_args[0][1]["env"]["HOME"] == str(tmp_path)
    assert run_args[0][1]["stderr"] == discovery.subprocess.DEVNULL
    assert "GH_TOKEN" not in run_args[0][1]["env"]
    assert "GITHUB_TOKEN" not in run_args[0][1]["env"]
    assert "GH_HOST" not in run_args[0][1]["env"]
    assert "GH_ENTERPRISE_TOKEN" not in run_args[0][1]["env"]
    assert "GITHUB_ENTERPRISE_TOKEN" not in run_args[0][1]["env"]


def test_unsafe_or_unsupported_cli_sources_are_not_scanned():
    source_modules = {module for module, _provider in discovery._SOURCES}

    assert source_modules.isdisjoint({"cursor", "kiro", "tabnine", "kimi_code"})
    assert {
        "qwen_code",
        "huggingface_login",
        "grok_consumer",
        "gemini_cli",
        "cody",
        "opencode",
        "aws_bedrock",
        "google_vertex",
    }.issubset(
        source_modules
    )


def test_grok_consumer_is_registered_before_generic_api_sources():
    source_modules = [module for module, _provider in discovery._SOURCES]
    assert source_modules.index("grok_consumer") < source_modules.index("compatible_api")


def test_gemini_cli_is_registered_before_generic_api_sources():
    source_modules = [module for module, _provider in discovery._SOURCES]
    assert source_modules.index("gemini_cli") < source_modules.index("compatible_api")


def test_new_provider_sources_are_registered_before_generic_api_sources():
    registrations = dict(discovery._SOURCES)
    generic_index = [module for module, _provider in discovery._SOURCES].index("compatible_api")

    assert registrations["cody"] == "sourcegraph-cody"
    assert registrations["opencode"] == "opencode:*"
    assert registrations["aws_bedrock"] == "aws-bedrock"
    assert registrations["google_vertex"] == "google-vertex"
    assert all(
        [module for module, _provider in discovery._SOURCES].index(module) < generic_index
        for module in ("cody", "opencode", "aws_bedrock", "google_vertex")
    )


def test_grok_consumer_candidate_uses_cli_model_factory(monkeypatch, tmp_path: Path):
    candidate = Candidate(
        provider="grok-consumer",
        auth_kind="subscription",
        source="grok CLI OIDC",
        model_name="grok-4.7",
        config={
            "integration": "grok-cli",
            "command": "/fake/bin/grok",
            "auth_home": str(tmp_path / ".grok"),
            "saved_login": "true",
        },
    )
    built = []

    def fake_import(name: str):
        if name == "trance.sources.grok_consumer":
            return SimpleNamespace(scan=lambda env, home: [candidate])
        if name == "trance.cli_model":
            return SimpleNamespace(
                build_cli_model=lambda item: built.append(item) or "grok-cli-model"
            )
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("grok_consumer", "grok-consumer"),))

    result = discovery.scan({}, tmp_path, providers=["grok-consumer"])

    assert len(result) == 1
    assert result[0].provider == "grok-consumer"
    assert result[0].model == "grok-cli-model"
    assert built == [candidate]


def test_gemini_cli_candidate_uses_cli_model_factory(monkeypatch, tmp_path: Path):
    candidate = Candidate(
        provider="gemini-cli",
        auth_kind="account",
        source="gemini-cli-login",
        model_name="auto",
        config={
            "command": "/fake/bin/gemini",
            "auth_home": str(tmp_path / ".gemini"),
            "saved_login": "true",
        },
    )
    built = []

    def fake_import(name: str):
        if name == "trance.sources.gemini_cli":
            return SimpleNamespace(scan=lambda env, home: [candidate])
        if name == "trance.cli_model":
            return SimpleNamespace(
                build_cli_model=lambda item: built.append(item) or "gemini-cli-model"
            )
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("gemini_cli", "gemini-cli"),))

    result = discovery.scan({}, tmp_path, providers=["gemini-cli"])

    assert len(result) == 1
    assert result[0].provider == "gemini-cli"
    assert result[0].model == "gemini-cli-model"
    assert built == [candidate]


def test_cody_and_opencode_candidates_use_cli_factory_with_dynamic_filter(
    monkeypatch, tmp_path: Path
):
    candidates = [
        Candidate(
            provider="sourcegraph-cody",
            auth_kind="subscription",
            source="cody-login",
            model_name="cody-default",
            config={"command": "/fake/bin/cody", "saved_login": "true"},
        ),
        Candidate(
            provider="opencode:openai",
            auth_kind="api_key",
            source="opencode-auth",
            model_name="openai/gpt-4o-mini",
            config={"integration": "opencode-cli", "command": "/fake/bin/opencode"},
        ),
        Candidate(
            provider="opencode:anthropic",
            auth_kind="account",
            source="opencode-auth",
            model_name="anthropic/claude-3-5-haiku-latest",
            config={"integration": "opencode-cli", "command": "/fake/bin/opencode"},
        ),
    ]
    built = []

    def fake_import(name: str):
        if name == "trance.sources.opencode":
            return SimpleNamespace(scan=lambda env, home: candidates[1:])
        if name == "trance.cli_model":
            return SimpleNamespace(build_cli_model=lambda item: built.append(item) or item.provider)
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(discovery, "_SOURCES", (("opencode", "opencode:*"),))

    result = discovery.scan({}, tmp_path, providers=["opencode:openai"])

    assert [item.provider for item in result] == ["opencode:openai"]
    assert [item.provider for item in built] == ["opencode:openai"]


def test_aws_and_vertex_candidates_use_standard_model_factory(monkeypatch):
    candidates = [
        Candidate(
            provider="aws-bedrock",
            auth_kind="account",
            source="aws-config",
            model_name="anthropic.claude-3-5-sonnet",
            config={"integration": "aws-bedrock"},
        ),
        Candidate(
            provider="google-vertex",
            auth_kind="account",
            source="google-adc",
            model_name="gemini-2.5-pro",
            config={"integration": "google-vertex"},
        ),
    ]
    built = []

    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: SimpleNamespace(build_model=lambda item: built.append(item) or item.provider),
    )

    assert [discovery._materialize(candidate) for candidate in candidates] == [
        "aws-bedrock",
        "google-vertex",
    ]
    assert built == candidates


def test_multi_provider_sources_are_scanned_with_candidate_level_filtering(
    monkeypatch, tmp_path: Path
):
    compatible = Candidate(
        provider="compatible-one",
        auth_kind="api_key",
        source="env:COMPATIBLE_ONE_KEY",
        model_name="compatible-model",
        secret="compatible-secret",
    )
    cloud = Candidate(
        provider="cloud-one",
        auth_kind="api_key",
        source="env:CLOUD_ONE_KEY",
        model_name="cloud-model",
        secret="cloud-secret",
    )
    compatible_other = Candidate(
        provider="compatible-two",
        auth_kind="api_key",
        source="env:COMPATIBLE_TWO_KEY",
        model_name="compatible-model-two",
        secret="compatible-secret-two",
    )
    cloud_other = Candidate(
        provider="cloud-two",
        auth_kind="api_key",
        source="env:CLOUD_TWO_KEY",
        model_name="cloud-model-two",
        secret="cloud-secret-two",
    )
    scanned = []
    built = []

    def fake_import(name: str):
        if name == "trance.sources.compatible_api":
            return SimpleNamespace(
                scan=lambda env, home: (
                    scanned.append("compatible_api") or [compatible, compatible_other]
                )
            )
        if name == "trance.sources.cloud_api":
            return SimpleNamespace(
                scan=lambda env, home: scanned.append("cloud_api") or [cloud, cloud_other]
            )
        if name == "trance.models":
            return SimpleNamespace(
                build_model=lambda candidate: built.append(candidate) or candidate.provider
            )
        raise AssertionError(name)

    monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        discovery,
        "_SOURCES",
        (("compatible_api", "*"), ("cloud_api", "*")),
    )

    found = discovery.scan({}, tmp_path, providers=["cloud-one"])

    assert scanned == ["compatible_api", "cloud_api"]
    assert [item.provider for item in found] == ["cloud-one"]
    assert [item.provider for item in built] == ["cloud-one"]


def test_claude_candidate_is_skipped_without_cli_on_supplied_path(monkeypatch, tmp_path: Path):
    candidate = Candidate(
        provider="claude-code",
        auth_kind="subscription",
        source="env:CLAUDE_CODE_OAUTH_TOKEN",
        model_name="claude-sonnet",
        secret="claude-token",
    )
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(scan=lambda env, home: [candidate])
            if name.startswith("trance.sources.")
            else SimpleNamespace(build_cli_model=lambda item: object())
        ),
    )
    monkeypatch.setattr(discovery, "_SOURCES", (("claude_code", "claude-code"),))

    assert discovery.scan({"PATH": ""}, tmp_path) == []
