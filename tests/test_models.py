import importlib
from types import SimpleNamespace

import pytest

from trance.types import Candidate

models = importlib.import_module("trance.models")


class FakeProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeModel:
    def __init__(self, model_name, *, provider=None):
        self.model_name = model_name
        self.provider = provider


def install_fake_pydantic_ai(monkeypatch):
    modules = {}

    def fake_import(path):
        module = modules.setdefault(path, SimpleNamespace())
        module.OpenAIChatModel = FakeModel
        module.OpenRouterModel = FakeModel
        module.GroqModel = FakeModel
        module.CerebrasModel = FakeModel
        module.AnthropicModel = FakeModel
        module.GoogleModel = FakeModel
        module.MistralModel = FakeModel
        module.CohereModel = FakeModel
        module.XaiModel = FakeModel
        module.HuggingFaceModel = FakeModel
        module.OpenAICodexModel = FakeModel
        module.GitHubCopilotModel = FakeModel
        for class_name in (
            "OpenAIProvider", "AnthropicProvider", "GoogleProvider", "OpenRouterProvider",
            "GroqProvider", "MistralProvider", "CohereProvider", "XaiProvider",
            "CerebrasProvider", "HuggingFaceProvider", "DeepSeekProvider", "FireworksProvider",
            "TogetherProvider", "SambaNovaProvider", "GitHubCopilotProvider",
        ):
            setattr(module, class_name, FakeProvider)
        return module

    monkeypatch.setattr(models, "import_module", fake_import)


@pytest.mark.parametrize("provider", sorted(models._API_PROVIDERS))
def test_builds_api_key_models_with_explicit_provider_key(monkeypatch, provider):
    install_fake_pydantic_ai(monkeypatch)
    config = {}
    if provider in {"qwen-code", "minimax-coding-plan"}:
        config = {
            "api_style": "openai",
            "base_url": next(iter(models._CONFIGURED_BASE_URLS[provider])),
        }
    if provider in models._DYNAMIC_BASE_URL_PROVIDERS:
        config = {
            "api_style": "openai",
            "base_url": {
                "azure-openai": "https://my-resource.openai.azure.com/openai/v1",
                "cloudflare": "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1",
                "databricks": "https://workspace.cloud.databricks.com/serving-endpoints",
                "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }[provider],
        }
    candidate = Candidate(provider, "api_key", "env:TEST_KEY", "model-x", "fake-key", config)

    model = models.build_model(candidate)

    assert model.model_name == "model-x"
    assert model.provider.kwargs["api_key"] == "fake-key"
    expected_base_url = models._API_PROVIDERS[provider].base_url
    if provider in {"qwen-code", "minimax-coding-plan"}:
        expected_base_url = config["base_url"]
    if provider in models._DYNAMIC_BASE_URL_PROVIDERS:
        expected_base_url = config["base_url"]
    if expected_base_url is not None:
        assert model.provider.kwargs["base_url"] == expected_base_url


def test_zai_general_and_coding_plan_use_distinct_official_endpoints(monkeypatch):
    install_fake_pydantic_ai(monkeypatch)

    general = models.build_model(
        Candidate("zai", "api_key", "env:ZAI_API_KEY", "glm-5.3", "general-key")
    )
    coding = models.build_model(
        Candidate(
            "zai-coding-plan",
            "subscription",
            "env:ZAI_CODING_PLAN_API_KEY",
            "glm-5.3",
            "coding-key",
        )
    )

    assert general.provider.kwargs["base_url"] == "https://api.z.ai/api/paas/v4"
    assert coding.provider.kwargs["base_url"] == "https://api.z.ai/api/coding/paas/v4"


def test_codex_uses_first_party_cli_resolver_without_copying_secret(monkeypatch):
    install_fake_pydantic_ai(monkeypatch)
    candidate = Candidate("openai-codex", "subscription", "codex-cli", "gpt-5.6", None)

    model = models.build_model(candidate)

    assert model.model_name == "gpt-5.6"
    assert model.provider is None


def test_codex_rejects_a_custom_path_unless_process_uses_same_codex_home(monkeypatch, tmp_path):
    other_home = tmp_path / "codex-home"
    candidate = Candidate(
        "openai-codex", "subscription", str(other_home / "auth.json"), "gpt-test",
        config={"codex_home": str(other_home)},
    )

    with pytest.raises(models.CredentialPathError):
        models.build_model(candidate)

    monkeypatch.setenv("CODEX_HOME", str(other_home))
    install_fake_pydantic_ai(monkeypatch)
    model = models.build_model(candidate)
    assert model.model_name == "gpt-test"


def test_copilot_uses_documented_token_provider(monkeypatch):
    install_fake_pydantic_ai(monkeypatch)
    candidate = Candidate(
        "github-copilot", "subscription", "env:GITHUB_COPILOT_API_KEY", "gpt-5.4", "fake-token"
    )

    model = models.build_model(candidate)

    assert model.model_name == "gpt-5.4"
    assert model.provider.kwargs == {"api_key": "fake-token"}


@pytest.mark.parametrize(
    "candidate",
    [
        Candidate("github-copilot", "subscription", "github-cli", "gpt-5.4", None),
        Candidate("github-copilot", "subscription", "env:GH_TOKEN", "gpt-5.4", "generic-token"),
        Candidate("github-copilot", "subscription", "env:GITHUB_TOKEN", "gpt-5.4", "generic-token"),
    ],
)
def test_copilot_rejects_unresolved_or_generic_github_credentials(candidate):
    with pytest.raises(models.MissingCredentialError):
        models.build_model(candidate)


def test_missing_dependency_error_does_not_leak_candidate_secret(monkeypatch):
    secret = "do-not-leak-this-value"

    def failing_import(_path):
        raise ModuleNotFoundError(f"optional SDK import failure with {secret}")

    monkeypatch.setattr(models, "import_module", failing_import)
    candidate = Candidate("openai", "api_key", "env:OPENAI_API_KEY", "gpt-x", secret)

    with pytest.raises(models.ModelDependencyError) as exc:
        models.build_model(candidate)

    assert secret not in str(exc.value)


def test_missing_optional_provider_dependency_is_explicit_and_sanitized(monkeypatch):
    def missing_import(_path):
        raise ModuleNotFoundError("optional SDK missing")

    monkeypatch.setattr(models, "import_module", missing_import)
    candidate = Candidate("openai", "api_key", "env:OPENAI_API_KEY", "gpt-x", "fake-key")

    with pytest.raises(models.ModelDependencyError, match="openai"):
        models.build_model(candidate)


def test_unknown_provider_has_a_clear_error():
    candidate = Candidate("unknown-provider", "api_key", "manual", "model-x", "fake-key")

    with pytest.raises(models.UnsupportedProviderError, match="unknown-provider"):
        models.build_model(candidate)


@pytest.mark.parametrize(
    "provider,config",
    [
        ("qwen-code", {"base_url": "https://attacker.example/v1", "api_style": "openai"}),
        ("minimax-coding-plan", {"base_url": "http://api.minimax.io/v1", "api_style": "openai"}),
        ("qwen-code", {"base_url": "https://coding.dashscope.aliyuncs.com/v1"}),
    ],
)
def test_subscription_compatible_endpoints_are_validated(monkeypatch, provider, config):
    install_fake_pydantic_ai(monkeypatch)
    candidate = Candidate(provider, "subscription", "manual", "model-x", "fake-key", config)

    with pytest.raises(models.ModelBuildError):
        models.build_model(candidate)


@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("azure-openai", "http://my-resource.openai.azure.com/openai/v1"),
        ("azure-openai", "https://evil.example/openai/v1"),
        ("azure-openai", "https://my-resource.openai.azure.com/openai/v1/extra"),
        ("azure-openai", "https://user@my-resource.openai.azure.com/openai/v1"),
        ("cloudflare", "https://api.cloudflare.com/client/v4/accounts/not-an-id/ai/v1"),
        ("cloudflare", "https://evil.example/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1"),
        ("cloudflare", "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1?x=1"),
        ("databricks", "https://evil.example/serving-endpoints"),
        ("databricks", "https://workspace.cloud.databricks.com/serving-endpoints/extra"),
        ("databricks", "https://workspace.azuredatabricks.net:443/serving-endpoints"),
        ("dashscope", "https://coding.dashscope.aliyuncs.com/v1"),
        ("dashscope", "https://evil.example/compatible-mode/v1"),
        ("dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1/extra"),
    ],
)
def test_dynamic_provider_endpoints_reject_unapproved_urls(monkeypatch, provider, base_url):
    install_fake_pydantic_ai(monkeypatch)
    candidate = Candidate(
        provider,
        "api_key",
        "manual",
        "model-x",
        "fake-key",
        {"api_style": "openai", "base_url": base_url},
    )

    with pytest.raises(models.ModelBuildError):
        models.build_model(candidate)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
    ],
)
def test_dynamic_dashscope_endpoints_accept_all_documented_regions(monkeypatch, base_url):
    install_fake_pydantic_ai(monkeypatch)
    candidate = Candidate(
        "dashscope",
        "api_key",
        "manual",
        "qwen-plus",
        "fake-key",
        {"api_style": "openai", "base_url": base_url},
    )

    model = models.build_model(candidate)

    assert model.provider.kwargs["base_url"] == base_url


@pytest.mark.parametrize("provider,base_url", sorted(models._OPENAI_COMPATIBLE_FALLBACKS.items()))
def test_native_optional_dependency_falls_back_to_official_openai_endpoint(
    monkeypatch, provider, base_url
):
    from pydantic_ai.models.openai import OpenAIChatModel

    native_spec = models._API_PROVIDERS[provider]
    real_load = models._load

    def missing_native(module, name, name_provider):
        if module == f"models.{native_spec.model_module}" and name == native_spec.model_class:
            raise models.ModelDependencyError("synthetic missing SDK")
        return real_load(module, name, name_provider)

    monkeypatch.setattr(models, "_load", missing_native)
    candidate = Candidate(provider, "api_key", "manual", "synthetic-model", "fake-key")

    model = models.build_model(candidate)

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "synthetic-model"
    assert str(model.provider.client.base_url).rstrip("/") == base_url.rstrip("/")


def test_fallback_does_not_hide_native_provider_constructor_errors(monkeypatch):
    fallback_requested = False

    def fake_load(module, name, provider):
        nonlocal fallback_requested
        if module == "models.groq":
            return FakeModel
        if module == "providers.groq":
            def fail_constructor(**_kwargs):
                raise RuntimeError("native provider constructor failed")

            return fail_constructor
        if module == "models.openai" or module == "providers.openai":
            fallback_requested = True
        return FakeModel if name == "OpenAIChatModel" else FakeProvider

    monkeypatch.setattr(models, "_load", fake_load)
    candidate = Candidate("groq", "api_key", "manual", "synthetic-model", "fake-key")

    with pytest.raises(RuntimeError, match="native provider constructor failed"):
        models.build_model(candidate)

    assert not fallback_requested


@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("crusoe", "https://api.inference.crusoecloud.com/v1"),
        ("siliconflow", "https://api.siliconflow.cn/v1"),
        ("venice", "https://api.venice.ai/api/v1"),
        ("chutes", "https://llm.chutes.ai/v1"),
        ("akashml", "https://api.akashml.com/v1"),
        ("scaleway", "https://api.scaleway.ai/v1"),
        ("friendli", "https://api.friendli.ai/serverless/v1"),
        ("clarifai", "https://api.clarifai.com/v2/ext/openai/v1"),
        ("meta-model-api", "https://api.meta.ai/v1"),
        ("parasail", "https://api.parasail.io/v1"),
        ("nscale", "https://inference.api.nscale.com/v1"),
    ],
)
def test_documented_compatible_provider_endpoints_are_fixed(monkeypatch, provider, base_url):
    install_fake_pydantic_ai(monkeypatch)
    candidate = Candidate(provider, "api_key", "env:TEST_KEY", "model-x", "fake-key")

    model = models.build_model(candidate)

    assert model.provider.kwargs == {"api_key": "fake-key", "base_url": base_url}


def test_bedrock_environment_credentials_create_explicit_client(monkeypatch):
    calls = {}

    class FakeBoto3:
        def client(self, service, **kwargs):
            calls["client"] = (service, kwargs)
            return "bedrock-client"

    class BedrockProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class BedrockConverseModel(FakeModel):
        pass

    modules = {
        "pydantic_ai.models.bedrock": SimpleNamespace(
            BedrockConverseModel=BedrockConverseModel
        ),
        "pydantic_ai.providers.bedrock": SimpleNamespace(BedrockProvider=BedrockProvider),
        "boto3": FakeBoto3(),
    }
    monkeypatch.setattr(models, "import_module", modules.__getitem__)
    candidate = Candidate(
        "aws-bedrock",
        "account",
        "env:AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY",
        "amazon.nova-micro-v1:0",
        config={
            "region": "us-west-2",
            "aws_access_key_id": "AKIA-example",
            "aws_secret_access_key": "secret-example",
            "aws_session_token": "session-example",
        },
    )

    model = models.build_model(candidate)

    assert model.model_name == "amazon.nova-micro-v1:0"
    assert model.provider.kwargs == {"bedrock_client": "bedrock-client"}
    assert calls["client"] == (
        "bedrock-runtime",
        {
            "region_name": "us-west-2",
            "aws_access_key_id": "AKIA-example",
            "aws_secret_access_key": "secret-example",
            "aws_session_token": "session-example",
        },
    )


def test_bedrock_profile_uses_injected_botocore_configuration(monkeypatch):
    calls = {}

    class FakeBotocoreSession:
        def __init__(self):
            self.settings = {}

        def set_config_variable(self, name, value):
            self.settings[name] = value

    class FakeBotocore:
        Session = FakeBotocoreSession

    class FakeBoto3Session:
        def __init__(self, *, botocore_session):
            calls["session"] = botocore_session

        def client(self, service, **kwargs):
            calls["client"] = (service, kwargs)
            return "profile-client"

    class FakeBoto3:
        Session = FakeBoto3Session

    class BedrockProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    modules = {
        "pydantic_ai.models.bedrock": SimpleNamespace(BedrockConverseModel=FakeModel),
        "pydantic_ai.providers.bedrock": SimpleNamespace(BedrockProvider=BedrockProvider),
        "boto3": FakeBoto3,
        "botocore.session": FakeBotocore,
    }
    monkeypatch.setattr(models, "import_module", modules.__getitem__)
    candidate = Candidate(
        "aws-bedrock",
        "account",
        "profile:research",
        "model-x",
        config={
            "region": "eu-west-1",
            "profile_name": "research",
            "shared_config_file": "/synthetic/config",
            "shared_credentials_file": "/synthetic/credentials",
        },
    )

    model = models.build_model(candidate)

    assert model.provider.kwargs == {"bedrock_client": "profile-client"}
    assert calls["session"].settings == {
        "config_file": "/synthetic/config",
        "credentials_file": "/synthetic/credentials",
        "profile": "research",
        "region": "eu-west-1",
    }
    assert calls["client"] == ("bedrock-runtime", {"region_name": "eu-west-1"})


def test_vertex_loads_adc_without_refreshing_credentials(monkeypatch):
    calls = {}

    class GoogleAuth:
        def load_credentials_from_file(self, path):
            calls["adc_file"] = path
            return "credentials", "project-from-file"

    class GoogleCloudProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    modules = {
        "google.auth": GoogleAuth(),
        "pydantic_ai.models.google": SimpleNamespace(GoogleModel=FakeModel),
        "pydantic_ai.providers.google_cloud": SimpleNamespace(
            GoogleCloudProvider=GoogleCloudProvider
        ),
    }
    monkeypatch.setattr(models, "import_module", modules.__getitem__)
    candidate = Candidate(
        "google-vertex",
        "account",
        "gcloud-adc",
        "gemini-2.5-flash",
        config={"project": "project", "location": "us-central1", "adc_file": "/adc.json"},
    )

    model = models.build_model(candidate)

    assert model.provider.kwargs == {
        "credentials": "credentials",
        "project": "project",
        "location": "us-central1",
    }
    assert calls["adc_file"] == "/adc.json"


@pytest.mark.parametrize("provider", ["aws-bedrock", "google-vertex"])
def test_cloud_provider_missing_optional_dependency_is_sanitized(monkeypatch, provider):
    def missing_import(_path):
        raise ModuleNotFoundError("optional cloud SDK missing")

    monkeypatch.setattr(models, "import_module", missing_import)
    config = {"region": "us-east-1"} if provider == "aws-bedrock" else {
        "project": "project", "location": "us-central1", "adc_file": "/adc.json"
    }
    candidate = Candidate(provider, "account", "fixture", "model-x", config=config)

    with pytest.raises(models.ModelDependencyError, match=provider):
        models.build_model(candidate)
