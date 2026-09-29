"""High value, offline edge cases for the model factory.

These tests deliberately fake provider modules.  Constructing a model must not
make a network request, import an optional SDK eagerly, or expose credentials
when one of those imports/configuration steps fails.
"""

from types import SimpleNamespace

import pytest

from trance import models
from trance.types import Candidate


class FakeProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeModel:
    def __init__(self, model_name, *, provider=None):
        self.model_name = model_name
        self.provider = provider


def _fake_loader(monkeypatch, *, missing=()):
    """Install a minimal loader for a selected provider and OpenAI fallback."""
    missing = set(missing)

    def load(module, name, provider):
        if (module, name) in missing:
            raise models.ModelDependencyError(f"missing {provider} support")
        if module == "models.groq" and name == "GroqModel":
            return FakeModel
        if module == "providers.groq" and name == "GroqProvider":
            return FakeProvider
        if module == "models.openai" and name == "OpenAIChatModel":
            return FakeModel
        if module == "providers.openai" and name == "OpenAIProvider":
            return FakeProvider
        raise AssertionError(f"unexpected load: {module}.{name}")

    monkeypatch.setattr(models, "_load", load)


def test_provider_and_model_validation_happens_before_optional_import(monkeypatch):
    calls = []

    def load(*args):
        calls.append(args)
        raise AssertionError("optional modules must not load for invalid candidates")

    monkeypatch.setattr(models, "_load", load)

    with pytest.raises(models.ModelBuildError, match="no model name"):
        models.build_model(Candidate("openai", "api_key", "manual", "  ", "secret"))
    with pytest.raises(models.MissingCredentialError, match="API key"):
        models.build_model(Candidate("openai", "api_key", "manual", "model", ""))

    assert calls == []


def test_provider_names_are_normalized_without_changing_candidate_data(monkeypatch):
    _fake_loader(monkeypatch)
    candidate = Candidate("GROQ", "api_key", "env:GROQ_API_KEY", "model", "token")

    model = models.build_model(candidate)

    assert model.model_name == "model"
    assert model.provider.kwargs == {"api_key": "token"}
    assert candidate.provider == "GROQ"


def test_missing_native_provider_module_falls_back_to_openai_compatibility(monkeypatch):
    _fake_loader(monkeypatch, missing={("providers.groq", "GroqProvider")})

    model = models.build_model(Candidate("groq", "api_key", "manual", "model", "token"))

    assert model.model_name == "model"
    assert model.provider.kwargs == {
        "api_key": "token",
        "base_url": models._OPENAI_COMPATIBLE_FALLBACKS["groq"],
    }


def test_openai_fallback_dependency_error_is_sanitized(monkeypatch):
    secret = "fallback-secret-must-not-appear"

    def failing_import(_module):
        raise ModuleNotFoundError(f"SDK unavailable while handling {secret}")

    monkeypatch.setattr(models, "import_module", failing_import)

    with pytest.raises(models.ModelDependencyError) as exc:
        models.build_model(Candidate("groq", "api_key", "manual", "model", secret))

    assert secret not in str(exc.value)


@pytest.mark.parametrize(
    "provider",
    ["azure-openai", "cloudflare", "databricks", "dashscope"],
)
def test_dynamic_endpoint_requires_openai_style_and_documented_url(monkeypatch, provider):
    _fake_loader(monkeypatch)
    candidate = Candidate(
        provider,
        "api_key",
        "manual",
        "model",
        "secret",
        {"api_style": "native", "base_url": "https://example.invalid"},
    )

    with pytest.raises(models.ModelBuildError, match="OpenAI-compatible"):
        models.build_model(candidate)

    candidate = Candidate(provider, "api_key", "manual", "model", "secret", {"api_style": "openai"})
    with pytest.raises(models.ModelBuildError, match="documented API endpoint"):
        models.build_model(candidate)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://workspace.cloud.databricks.com:443/serving-endpoints",
        "https://workspace.cloud.databricks.com/serving-endpoints?token=secret",
        "https://user:password@workspace.cloud.databricks.com/serving-endpoints",
        "https://workspace.cloud.databricks.com/%zz",
        "https://workspace.cloud.databricks.com/serving-endpoints#fragment",
    ],
)
def test_dynamic_endpoint_rejects_credentials_ports_queries_and_fragments(monkeypatch, base_url):
    _fake_loader(monkeypatch)
    candidate = Candidate(
        "databricks",
        "api_key",
        "manual",
        "model",
        "secret",
        {"api_style": "openai", "base_url": base_url},
    )

    with pytest.raises(models.ModelBuildError, match="API endpoint") as exc:
        models.build_model(candidate)

    assert "secret" not in str(exc.value)


def test_bedrock_constructor_failures_are_redacted(monkeypatch):
    secret = "bedrock-secret-value"

    class Boto3:
        def client(self, *_args, **_kwargs):
            raise RuntimeError(f"bad credentials: {secret}")

    modules = {
        "pydantic_ai.models.bedrock": SimpleNamespace(BedrockConverseModel=FakeModel),
        "pydantic_ai.providers.bedrock": SimpleNamespace(BedrockProvider=FakeProvider),
        "boto3": Boto3(),
    }
    monkeypatch.setattr(models, "import_module", modules.__getitem__)
    candidate = Candidate(
        "aws-bedrock",
        "account",
        "manual",
        "model",
        config={
            "region": "us-east-1",
            "aws_access_key_id": "access",
            "aws_secret_access_key": secret,
        },
    )

    with pytest.raises(models.ModelBuildError, match="Invalid AWS Bedrock") as exc:
        models.build_model(candidate)

    assert secret not in str(exc.value)


def test_vertex_constructor_failures_are_redacted(monkeypatch):
    secret = "vertex-secret-value"

    class GoogleAuth:
        def load_credentials_from_file(self, _path):
            raise RuntimeError(f"ADC leaked {secret}")

    modules = {
        "google.auth": GoogleAuth(),
        "pydantic_ai.models.google": SimpleNamespace(GoogleModel=FakeModel),
        "pydantic_ai.providers.google_cloud": SimpleNamespace(GoogleCloudProvider=FakeProvider),
    }
    monkeypatch.setattr(models, "import_module", modules.__getitem__)
    candidate = Candidate(
        "google-vertex",
        "account",
        "manual",
        "model",
        config={"project": "project", "location": "global", "adc_file": secret},
    )

    with pytest.raises(models.ModelBuildError, match="Invalid Google Vertex") as exc:
        models.build_model(candidate)

    assert secret not in str(exc.value)
