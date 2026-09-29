"""Construct Pydantic AI models from discovered credentials.

Pydantic AI and provider SDKs are imported only when a model is requested. The
factory uses first-party providers where available and their official
OpenAI-compatible provider for Perplexity.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from trance.types import Candidate

if TYPE_CHECKING:
    from pydantic_ai.models import Model


class ModelBuildError(RuntimeError):
    """Base class for failures constructing a provider model."""


class UnsupportedProviderError(ModelBuildError):
    """The candidate names a provider this factory does not support."""


class MissingCredentialError(ModelBuildError):
    """The candidate does not contain credentials needed to construct a model."""


class ModelDependencyError(ModelBuildError):
    """Pydantic AI or the selected provider's optional dependency is missing."""


class CredentialPathError(ModelBuildError):
    """A provider's documented credential resolver points at another location."""


@dataclass(frozen=True, slots=True)
class _ProviderSpec:
    model_module: str
    model_class: str
    provider_module: str
    provider_class: str
    base_url: str | None = None


_API_PROVIDERS: dict[str, _ProviderSpec] = {
    "openai": _ProviderSpec("openai", "OpenAIChatModel", "openai", "OpenAIProvider"),
    "anthropic": _ProviderSpec("anthropic", "AnthropicModel", "anthropic", "AnthropicProvider"),
    "google": _ProviderSpec("google", "GoogleModel", "google", "GoogleProvider"),
    "gemini": _ProviderSpec("google", "GoogleModel", "google", "GoogleProvider"),
    "openrouter": _ProviderSpec(
        "openrouter", "OpenRouterModel", "openrouter", "OpenRouterProvider"
    ),
    "groq": _ProviderSpec("groq", "GroqModel", "groq", "GroqProvider"),
    "mistral": _ProviderSpec("mistral", "MistralModel", "mistral", "MistralProvider"),
    "cohere": _ProviderSpec("cohere", "CohereModel", "cohere", "CohereProvider"),
    "xai": _ProviderSpec("xai", "XaiModel", "xai", "XaiProvider"),
    "cerebras": _ProviderSpec("cerebras", "CerebrasModel", "cerebras", "CerebrasProvider"),
    "huggingface": _ProviderSpec(
        "huggingface", "HuggingFaceModel", "huggingface", "HuggingFaceProvider"
    ),
    "deepseek": _ProviderSpec("openai", "OpenAIChatModel", "deepseek", "DeepSeekProvider"),
    "fireworks": _ProviderSpec("openai", "OpenAIChatModel", "fireworks", "FireworksProvider"),
    "together": _ProviderSpec("openai", "OpenAIChatModel", "together", "TogetherProvider"),
    "sambanova": _ProviderSpec("openai", "OpenAIChatModel", "sambanova", "SambaNovaProvider"),
    "perplexity": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.perplexity.ai"
    ),
    "poe": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.poe.com/v1/"
    ),
    "qwen-code": _ProviderSpec("openai", "OpenAIChatModel", "openai", "OpenAIProvider"),
    "minimax-coding-plan": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider"
    ),
    "moonshot": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.moonshot.ai/v1"
    ),
    "nebius": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.tokenfactory.nebius.com/v1",
    ),
    "deepinfra": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.deepinfra.com/v1/openai",
    ),
    "nvidia": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://integrate.api.nvidia.com/v1",
    ),
    "novita": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.novita.ai/openai"
    ),
    "aimlapi": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.aimlapi.com/v1"
    ),
    "ovhcloud": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    ),
    "helicone": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://ai-gateway.helicone.ai"
    ),
    "requesty": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://router.requesty.ai/v1"
    ),
    "featherless": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.featherless.ai/v1",
    ),
    "hyperbolic": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.hyperbolic.xyz/v1",
    ),
    "crusoe": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.inference.crusoecloud.com/v1",
    ),
    "siliconflow": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.siliconflow.cn/v1",
    ),
    "venice": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.venice.ai/api/v1",
    ),
    "chutes": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://llm.chutes.ai/v1",
    ),
    "akashml": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.akashml.com/v1"
    ),
    "scaleway": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.scaleway.ai/v1"
    ),
    "friendli": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.friendli.ai/serverless/v1"
    ),
    "clarifai": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.clarifai.com/v2/ext/openai/v1"
    ),
    "meta-model-api": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.meta.ai/v1"
    ),
    "parasail": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.parasail.io/v1"
    ),
    "nscale": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://inference.api.nscale.com/v1"
    ),
    "azure-openai": _ProviderSpec("openai", "OpenAIChatModel", "openai", "OpenAIProvider"),
    "cloudflare": _ProviderSpec("openai", "OpenAIChatModel", "openai", "OpenAIProvider"),
    "databricks": _ProviderSpec("openai", "OpenAIChatModel", "openai", "OpenAIProvider"),
    "dashscope": _ProviderSpec("openai", "OpenAIChatModel", "openai", "OpenAIProvider"),
    "zai": _ProviderSpec(
        "openai", "OpenAIChatModel", "openai", "OpenAIProvider", "https://api.z.ai/api/paas/v4"
    ),
    "zai-coding-plan": _ProviderSpec(
        "openai",
        "OpenAIChatModel",
        "openai",
        "OpenAIProvider",
        "https://api.z.ai/api/coding/paas/v4",
    ),
}

_CONFIGURED_BASE_URLS = {
    "qwen-code": frozenset(
        {
            "https://coding.dashscope.aliyuncs.com/v1",
            "https://coding-intl.dashscope.aliyuncs.com/v1",
        }
    ),
    "minimax-coding-plan": frozenset(
        {"https://api.minimax.io/v1", "https://api.minimaxi.com/v1"}
    ),
}

_DYNAMIC_BASE_URL_PROVIDERS = frozenset(
    {"azure-openai", "cloudflare", "databricks", "dashscope"}
)

_DASHSCOPE_BASE_URLS = frozenset(
    {
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
    }
)


def _validated_dynamic_base_url(provider: str, candidate: Candidate) -> str:
    """Accept only documented endpoints for providers with configurable URLs."""
    if candidate.config.get("api_style") != "openai":
        raise ModelBuildError(f"Provider {provider!r} requires an OpenAI-compatible API style.")
    value = candidate.config.get("base_url")
    if not isinstance(value, str):
        raise ModelBuildError(f"Provider {provider!r} requires a documented API endpoint.")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        raise ModelBuildError(f"Provider {provider!r} has an invalid API endpoint.") from None
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ModelBuildError(f"Provider {provider!r} has an unsupported API endpoint.")

    valid = False
    if provider == "azure-openai":
        valid = bool(
            re.fullmatch(
                r"https://[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.openai\.azure\.com/openai/v1",
                value,
            )
        )
    elif provider == "cloudflare":
        valid = bool(
            re.fullmatch(
                r"https://api\.cloudflare\.com/client/v4/accounts/[0-9a-fA-F]{32}/ai/v1",
                value,
            )
        )
    elif provider == "databricks":
        valid = bool(
            re.fullmatch(
                r"https://(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+cloud\.databricks\.com/serving-endpoints",
                value,
            )
            or re.fullmatch(
                r"https://(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+azuredatabricks\.net/serving-endpoints",
                value,
            )
        )
    elif provider == "dashscope":
        valid = value in _DASHSCOPE_BASE_URLS
    if not valid:
        raise ModelBuildError(f"Provider {provider!r} has an unsupported API endpoint.")
    return value

_OPENAI_COMPATIBLE_FALLBACKS = {
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "cohere": "https://api.cohere.ai/compatibility/v1",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "huggingface": "https://router.huggingface.co/v1",
}

_COPILOT_ENV_SOURCES = {
    "env:GITHUB_COPILOT_API_KEY",
    "env:GITHUB_COPILOT_API_TOKEN",
    "env:COPILOT_GITHUB_TOKEN",
}


def _load(module: str, name: str, provider: str) -> Any:
    try:
        return getattr(import_module(f"pydantic_ai.{module}"), name)
    except (ImportError, AttributeError):
        # Keep import errors concise: exceptions from optional SDKs can include
        # implementation details, and are not useful to callers here.
        raise ModelDependencyError(
            f"Pydantic AI support for {provider!r} is unavailable; install "
            "pydantic-ai-slim with that provider's optional dependency."
        ) from None


def _load_module(module: str, provider: str) -> Any:
    """Load an optional provider SDK without exposing import details."""
    try:
        return import_module(module)
    except (ImportError, AttributeError):
        raise ModelDependencyError(
            f"Support for {provider!r} is unavailable; install the provider extra."
        ) from None


def _build_bedrock(candidate: Candidate) -> Model:
    """Build a Bedrock Converse model from explicit or profile credentials."""
    config = candidate.config
    region = config.get("region")
    if not isinstance(region, str) or not region.strip():
        raise ModelBuildError("AWS Bedrock candidate has no region.")

    bedrock_models = _load_module("pydantic_ai.models.bedrock", "aws-bedrock")
    bedrock_providers = _load_module("pydantic_ai.providers.bedrock", "aws-bedrock")
    try:
        boto3 = _load_module("boto3", "aws-bedrock")
        if config.get("aws_access_key_id") and config.get("aws_secret_access_key"):
            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=config["aws_access_key_id"],
                aws_secret_access_key=config["aws_secret_access_key"],
                **(
                    {"aws_session_token": config["aws_session_token"]}
                    if config.get("aws_session_token")
                    else {}
                ),
            )
        else:
            profile = config.get("profile_name")
            config_file = config.get("shared_config_file")
            credentials_file = config.get("shared_credentials_file")
            if not profile or not config_file or not credentials_file:
                raise ModelBuildError("AWS Bedrock candidate has incomplete profile configuration.")
            botocore = _load_module("botocore.session", "aws-bedrock")
            session = botocore.Session()
            session.set_config_variable("config_file", config_file)
            session.set_config_variable("credentials_file", credentials_file)
            session.set_config_variable("profile", profile)
            session.set_config_variable("region", region)
            client = boto3.Session(botocore_session=session).client(
                "bedrock-runtime", region_name=region
            )
        provider = bedrock_providers.BedrockProvider(bedrock_client=client)
        return bedrock_models.BedrockConverseModel(candidate.model_name, provider=provider)
    except ModelBuildError:
        raise
    except Exception:
        raise ModelBuildError("Invalid AWS Bedrock credential configuration.") from None


def _build_vertex(candidate: Candidate) -> Model:
    """Build a Vertex model using credentials loaded from the candidate ADC file."""
    config = candidate.config
    project = config.get("project")
    location = config.get("location")
    adc_file = config.get("adc_file")
    if not all(isinstance(value, str) and value.strip() for value in (project, location, adc_file)):
        raise ModelBuildError("Google Vertex candidate has incomplete configuration.")

    google_auth = _load_module("google.auth", "google-vertex")
    google_models = _load_module("pydantic_ai.models.google", "google-vertex")
    google_providers = _load_module("pydantic_ai.providers.google_cloud", "google-vertex")
    try:
        credentials, _ = google_auth.load_credentials_from_file(adc_file)
        provider = google_providers.GoogleCloudProvider(
            credentials=credentials, project=project, location=location
        )
        return google_models.GoogleModel(candidate.model_name, provider=provider)
    except ModelBuildError:
        raise
    except Exception:
        raise ModelBuildError("Invalid Google Vertex ADC configuration.") from None


def build_model(candidate: Candidate) -> Model:
    """Build a Pydantic AI model for ``candidate`` without making a request.

    API-key providers receive the candidate's secret directly. OpenAI Codex
    delegates credential loading and refresh to Pydantic AI's official
    ``~/.codex/auth.json`` resolver. GitHub Copilot accepts only its documented
    dedicated environment variables or an explicit in-memory credential.
    """
    provider = candidate.provider.lower().replace("_", "-")
    if not isinstance(candidate.model_name, str) or not candidate.model_name.strip():
        raise ModelBuildError(f"Provider {candidate.provider!r} has no model name.")

    if provider in {"openai-codex", "codex"}:
        configured_home = candidate.config.get("codex_home")
        if configured_home:
            active_codex_home = os.environ.get("CODEX_HOME") or (Path.home() / ".codex")
            actual_home = Path(active_codex_home).expanduser()
            if Path(configured_home).expanduser().resolve() != actual_home.resolve():
                raise CredentialPathError(
                    "Codex credentials were discovered under a custom CODEX_HOME; set "
                    "CODEX_HOME in the process environment before scanning so Pydantic AI "
                    "uses the same credential location."
                )
        model_cls = _load("models.openai_codex", "OpenAICodexModel", provider)
        return model_cls(candidate.model_name)

    if provider in {"github-copilot", "copilot"}:
        if candidate.source == "github-cli" or candidate.secret is None:
            raise MissingCredentialError(
                "GitHub Copilot model requires an explicit token; resolve the "
                "GitHub CLI token through the CLI credential adapter first."
            )
        if candidate.source.startswith("env:") and candidate.source not in _COPILOT_ENV_SOURCES:
            raise MissingCredentialError(
                "GitHub Copilot ignores generic GitHub tokens; provide a "
                "dedicated Copilot token or an explicit credential."
            )
        model_cls = _load("models.github_copilot", "GitHubCopilotModel", provider)
        provider_cls = _load("providers.github_copilot", "GitHubCopilotProvider", provider)
        return model_cls(candidate.model_name, provider=provider_cls(api_key=candidate.secret))

    if provider == "aws-bedrock":
        return _build_bedrock(candidate)
    if provider == "google-vertex":
        return _build_vertex(candidate)

    spec = _API_PROVIDERS.get(provider)
    if spec is None:
        raise UnsupportedProviderError(f"Unsupported model provider: {candidate.provider!r}")
    if not candidate.secret:
        raise MissingCredentialError(
            f"Provider {candidate.provider!r} requires an API key in the candidate."
        )

    try:
        model_cls = _load(f"models.{spec.model_module}", spec.model_class, provider)
        provider_cls = _load(f"providers.{spec.provider_module}", spec.provider_class, provider)
    except ModelDependencyError:
        fallback_base_url = _OPENAI_COMPATIBLE_FALLBACKS.get(provider)
        if fallback_base_url is None:
            raise
        model_cls = _load("models.openai", "OpenAIChatModel", provider)
        provider_cls = _load("providers.openai", "OpenAIProvider", provider)
        spec = _ProviderSpec(
            model_module="openai",
            model_class="OpenAIChatModel",
            provider_module="openai",
            provider_class="OpenAIProvider",
            base_url=fallback_base_url,
        )
    provider_kwargs: dict[str, Any] = {"api_key": candidate.secret}
    base_url = spec.base_url
    if provider in _DYNAMIC_BASE_URL_PROVIDERS:
        base_url = _validated_dynamic_base_url(provider, candidate)
    if provider in _CONFIGURED_BASE_URLS:
        if candidate.config.get("api_style") != "openai":
            raise ModelBuildError(f"Provider {provider!r} requires an OpenAI-compatible API style.")
        configured_base_url = candidate.config.get("base_url", "").rstrip("/")
        if configured_base_url not in _CONFIGURED_BASE_URLS[provider]:
            raise ModelBuildError(f"Provider {provider!r} has an unsupported API endpoint.")
        base_url = configured_base_url
    if base_url is not None:
        provider_kwargs["base_url"] = base_url
    provider_obj = provider_cls(**provider_kwargs)
    return model_cls(candidate.model_name, provider=provider_obj)
