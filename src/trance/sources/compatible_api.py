"""Discover API keys for selected OpenAI-compatible model endpoints.

This module only reads explicitly documented environment variables. It does
not validate keys over the network or import any provider SDKs; the resolver
uses the provider id and ``api_style`` metadata to construct a client.
"""

from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

# (provider id, environment names in priority order, default model).  Endpoint
# URLs intentionally belong to the model factory, not the credential source.
_PROVIDERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("moonshot", ("MOONSHOT_API_KEY",), "kimi-k2.6"),
    ("nebius", ("NEBIUS_API_KEY",), "Qwen/Qwen3.5-397B-A17B"),
    ("deepinfra", ("DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY"), "Qwen/Qwen3.8-27B"),
    ("nvidia", ("NVIDIA_API_KEY",), "openai/gpt-oss-20b"),
    ("novita", ("NOVITA_API_KEY",), "deepseek/deepseek-v3.2"),
    ("aimlapi", ("AIML_API_KEY",), "gpt-4o"),
    ("ovhcloud", ("OVH_AI_ENDPOINTS_ACCESS_TOKEN",), "Meta-Llama-3_3-70B-Instruct"),
    ("helicone", ("HELICONE_API_KEY",), "gpt-4o-mini"),
    ("requesty", ("REQUESTY_API_KEY",), "openai/gpt-4o"),
    ("featherless", ("FEATHERLESS_API_KEY",), "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("hyperbolic", ("HYPERBOLIC_API_KEY",), "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("crusoe", ("CRUSOE_API_KEY",), "meta-llama/Llama-3.3-70B-Instruct"),
    ("siliconflow", ("SILICONFLOW_API_KEY",), "deepseek-ai/DeepSeek-V4-Flash"),
    ("venice", ("VENICE_API_KEY",), "venice-uncensored"),
    ("chutes", ("CHUTES_API_KEY",), "Qwen/Qwen3-32B-TEE"),
    ("akashml", ("AKASH_API_KEY",), "zai-org/GLM-5.3"),
    ("scaleway", ("SCW_SECRET_KEY",), "llama-3.3-70b-instruct"),
    ("friendli", ("FRIENDLI_API_KEY",), "zai-org/GLM-5.3-Flash"),
    (
        "clarifai",
        ("CLARIFAI_PAT",),
        "https://clarifai.com/openai/chat-completion/models/gpt-oss-120b",
    ),
    ("meta-model-api", ("MODEL_API_KEY",), "muse-spark-1.3"),
    ("parasail", ("PARASAIL_API_KEY",), "parasail-minimax-m3"),
    ("nscale", ("NSCALE_API_KEY",), "Qwen/Qwen2.5-Coder-32B-Instruct"),
)


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return candidates for configured keys; do not inspect files or network."""
    del home
    candidates: list[Candidate] = []
    for provider, variables, default_model in _PROVIDERS:
        seen: set[str] = set()
        override = environ.get(f"TRANCE_MODEL_{provider.upper()}", "").strip()
        for variable in variables:
            secret = environ.get(variable, "").strip()
            if not secret or secret in seen:
                continue
            seen.add(secret)
            candidates.append(
                Candidate(
                    provider=provider,
                    auth_kind="api_key",
                    source=f"env:{variable}",
                    model_name=override or default_model,
                    secret=secret,
                    config={"api_style": "openai"},
                )
            )
    return candidates
