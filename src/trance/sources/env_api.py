"""Discover documented model-provider API keys from environment variables.

This source intentionally uses an explicit registry instead of guessing from
variable suffixes. It never performs network validation and keeps values only
in ``Candidate.secret``. Pydantic AI provider ids are used so the resolver can
construct the matching provider without importing every provider SDK here.
"""

from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

# (environment variable, Pydantic AI provider id, usable default model).
# Environment names and provider defaults follow the providers' documented
# setup instructions and Pydantic AI's model/provider integrations.
_API_KEYS: tuple[tuple[str, str, str], ...] = (
    ("OPENAI_API_KEY", "openai", "gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic", "claude-haiku-4-5-20251001"),
    ("GOOGLE_API_KEY", "google", "gemini-2.5-flash"),
    ("GEMINI_API_KEY", "google", "gemini-2.5-flash"),
    ("XAI_API_KEY", "xai", "grok-4.7"),
    ("GROQ_API_KEY", "groq", "openai/gpt-oss-120b"),
    ("MISTRAL_API_KEY", "mistral", "mistral-small-latest"),
    ("COHERE_API_KEY", "cohere", "command-r-08-2024"),
    ("CEREBRAS_API_KEY", "cerebras", "llama-3.3-70b"),
    ("HF_TOKEN", "huggingface", "Qwen/Qwen2.5-72B-Instruct"),
    ("HUGGINGFACE_API_KEY", "huggingface", "Qwen/Qwen2.5-72B-Instruct"),
    ("OPENROUTER_API_KEY", "openrouter", "openai/gpt-4o-mini"),
    ("DEEPSEEK_API_KEY", "deepseek", "deepseek-chat"),
    ("FIREWORKS_API_KEY", "fireworks", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
    ("TOGETHER_API_KEY", "together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("PERPLEXITY_API_KEY", "perplexity", "sonar"),
    ("SAMBANOVA_API_KEY", "sambanova", "Meta-Llama-3.3-70B-Instruct"),
)


def _model_override(environ: Mapping[str, str], provider: str) -> str | None:
    name = "TRANCE_MODEL_" + provider.upper().replace("-", "_")
    value = environ.get(name, "").strip()
    return value or None


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return candidates for explicitly supported API-key variables.

    ``home`` is accepted to share the source-adapter interface; environment
    scanning does not need to inspect the filesystem.
    """
    del home
    candidates: list[Candidate] = []
    for variable, provider, default_model in _API_KEYS:
        secret = environ.get(variable, "").strip()
        if not secret:
            continue
        candidates.append(
            Candidate(
                provider=provider,
                auth_kind="api_key",
                source=f"env:{variable}",
                model_name=_model_override(environ, provider) or default_model,
                secret=secret,
            )
        )
    return candidates
