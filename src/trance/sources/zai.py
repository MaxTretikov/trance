"""Discover explicitly configured Z.AI API and Coding Plan keys.

Z.AI's official documentation distinguishes the general OpenAI-compatible
 endpoint from the Coding Plan endpoint.  The environment names below are
 library-defined aliases: ``ZAI_API_KEY`` selects the general API and
 ``ZAI_CODING_PLAN_API_KEY`` explicitly selects the subscription endpoint.
 A generic key is never classified as a subscription credential.
"""

from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_GENERAL_KEY = "ZAI_API_KEY"
_CODING_KEY = "ZAI_CODING_PLAN_API_KEY"
_GENERAL_BASE_URL = "https://api.z.ai/api/paas/v4"
_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
_DEFAULT_MODEL = "glm-5.3"


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return candidates for configured general and Coding Plan Z.AI keys."""
    del home
    candidates: list[Candidate] = []
    general = environ.get(_GENERAL_KEY, "").strip()
    if general:
        candidates.append(
            Candidate(
                provider="zai",
                auth_kind="api_key",
                source=f"env:{_GENERAL_KEY}",
                model_name=environ.get("TRANCE_MODEL_ZAI", "").strip() or _DEFAULT_MODEL,
                secret=general,
                config={"base_url": _GENERAL_BASE_URL, "api_style": "openai"},
            )
        )
    coding = environ.get(_CODING_KEY, "").strip()
    if coding:
        candidates.append(
            Candidate(
                provider="zai-coding-plan",
                auth_kind="subscription",
                source=f"env:{_CODING_KEY}",
                model_name=(
                    environ.get("TRANCE_MODEL_ZAI_CODING_PLAN", "").strip()
                    or _DEFAULT_MODEL
                ),
                secret=coding,
                config={"base_url": _CODING_BASE_URL, "api_style": "openai"},
            )
        )
    return candidates
