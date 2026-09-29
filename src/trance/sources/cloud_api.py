"""Discover configured OpenAI-compatible cloud API credentials.

Only the documented environment variables for each endpoint are read. This
adapter performs no network requests and accepts only provider-owned hosts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from trance.types import Candidate

_AZURE_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.openai\.azure\.com\Z")
_DATABRICKS_HOST = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:cloud\.databricks\.com|azuredatabricks\.net)\Z"
)
_DASHSCOPE_BASES = {
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
}
_CLOUDFLARE_MODEL = "@cf/meta/llama-3.1-8b-instruct"


def _https_host(value: str) -> str | None:
    """Return a normalized HTTPS origin, rejecting paths and URL tricks."""
    try:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            return None
    except ValueError:
        return None
    return f"https://{parsed.hostname.lower()}"


def _nonblank(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def _candidate(
    provider: str, source: str, model: str, secret: str, base_url: str
) -> Candidate:
    return Candidate(
        provider=provider,
        auth_kind="api_key",
        source=source,
        model_name=model,
        secret=secret,
        config={"base_url": base_url, "api_style": "openai"},
    )


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return candidates for complete, explicitly configured cloud APIs."""
    del home
    found: list[Candidate] = []

    key = _nonblank(environ, "AZURE_OPENAI_API_KEY")
    endpoint = _nonblank(environ, "AZURE_OPENAI_ENDPOINT")
    deployment = _nonblank(environ, "AZURE_OPENAI_DEPLOYMENT")
    origin = _https_host(endpoint) if endpoint else None
    if key and deployment and origin:
        host = origin.removeprefix("https://")
        if _AZURE_HOST.fullmatch(host):
            found.append(
                _candidate(
                    "azure-openai",
                    "env:AZURE_OPENAI_API_KEY",
                    deployment,
                    key,
                    f"{origin}/openai/v1",
                )
            )

    account = _nonblank(environ, "CLOUDFLARE_ACCOUNT_ID")
    key = _nonblank(environ, "CLOUDFLARE_API_KEY")
    if account and key and re.fullmatch(r"[a-fA-F0-9]{32}", account):
        model = _nonblank(environ, "TRANCE_MODEL_CLOUDFLARE") or _CLOUDFLARE_MODEL
        found.append(
            _candidate(
                "cloudflare",
                "env:CLOUDFLARE_API_KEY",
                model,
                key,
                f"https://api.cloudflare.com/client/v4/accounts/{account.lower()}/ai/v1",
            )
        )

    token = _nonblank(environ, "DATABRICKS_TOKEN")
    host_value = _nonblank(environ, "DATABRICKS_HOST")
    model = _nonblank(environ, "DATABRICKS_MODEL")
    origin = _https_host(host_value) if host_value else None
    if token and model and origin:
        host = origin.removeprefix("https://")
        if _DATABRICKS_HOST.fullmatch(host):
            found.append(
                _candidate(
                    "databricks",
                    "env:DATABRICKS_TOKEN",
                    model,
                    token,
                    f"{origin}/serving-endpoints",
                )
            )

    key = _nonblank(environ, "DASHSCOPE_API_KEY")
    if key:
        base_url = _nonblank(environ, "DASHSCOPE_BASE_URL") or (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        if base_url.rstrip("/") in _DASHSCOPE_BASES:
            found.append(
                _candidate(
                    "dashscope",
                    "env:DASHSCOPE_API_KEY",
                    _nonblank(environ, "TRANCE_MODEL_DASHSCOPE") or "qwen-plus",
                    key,
                    base_url.rstrip("/"),
                )
            )

    return found
