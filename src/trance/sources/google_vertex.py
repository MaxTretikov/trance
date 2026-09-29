"""Discover local Google Vertex AI Application Default Credentials (ADC).

This source only proves that a local ADC configuration exists.  It does not
invoke ``google.auth.default`` (which can consult metadata servers) and never
returns credential material.  Gemini API keys are handled by ``env_api`` and
are deliberately not considered here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_LOCATION = "us-central1"
_MAX_METADATA_BYTES = 64 * 1024
_ADC_FILENAME = "application_default_credentials.json"
_PROJECT_ENV_NAMES = (
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "GCP_PROJECT",
    "VERTEXAI_PROJECT",
)
_LOCATION_ENV_NAMES = (
    "GOOGLE_CLOUD_LOCATION",
    "VERTEXAI_LOCATION",
    "GOOGLE_VERTEX_LOCATION",
    "TRANCE_GOOGLE_VERTEX_LOCATION",
)


def _value(environ: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = environ.get(name, "").strip()
        if value:
            return value
    return None


def _supplied_home_path(value: str, home: Path) -> Path:
    """Expand only against the caller-supplied home, never process HOME."""
    if value == "~":
        return home
    if value.startswith("~/"):
        return home / value[2:]
    path = Path(value)
    return path if path.is_absolute() else home / path


def _adc_path(environ: Mapping[str, str], home: Path) -> tuple[Path, str] | None:
    configured = environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if configured:
        return _supplied_home_path(configured, home), "env:GOOGLE_APPLICATION_CREDENTIALS"

    config_home = environ.get("XDG_CONFIG_HOME", "").strip()
    if config_home:
        base = _supplied_home_path(config_home, home)
    else:
        base = home / ".config"
    return base / "gcloud" / _ADC_FILENAME, "gcloud-adc"


def _read_metadata(path: Path) -> Mapping[str, object] | None:
    try:
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            payload = stream.read(_MAX_METADATA_BYTES + 1)
    except (OSError, UnicodeError):
        return None
    if len(payload) > _MAX_METADATA_BYTES:
        return None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _credential_metadata(metadata: Mapping[str, object]) -> tuple[bool, str | None]:
    """Validate a known ADC shape and return its non-secret project metadata."""
    credential_type = metadata.get("type")
    if not isinstance(credential_type, str):
        return False, None

    project = metadata.get("project_id")
    project_id = project.strip() if isinstance(project, str) else None
    if credential_type == "authorized_user":
        valid = all(
            isinstance(metadata.get(name), str) and bool(metadata[name].strip())
            for name in ("client_id", "refresh_token")
        )
        quota_project = metadata.get("quota_project_id")
        if not project_id and isinstance(quota_project, str) and quota_project.strip():
            project_id = quota_project.strip()
        return valid, project_id
    if credential_type == "service_account":
        valid = all(
            isinstance(metadata.get(name), str) and bool(metadata[name].strip())
            for name in ("client_email", "private_key")
        )
        return valid, project_id
    if credential_type in {"external_account", "external_account_authorized_user"}:
        valid = isinstance(metadata.get("audience"), str) and isinstance(
            metadata.get("credential_source"), Mapping
        )
        return valid, project_id
    if credential_type == "impersonated_service_account":
        valid = isinstance(metadata.get("source_credentials"), Mapping) and isinstance(
            metadata.get("service_account_impersonation_url"), str
        )
        return valid, project_id
    return False, None


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return a secretless Vertex candidate for a local, valid ADC file."""
    path_info = _adc_path(environ, Path(home).absolute())
    if path_info is None:
        return []
    adc_file, source = path_info
    metadata = _read_metadata(adc_file)
    if metadata is None:
        return []

    valid, metadata_project = _credential_metadata(metadata)
    if not valid:
        return []
    project = _value(environ, _PROJECT_ENV_NAMES) or metadata_project
    location = _value(environ, _LOCATION_ENV_NAMES) or _DEFAULT_LOCATION
    if not project:
        return []

    model = environ.get("TRANCE_MODEL_GOOGLE_VERTEX", "").strip() or _DEFAULT_MODEL
    return [
        Candidate(
            provider="google-vertex",
            auth_kind="account",
            source=source,
            model_name=model,
            secret=None,
            config={
                "project": project,
                "location": location,
                "adc_file": str(adc_file),
                "integration": "google-vertex",
            },
        )
    ]


__all__ = ["scan"]
