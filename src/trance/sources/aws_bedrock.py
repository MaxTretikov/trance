"""Discover local AWS credential-chain configuration for Amazon Bedrock.

This adapter deliberately inspects only the environment and bounded AWS INI
files.  It does not construct a boto3 session, call AWS, invoke a credential
process, or query instance metadata.  Environment credentials are copied into
the candidate configuration so a later model adapter can use the supplied
scan context rather than unrelated process-global environment state.  The
candidate representation redacts those values.
"""

from __future__ import annotations

import configparser
from collections.abc import Mapping
from pathlib import Path

from trance.types import Candidate

_MODEL_ENV = "TRANCE_MODEL_AWS_BEDROCK"
_DEFAULT_MODEL = "us.amazon.nova-micro-v1:0"
_DEFAULT_REGION_ENV = ("AWS_REGION", "AWS_DEFAULT_REGION")
_MAX_INI_BYTES = 1024 * 1024


def _value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name, "")
    value = value.strip() if isinstance(value, str) else ""
    return value or None


def _path(value: str | None, home: Path, default: Path) -> Path:
    """Resolve AWS path settings without consulting the process user's home."""
    if not value:
        return default.resolve()
    # ``Path.expanduser`` would use the real process account, which is unsafe
    # for discovery's injected synthetic homes.  Interpret ~/ ourselves.
    if value == "~":
        return home.resolve()
    if value.startswith("~/"):
        return (home / value[2:]).resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = home / candidate
    return candidate.resolve()


def _read_ini(path: Path) -> configparser.RawConfigParser:
    parser = configparser.RawConfigParser()
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_INI_BYTES + 1)
        if len(raw) > _MAX_INI_BYTES:
            return parser
        parser.read_string(raw.decode("utf-8"), source=str(path))
    except (OSError, UnicodeError, configparser.Error):
        return configparser.RawConfigParser()
    return parser


def _profile_section(parser: configparser.RawConfigParser, profile: str) -> str | None:
    if profile == "default" and parser.has_section("default"):
        return "default"
    section = f"profile {profile}"
    if parser.has_section(section):
        return section
    if parser.has_section(profile):
        return profile
    return None


def _section_values(
    parser: configparser.RawConfigParser, profile: str
) -> Mapping[str, str]:
    section = _profile_section(parser, profile)
    return parser[section] if section is not None else {}


def _static_credentials(
    credentials: configparser.RawConfigParser,
    config: configparser.RawConfigParser,
    profile: str,
) -> bool:
    for parser in (credentials, config):
        values = _section_values(parser, profile)
        if _value(values, "aws_access_key_id") and _value(values, "aws_secret_access_key"):
            return True
    return False


def _sso_profile(config: configparser.RawConfigParser, profile: str) -> bool:
    values = _section_values(config, profile)
    if not values:
        return False
    start_url = _value(values, "sso_start_url")
    account = _value(values, "sso_account_id")
    role = _value(values, "sso_role_name")
    if start_url and account and role:
        return True
    session_name = _value(values, "sso_session")
    if not session_name:
        return False
    if not account or not role:
        return False
    session_section = f"sso-session {session_name}"
    return config.has_section(session_section) and bool(
        _value(config[session_section], "sso_start_url")
    )


def _region(
    environ: Mapping[str, str], config: configparser.RawConfigParser, profile: str
) -> str | None:
    for name in _DEFAULT_REGION_ENV:
        region = _value(environ, name)
        if region:
            return region
    return _value(_section_values(config, profile), "region")


def scan(environ: Mapping[str, str], home: Path) -> list[Candidate]:
    """Return one local Bedrock credential-chain candidate.

    Environment-mode candidates carry explicit AWS credential values in their
    protected candidate configuration; profile and SSO candidates carry no
    credential values.
    """
    access_key = _value(environ, "AWS_ACCESS_KEY_ID")
    secret_key = _value(environ, "AWS_SECRET_ACCESS_KEY")
    env_region = _value(environ, "AWS_REGION") or _value(environ, "AWS_DEFAULT_REGION")
    config_override = _value(environ, "AWS_CONFIG_FILE")
    credentials_override = _value(environ, "AWS_SHARED_CREDENTIALS_FILE")

    # Environment credentials and region are sufficient for the credential
    # chain. Avoid parsing either INI file in this common path.
    if access_key and secret_key and env_region:
        home = home.resolve()
        config_path = _path(config_override, home, home / ".aws" / "config")
        credentials_path = _path(
            credentials_override, home, home / ".aws" / "credentials"
        )
        metadata = {
            "region": env_region,
            "shared_config_file": str(config_path),
            "shared_credentials_file": str(credentials_path),
            "integration": "aws-bedrock",
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        }
        session_token = _value(environ, "AWS_SESSION_TOKEN")
        if session_token:
            metadata["aws_session_token"] = session_token
        return [
            Candidate(
                provider="aws-bedrock",
                auth_kind="account",
                source="env:AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY",
                model_name=_value(environ, _MODEL_ENV) or _DEFAULT_MODEL,
                secret=None,
                config=metadata,
            )
        ]

    # Most scans have no AWS setup at all. A stat is enough to distinguish
    # that case; defer resolving paths and constructing INI parsers until a
    # configured override or a default AWS file is actually present.
    if not config_override and not credentials_override:
        default_config = home / ".aws" / "config"
        default_credentials = home / ".aws" / "credentials"
        if not default_config.is_file() and not default_credentials.is_file():
            return []

    home = home.resolve()
    config_path = _path(
        _value(environ, "AWS_CONFIG_FILE"), home, home / ".aws" / "config"
    )
    credentials_path = _path(
        _value(environ, "AWS_SHARED_CREDENTIALS_FILE"),
        home,
        home / ".aws" / "credentials",
    )
    config = _read_ini(config_path)
    credentials = _read_ini(credentials_path)

    profile = _value(environ, "AWS_PROFILE") or "default"
    region = _region(environ, config, profile)
    if not region:
        return []

    if access_key and secret_key:
        source = "env:AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY"
    elif _static_credentials(credentials, config, profile):
        source = f"profile:{profile}"
    elif _sso_profile(config, profile):
        source = f"profile:{profile}"
    else:
        return []

    metadata: dict[str, str] = {
        "region": region,
        "shared_config_file": str(config_path),
        "shared_credentials_file": str(credentials_path),
        "integration": "aws-bedrock",
    }
    if source.startswith("profile:"):
        metadata["profile_name"] = profile
    else:
        metadata["aws_access_key_id"] = access_key
        metadata["aws_secret_access_key"] = secret_key
        session_token = _value(environ, "AWS_SESSION_TOKEN")
        if session_token:
            metadata["aws_session_token"] = session_token
    return [
        Candidate(
            provider="aws-bedrock",
            auth_kind="account",
            source=source,
            model_name=_value(environ, _MODEL_ENV) or _DEFAULT_MODEL,
            secret=None,
            config=metadata,
        )
    ]
