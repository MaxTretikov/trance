from pathlib import Path

from trance.sources.aws_bedrock import scan


def _aws_files(home: Path, *, config: str = "", credentials: str = "") -> None:
    aws = home / ".aws"
    aws.mkdir(parents=True)
    if config:
        (aws / "config").write_text(config, encoding="utf-8")
    if credentials:
        (aws / "credentials").write_text(credentials, encoding="utf-8")


def test_environment_credentials_are_explicit_but_redacted(tmp_path: Path):
    candidates = scan(
        {
            "AWS_ACCESS_KEY_ID": "AKIA-example",
            "AWS_SECRET_ACCESS_KEY": "secret-example",
            "AWS_SESSION_TOKEN": "session-example",
            "AWS_REGION": "us-west-2",
            "TRANCE_MODEL_AWS_BEDROCK": "custom-model",
        },
        tmp_path,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "aws-bedrock"
    assert candidate.auth_kind == "account"
    assert candidate.source == "env:AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY"
    assert candidate.model_name == "custom-model"
    assert candidate.secret is None
    assert candidate.config["aws_access_key_id"] == "AKIA-example"
    assert candidate.config["aws_secret_access_key"] == "secret-example"
    assert candidate.config["aws_session_token"] == "session-example"
    assert candidate.config["region"] == "us-west-2"
    assert candidate.config["integration"] == "aws-bedrock"
    assert "secret-example" not in repr(candidate)
    assert "AKIA-example" not in repr(candidate)
    assert "session-example" not in repr(candidate)


def test_named_profile_static_credentials_and_profile_region(tmp_path: Path):
    _aws_files(
        tmp_path,
        config="[profile research]\nregion = eu-west-1\n",
        credentials=(
            "[research]\naws_access_key_id = AKIA-example\n"
            "aws_secret_access_key = secret-example\n"
        ),
    )

    candidate = scan({"AWS_PROFILE": "research"}, tmp_path)[0]

    assert candidate.source == "profile:research"
    assert candidate.config["profile_name"] == "research"
    assert candidate.config["region"] == "eu-west-1"
    assert candidate.config["shared_config_file"] == str((tmp_path / ".aws/config").resolve())
    assert candidate.config["shared_credentials_file"] == str(
        (tmp_path / ".aws/credentials").resolve()
    )


def test_default_profile_can_read_credentials_from_config(tmp_path: Path):
    _aws_files(
        tmp_path,
        config=(
            "[default]\nregion = us-east-1\n"
            "aws_access_key_id = AKIA-example\naws_secret_access_key = secret-example\n"
        ),
    )

    candidate = scan({}, tmp_path)[0]

    assert candidate.source == "profile:default"
    assert candidate.config["profile_name"] == "default"
    assert candidate.model_name == "us.amazon.nova-micro-v1:0"


def test_sso_profile_is_local_evidence(tmp_path: Path):
    _aws_files(
        tmp_path,
        config=(
            "[profile sso]\nregion = us-east-2\nsso_session = work\n"
            "sso_account_id = 123456789012\nsso_role_name = role\n"
            "[sso-session work]\nsso_start_url = https://example.awsapps.com/start\n"
        ),
    )

    candidate = scan({"AWS_PROFILE": "sso"}, tmp_path)[0]

    assert candidate.source == "profile:sso"
    assert candidate.secret is None
    assert "aws_access_key_id" not in candidate.config
    assert "aws_secret_access_key" not in candidate.config


def test_sso_session_without_account_or_role_is_not_login(tmp_path: Path):
    _aws_files(
        tmp_path,
        config=(
            "[profile incomplete]\nregion = us-east-2\nsso_session = work\n"
            "[sso-session work]\nsso_start_url = https://example.awsapps.com/start\n"
        ),
    )

    assert scan({"AWS_PROFILE": "incomplete"}, tmp_path) == []


def test_directory_or_region_only_is_not_login(tmp_path: Path):
    _aws_files(tmp_path, config="[profile empty]\nregion = us-west-2\n")
    assert scan({"AWS_PROFILE": "empty"}, tmp_path) == []
    assert scan({"AWS_REGION": "us-west-2"}, tmp_path) == []


def test_relative_file_overrides_are_under_injected_home(tmp_path: Path):
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "cfg").write_text(
        "[profile named]\nregion = ap-southeast-1\nsso_start_url = https://example/start\n"
        "sso_account_id = 123456789012\nsso_role_name = role\n",
        encoding="utf-8",
    )
    (tmp_path / "custom" / "creds").write_text("", encoding="utf-8")

    candidate = scan(
        {
            "AWS_PROFILE": "named",
            "AWS_CONFIG_FILE": "custom/cfg",
            "AWS_SHARED_CREDENTIALS_FILE": "custom/creds",
        },
        tmp_path,
    )[0]

    assert candidate.config["shared_config_file"] == str((tmp_path / "custom/cfg").resolve())
    assert candidate.config["shared_credentials_file"] == str(
        (tmp_path / "custom/creds").resolve()
    )


def test_region_is_required(tmp_path: Path):
    _aws_files(
        tmp_path,
        credentials="[default]\naws_access_key_id = key\naws_secret_access_key = secret\n",
    )
    assert scan({}, tmp_path) == []


def test_empty_home_skips_ini_reader(tmp_path: Path, monkeypatch):
    import trance.sources.aws_bedrock as source

    monkeypatch.setattr(source, "_read_ini", lambda path: (_ for _ in ()).throw(
        AssertionError("empty scan read an AWS INI file")
    ))

    assert scan({}, tmp_path) == []


def test_environment_credentials_with_region_skip_ini_reader(tmp_path: Path, monkeypatch):
    import trance.sources.aws_bedrock as source

    monkeypatch.setattr(source, "_read_ini", lambda path: (_ for _ in ()).throw(
        AssertionError("environment scan read an AWS INI file")
    ))

    candidates = scan(
        {
            "AWS_ACCESS_KEY_ID": "AKIA-example",
            "AWS_SECRET_ACCESS_KEY": "secret-example",
            "AWS_DEFAULT_REGION": "us-west-2",
        },
        tmp_path,
    )

    assert len(candidates) == 1
    assert candidates[0].config["region"] == "us-west-2"


def test_oversized_aws_file_is_ignored(tmp_path: Path):
    aws = tmp_path / ".aws"
    aws.mkdir()
    (aws / "config").write_text(
        "[default]\nregion = us-east-1\n" + ("# padding\n" * 120_000),
        encoding="utf-8",
    )
    (aws / "credentials").write_text(
        "[default]\naws_access_key_id = key\naws_secret_access_key = secret\n",
        encoding="utf-8",
    )

    assert scan({}, tmp_path) == []
