import json
from pathlib import Path

from trance.sources.google_vertex import scan


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scan_discovers_gcloud_adc_with_project_and_location(tmp_path: Path) -> None:
    adc = tmp_path / ".config/gcloud/application_default_credentials.json"
    _write(
        adc,
        {
            "type": "authorized_user",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "refresh_token": "refresh-token",
            "quota_project_id": "fixture-project",
        },
    )

    [candidate] = scan({"GOOGLE_CLOUD_LOCATION": "us-central1"}, tmp_path)

    assert candidate.provider == "google-vertex"
    assert candidate.auth_kind == "account"
    assert candidate.source == "gcloud-adc"
    assert candidate.model_name == "gemini-2.5-flash"
    assert candidate.secret is None
    assert candidate.config == {
        "project": "fixture-project",
        "location": "us-central1",
        "adc_file": str(adc),
        "integration": "google-vertex",
    }
    assert "refresh-token" not in repr(candidate)


def test_scan_honors_explicit_adc_path_env_and_model_override(tmp_path: Path) -> None:
    adc = tmp_path / "credentials.json"
    _write(
        adc,
        {
            "type": "service_account",
            "project_id": "service-project",
            "client_email": "vertex@example.iam.gserviceaccount.com",
            "private_key": "private-key",
        },
    )

    [candidate] = scan(
        {
            "GOOGLE_APPLICATION_CREDENTIALS": str(adc),
            "GOOGLE_CLOUD_LOCATION": "europe-west4",
            "TRANCE_MODEL_GOOGLE_VERTEX": "gemini-2.5-pro",
        },
        tmp_path,
    )

    assert candidate.source == "env:GOOGLE_APPLICATION_CREDENTIALS"
    assert candidate.model_name == "gemini-2.5-pro"
    assert candidate.config["project"] == "service-project"


def test_scan_honors_xdg_config_home_and_explicit_project(tmp_path: Path) -> None:
    adc = tmp_path / "xdg/gcloud/application_default_credentials.json"
    _write(
        adc,
        {
            "type": "external_account",
            "audience": (
                "//iam.googleapis.com/projects/1/locations/global/"
                "workloadIdentityPools/pool/providers/provider"
            ),
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "credential_source": {"file": "/tmp/token"},
        },
    )

    [candidate] = scan(
        {
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
            "GOOGLE_CLOUD_PROJECT": "explicit-project",
            "GOOGLE_CLOUD_LOCATION": "asia-northeast1",
        },
        tmp_path,
    )

    assert candidate.config["project"] == "explicit-project"
    assert candidate.config["adc_file"] == str(adc)


def test_scan_requires_valid_adc_project_and_location(tmp_path: Path) -> None:
    adc = tmp_path / ".config/gcloud/application_default_credentials.json"
    _write(adc, {"type": "authorized_user", "refresh_token": "token"})
    assert scan({"GOOGLE_CLOUD_LOCATION": "us-central1"}, tmp_path) == []

    _write(
        adc,
        {
            "type": "authorized_user",
            "client_id": "client",
            "refresh_token": "token",
            "quota_project_id": "project",
        },
    )
    assert scan({}, tmp_path)
    assert scan({"GOOGLE_CLOUD_LOCATION": "us-central1"}, tmp_path)


def test_scan_uses_vertex_default_location_when_project_is_available(tmp_path: Path) -> None:
    adc = tmp_path / ".config/gcloud/application_default_credentials.json"
    _write(
        adc,
        {
            "type": "authorized_user",
            "client_id": "client",
            "refresh_token": "token",
            "quota_project_id": "project",
        },
    )

    [candidate] = scan({}, tmp_path)

    assert candidate.config["location"] == "us-central1"


def test_scan_does_not_treat_gemini_api_keys_as_vertex_adc(tmp_path: Path) -> None:
    assert scan(
        {
            "GEMINI_API_KEY": "api-key",
            "GOOGLE_API_KEY": "api-key",
            "GOOGLE_CLOUD_PROJECT": "project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
        tmp_path,
    ) == []


def test_scan_rejects_oversized_or_malformed_adc(tmp_path: Path) -> None:
    adc = tmp_path / ".config/gcloud/application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{" + "x" * 70000, encoding="utf-8")
    assert scan({"GOOGLE_CLOUD_LOCATION": "us-central1"}, tmp_path) == []
