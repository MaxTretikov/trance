from __future__ import annotations

import pytest

from trance.types import Candidate, FoundModel


def test_candidate_redacts_secret_and_config_values_in_repr() -> None:
    candidate = Candidate(
        provider="example",
        auth_kind="subscription",
        source="test",
        model_name="example-model",
        secret="secret-value",
        config={"access_token": "config-secret", "region": "us-test"},
    )

    rendered = repr(candidate)
    assert "secret-value" not in rendered
    assert "config-secret" not in rendered
    assert "us-test" not in rendered
    assert "<redacted>" in rendered


def test_candidate_config_is_copied_and_immutable() -> None:
    config = {"region": "us-test"}
    candidate = Candidate("example", "api_key", "env:X", "model", config=config)

    config["region"] = "mutated"
    assert candidate.config["region"] == "us-test"
    with pytest.raises(TypeError):
        candidate.config["region"] = "mutated"  # type: ignore[index]


def test_candidate_has_an_empty_config_by_default() -> None:
    candidate = Candidate("example", "subscription", "test", "model")

    assert candidate.config == {}
    assert repr(candidate).endswith("config={})")


def test_found_model_hides_model_representation() -> None:
    model = object()
    found = FoundModel("example", "subscription", "test", "model", model)  # type: ignore[arg-type]

    assert found.model is model
    assert "model=<redacted>" in repr(found)
    assert repr(model) not in repr(found)
