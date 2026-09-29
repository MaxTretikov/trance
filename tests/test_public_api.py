"""Public package API tests using synthetic models and candidates only."""

from __future__ import annotations

import subprocess
import sys

from trance import Candidate, FoundModel, clients, scan


def test_public_exports_are_available() -> None:
    assert callable(scan)
    assert callable(clients)
    candidate = Candidate("example", "api-key", "test", "example-model", "secret")
    assert candidate.provider == "example"
    assert FoundModel.__name__ == "FoundModel"


def test_package_import_does_not_load_pydantic_ai() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, trance; assert 'pydantic_ai' not in sys.modules; "
            "assert 'trance.discovery' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
