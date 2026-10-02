from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from interview_evidence.runtime.controlproof_model_substitute import (
    FIXTURE_DIGEST,
    controlproof_health,
    validate_controlproof_test_controls,
)


def test_repository_defaults_keep_both_controls_disabled() -> None:
    root = Path(__file__).resolve().parents[3]
    env = dict(
        line.split("=", 1)
        for line in (root / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    compose = yaml.safe_load((root / "compose.yaml").read_text(encoding="utf-8"))

    assert env["CONTROLPROOF_TEST_HOOKS_ENABLED"] == "false"
    assert env["CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED"] == "false"
    assert compose["x-controlproof-test-controls"]["CONTROLPROOF_TEST_HOOKS_ENABLED"].endswith(
        ":-false}"
    )
    assert compose["x-controlproof-test-controls"][
        "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED"
    ].endswith(":-false}")


@pytest.mark.parametrize(
    "name",
    ["CONTROLPROOF_TEST_HOOKS_ENABLED", "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED"],
)
def test_production_rejects_each_opt_in(name) -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_controlproof_test_controls({"APP_ENVIRONMENT": "production", name: "true"})


def test_local_health_exposes_no_secret_or_model_output() -> None:
    result = controlproof_health(
        {
            "APP_ENVIRONMENT": "local",
            "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED": "true",
            "AI_PROVIDER": "aws",
            "EMBEDDING_PROVIDER": "aws",
            "STT_PROVIDER": "disabled",
            "TTS_PROVIDER": "text_only",
            "BEDROCK_RUNTIME_ENDPOINT_URL": "http://127.0.0.1:4566",
            "TRANSCRIBE_ENDPOINT_URL": "http://127.0.0.1:4566",
            "POLLY_ENDPOINT_URL": "http://127.0.0.1:4566",
            "GCP_DOCUMENT_AI_API_ENDPOINT": "127.0.0.1:4566",
        }
    )
    assert set(result) == {
        "fault_hooks_enabled",
        "fault_root_digest",
        "model_substitute_enabled",
        "fixture_id",
        "fixture_digest",
        "external_ai_isolated",
        "ai_isolation_digest",
    }
    assert result["fault_root_digest"] is None
    assert result["fixture_digest"] == FIXTURE_DIGEST
