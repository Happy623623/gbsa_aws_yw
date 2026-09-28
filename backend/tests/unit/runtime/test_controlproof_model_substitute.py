from __future__ import annotations

import hashlib
import json
from uuid import uuid4

import pytest
from interview_evidence.runtime.controlproof_model_substitute import (
    FIXTURE_DIGEST,
    FIXTURE_ID,
    ControlProofFixedEmbedder,
    ControlProofFixedModel,
    controlproof_health,
    resolve_controlproof_embedder,
    resolve_controlproof_model,
    validate_controlproof_test_controls,
)
from interview_evidence.shared.tenant import ActorType, TenantContext


def _context() -> TenantContext:
    company_id = uuid4()
    return TenantContext(
        company_id=company_id,
        actor_type=ActorType.SYSTEM,
        actor_id=uuid4(),
        request_id=uuid4(),
        trace_id="controlproof-test",
    )


def _input(payload):
    return {
        "messages": [{"role": "user", "content": [{"type": "text", "text": json.dumps(payload)}]}]
    }


def test_model_substitute_is_disabled_by_default() -> None:
    fallback = object()
    assert resolve_controlproof_model({"APP_ENVIRONMENT": "local"}, fallback) is fallback
    assert resolve_controlproof_embedder({"APP_ENVIRONMENT": "local"}, fallback) is fallback


@pytest.mark.parametrize(
    "control",
    ["CONTROLPROOF_TEST_HOOKS_ENABLED", "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED"],
)
def test_test_controls_are_rejected_in_production(control) -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_controlproof_test_controls({"APP_ENVIRONMENT": "production", control: "true"})


def test_health_exposes_only_control_state_and_fixture_identity() -> None:
    health = controlproof_health(
        {
            "APP_ENVIRONMENT": "test",
            "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED": "true",
        }
    )
    assert health == {
        "fault_hooks_enabled": False,
        "fault_root_digest": None,
        "model_substitute_enabled": True,
        "fixture_id": FIXTURE_ID,
        "fixture_digest": FIXTURE_DIGEST,
    }


def test_health_exposes_only_a_digest_for_the_shared_fault_root(tmp_path) -> None:
    fault_root = tmp_path / "faults"
    health = controlproof_health(
        {
            "APP_ENVIRONMENT": "test",
            "CONTROLPROOF_TEST_HOOKS_ENABLED": "true",
            "CONTROLPROOF_FAULT_ROOT": str(fault_root),
        }
    )

    expected = hashlib.sha256(
        fault_root.resolve().as_posix().casefold().encode("utf-8")
    ).hexdigest()
    assert health["fault_root_digest"] == expected
    assert str(fault_root) not in json.dumps(health)


def test_fixed_model_is_deterministic_and_cites_input_evidence() -> None:
    criterion_id = str(uuid4())
    evidence_id = str(uuid4())
    payload = {
        "task": "assess_interview_criterion",
        "criterion": {"criterion_id": criterion_id},
        "axes": [{"key": "correctness"}, {"key": "depth"}],
        "provided_answers": [{"evidence_id": evidence_id}],
    }
    model = ControlProofFixedModel()
    first = model.generate(_context(), _input(payload))
    second = model.generate(_context(), _input(payload))

    assert first == second
    assert first["criterion_id"] == criterion_id
    assert {tuple(axis["quoted_evidence_ids"]) for axis in first["axis_scores"]} == {(evidence_id,)}


def test_fixed_requirement_result_never_calls_external_fallback() -> None:
    evidence_id = str(uuid4())
    model = resolve_controlproof_model(
        {
            "APP_ENVIRONMENT": "local",
            "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED": "true",
            "CONTROLPROOF_MODEL_FIXTURE_ID": FIXTURE_ID,
        },
        fallback=object(),
    )
    result = model.generate(
        _context(),
        _input(
            {
                "task": "assess_job_requirement",
                "evidence_candidates": [{"evidence_id": evidence_id}],
            }
        ),
    )
    assert result["signals"][0]["evidence_id"] == evidence_id
    assert result["signals"][0]["relation"] == "partially_supports"


def test_fixed_embedder_is_deterministic_and_has_requested_dimensions() -> None:
    embedder = ControlProofFixedEmbedder()

    first = embedder.embed(_context(), "합성 지원자 리포트", dimensions=32)
    second = embedder.embed(_context(), "합성 지원자 리포트", dimensions=32)

    assert first == second
    assert len(first) == 32
    assert sum(value * value for value in first) == pytest.approx(1.0)
