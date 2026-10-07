from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from interview_evidence.runtime.controlproof_model_substitute import (
    FIXTURE_DIGEST,
    FIXTURE_ID,
    SPEC004_FIXTURE_DIGEST,
    SPEC004_FIXTURE_ID,
    ControlProofFixedEmbedder,
    ControlProofFixedModel,
    ControlProofSpec004Model,
    controlproof_ai_isolation_digest,
    controlproof_health,
    resolve_controlproof_embedder,
    resolve_controlproof_model,
    validate_controlproof_test_controls,
)
from interview_evidence.shared.ids import new_uuid7
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
    health = controlproof_health(_isolated_environment())
    assert health == {
        "fault_hooks_enabled": False,
        "fault_root_digest": None,
        "model_substitute_enabled": True,
        "fixture_id": FIXTURE_ID,
        "fixture_digest": FIXTURE_DIGEST,
        "external_ai_isolated": True,
        "ai_isolation_digest": health["ai_isolation_digest"],
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
        _isolated_environment() | {"APP_ENVIRONMENT": "local"},
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


def _isolated_environment() -> dict[str, str]:
    return {
        "APP_ENVIRONMENT": "test",
        "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED": "true",
        "CONTROLPROOF_EXTERNAL_AI_ALLOWED": "false",
        "AI_PROVIDER": "aws",
        "EMBEDDING_PROVIDER": "aws",
        "STT_PROVIDER": "disabled",
        "TTS_PROVIDER": "text_only",
        "BEDROCK_RUNTIME_ENDPOINT_URL": "http://127.0.0.1:4566",
        "TRANSCRIBE_ENDPOINT_URL": "http://127.0.0.1:4566",
        "POLLY_ENDPOINT_URL": "http://127.0.0.1:4566",
        "GCP_DOCUMENT_AI_API_ENDPOINT": "127.0.0.1:4566",
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("BEDROCK_RUNTIME_ENDPOINT_URL", ""),
        ("TRANSCRIBE_ENDPOINT_URL", "https://transcribe.amazonaws.com"),
        ("POLLY_ENDPOINT_URL", "http://example.com"),
        ("GCP_DOCUMENT_AI_API_ENDPOINT", "us-documentai.googleapis.com"),
        ("AI_PROVIDER", "gcp"),
        ("EMBEDDING_PROVIDER", "gcp"),
        ("STT_PROVIDER", "gcp_streaming"),
        ("TTS_PROVIDER", "gcp_unary"),
        ("CONTROLPROOF_EXTERNAL_AI_ALLOWED", "true"),
    ],
)
def test_fixed_profile_rejects_external_ai_routes(name: str, value: str) -> None:
    environment = _isolated_environment()
    environment[name] = value
    with pytest.raises(RuntimeError):
        validate_controlproof_test_controls(environment)


def test_fixed_profile_health_attests_isolation_without_endpoint_disclosure() -> None:
    health = controlproof_health(_isolated_environment())
    assert health["external_ai_isolated"] is True
    assert len(health["ai_isolation_digest"]) == 64
    assert "127.0.0.1" not in json.dumps(health)


# --- Spec 004 fixture `spec004-report-v1` -------------------------------------------------
# Contract: ControlProof specs/004-e01-e02-score-evidence/contracts/whyyou-spec004-fixture.md

_H03_DIGEST = "ce09b95403b34e1390502c90f5c5edc518ddf65d38c8ce881617a37cac6d16b1"
_SPEC004_DIGEST = "e15ec3790b64b2fba10e0caa9372f08c917edbbaa99ce308076952b838668b3f"
_AXES = [{"key": "correctness"}, {"key": "depth"}]
_AT = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def _spec004_environment(**overrides: str) -> dict[str, str]:
    return (
        _isolated_environment() | {"CONTROLPROOF_MODEL_FIXTURE_ID": SPEC004_FIXTURE_ID} | overrides
    )


def _criterion_payload(criterion_id, text, evidence_ids):
    return {
        "task": "assess_interview_criterion",
        "criterion": {"criterion_id": str(criterion_id), "name": "기준", "text": text},
        "axes": _AXES,
        "provided_answers": [{"evidence_id": str(item)} for item in evidence_ids],
    }


def _quoted(result) -> set[tuple[str, ...]]:
    return {tuple(axis["quoted_evidence_ids"]) for axis in result["axis_scores"]}


def _scores(result) -> set[int | None]:
    return {axis["score"] for axis in result["axis_scores"]}


def _receipts(root: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "model").glob("*.json"))
    ]


def _generate(model, criterion_id, text, evidence_ids, *, context=None):
    return model.generate(
        context or _context(), _input(_criterion_payload(criterion_id, text, evidence_ids))
    )


def test_h03_fixture_identity_is_unchanged() -> None:
    assert FIXTURE_ID == "h03-report-v1"
    assert FIXTURE_DIGEST == _H03_DIGEST
    assert controlproof_health(_isolated_environment())["fixture_digest"] == _H03_DIGEST


def test_h03_fixture_ignores_spec004_markers() -> None:
    evidence_id = new_uuid7(_AT)
    marker = "[controlproof-spec004 mode=EMPTY score=10] 설명"
    result = _generate(ControlProofFixedModel(), uuid4(), marker, [evidence_id])
    assert _quoted(result) == {(str(evidence_id),)}
    assert _scores(result) == {72}


def test_spec004_identity_is_derived_from_its_seed() -> None:
    assert SPEC004_FIXTURE_ID == "spec004-report-v1"
    assert SPEC004_FIXTURE_DIGEST == _SPEC004_DIGEST


def test_spec004_is_selected_by_fixture_id_and_reported_in_health() -> None:
    environment = _spec004_environment()
    assert isinstance(resolve_controlproof_model(environment, object()), ControlProofSpec004Model)
    assert isinstance(
        resolve_controlproof_embedder(environment, object()), ControlProofFixedEmbedder
    )
    health = controlproof_health(environment)
    assert health["fixture_id"] == SPEC004_FIXTURE_ID
    assert health["fixture_digest"] == SPEC004_FIXTURE_DIGEST
    assert controlproof_ai_isolation_digest(environment) != controlproof_ai_isolation_digest(
        _isolated_environment()
    )


def test_unknown_fixture_id_is_rejected_at_startup() -> None:
    with pytest.raises(RuntimeError, match="fixture"):
        validate_controlproof_test_controls(
            _isolated_environment() | {"CONTROLPROOF_MODEL_FIXTURE_ID": "spec999-report-v1"}
        )


def test_spec004_without_marker_matches_h03_output() -> None:
    payload = _criterion_payload(uuid4(), "마커 없는 합성 기준 설명", [new_uuid7(_AT)])
    assert ControlProofSpec004Model().generate(_context(), _input(payload)) == (
        ControlProofFixedModel().generate(_context(), _input(payload))
    )


def test_spec004_requirement_task_matches_h03_output() -> None:
    payload = {
        "task": "assess_job_requirement",
        "evidence_candidates": [{"evidence_id": str(uuid4())}],
    }
    assert ControlProofSpec004Model().generate(_context(), _input(payload)) == (
        ControlProofFixedModel().generate(_context(), _input(payload))
    )


def test_spec004_valid_mode_cites_first_answer_with_marker_score() -> None:
    first, second = new_uuid7(_AT), new_uuid7(_AT)
    marker = "[controlproof-spec004 mode=VALID score=73] 설명"
    result = _generate(ControlProofSpec004Model(), uuid4(), marker, [first, second])
    assert _quoted(result) == {(str(first),)}
    assert _scores(result) == {73}
    assert result["assessment_state"] == "confirmed"


def test_spec004_empty_mode_keeps_a_score_without_citation() -> None:
    marker = "[controlproof-spec004 mode=EMPTY]"
    result = _generate(ControlProofSpec004Model(), uuid4(), marker, [new_uuid7(_AT)])
    assert _quoted(result) == {()}
    assert _scores(result) == {72}


@pytest.mark.parametrize("mode", ["NONEXISTENT", "OTHER_APPLICANT"])
def test_spec004_marker_argument_modes_cite_only_the_argument(mode: str) -> None:
    argument = uuid4()
    marker = f"[controlproof-spec004 mode={mode} arg={argument}]"
    result = _generate(ControlProofSpec004Model(), uuid4(), marker, [new_uuid7(_AT)])
    assert _quoted(result) == {(str(argument),)}
    assert _scores(result) == {72}


def test_spec004_other_criterion_uses_the_same_call_memory() -> None:
    model = ControlProofSpec004Model()
    context = _context()
    valid_criterion, valid_evidence = uuid4(), new_uuid7(_AT)
    _generate(
        model,
        valid_criterion,
        "[controlproof-spec004 mode=VALID]",
        [valid_evidence],
        context=context,
    )
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={valid_criterion}]"
    result = _generate(model, uuid4(), marker, [new_uuid7(_AT)], context=context)
    assert _quoted(result) == {(str(valid_evidence),)}
    assert _scores(result) == {72}


def test_spec004_other_criterion_refuses_memory_from_another_call() -> None:
    model = ControlProofSpec004Model()
    context = _context()
    valid_criterion = uuid4()
    _generate(
        model,
        valid_criterion,
        "[controlproof-spec004 mode=VALID]",
        [new_uuid7(_AT)],
        context=context,
    )
    later = new_uuid7(_AT + timedelta(milliseconds=1))
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={valid_criterion}]"
    result = _generate(model, uuid4(), marker, [later], context=context)
    assert _quoted(result) == {()}
    assert _scores(result) == {None}


def test_spec004_other_criterion_without_memory_emits_nothing() -> None:
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={uuid4()}]"
    result = _generate(ControlProofSpec004Model(), uuid4(), marker, [new_uuid7(_AT)])
    assert _quoted(result) == {()}
    assert _scores(result) == {None}


def test_spec004_other_applicant_never_reads_memory() -> None:
    model = ControlProofSpec004Model()
    remembered_criterion = uuid4()
    _generate(model, remembered_criterion, "[controlproof-spec004 mode=VALID]", [new_uuid7(_AT)])
    marker = f"[controlproof-spec004 mode=OTHER_APPLICANT arg={remembered_criterion}]"
    result = _generate(model, uuid4(), marker, [new_uuid7(_AT)])
    assert _quoted(result) == {(str(remembered_criterion),)}


def test_spec004_memory_is_bounded() -> None:
    model = ControlProofSpec004Model()
    context = _context()
    oldest = uuid4()
    _generate(model, oldest, "[controlproof-spec004 mode=VALID]", [new_uuid7(_AT)], context=context)
    for _ in range(256):
        _generate(
            model, uuid4(), "[controlproof-spec004 mode=VALID]", [new_uuid7(_AT)], context=context
        )
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={oldest}]"
    assert _quoted(_generate(model, uuid4(), marker, [new_uuid7(_AT)], context=context)) == {()}


@pytest.mark.parametrize("boundary", ["company_id", "request_id"])
def test_spec004_same_millisecond_cannot_cross_report_scope(boundary, tmp_path):
    model = ControlProofSpec004Model(observer_root=tmp_path)
    context = _context()
    other = context.model_copy(update={boundary: uuid4()})
    criterion = uuid4()
    _generate(
        model, criterion, "[controlproof-spec004 mode=VALID]", [new_uuid7(_AT)], context=context
    )
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={criterion}]"
    result = _generate(model, uuid4(), marker, [new_uuid7(_AT)], context=other)
    assert _quoted(result) == {()}
    assert _scores(result) == {None}
    assert any(row["mode_status"] == "MODE_SOURCE_MISSING" for row in _receipts(tmp_path))


def test_spec004_interleaved_reports_keep_separate_criterion_memory():
    model = ControlProofSpec004Model()
    first = _context()
    second = first.model_copy(update={"request_id": uuid4()})
    criterion = uuid4()
    first_evidence, second_evidence = new_uuid7(_AT), new_uuid7(_AT)
    for context, evidence in ((first, first_evidence), (second, second_evidence)):
        _generate(
            model, criterion, "[controlproof-spec004 mode=VALID]", [evidence], context=context
        )
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={criterion}]"
    for context, evidence in ((first, first_evidence), (second, second_evidence)):
        result = _generate(model, uuid4(), marker, [new_uuid7(_AT)], context=context)
        assert _quoted(result) == {(str(evidence),)}


def test_spec004_without_answers_scores_nothing_in_every_mode() -> None:
    marker = "[controlproof-spec004 mode=EMPTY score=90]"
    result = _generate(ControlProofSpec004Model(), uuid4(), marker, [])
    assert _quoted(result) == {()}
    assert _scores(result) == {None}


@pytest.mark.parametrize(
    "marker",
    [
        "[controlproof-spec004 mode=UNKNOWN]",
        "[controlproof-spec004 mode=NONEXISTENT]",
        "[controlproof-spec004 mode=OTHER_APPLICANT arg=not-a-uuid]",
        "[controlproof-spec004 mode=VALID score=101]",
        "[controlproof-spec004 mode=VALID arg=00000000-0000-0000-0000-000000000001]",
        "[controlproof-spec004 mode=VALID mode=EMPTY]",
        "[controlproof-spec004 score=50]",
        "[controlproof-spec004 mode=EMPTY",
        "[controlproof-spec004mode=EMPTY]",
    ],
)
def test_spec004_invalid_marker_falls_back_to_default(marker: str, tmp_path: Path) -> None:
    evidence_id = new_uuid7(_AT)
    model = ControlProofSpec004Model(observer_root=tmp_path)
    result = _generate(model, uuid4(), marker, [evidence_id])
    assert _quoted(result) == {(str(evidence_id),)}
    assert _scores(result) == {72}
    (receipt,) = _receipts(tmp_path)
    assert receipt["mode"] == "DEFAULT"
    assert receipt["mode_status"] == "MARKER_INVALID"


def test_spec004_other_criterion_cannot_reference_itself(tmp_path: Path) -> None:
    criterion_id = uuid4()
    model = ControlProofSpec004Model(observer_root=tmp_path)
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={criterion_id}]"
    _generate(model, criterion_id, marker, [new_uuid7(_AT)])
    assert _receipts(tmp_path)[0]["mode_status"] == "MARKER_INVALID"


def test_spec004_writes_a_sanitized_emission_receipt(tmp_path: Path) -> None:
    criterion_id, argument, evidence_id = uuid4(), uuid4(), new_uuid7(_AT)
    model = ControlProofSpec004Model(observer_root=tmp_path)
    marker = f"[controlproof-spec004 mode=OTHER_APPLICANT arg={argument} score=74] 합성 기준 본문"
    payload = _criterion_payload(criterion_id, marker, [evidence_id])
    payload["provided_answers"][0] |= {"question": "합성 질문", "answer_text": "합성 답변"}
    model.generate(_context(), _input(payload))

    (receipt,) = _receipts(tmp_path)
    assert receipt == {
        "schema_version": "controlproof.spec004-model-emission.v1",
        "receipt_id": receipt["receipt_id"],
        "fixture_id": SPEC004_FIXTURE_ID,
        "criterion_id": str(criterion_id),
        "mode": "OTHER_APPLICANT",
        "mode_status": "EMITTED",
        "provided_evidence_ids": [str(evidence_id)],
        "emitted_quoted_ids": [str(argument)],
        "emitted_score": 74,
        "emitted_at": receipt["emitted_at"],
    }
    datetime.fromisoformat(receipt["emitted_at"])
    raw = (tmp_path / "model" / f"{receipt['receipt_id']}.json").read_text(encoding="utf-8")
    for text in ("합성 기준 본문", "합성 질문", "합성 답변"):
        assert text not in raw
    assert not list((tmp_path / "model").glob("*.tmp"))


def test_spec004_receipt_records_missing_memory_source(tmp_path: Path) -> None:
    model = ControlProofSpec004Model(observer_root=tmp_path)
    marker = f"[controlproof-spec004 mode=OTHER_CRITERION arg={uuid4()}]"
    _generate(model, uuid4(), marker, [new_uuid7(_AT)])
    (receipt,) = _receipts(tmp_path)
    assert receipt["mode_status"] == "MODE_SOURCE_MISSING"
    assert receipt["emitted_quoted_ids"] == []
    assert receipt["emitted_score"] is None


def test_spec004_writes_no_receipt_without_observer_root(tmp_path: Path) -> None:
    _generate(ControlProofSpec004Model(), uuid4(), "설명", [new_uuid7(_AT)])
    assert not (tmp_path / "model").exists()


def test_spec004_receipt_write_failure_does_not_block_the_response(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x", encoding="utf-8")
    evidence_id = new_uuid7(_AT)
    result = _generate(
        ControlProofSpec004Model(observer_root=blocked), uuid4(), "설명", [evidence_id]
    )
    assert _quoted(result) == {(str(evidence_id),)}


def test_spec004_resolution_passes_the_observer_root(tmp_path: Path) -> None:
    model = resolve_controlproof_model(
        _spec004_environment(CONTROLPROOF_OBSERVER_ROOT=str(tmp_path)), object()
    )
    _generate(model, uuid4(), "설명", [new_uuid7(_AT)])
    assert len(_receipts(tmp_path)) == 1
