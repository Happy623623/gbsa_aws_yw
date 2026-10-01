from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from interview_evidence.runtime.controlproof_consent import (
    ControlProofConsentFaultGuard,
    ControlProofConsentFaultTriggered,
    ControlProofProcessingObserver,
)

RUN_ID = UUID("00000000-0000-7000-8000-000000000101")
INVITATION_ID = UUID("00000000-0000-7000-8000-000000000102")
APPLICANT_ID = UUID("00000000-0000-7000-8000-000000000103")
REQUEST_ID = UUID("00000000-0000-7000-8000-000000000104")
SUBJECT_REF = "synthetic-consent-fault"


def _enabled_guard(tmp_path: Path) -> ControlProofConsentFaultGuard:
    return ControlProofConsentFaultGuard.from_environment(
        {
            "APP_ENVIRONMENT": "test",
            "CONTROLPROOF_TEST_HOOKS_ENABLED": "true",
            "CONTROLPROOF_FAULT_ROOT": str(tmp_path / "faults"),
        }
    )


def _write_marker(
    guard: ControlProofConsentFaultGuard,
    *,
    run_id: UUID = RUN_ID,
    invitation_id: UUID = INVITATION_ID,
    applicant_id: UUID = APPLICANT_ID,
    subject_ref: str = SUBJECT_REF,
    issued_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> None:
    now = datetime.now(UTC)
    marker = {
        "schema_version": "controlproof.whyyou-consent-fault.v1",
        "run_id": str(run_id),
        "lane_id": "CONSENT_FAULT_RECOVERY",
        "subject_ref": subject_ref,
        "invitation_id": str(invitation_id),
        "applicant_id": str(applicant_id),
        "fault_type": "consent_after_record_before_state_v1",
        "fault_variant": "AFTER_CONSENT_RECORD_BEFORE_STATE",
        "issued_at": (issued_at or now - timedelta(seconds=1)).isoformat(),
        "expires_at": (expires_at or now + timedelta(minutes=5)).isoformat(),
        "one_shot": True,
    }
    path = guard.root / "consent" / f"{invitation_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker), encoding="utf-8")


def _trigger(guard: ControlProofConsentFaultGuard, *, run_id: UUID = RUN_ID) -> bool:
    return guard.trigger_if_configured(
        invitation_id=str(INVITATION_ID),
        applicant_id=str(APPLICANT_ID),
        request_id=str(REQUEST_ID),
        trace_id=f"controlproof:{run_id}:CONSENT_FAULT_RECOVERY:{SUBJECT_REF}",
    )


def test_fault_and_observer_are_disabled_by_default(tmp_path: Path) -> None:
    environment = {
        "APP_ENVIRONMENT": "local",
        "CONTROLPROOF_FAULT_ROOT": str(tmp_path / "faults"),
        "CONTROLPROOF_OBSERVER_ROOT": str(tmp_path / "observers"),
    }
    guard = ControlProofConsentFaultGuard.from_environment(environment)
    observer = ControlProofProcessingObserver.from_environment(environment)
    assert guard.enabled is False
    assert observer.enabled is False
    assert guard.evaluate(invitation_id="not-a-uuid", applicant_id="not-a-uuid") is None
    assert observer.record(
        trace_id="invalid", path_id="DOCUMENT_ANALYSIS", boundary="ANALYSIS_HANDLER_ENTERED"
    ) is False
    assert not (tmp_path / "faults").exists()
    assert not (tmp_path / "observers").exists()


@pytest.mark.parametrize("profile", ["production", "staging", ""])
@pytest.mark.parametrize(
    "flag", ["CONTROLPROOF_TEST_HOOKS_ENABLED", "CONTROLPROOF_OBSERVER_ENABLED"]
)
def test_enabled_controls_are_rejected_outside_local_or_test(
    tmp_path: Path, profile: str, flag: str
) -> None:
    environment = {
        "APP_ENVIRONMENT": profile,
        flag: "true",
        "CONTROLPROOF_FAULT_ROOT": str(tmp_path / "faults"),
        "CONTROLPROOF_OBSERVER_ROOT": str(tmp_path / "observers"),
    }
    factory = (
        ControlProofConsentFaultGuard.from_environment
        if flag == "CONTROLPROOF_TEST_HOOKS_ENABLED"
        else ControlProofProcessingObserver.from_environment
    )
    with pytest.raises(RuntimeError, match="outside local/test"):
        factory(environment)


@pytest.mark.parametrize(
    "key,value",
    [
        ("CONTROLPROOF_FAULT_ROOT", ""),
        ("CONTROLPROOF_OBSERVER_ROOT", ""),
    ],
)
def test_enabled_control_requires_a_bounded_root(
    tmp_path: Path, key: str, value: str
) -> None:
    is_fault = key == "CONTROLPROOF_FAULT_ROOT"
    environment = {
        "APP_ENVIRONMENT": "test",
        "CONTROLPROOF_TEST_HOOKS_ENABLED" if is_fault else "CONTROLPROOF_OBSERVER_ENABLED": "true",
        key: value,
    }
    factory = (
        ControlProofConsentFaultGuard.from_environment
        if is_fault
        else ControlProofProcessingObserver.from_environment
    )
    with pytest.raises(RuntimeError, match="ROOT"):
        factory(environment)


@pytest.mark.parametrize("variant", ["expired", "long_ttl", "wrong_subject", "wrong_run"])
def test_invalid_or_foreign_marker_is_a_safe_noop(tmp_path: Path, variant: str) -> None:
    guard = _enabled_guard(tmp_path)
    now = datetime.now(UTC)
    if variant == "expired":
        _write_marker(
            guard,
            issued_at=now - timedelta(minutes=2),
            expires_at=now - timedelta(minutes=1),
        )
    elif variant == "long_ttl":
        _write_marker(
            guard,
            issued_at=now,
            expires_at=now + timedelta(minutes=11),
        )
    elif variant == "wrong_subject":
        _write_marker(guard, applicant_id=UUID(int=999))
    else:
        _write_marker(guard)
    triggered = _trigger(guard, run_id=UUID(int=998) if variant == "wrong_run" else RUN_ID)
    assert triggered is False
    assert not (guard.root / "receipts" / f"{RUN_ID}.jsonl").exists()


def test_one_shot_concurrency_emits_exactly_one_fsynced_receipt(tmp_path: Path) -> None:
    guard = _enabled_guard(tmp_path)
    _write_marker(guard)

    def invoke() -> str:
        try:
            _trigger(guard)
        except ControlProofConsentFaultTriggered:
            return "triggered"
        return "noop"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _index: invoke(), range(8)))
    assert results.count("triggered") == 1
    receipt_path = guard.root / "receipts" / f"{RUN_ID}.jsonl"
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert len(receipts) == 1
    assert receipts[0]["request_id"] == str(REQUEST_ID)
    assert receipts[0]["one_shot_consumed"] is True


def test_receipt_fsync_failure_does_not_trigger_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _enabled_guard(tmp_path)
    _write_marker(guard)
    monkeypatch.setattr(os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync")))
    assert _trigger(guard) is False
    assert not (
        guard.root / "consumed" / f"{RUN_ID}-{INVITATION_ID}.consent"
    ).exists()
