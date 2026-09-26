from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from interview_evidence.runtime.controlproof_faults import (
    FAULT_SCHEMA,
    FAULT_TYPE,
    ControlProofReportingFaultGuard,
)
from interview_evidence.shared.messaging.outbox import OutboxEvent


def _event(session_id: UUID, *, attempt: int = 2) -> OutboxEvent:
    now = datetime.now(UTC)
    return OutboxEvent(
        outbox_event_id=uuid4(),
        company_id=uuid4(),
        aggregate_type="interview_session",
        aggregate_id=session_id,
        aggregate_version=1,
        event_type="report.generation_requested",
        event_version=1,
        payload={"interview_session_id": str(session_id)},
        idempotency_key=f"report-{session_id}",
        trace_id="controlproof-test",
        occurred_at=now,
        delivery_attempt=attempt,
    )


def _write_marker(root, session_id: UUID, **overrides) -> UUID:
    run_id = uuid4()
    now = datetime.now(UTC)
    marker = {
        "schema_version": FAULT_SCHEMA,
        "run_id": str(run_id),
        "interview_session_id": str(session_id),
        "fault_type": FAULT_TYPE,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=5)).isoformat(),
        **overrides,
    }
    path = root / "reporting" / f"{session_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(marker), encoding="utf-8")
    return run_id


def test_hook_is_disabled_by_default(tmp_path) -> None:
    guard = ControlProofReportingFaultGuard.from_environment(
        {"APP_ENVIRONMENT": "local", "CONTROLPROOF_FAULT_ROOT": str(tmp_path)}
    )
    guard.before_report_side_effect(_event(uuid4()))
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("profile", ["production", "staging", ""])
def test_enabled_hook_is_rejected_outside_local_or_test(tmp_path, profile) -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        ControlProofReportingFaultGuard.from_environment(
            {
                "APP_ENVIRONMENT": profile,
                "CONTROLPROOF_TEST_HOOKS_ENABLED": "true",
                "CONTROLPROOF_FAULT_ROOT": str(tmp_path),
            }
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "unknown"},
        {"interview_session_id": str(uuid4())},
        {"expires_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()},
        {"expires_at": (datetime.now(UTC) + timedelta(minutes=20)).isoformat()},
    ],
)
def test_malformed_expired_or_wrong_session_marker_is_ignored(tmp_path, overrides) -> None:
    session_id = uuid4()
    _write_marker(tmp_path, session_id, **overrides)
    guard = ControlProofReportingFaultGuard.from_environment(
        {
            "APP_ENVIRONMENT": "test",
            "CONTROLPROOF_TEST_HOOKS_ENABLED": "true",
            "CONTROLPROOF_FAULT_ROOT": str(tmp_path),
        }
    )
    guard.before_report_side_effect(_event(session_id))
    assert not (tmp_path / "receipts").exists()


def test_matching_marker_appends_and_fsyncs_receipt_before_timeout(
    tmp_path,
    caplog,
) -> None:
    session_id = uuid4()
    run_id = _write_marker(tmp_path, session_id)
    event = _event(session_id, attempt=3)
    guard = ControlProofReportingFaultGuard.from_environment(
        {
            "APP_ENVIRONMENT": "local",
            "CONTROLPROOF_TEST_HOOKS_ENABLED": "true",
            "CONTROLPROOF_FAULT_ROOT": str(tmp_path),
        }
    )

    with caplog.at_level(logging.WARNING), pytest.raises(TimeoutError, match="fault triggered"):
        guard.before_report_side_effect(event)

    lines = (tmp_path / "receipts" / f"{run_id}.jsonl").read_text().splitlines()
    receipt = json.loads(lines[0])
    assert receipt["run_id"] == str(run_id)
    assert receipt["session_id"] == str(session_id)
    assert receipt["outbox_event_id"] == str(event.outbox_event_id)
    assert receipt["delivery_attempt"] == 3
    assert receipt["fault_type"] == FAULT_TYPE
    assert "CONTROLPROOF_FAULT_TRIGGERED" in caplog.messages


def test_worker_invokes_guard_before_first_report_side_effect() -> None:
    worker = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "interview_evidence"
        / "runtime"
        / "worker.py"
    ).read_text(encoding="utf-8")
    handler = worker.split("class ReportRequestedEventHandler:", 1)[1]
    call = handler.index("before_report_side_effect(event)")
    first_read = handler.index("get_session_snapshot(")
    assert call < first_read
