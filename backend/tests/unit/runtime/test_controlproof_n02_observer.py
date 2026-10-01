from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from interview_evidence.runtime.controlproof_consent import (
    ControlProofProcessingObserver,
)


def _observer(tmp_path: Path) -> ControlProofProcessingObserver:
    return ControlProofProcessingObserver.from_environment(
        {
            "APP_ENVIRONMENT": "test",
            "CONTROLPROOF_OBSERVER_ENABLED": "true",
            "CONTROLPROOF_OBSERVER_ROOT": str(tmp_path),
        }
    )


def test_valid_receipt_is_sanitized_and_bound_to_run_lane(tmp_path: Path) -> None:
    run_id = uuid4()
    request_id = uuid4()
    trace_id = f"controlproof:{run_id}:DOCUMENT_BYPASS:document"
    observer = _observer(tmp_path)

    assert observer.record(
        trace_id=trace_id,
        path_id="DOCUMENT_ANALYSIS",
        boundary="ANALYSIS_HANDLER_ENTERED",
        subject_ref="synthetic-document-bypass",
        request_or_event_id=str(request_id),
    )

    payload = json.loads(
        (tmp_path / "receipts" / f"{run_id}.jsonl").read_text(encoding="utf-8")
    )
    assert payload["run_id"] == str(run_id)
    assert payload["lane_id"] == "DOCUMENT_BYPASS"
    assert payload["request_or_event_id"] == str(request_id)
    assert trace_id not in json.dumps(payload)
    assert set(payload) == {
        "schema_version",
        "receipt_id",
        "run_id",
        "lane_id",
        "subject_ref",
        "path_id",
        "boundary",
        "request_or_event_id",
        "trace_id_digest",
        "observed_at",
    }


@pytest.mark.parametrize(
    "trace_id",
    [
        "controlproof:not-a-uuid:DOCUMENT_BYPASS:document",
        f"controlproof:{uuid4()}:FOREIGN_LANE:document",
        f"foreign:{uuid4()}:DOCUMENT_BYPASS:document",
    ],
)
def test_invalid_or_foreign_trace_is_rejected(tmp_path: Path, trace_id: str) -> None:
    assert not _observer(tmp_path).record(
        trace_id=trace_id,
        path_id="DOCUMENT_ANALYSIS",
        boundary="ANALYSIS_HANDLER_ENTERED",
        subject_ref="synthetic-document-bypass",
        request_or_event_id=str(uuid4()),
    )
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize(
    "subject_ref",
    ["person@example.com", "010/1234", r"C:\secret", ""],
)
def test_subject_reference_cannot_contain_direct_pii(
    tmp_path: Path, subject_ref: str
) -> None:
    assert not _observer(tmp_path).record(
        trace_id=f"controlproof:{uuid4()}:DOCUMENT_BYPASS:document",
        path_id="DOCUMENT_ANALYSIS",
        boundary="ANALYSIS_HANDLER_ENTERED",
        subject_ref=subject_ref,
        request_or_event_id=str(uuid4()),
    )


def test_writer_io_failure_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny_write(*_args, **_kwargs):
        raise OSError("synthetic observer failure")

    monkeypatch.setattr(Path, "open", deny_write)
    assert not _observer(tmp_path).record(
        trace_id=f"controlproof:{uuid4()}:DOCUMENT_BYPASS:document",
        path_id="DOCUMENT_ANALYSIS",
        boundary="ANALYSIS_HANDLER_ENTERED",
        subject_ref="synthetic-document-bypass",
        request_or_event_id=str(uuid4()),
    )
