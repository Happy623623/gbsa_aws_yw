from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src" / "interview_evidence"


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_analysis_and_report_receipts_precede_protected_handler_work() -> None:
    analysis = _source("workers/analysis/event_handler.py")
    worker = _source("runtime/worker.py")
    assert analysis.index('boundary="ANALYSIS_HANDLER_ENTERED"') < analysis.index(
        "get_submission(context, submission_id)"
    )
    assert worker.index('boundary="REPORT_HANDLER_ENTERED"') < worker.index(
        "get_session_snapshot(context, session_id=session_id)"
    )


def test_session_receipts_follow_each_successful_boundary() -> None:
    source = _source("interview_engine/application/session_service.py")
    assert source.index('boundary="INTERVIEW_SESSION_CREATED"') > source.index(
        "self._checkpoints.create("
    )
    start_method = source.index("def _start_session_once(")
    assert source.index('boundary="INTERVIEW_SESSION_STARTED"', start_method) > source.index(
        "self._repository.save_session(context, started)", start_method
    )
    confirm_method = source.index("def confirm_recording_upload(")
    assert source.index('boundary="RECORDING_CONFIRMED"', confirm_method) > source.index(
        "verify_uploaded_chunk(context, intent=intent)", confirm_method
    )


def test_observer_is_optional_and_wired_only_through_runtime_composition() -> None:
    api = _source("interview_engine/api/__init__.py")
    production = _source("runtime/production.py")
    worker = _source("runtime/worker.py")
    assert "processing_observer: ProcessingObserverPort | None = None" in api
    assert "processing_observer=processing_observer" in api
    assert "ControlProofProcessingObserver.from_environment(environment)" in production
    assert "ControlProofProcessingObserver.from_environment(environment)" in worker
