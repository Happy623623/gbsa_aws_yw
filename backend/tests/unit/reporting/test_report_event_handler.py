from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
from interview_evidence.runtime.worker import (
    ReportRequestedEventHandler,
    _criterion_answers_by_criterion,
)
from interview_evidence.shared.ids import FrozenClock
from interview_evidence.shared.messaging.outbox import OutboxEvent
from interview_evidence.shared.tenant import ActorType, TenantContext

NOW = datetime(2026, 8, 21, tzinfo=UTC)
COMPANY_ID = UUID("00000000-0000-7000-8000-000000000001")
SESSION_ID = UUID("00000000-0000-7000-8000-000000000002")
INVITATION_ID = UUID("00000000-0000-7000-8000-000000000003")
VERSION_ID = UUID("00000000-0000-7000-8000-000000000004")
CRITERION_ID = UUID("00000000-0000-7000-8000-000000000005")
QUESTION_TURN_ID = UUID("00000000-0000-7000-8000-000000000006")
ANSWER_TURN_ID = UUID("00000000-0000-7000-8000-000000000007")


@pytest.mark.parametrize("fault_guard_blocks", [False, True])
def test_controlproof_report_start_receipt_follows_fault_guard(fault_guard_blocks: bool) -> None:
    context = TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.SYSTEM,
        actor_id=UUID("00000000-0000-7000-8000-000000000008"),
        request_id=UUID("00000000-0000-7000-8000-000000000009"),
        trace_id="controlproof:00000000-0000-7000-8000-000000000012:ASSESSMENT_BOUNDARY_PROBE:synthetic",
    )
    event = OutboxEvent(
        outbox_event_id=UUID("00000000-0000-7000-8000-000000000010"),
        company_id=COMPANY_ID,
        aggregate_type="interview_session",
        aggregate_id=SESSION_ID,
        aggregate_version=1,
        event_type="report.generation_requested",
        event_version=1,
        payload={"interview_session_id": str(SESSION_ID)},
        idempotency_key="report-generation-request",
        trace_id=context.trace_id,
        occurred_at=NOW,
    )
    observer = Mock()
    interview = Mock()
    interview.get_session_snapshot.return_value = SimpleNamespace(
        invitation_id=INVITATION_ID, competency_model_version_id=VERSION_ID
    )
    company = Mock()
    company.get_consent_authorization.return_value = SimpleNamespace(authorized=True)
    company.get_criterion_version.side_effect = RuntimeError("stop after start boundary")
    guard = Mock()
    if fault_guard_blocks:
        guard.before_report_side_effect.side_effect = RuntimeError("blocked")
    handler = ReportRequestedEventHandler(
        company=company,
        interview=interview,
        reporting=Mock(),
        generator=Mock(),
        clock=FrozenClock(NOW),
        processing_observer=observer,
        controlproof_fault_guard=guard,
    )
    with pytest.raises(RuntimeError):
        handler(context, event)
    boundaries = [call.kwargs["boundary"] for call in observer.record.call_args_list]
    assert boundaries == (
        ["REPORT_HANDLER_ENTERED"]
        if fault_guard_blocks
        else ["REPORT_HANDLER_ENTERED", "REPORT_ASSESSMENT_STARTED"]
    )


def test_report_handler_refuses_without_active_ai_assessment_consent() -> None:
    """T083: ControlProof N-02 child e2e8e71d recorded REPORT_ASSESSMENT_STARTED for
    unconsented subjects. The handler must refuse before the start boundary, record the
    refusal and finish without retry or any report read, write or model call."""
    context = TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.SYSTEM,
        actor_id=UUID("00000000-0000-7000-8000-000000000008"),
        request_id=UUID("00000000-0000-7000-8000-000000000009"),
        trace_id="controlproof:00000000-0000-7000-8000-000000000012:ASSESSMENT_BOUNDARY_PROBE:synthetic",
    )
    event = OutboxEvent(
        outbox_event_id=UUID("00000000-0000-7000-8000-000000000010"),
        company_id=COMPANY_ID,
        aggregate_type="interview_session",
        aggregate_id=SESSION_ID,
        aggregate_version=1,
        event_type="report.generation_requested",
        event_version=1,
        payload={"interview_session_id": str(SESSION_ID)},
        idempotency_key="report-generation-request",
        trace_id=context.trace_id,
        occurred_at=NOW,
    )
    observer, interview, company, reporting, generator = Mock(), Mock(), Mock(), Mock(), Mock()
    interview.get_session_snapshot.return_value = SimpleNamespace(
        invitation_id=INVITATION_ID, competency_model_version_id=VERSION_ID
    )
    company.get_consent_authorization.return_value = SimpleNamespace(authorized=False)
    handler = ReportRequestedEventHandler(
        company=company,
        interview=interview,
        reporting=reporting,
        generator=generator,
        clock=FrozenClock(NOW),
        processing_observer=observer,
    )

    assert handler(context, event) is None
    assert [call.kwargs["boundary"] for call in observer.record.call_args_list] == [
        "REPORT_HANDLER_ENTERED",
        "REPORT_ASSESSMENT_REFUSED",
    ]
    company.get_consent_authorization.assert_called_once_with(
        context, INVITATION_ID, required_purposes=frozenset({"ai_assessment"})
    )
    company.get_criterion_version.assert_not_called()
    assert reporting.method_calls == [] and generator.method_calls == []


def test_report_request_uses_transcript_range_for_evidence() -> None:
    context = TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.SYSTEM,
        actor_id=UUID("00000000-0000-7000-8000-000000000008"),
        request_id=UUID("00000000-0000-7000-8000-000000000009"),
        trace_id="trace-report-request",
    )
    transcript = SimpleNamespace(
        transcript_segment_id=UUID("00000000-0000-7000-8000-000000000011"),
        turn_id=ANSWER_TURN_ID,
        text="로그와 지표를 비교해 원인을 좁히고 복구했습니다.",
        session_start_ms=1_250,
        session_end_ms=4_750,
    )
    repository = Mock()
    repository.get_report_for_session.return_value = None
    repository.list_transcripts.return_value = (transcript,)
    repository.list_recording_assets.return_value = (
        SimpleNamespace(asset_type="final_video"),
    )
    repository.list_session_events.return_value = ()

    company = Mock()
    company.get_criterion_version.return_value = SimpleNamespace(
        criteria=(
            SimpleNamespace(
                criterion_id=CRITERION_ID,
                name="운영 문제 해결",
                description="운영 문제를 분석하고 복구할 수 있다.",
                weight=1.0,
            ),
        ),
        interview_level="entry",
        axis_weights={},
        job_requirements=(),
    )
    company.get_recruiting_assistant_subject.return_value = SimpleNamespace(
        applicant_id=UUID("00000000-0000-7000-8000-000000000012")
    )

    interview = Mock()
    interview.get_session_snapshot.return_value = SimpleNamespace(
        competency_model_version_id=VERSION_ID,
        invitation_id=INVITATION_ID,
    )
    interview.list_final_turns.return_value = (
        SimpleNamespace(
            turn_id=QUESTION_TURN_ID,
            speaker=SimpleNamespace(value="interviewer"),
            text="운영 문제를 해결한 경험을 설명해 주세요.",
        ),
        SimpleNamespace(
            turn_id=ANSWER_TURN_ID,
            speaker=SimpleNamespace(value="applicant"),
            text="로그와 지표를 비교해 원인을 좁히고 복구했습니다.",
        ),
    )
    interview.list_question_rationales.return_value = (
        SimpleNamespace(
            question_turn_id=QUESTION_TURN_ID,
            criterion_id=CRITERION_ID,
            interview_stage="technical",
        ),
    )

    generated_report = object()
    generator = Mock()
    generator.generate.return_value = generated_report
    handler = ReportRequestedEventHandler(
        company=company,
        interview=interview,
        reporting=SimpleNamespace(repository=repository),
        generator=generator,
        clock=FrozenClock(NOW),
    )

    result = handler(
        context,
        OutboxEvent(
            outbox_event_id=UUID("00000000-0000-7000-8000-000000000010"),
            company_id=COMPANY_ID,
            aggregate_type="interview_session",
            aggregate_id=SESSION_ID,
            aggregate_version=1,
            event_type="report.generation_requested",
            event_version=1,
            payload={"interview_session_id": str(SESSION_ID)},
            idempotency_key="report-generation-request",
            trace_id=context.trace_id,
            occurred_at=NOW,
        ),
    )

    assert result is generated_report
    criterion_input = generator.generate.call_args.kwargs["criteria"][0]
    assert len(criterion_input.answers) == 1
    assert criterion_input.answers[0].video_start_ms == transcript.session_start_ms
    assert criterion_input.answers[0].video_end_ms == transcript.session_end_ms


def test_repeated_question_keeps_only_answers_that_add_information() -> None:
    question_ids = tuple(
        UUID(f"00000000-0000-7000-8000-00000000002{index}") for index in range(3)
    )
    answer_ids = tuple(
        UUID(f"00000000-0000-7000-8000-00000000003{index}") for index in range(3)
    )
    question_text = "운영 문제를 해결한 경험을 설명해 주세요."
    repeated_answer = "로그를 비교해 원인을 찾고 복구했습니다."
    new_answer = "복구 후 오류율을 측정하고 재발 방지 알림을 추가했습니다."
    turns: list[SimpleNamespace] = []
    transcripts: dict[UUID, SimpleNamespace] = {}
    rationales: list[SimpleNamespace] = []
    for index, (question_id, answer_id, answer_text) in enumerate(
        zip(
            question_ids,
            answer_ids,
            (repeated_answer, repeated_answer, new_answer),
            strict=True,
        )
    ):
        turns.extend(
            (
                SimpleNamespace(
                    turn_id=question_id,
                    speaker=SimpleNamespace(value="interviewer"),
                    text=question_text,
                ),
                SimpleNamespace(
                    turn_id=answer_id,
                    speaker=SimpleNamespace(value="applicant"),
                    text=answer_text,
                ),
            )
        )
        transcripts[answer_id] = SimpleNamespace(
            transcript_segment_id=UUID(
                f"00000000-0000-7000-8000-00000000004{index}"
            ),
            turn_id=answer_id,
            text=answer_text,
            session_start_ms=index * 1_000,
            session_end_ms=(index + 1) * 1_000,
        )
        rationales.append(
            SimpleNamespace(
                question_turn_id=question_id,
                criterion_id=CRITERION_ID,
                interview_stage="technical",
            )
        )

    grouped = _criterion_answers_by_criterion(turns, rationales, transcripts)

    assert [answer.transcript.text for answer in grouped[CRITERION_ID]] == [
        repeated_answer,
        new_answer,
    ]
