from __future__ import annotations

from typing import Protocol
from uuid import UUID

from interview_evidence.company_management.application.company_service import (
    CompanyManagementPublic,
)
from interview_evidence.shared.ids import CommandMeta
from interview_evidence.shared.messaging.outbox import OutboxEvent
from interview_evidence.shared.tenant import TenantContext
from interview_evidence.submission_analysis.api import LaneBRuntime
from interview_evidence.submission_analysis.domain.submission import (
    SourceType,
    Submission,
    SubmissionStatus,
)
from interview_evidence.workers.analysis.handlers import (
    AnalysisJob,
    AnalysisJobHandler,
    JobStatus,
)

ANALYSIS_RETRY_EXHAUSTED_SUMMARY = (
    "분석 서비스 연결이 반복해서 실패해 자료를 처리하지 못했습니다. 다시 제출해 주세요."
)


class InvitationAnalysisFinalizer(Protocol):
    def finalize_invitation(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
        applicant_id: UUID,
        submission_ids: frozenset[UUID],
    ) -> bool: ...


class ProcessingObserver(Protocol):
    def record(
        self,
        *,
        trace_id: str,
        path_id: str,
        boundary: str,
        subject_ref: str = "",
        request_or_event_id: str = "",
    ) -> bool: ...


class AnalysisRequestedEventHandler:
    def __init__(
        self,
        runtime: LaneBRuntime,
        handler: AnalysisJobHandler,
        company: CompanyManagementPublic | None = None,
        processing_observer: ProcessingObserver | None = None,
    ) -> None:
        self._runtime = runtime
        self._handler = handler
        self._company = company
        self._processing_observer = processing_observer

    def __call__(self, context: TenantContext, event: OutboxEvent) -> object:
        submission_id = UUID(str(event.payload["submission_id"]))
        if self._processing_observer is not None:
            try:
                parts = event.trace_id.split(":", 3)
                self._processing_observer.record(
                    trace_id=event.trace_id,
                    path_id="DOCUMENT_ANALYSIS",
                    boundary="ANALYSIS_HANDLER_ENTERED",
                    subject_ref=parts[3] if len(parts) == 4 else "",
                    request_or_event_id=str(event.outbox_event_id),
                )
            except Exception:  # noqa: BLE001 - observer cannot fail analysis
                pass
        submission = self._runtime.repository.get_submission(context, submission_id)
        if self._company is not None and _owns_analysis_state_transition(
            submission,
            self._runtime.repository.list_submissions_for_invitation(
                context,
                submission.invitation_id,
            ),
        ):
            snapshot = self._company.authorize_invitation(
                context,
                submission.invitation_id,
                required_state=frozenset({"materials_submitted", "analyzing", "ready"}),
            )
            if snapshot.state == "materials_submitted":
                self._company.advance_invitation_state(
                    context,
                    submission.invitation_id,
                    from_state="materials_submitted",
                    to_state="analyzing",
                    meta=CommandMeta.create(
                        f"analysis-started-{submission.invitation_id}",
                        expected_version=snapshot.row_version,
                    ),
                )
        outcome = self._handler.handle(
            context,
            AnalysisJob(
                submission_id=submission_id,
                invitation_id=submission.invitation_id,
                applicant_id=submission.applicant_id,
                analysis_version=int(event.payload["analysis_version"]),
                source_type=SourceType(str(event.payload["source_type"])),
                source_object_id=UUID(str(event.payload["source_object_id"])),
                idempotency_key=event.idempotency_key,
            ),
            attempt_number=event.delivery_attempt,
        )
        if outcome.status is JobStatus.DLQ:
            current = self._runtime.repository.get_submission(context, submission_id)
            if current.status in {
                SubmissionStatus.RECEIVED,
                SubmissionStatus.VALIDATING,
                SubmissionStatus.ANALYZING,
            }:
                self._runtime.repository.save_submission(
                    context,
                    current.transition(
                        SubmissionStatus.FAILED,
                        failure_code=outcome.failure_code or "analysis_retry_exhausted",
                        impact_summary=ANALYSIS_RETRY_EXHAUSTED_SUMMARY,
                    ),
                )
        return outcome


def _owns_analysis_state_transition(
    submission: Submission,
    submissions: tuple[Submission, ...],
) -> bool:
    """Elects one submission of a batch to move the invitation to `analyzing`.

    Note that the elected submission usually cannot: the invitation only reaches
    `materials_submitted` once the last required material arrives, which is after the earliest
    submission's analysis event. `AnalysisCompletedEventHandler` therefore makes the move itself
    when it finds the invitation still at `materials_submitted`, and this election only takes
    effect for a submission added to an invitation that already reached that state.
    """
    return submission.submission_id == min(item.submission_id for item in submissions)


class AnalysisCompletedEventHandler:
    def __init__(
        self,
        runtime: LaneBRuntime,
        finalizer: InvitationAnalysisFinalizer,
        company: CompanyManagementPublic,
    ) -> None:
        self._runtime = runtime
        self._finalizer = finalizer
        self._company = company

    def __call__(self, context: TenantContext, event: OutboxEvent) -> object:
        invitation_id = UUID(str(event.payload["invitation_id"]))
        submissions = self._runtime.repository.list_submissions_for_invitation(
            context,
            invitation_id,
        )
        if not submissions or any(
            submission.status
            in {
                SubmissionStatus.RECEIVED,
                SubmissionStatus.VALIDATING,
                SubmissionStatus.ANALYZING,
            }
            for submission in submissions
        ):
            return {"status": "waiting_for_submissions"}
        requirements = self._company.get_submission_requirements(
            context,
            invitation_id,
        ).requirements
        required_materials = {
            requirement.material_type
            for requirement in requirements
            if requirement.enabled and requirement.required
        }
        ready_materials = {
            submission.material_type
            for submission in submissions
            if submission.status in {SubmissionStatus.READY, SubmissionStatus.PARTIAL}
        }
        if not required_materials.issubset(ready_materials):
            return {"status": "required_submission_failed"}
        included = tuple(
            submission
            for submission in submissions
            if submission.status in {SubmissionStatus.READY, SubmissionStatus.PARTIAL}
        )
        if not included or not self._finalizer.finalize_invitation(
            context,
            invitation_id=invitation_id,
            applicant_id=included[0].applicant_id,
            submission_ids=frozenset(submission.submission_id for submission in included),
        ):
            return {"status": "strategy_not_ready"}
        authorization = self._company.authorize_invitation(
            context,
            invitation_id,
            required_state=frozenset({"materials_submitted", "analyzing", "ready"}),
        )
        state = authorization.state
        row_version = authorization.row_version
        if state == "materials_submitted":
            # `AnalysisRequestedEventHandler` is supposed to have made this move when the first
            # analysis started, and cannot: the invitation only reaches `materials_submitted`
            # once every required material is in, which is after the earliest submission's
            # analysis event -- the one that handler gives the transition to -- has been and
            # gone. Every analysis is finished by the time we get here, so `analyzing` is a
            # formality; without it `ready` is unreachable and the applicant is never
            # interviewable.
            advanced = self._company.advance_invitation_state(
                context,
                invitation_id,
                from_state="materials_submitted",
                to_state="analyzing",
                meta=CommandMeta.create(
                    f"analysis-started-{invitation_id}",
                    expected_version=row_version,
                ),
            )
            state = advanced.state
            row_version = advanced.row_version
        if state != "analyzing":
            return {"status": state}
        self._company.advance_invitation_state(
            context,
            invitation_id,
            from_state="analyzing",
            to_state="ready",
            meta=CommandMeta.create(
                f"analysis-ready-{invitation_id}",
                expected_version=row_version,
            ),
        )
        return {"status": "ready"}
