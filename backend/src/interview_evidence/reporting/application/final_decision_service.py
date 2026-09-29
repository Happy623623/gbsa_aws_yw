from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from interview_evidence.reporting.application.review_service import (
    InvitationDecisionWriter,
    RecruitingStageDecision,
    ReviewService,
    close_invitation_review,
)
from interview_evidence.reporting.domain.review import HumanReview
from interview_evidence.reporting.repositories.postgres import ReportingRepository
from interview_evidence.shared.audit import AuditAppender
from interview_evidence.shared.idempotency import ResourceIdempotencyStore
from interview_evidence.shared.ids import Clock
from interview_evidence.shared.tenant import TenantContext


class FinalDecisionReportUnavailable(RuntimeError):
    pass


class FinalDecisionIdempotencyConflict(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FinalDecisionResult:
    review: HumanReview
    stage: RecruitingStageDecision
    invitation_state: str


class FinalDecisionService:
    """Own one atomic, idempotent human final-decision command."""

    def __init__(
        self,
        *,
        repository: ReportingRepository,
        invitations: InvitationDecisionWriter,
        audit: AuditAppender,
        clock: Clock,
        idempotency: ResourceIdempotencyStore,
    ) -> None:
        self._repository = repository
        self._invitations = invitations
        self._audit = audit
        self._clock = clock
        self._idempotency = idempotency
        self._reviews = ReviewService(repository)

    def record(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
        recruiting_stage_id: UUID,
        expected_pipeline_version: int,
        idempotency_key: str,
    ) -> FinalDecisionResult:
        report = self._repository.get_report_for_invitation(context, invitation_id)
        if report is None:
            raise FinalDecisionReportUnavailable
        operation = f"reporting.final_decision:{invitation_id}"
        existing_review_id = self._idempotency.get(
            context,
            operation=operation,
            idempotency_key=idempotency_key,
        )
        if existing_review_id is not None:
            return self._replay(
                context,
                invitation_id=invitation_id,
                recruiting_stage_id=recruiting_stage_id,
                expected_pipeline_version=expected_pipeline_version,
                review_id=existing_review_id,
            )

        occurred_at = self._clock.now()
        stage = self._invitations.move_to_recruiting_stage(
            context,
            invitation_id,
            recruiting_stage_id=recruiting_stage_id,
            expected_pipeline_version=expected_pipeline_version,
        )
        review = self._reviews.record_recruiting_stage_decision(
            context,
            report_id=report.report_id,
            invitation_id=invitation_id,
            recruiting_stage_id=stage.recruiting_stage_id,
            recruiting_stage_name=stage.recruiting_stage_name,
            expected_pipeline_version=expected_pipeline_version,
            occurred_at=occurred_at,
        )
        invitation_state = close_invitation_review(
            self._invitations,
            context,
            invitation_id=invitation_id,
            occurred_at=occurred_at,
        )
        self._audit.append(
            context,
            action="final_decision.create",
            resource_type="human_review",
            resource_id=review.human_review_id,
            result="created",
            metadata={
                "invitation_id": str(invitation_id),
                "recruiting_stage_id": str(stage.recruiting_stage_id),
                "recruiting_stage_name": stage.recruiting_stage_name,
                "invitation_state": invitation_state or "unchanged",
            },
        )
        self._idempotency.put(
            context,
            operation=operation,
            idempotency_key=idempotency_key,
            resource_id=review.human_review_id,
        )
        return FinalDecisionResult(
            review=review,
            stage=stage,
            invitation_state=invitation_state,
        )

    def _replay(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
        recruiting_stage_id: UUID,
        expected_pipeline_version: int,
        review_id: UUID,
    ) -> FinalDecisionResult:
        review = self._repository.get_review(context, review_id)
        stored_stage_id = review.value.get("recruiting_stage_id")
        stored_version = review.value.get("expected_pipeline_version")
        if (
            review.target_id != invitation_id
            or stored_stage_id != str(recruiting_stage_id)
            or stored_version != str(expected_pipeline_version)
        ):
            raise FinalDecisionIdempotencyConflict
        stage = self._invitations.get_recruiting_stage_decision(context, invitation_id)
        if stage.recruiting_stage_id != recruiting_stage_id:
            raise FinalDecisionIdempotencyConflict
        state = self._invitations.authorize_invitation(
            context,
            invitation_id,
            required_state=frozenset({"completed", "reviewed"}),
        ).state
        return FinalDecisionResult(review=review, stage=stage, invitation_state=state)
