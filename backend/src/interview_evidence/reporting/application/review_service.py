from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from interview_evidence.reporting.domain.report import (
    AssessmentState,
    RequirementAssessmentStatus,
)
from interview_evidence.reporting.domain.review import (
    Decision,
    HumanReview,
    ReviewType,
)
from interview_evidence.reporting.repositories.postgres import ReportingRepository
from interview_evidence.shared.ids import CommandMeta, new_uuid7
from interview_evidence.shared.tenant import ActorType, TenantContext


class InvitationReviewState(Protocol):
    # Read-only: the hiring lane returns frozen models, which a mutable attribute would reject.
    @property
    def state(self) -> str: ...

    @property
    def row_version(self) -> int: ...


class InvitationStateAdvancer(Protocol):
    """The part of hiring that review needs: read an invitation's state and move it on.

    Declared here as a structural type, the way `QuestionRationaleProvider` is, so reporting
    states what it needs without importing the hiring lane.
    """

    def authorize_invitation(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        required_state: str | frozenset[str],
    ) -> InvitationReviewState: ...

    def advance_invitation_state(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        from_state: str,
        to_state: str,
        meta: CommandMeta,
    ) -> InvitationReviewState: ...


class RecruitingStageDecision(Protocol):
    @property
    def invitation_id(self) -> UUID: ...

    @property
    def position_id(self) -> UUID: ...

    @property
    def recruiting_stage_id(self) -> UUID: ...

    @property
    def recruiting_stage_name(self) -> str: ...

    @property
    def pipeline_row_version(self) -> int: ...


class InvitationDecisionWriter(InvitationStateAdvancer, Protocol):
    """Hiring-side write boundary used by the final-decision transaction."""

    def move_to_recruiting_stage(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        recruiting_stage_id: UUID,
        expected_pipeline_version: int,
    ) -> RecruitingStageDecision: ...

    def get_recruiting_stage_decision(
        self,
        context: TenantContext,
        invitation_id: UUID,
    ) -> RecruitingStageDecision: ...


def close_invitation_review(
    invitations: InvitationStateAdvancer,
    context: TenantContext,
    *,
    invitation_id: UUID,
    occurred_at: datetime,
) -> str:
    """Mark an invitation reviewed once a human has recorded the final decision.

    Recording the decision used to write a `HumanReview` row and stop there, so the applicant
    stayed at "검토 대기" in the console no matter how many decisions were made — the counter for
    "검토 완료" reads `status == "reviewed"`, and nothing in the codebase produced that state.

    Returns the state the invitation ended in, so the caller can report what happened. Errors
    are intentionally propagated so the HTTP transaction can roll the stage, audit review and
    invitation-state writes back together.
    """
    authorization = invitations.authorize_invitation(
        context,
        invitation_id,
        required_state=frozenset({"completed", "reviewed"}),
    )
    if authorization.state != "completed":
        return authorization.state
    return invitations.advance_invitation_state(
        context,
        invitation_id,
        from_state="completed",
        to_state="reviewed",
        meta=CommandMeta(
            idempotency_key=f"invitation-reviewed-{invitation_id}",
            expected_version=authorization.row_version,
            occurred_at=occurred_at,
        ),
    ).state


class ReviewService:
    def __init__(self, repository: ReportingRepository) -> None:
        self._repository = repository

    def override_assessment(
        self,
        context: TenantContext,
        *,
        report_id: UUID,
        report_item_id: UUID,
        assessment_state: str,
        reason: str,
        occurred_at: datetime,
    ) -> HumanReview:
        if context.actor_type is not ActorType.COMPANY_USER:
            raise PermissionError("assessment review requires a company user")
        self._repository.get_report(context, report_id)
        validated_state = AssessmentState(assessment_state)
        return self._repository.save_review(
            context,
            HumanReview.assessment_override(
                human_review_id=new_uuid7(occurred_at),
                company_id=context.company_id,
                report_id=report_id,
                company_user_id=context.actor_id,
                report_item_id=report_item_id,
                assessment_state=validated_state.value,
                reason=reason,
                created_at=occurred_at,
            ),
        )

    def create_artifact(
        self,
        context: TenantContext,
        *,
        report_id: UUID,
        target_id: UUID,
        review_type: ReviewType,
        value: str,
        occurred_at: datetime,
    ) -> HumanReview:
        if context.actor_type is not ActorType.COMPANY_USER:
            raise PermissionError("review artifact requires a company user")
        if review_type is not ReviewType.NOTE:
            raise ValueError("unsupported review artifact")
        review = HumanReview(
            human_review_id=new_uuid7(occurred_at),
            company_id=context.company_id,
            report_id=report_id,
            company_user_id=context.actor_id,
            review_type=review_type,
            target_id=target_id,
            value={"text": value},
            reason=None,
            created_at=occurred_at,
        )
        return self._repository.save_review(context, review)

    def override_requirement(
        self,
        context: TenantContext,
        *,
        report_id: UUID,
        requirement_assessment_id: UUID,
        requirement_status: str,
        reason: str,
        occurred_at: datetime,
    ) -> HumanReview:
        if context.actor_type is not ActorType.COMPANY_USER:
            raise PermissionError("requirement review requires a company user")
        report = self._repository.get_report(context, report_id)
        if requirement_assessment_id not in {
            item.requirement_assessment_id for item in report.requirement_assessments
        }:
            raise LookupError("requirement assessment not found")
        validated_status = RequirementAssessmentStatus(requirement_status)
        return self._repository.save_review(
            context,
            HumanReview.requirement_override(
                human_review_id=new_uuid7(occurred_at),
                company_id=context.company_id,
                report_id=report_id,
                company_user_id=context.actor_id,
                requirement_assessment_id=requirement_assessment_id,
                requirement_status=validated_status.value,
                reason=reason,
                created_at=occurred_at,
            ),
        )

    def record_final_decision(
        self,
        context: TenantContext,
        *,
        report_id: UUID,
        invitation_id: UUID,
        decision: Decision,
        reason: str,
        occurred_at: datetime,
    ) -> HumanReview:
        self._repository.get_report(context, report_id)
        review = HumanReview.final_decision(
            human_review_id=new_uuid7(occurred_at),
            company_id=context.company_id,
            report_id=report_id,
            company_user_id=context.actor_id,
            invitation_id=invitation_id,
            actor_type=context.actor_type,
            decision=decision,
            reason=reason,
            created_at=occurred_at,
        )
        return self._repository.save_review(context, review)

    def record_recruiting_stage_decision(
        self,
        context: TenantContext,
        *,
        report_id: UUID,
        invitation_id: UUID,
        recruiting_stage_id: UUID,
        recruiting_stage_name: str,
        expected_pipeline_version: int,
        occurred_at: datetime,
    ) -> HumanReview:
        self._repository.get_report(context, report_id)
        review = HumanReview.recruiting_stage_decision(
            human_review_id=new_uuid7(occurred_at),
            company_id=context.company_id,
            report_id=report_id,
            company_user_id=context.actor_id,
            invitation_id=invitation_id,
            actor_type=context.actor_type,
            recruiting_stage_id=recruiting_stage_id,
            recruiting_stage_name=recruiting_stage_name,
            expected_pipeline_version=expected_pipeline_version,
            created_at=occurred_at,
        )
        return self._repository.save_review(context, review)
