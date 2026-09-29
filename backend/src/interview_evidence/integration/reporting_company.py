from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from interview_evidence.company_management.application.company_service import (
    CompanyManagementPublic,
    InvitationAuthorization,
    InvitationStateSnapshot,
)
from interview_evidence.company_management.application.hiring_service import (
    ApplicantPipelineMove,
    HiringService,
)
from interview_evidence.reporting.application.public import ReportingPublic
from interview_evidence.shared.ids import CommandMeta
from interview_evidence.shared.tenant import TenantContext


@dataclass(frozen=True, slots=True)
class CompanyReviewProjection:
    invitation_id: UUID
    interview_session_id: UUID
    report_id: UUID | None
    report_status: str
    human_decision_status: str | None
    #: The weighted score and how much of the configuration it covers, so Lane A's invitation
    #: list can rank on it without reading Lane D's domain. The counts travel with the score
    #: because a ranked column of numbers taken over different subsets of the criteria is a
    #: comparison that does not hold, and the screen has to be able to say so.
    overall_score: int | None = None
    scored_criteria_count: int = 0
    total_criteria_count: int = 0


@dataclass(frozen=True, slots=True)
class CompanyDeletionProjection:
    deletion_request_id: UUID
    manifest_id: UUID
    status: str
    expected_targets: int
    verified_targets: int


@dataclass(frozen=True, slots=True)
class RecruitingStageDecisionProjection:
    invitation_id: UUID
    position_id: UUID
    recruiting_stage_id: UUID
    recruiting_stage_name: str
    pipeline_row_version: int


class ReportingHiringBoundary:
    """Adapt hiring writes needed by reporting without importing Lane A into Lane D."""

    def __init__(
        self,
        company: CompanyManagementPublic,
        hiring: HiringService,
    ) -> None:
        self._company = company
        self._hiring = hiring

    def authorize_invitation(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        required_state: str | frozenset[str],
    ) -> InvitationAuthorization:
        return self._company.authorize_invitation(
            context,
            invitation_id,
            required_state=required_state,
        )

    def advance_invitation_state(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        from_state: str,
        to_state: str,
        meta: CommandMeta,
    ) -> InvitationStateSnapshot:
        return self._company.advance_invitation_state(
            context,
            invitation_id,
            from_state=from_state,
            to_state=to_state,
            meta=meta,
        )

    def move_to_recruiting_stage(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        recruiting_stage_id: UUID,
        expected_pipeline_version: int,
    ) -> RecruitingStageDecisionProjection:
        current = self._hiring.get_applicant_recruiting_state(context, invitation_id)
        moved = self._hiring.move_applicants(
            context,
            position_id=current.invitation.position_id,
            target_stage_id=recruiting_stage_id,
            moves=(
                ApplicantPipelineMove(
                    invitation_id=invitation_id,
                    expected_version=expected_pipeline_version,
                ),
            ),
        )[0]
        stage = next(
            candidate
            for candidate in current.stages
            if candidate.recruiting_stage_id == recruiting_stage_id
        )
        return RecruitingStageDecisionProjection(
            invitation_id=moved.invitation_id,
            position_id=moved.position_id,
            recruiting_stage_id=recruiting_stage_id,
            recruiting_stage_name=stage.name,
            pipeline_row_version=moved.pipeline_row_version,
        )

    def get_recruiting_stage_decision(
        self,
        context: TenantContext,
        invitation_id: UUID,
    ) -> RecruitingStageDecisionProjection:
        current = self._hiring.get_applicant_recruiting_state(context, invitation_id)
        stage = next(
            candidate
            for candidate in current.stages
            if candidate.recruiting_stage_id == current.invitation.recruiting_stage_id
        )
        return RecruitingStageDecisionProjection(
            invitation_id=current.invitation.invitation_id,
            position_id=current.invitation.position_id,
            recruiting_stage_id=stage.recruiting_stage_id,
            recruiting_stage_name=stage.name,
            pipeline_row_version=current.invitation.pipeline_row_version,
        )


class ReportingCompanyBoundary:
    """Adapt Lane D's public projections for company-facing read models."""

    def __init__(self, reporting: ReportingPublic) -> None:
        self._reporting = reporting

    def get_invitation_review(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
    ) -> CompanyReviewProjection | None:
        projection = self._reporting.get_review_projection(
            context,
            invitation_id=invitation_id,
        )
        if projection is None:
            return None
        return CompanyReviewProjection(
            invitation_id=projection.invitation_id,
            interview_session_id=projection.interview_session_id,
            report_id=projection.report_id,
            report_status=projection.report_status,
            human_decision_status=projection.human_decision_status,
            overall_score=projection.overall_score,
            scored_criteria_count=projection.scored_criteria_count,
            total_criteria_count=projection.total_criteria_count,
        )

    def get_deletion_progress(
        self,
        context: TenantContext,
        *,
        deletion_request_id: UUID,
    ) -> CompanyDeletionProjection:
        manifest = self._reporting.get_deletion_status(
            context,
            deletion_request_id=deletion_request_id,
        )
        return CompanyDeletionProjection(
            deletion_request_id=manifest.deletion_request_id,
            manifest_id=manifest.manifest_id,
            status=manifest.status.value,
            expected_targets=len(manifest.targets),
            verified_targets=manifest.verified_targets,
        )
