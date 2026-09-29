from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from interview_evidence.company_management.adapters.applicant_session import (
    ApplicantSessionAdapter,
    IssuedInvitationToken,
)
from interview_evidence.company_management.domain.hiring import (
    DEFAULT_RECRUITING_STAGE_NAMES,
    MAX_RECRUITING_STAGES,
    Invitation,
    InvitationStatus,
    RecruitingStage,
)
from interview_evidence.company_management.repositories.postgres import CompanyRepository
from interview_evidence.shared.idempotency import ResourceIdempotencyStore
from interview_evidence.shared.ids import Clock, new_uuid7
from interview_evidence.shared.tenant import TenantContext


@dataclass(frozen=True, slots=True)
class ApplicantInvitationInput:
    email: str
    display_name: str


@dataclass(frozen=True, slots=True)
class InvitationIssuance:
    invitation: Invitation
    token: IssuedInvitationToken


@dataclass(frozen=True, slots=True)
class ApplicantPipelineMove:
    invitation_id: UUID
    expected_version: int


@dataclass(frozen=True, slots=True)
class ApplicantRecruitingState:
    """The single live hiring-stage projection shared by every company view."""

    invitation: Invitation
    stages: tuple[RecruitingStage, ...]


class InvitationReviewSnapshot(Protocol):
    @property
    def report_status(self) -> str: ...


class InvitationReviewResolver(Protocol):
    def get_invitation_review(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
    ) -> InvitationReviewSnapshot | None: ...


class HiringService:
    def __init__(
        self,
        repository: CompanyRepository,
        sessions: ApplicantSessionAdapter,
        clock: Clock,
        idempotency: ResourceIdempotencyStore,
    ) -> None:
        self._repository = repository
        self._sessions = sessions
        self._clock = clock
        self._idempotency = idempotency

    def issue_invitations(
        self,
        context: TenantContext,
        *,
        position_id: UUID,
        applicants: tuple[ApplicantInvitationInput, ...],
        expires_at: datetime,
    ) -> tuple[InvitationIssuance, ...]:
        position = self._repository.get_position(context, position_id)
        if not position.accepts_new_applications_on(self._clock.now().date()):
            raise ValueError("position is not accepting new applications")
        stages = self.ensure_default_stages(context, position_id)
        initial_stage = next(stage for stage in stages if stage.name == "검토")
        versions = self._repository.list_criterion_versions(context, position_id)
        published = [version for version in versions if version.status == "published"]
        if not published:
            raise ValueError("invitations require a published competency model")
        version = max(published, key=lambda item: item.version_number)
        issuances: list[InvitationIssuance] = []
        for applicant in applicants:
            invitation_id = new_uuid7(self._clock.now())
            applicant_id = new_uuid7(self._clock.now())
            token = self._sessions.issue_token(
                invitation_id=invitation_id,
                company_id=context.company_id,
                applicant_id=applicant_id,
                expires_at=expires_at,
            )
            invitation = Invitation.create(
                invitation_id=invitation_id,
                company_id=context.company_id,
                position_id=position_id,
                competency_model_version_id=version.competency_model_version_id,
                applicant_id=applicant_id,
                applicant_email=applicant.email,
                applicant_display_name=applicant.display_name,
                submission_requirements=position.submission_requirements,
                token_hash=token.token_hash,
                expires_at=expires_at,
                recruiting_stage_id=initial_stage.recruiting_stage_id,
            )
            self._repository.save_invitation(context, invitation)
            issuances.append(InvitationIssuance(invitation=invitation, token=token))
        return tuple(issuances)

    def list_invitations(self, context: TenantContext, position_id: UUID) -> tuple[Invitation, ...]:
        self.ensure_default_stages(context, position_id)
        return self._repository.list_invitations(context, position_id)

    def ensure_default_stages(
        self,
        context: TenantContext,
        position_id: UUID,
    ) -> tuple[RecruitingStage, ...]:
        self._repository.get_position(context, position_id)
        stages = self._repository.list_recruiting_stages(context, position_id)
        if not stages:
            stages = tuple(
                self._repository.save_recruiting_stage(
                    context,
                    RecruitingStage(
                        recruiting_stage_id=new_uuid7(self._clock.now()),
                        company_id=context.company_id,
                        position_id=position_id,
                        name=name,
                        sort_order=sort_order,
                    ),
                )
                for sort_order, name in enumerate(DEFAULT_RECRUITING_STAGE_NAMES)
            )
        by_name = {stage.name: stage for stage in stages}
        for invitation in self._repository.list_invitations(context, position_id):
            if invitation.recruiting_stage_id is not None:
                continue
            stage = by_name[_default_stage_name(invitation.status)]
            self._repository.save_invitation(
                context,
                invitation.move_to_recruiting_stage(
                    stage.recruiting_stage_id,
                    expected_version=invitation.pipeline_row_version,
                ),
            )
        return stages

    def list_recruiting_stages(
        self,
        context: TenantContext,
        position_id: UUID | None = None,
    ) -> tuple[RecruitingStage, ...]:
        if position_id is not None:
            return self.ensure_default_stages(context, position_id)
        for position in self._repository.list_positions(context):
            self.ensure_default_stages(context, position.position_id)
        return self._repository.list_recruiting_stages(context)

    def get_applicant_recruiting_state(
        self,
        context: TenantContext,
        invitation_id: UUID,
    ) -> ApplicantRecruitingState:
        """Return the invitation's current pipeline assignment and its position stages.

        The assignment on ``Invitation`` is authoritative. Reports and kanban views must read
        this projection instead of persisting their own copy of a stage name or decision.
        ``ensure_default_stages`` also backfills legacy invitations that predate the configurable
        pipeline, so the returned assignment is always usable by the company console.
        """
        invitation = self._repository.get_invitation(context, invitation_id)
        stages = self.ensure_default_stages(context, invitation.position_id)
        invitation = self._repository.get_invitation(context, invitation_id)
        if invitation.recruiting_stage_id is None:
            raise RuntimeError("applicant recruiting stage was not initialized")
        return ApplicantRecruitingState(invitation=invitation, stages=stages)

    def create_recruiting_stage(
        self,
        context: TenantContext,
        *,
        position_id: UUID,
        name: str,
    ) -> RecruitingStage:
        stages = self.ensure_default_stages(context, position_id)
        normalized = " ".join(name.split())
        if len(stages) >= MAX_RECRUITING_STAGES:
            raise ValueError("a position can have at most 20 recruiting stages")
        if normalized.casefold() in {stage.name.casefold() for stage in stages}:
            raise ValueError("recruiting stage names must be unique within a position")
        return self._repository.save_recruiting_stage(
            context,
            RecruitingStage(
                recruiting_stage_id=new_uuid7(self._clock.now()),
                company_id=context.company_id,
                position_id=position_id,
                name=normalized,
                sort_order=len(stages),
            ),
        )

    def rename_recruiting_stage(
        self,
        context: TenantContext,
        *,
        position_id: UUID,
        stage_id: UUID,
        name: str,
        expected_version: int,
    ) -> RecruitingStage:
        stages = self.ensure_default_stages(context, position_id)
        current = _stage_in_position(stages, stage_id, position_id)
        normalized = " ".join(name.split())
        if normalized.casefold() in {
            stage.name.casefold() for stage in stages if stage.recruiting_stage_id != stage_id
        }:
            raise ValueError("recruiting stage names must be unique within a position")
        updated = current.rename(normalized, expected_version=expected_version)
        return self._repository.save_recruiting_stage(context, updated)

    def reorder_recruiting_stages(
        self,
        context: TenantContext,
        *,
        position_id: UUID,
        ordered_stage_ids: tuple[UUID, ...],
    ) -> tuple[RecruitingStage, ...]:
        stages = self.ensure_default_stages(context, position_id)
        if len(set(ordered_stage_ids)) != len(ordered_stage_ids) or set(ordered_stage_ids) != {
            stage.recruiting_stage_id for stage in stages
        }:
            raise ValueError("ordered_stage_ids must contain every stage exactly once")
        by_id = {stage.recruiting_stage_id: stage for stage in stages}
        return tuple(
            self._repository.save_recruiting_stage(
                context,
                by_id[stage_id].reorder(sort_order),
            )
            for sort_order, stage_id in enumerate(ordered_stage_ids)
        )

    def delete_recruiting_stage(
        self,
        context: TenantContext,
        *,
        position_id: UUID,
        stage_id: UUID,
        replacement_stage_id: UUID,
    ) -> tuple[RecruitingStage, ...]:
        stages = self.ensure_default_stages(context, position_id)
        if len(stages) <= 1:
            raise ValueError("a position must keep at least one recruiting stage")
        _stage_in_position(stages, stage_id, position_id)
        _stage_in_position(stages, replacement_stage_id, position_id)
        if stage_id == replacement_stage_id:
            raise ValueError("replacement stage must differ from the deleted stage")
        moves = tuple(
            ApplicantPipelineMove(
                invitation_id=invitation.invitation_id,
                expected_version=invitation.pipeline_row_version,
            )
            for invitation in self._repository.list_invitations(context, position_id)
            if invitation.recruiting_stage_id == stage_id
        )
        if moves:
            self.move_applicants(
                context,
                position_id=position_id,
                target_stage_id=replacement_stage_id,
                moves=moves,
            )
        self._repository.delete_recruiting_stage(context, stage_id)
        remaining = tuple(stage for stage in stages if stage.recruiting_stage_id != stage_id)
        return tuple(
            self._repository.save_recruiting_stage(
                context,
                stage.reorder(sort_order),
            )
            for sort_order, stage in enumerate(remaining)
        )

    def move_applicants(
        self,
        context: TenantContext,
        *,
        position_id: UUID,
        target_stage_id: UUID,
        moves: tuple[ApplicantPipelineMove, ...],
        invitation_reviews: InvitationReviewResolver | None = None,
        require_final_report: bool = False,
    ) -> tuple[Invitation, ...]:
        if not moves or len(moves) > 1000:
            raise ValueError("between 1 and 1000 applicants must be moved")
        if len({move.invitation_id for move in moves}) != len(moves):
            raise ValueError("each applicant may only appear once")
        stages = self.ensure_default_stages(context, position_id)
        target_stage = _stage_in_position(stages, target_stage_id, position_id)
        final_stage = target_stage.name in {"최종합격", "불합격"}
        pending: list[Invitation] = []
        for move in moves:
            invitation = self._repository.get_invitation_for_update(
                context,
                move.invitation_id,
            )
            if invitation.position_id != position_id:
                raise ValueError("applicant does not belong to the selected position")
            if final_stage and require_final_report:
                review = (
                    invitation_reviews.get_invitation_review(
                        context,
                        invitation_id=invitation.invitation_id,
                    )
                    if invitation_reviews is not None
                    else None
                )
                if review is None or review.report_status != "ready":
                    raise ValueError("final report is not available")
            moved = invitation.move_to_recruiting_stage(
                target_stage_id,
                expected_version=move.expected_version,
            )
            pending.append(moved)
        return tuple(self._repository.save_invitation(context, moved) for moved in pending)


def _default_stage_name(status: InvitationStatus) -> str:
    if status in {
        InvitationStatus.INTERRUPTED,
        InvitationStatus.EXPIRED,
        InvitationStatus.REVOKED,
    }:
        return "보류"
    if status is InvitationStatus.COMPLETED:
        return "1차 합격"
    if status in {InvitationStatus.REVIEWED, InvitationStatus.DELETED}:
        return "최종합격"
    return "검토"


def _stage_in_position(
    stages: tuple[RecruitingStage, ...],
    stage_id: UUID,
    position_id: UUID,
) -> RecruitingStage:
    stage = next(
        (candidate for candidate in stages if candidate.recruiting_stage_id == stage_id),
        None,
    )
    if stage is None or stage.position_id != position_id:
        raise ValueError("recruiting stage does not belong to the selected position")
    return stage
