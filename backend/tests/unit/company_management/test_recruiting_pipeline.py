from datetime import UTC, datetime
from uuid import UUID

import pytest
from interview_evidence.company_management.application.hiring_service import (
    ApplicantPipelineMove,
    HiringService,
)
from interview_evidence.company_management.domain.company import Position, PositionStatus
from interview_evidence.company_management.domain.hiring import (
    DEFAULT_RECRUITING_STAGE_NAMES,
    Invitation,
    InvitationStateError,
    InvitationStatus,
    RecruitingStage,
)
from interview_evidence.shared.ids import FrozenClock
from interview_evidence.shared.submission_materials import DEFAULT_SUBMISSION_REQUIREMENTS
from interview_evidence.shared.tenant import ActorType, TenantContext

COMPANY_ID = UUID("00000000-0000-7000-8000-000000000201")
POSITION_ID = UUID("00000000-0000-7000-8000-000000000202")
FIRST_STAGE_ID = UUID("00000000-0000-7000-8000-000000000203")
NEXT_STAGE_ID = UUID("00000000-0000-7000-8000-000000000204")
NOW = datetime(2026, 8, 24, 3, 0, tzinfo=UTC)


class PipelineRepository:
    def __init__(self, current_invitation: Invitation) -> None:
        self.position = Position(
            position_id=POSITION_ID,
            company_id=COMPANY_ID,
            title="백엔드 엔지니어",
            description="서비스 개발",
            created_by=UUID("00000000-0000-7000-8000-000000000208"),
            status=PositionStatus.ACTIVE,
            created_at=NOW,
        )
        self.stages: list[RecruitingStage] = []
        self.invitations = [current_invitation]

    def get_position(self, context: TenantContext, position_id: UUID) -> Position:
        context.assert_company(COMPANY_ID)
        if position_id != POSITION_ID:
            raise LookupError("position not found")
        return self.position

    def list_recruiting_stages(
        self,
        context: TenantContext,
        position_id: UUID | None = None,
    ) -> tuple[RecruitingStage, ...]:
        context.assert_company(COMPANY_ID)
        return tuple(
            stage
            for stage in self.stages
            if position_id is None or stage.position_id == position_id
        )

    def save_recruiting_stage(
        self,
        context: TenantContext,
        stage: RecruitingStage,
    ) -> RecruitingStage:
        context.assert_company(stage.company_id)
        self.stages = [
            candidate
            for candidate in self.stages
            if candidate.recruiting_stage_id != stage.recruiting_stage_id
        ]
        self.stages.append(stage)
        return stage

    def list_invitations(
        self,
        context: TenantContext,
        position_id: UUID,
    ) -> tuple[Invitation, ...]:
        context.assert_company(COMPANY_ID)
        return tuple(item for item in self.invitations if item.position_id == position_id)

    def get_invitation_for_update(
        self,
        context: TenantContext,
        invitation_id: UUID,
    ) -> Invitation:
        context.assert_company(COMPANY_ID)
        return next(item for item in self.invitations if item.invitation_id == invitation_id)

    def get_invitation(
        self,
        context: TenantContext,
        invitation_id: UUID,
    ) -> Invitation:
        return self.get_invitation_for_update(context, invitation_id)

    def save_invitation(
        self,
        context: TenantContext,
        updated: Invitation,
    ) -> Invitation:
        context.assert_company(updated.company_id)
        self.invitations = [
            updated if item.invitation_id == updated.invitation_id else item
            for item in self.invitations
        ]
        return updated


class ReadyReportResolver:
    class Snapshot:
        report_status = "ready"

    def get_invitation_review(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
    ) -> Snapshot:
        context.assert_company(COMPANY_ID)
        assert invitation_id == UUID("00000000-0000-7000-8000-000000000205")
        return self.Snapshot()


def context() -> TenantContext:
    return TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.COMPANY_USER,
        actor_id=UUID("00000000-0000-7000-8000-000000000208"),
        request_id=UUID("00000000-0000-7000-8000-000000000209"),
        trace_id="recruiting-pipeline-test",
    )


def invitation() -> Invitation:
    return Invitation.create(
        invitation_id=UUID("00000000-0000-7000-8000-000000000205"),
        company_id=COMPANY_ID,
        position_id=POSITION_ID,
        competency_model_version_id=UUID("00000000-0000-7000-8000-000000000206"),
        applicant_id=UUID("00000000-0000-7000-8000-000000000207"),
        applicant_email="candidate@example.com",
        applicant_display_name="지원자",
        submission_requirements=DEFAULT_SUBMISSION_REQUIREMENTS,
        token_hash="a" * 64,
        expires_at=datetime(2026, 9, 1, tzinfo=UTC),
        recruiting_stage_id=FIRST_STAGE_ID,
    )


def test_default_recruiting_stages_match_the_product_order() -> None:
    assert DEFAULT_RECRUITING_STAGE_NAMES == (
        "보류",
        "검토",
        "1차 합격",
        "최종합격",
        "불합격",
    )


def test_pipeline_move_changes_only_latest_stage_and_pipeline_version() -> None:
    current = invitation()

    moved = current.move_to_recruiting_stage(NEXT_STAGE_ID, expected_version=1)

    assert moved.recruiting_stage_id == NEXT_STAGE_ID
    assert moved.pipeline_row_version == 2
    assert moved.status == current.status
    assert moved.row_version == current.row_version
    assert current.recruiting_stage_id == FIRST_STAGE_ID


def test_pipeline_move_rejects_a_stale_version() -> None:
    with pytest.raises(InvitationStateError, match="stale applicant pipeline"):
        invitation().move_to_recruiting_stage(NEXT_STAGE_ID, expected_version=9)


def test_stage_names_are_normalized_and_edits_are_versioned() -> None:
    stage = RecruitingStage(
        recruiting_stage_id=FIRST_STAGE_ID,
        company_id=COMPANY_ID,
        position_id=POSITION_ID,
        name="  1차   실무 면접  ",
        sort_order=1,
    )

    renamed = stage.rename("  2차   문화 면접 ", expected_version=1)
    reordered = renamed.reorder(0)

    assert stage.name == "1차 실무 면접"
    assert renamed.name == "2차 문화 면접"
    assert renamed.row_version == 2
    assert reordered.sort_order == 0
    assert reordered.row_version == 3


def test_service_backfills_from_system_progress_and_moves_the_latest_state() -> None:
    current = invitation().model_copy(
        update={
            "status": InvitationStatus.COMPLETED,
            "recruiting_stage_id": None,
        }
    )
    repository = PipelineRepository(current)
    service = HiringService(
        repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        FrozenClock(NOW),
        object(),  # type: ignore[arg-type]
    )

    stages = service.ensure_default_stages(context(), POSITION_ID)
    first_pass = next(stage for stage in stages if stage.name == "1차 합격")
    final_pass = next(stage for stage in stages if stage.name == "최종합격")
    assert repository.invitations[0].recruiting_stage_id == first_pass.recruiting_stage_id

    moved = service.move_applicants(
        context(),
        position_id=POSITION_ID,
        target_stage_id=final_pass.recruiting_stage_id,
        moves=(
            ApplicantPipelineMove(
                invitation_id=current.invitation_id,
                expected_version=2,
            ),
        ),
        invitation_reviews=ReadyReportResolver(),
    )

    assert moved[0].recruiting_stage_id == final_pass.recruiting_stage_id
    assert moved[0].pipeline_row_version == 3
    assert moved[0].status.value == "completed"


def test_recruiting_state_uses_the_invitation_assignment_as_the_single_source() -> None:
    current = invitation().model_copy(
        update={
            "status": InvitationStatus.COMPLETED,
            "recruiting_stage_id": None,
        }
    )
    repository = PipelineRepository(current)
    service = HiringService(
        repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        FrozenClock(NOW),
        object(),  # type: ignore[arg-type]
    )

    state = service.get_applicant_recruiting_state(context(), current.invitation_id)

    assigned = next(
        stage
        for stage in state.stages
        if stage.recruiting_stage_id == state.invitation.recruiting_stage_id
    )
    assert assigned.name == "1차 합격"
    assert state.invitation.pipeline_row_version == 2
