from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from interview_evidence.company_management.application.hiring_service import (
    ApplicantPipelineMove,
    HiringService,
)
from interview_evidence.company_management.domain.company import Position, PositionStatus
from interview_evidence.company_management.domain.hiring import Invitation, RecruitingStage
from interview_evidence.reporting.api import create_lane_d_runtime
from interview_evidence.reporting.application.deletion_service import DeletionService
from interview_evidence.reporting.application.public import ReportingPublic
from interview_evidence.reporting.application.review_service import ReviewService
from interview_evidence.reporting.domain.report import Report, ReportKind, ReportStatus
from interview_evidence.reporting.domain.review import Decision
from interview_evidence.reporting.repositories.postgres import (
    Base,
    SQLAlchemyReportingRepository,
)
from interview_evidence.shared.audit import InMemoryAuditAppender
from interview_evidence.shared.ids import CommandMeta, FrozenClock
from interview_evidence.shared.security.principals import CompanyPrincipal, FakePrincipalProvider
from interview_evidence.shared.submission_materials import DEFAULT_SUBMISSION_REQUIREMENTS
from interview_evidence.shared.tenant import ActorType, TenantContext
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

NOW = datetime(2026, 8, 24, 9, 0, tzinfo=UTC)
COMPANY_ID = UUID("00000000-0000-7000-8000-000000000001")
USER_ID = UUID("00000000-0000-7000-8000-000000000002")
REPORT_ID = UUID("00000000-0000-7000-8000-000000000003")
SESSION_ID = UUID("00000000-0000-7000-8000-000000000004")
INVITATION_ID = UUID("00000000-0000-7000-8000-000000000005")
POSITION_ID = UUID("00000000-0000-7000-8000-000000000006")
STAGE_ID = UUID("00000000-0000-7000-8000-000000000007")
REVIEW_STAGE_ID = UUID("00000000-0000-7000-8000-000000000008")
REJECT_STAGE_ID = UUID("00000000-0000-7000-8000-000000000009")


@dataclass(frozen=True)
class InvitationState:
    state: str
    row_version: int


@dataclass(frozen=True)
class StageDecision:
    invitation_id: UUID
    position_id: UUID
    recruiting_stage_id: UUID
    recruiting_stage_name: str
    pipeline_row_version: int


class DecisionWriter:
    def __init__(self, *, fail_move: bool = False) -> None:
        self.fail_move = fail_move
        self.moves: list[tuple[UUID, UUID, int]] = []
        self.advances: list[CommandMeta] = []
        self.state = "completed"
        self.current_stage: StageDecision | None = None

    def move_to_recruiting_stage(
        self,
        _context: TenantContext,
        invitation_id: UUID,
        *,
        recruiting_stage_id: UUID,
        expected_pipeline_version: int,
    ) -> StageDecision:
        if self.fail_move:
            raise ValueError("stale applicant pipeline version")
        self.moves.append((invitation_id, recruiting_stage_id, expected_pipeline_version))
        self.current_stage = StageDecision(
            invitation_id=invitation_id,
            position_id=POSITION_ID,
            recruiting_stage_id=recruiting_stage_id,
            recruiting_stage_name="최종 합격",
            pipeline_row_version=expected_pipeline_version + 1,
        )
        return self.current_stage

    def get_recruiting_stage_decision(
        self,
        _context: TenantContext,
        invitation_id: UUID,
    ) -> StageDecision:
        if self.current_stage is None or self.current_stage.invitation_id != invitation_id:
            raise LookupError("recruiting stage decision not found")
        return self.current_stage

    def authorize_invitation(
        self,
        _context: TenantContext,
        _invitation_id: UUID,
        *,
        required_state: str | frozenset[str],
    ) -> InvitationState:
        del required_state
        return InvitationState(state=self.state, row_version=4)

    def advance_invitation_state(
        self,
        _context: TenantContext,
        _invitation_id: UUID,
        *,
        from_state: str,
        to_state: str,
        meta: CommandMeta,
    ) -> InvitationState:
        assert from_state == "completed"
        assert to_state == "reviewed"
        self.advances.append(meta)
        self.state = "reviewed"
        return InvitationState(state="reviewed", row_version=5)


class BatchDecisionRepository:
    def __init__(self) -> None:
        self.position = Position(
            position_id=POSITION_ID,
            company_id=COMPANY_ID,
            title="백엔드 엔지니어",
            description="서비스 개발",
            created_by=USER_ID,
            status=PositionStatus.ACTIVE,
            created_at=NOW,
        )
        self.stages = (
            RecruitingStage(
                recruiting_stage_id=REVIEW_STAGE_ID,
                company_id=COMPANY_ID,
                position_id=POSITION_ID,
                name="검토",
                sort_order=0,
            ),
            RecruitingStage(
                recruiting_stage_id=STAGE_ID,
                company_id=COMPANY_ID,
                position_id=POSITION_ID,
                name="최종합격",
                sort_order=1,
            ),
            RecruitingStage(
                recruiting_stage_id=REJECT_STAGE_ID,
                company_id=COMPANY_ID,
                position_id=POSITION_ID,
                name="불합격",
                sort_order=2,
            ),
        )
        self.invitation = Invitation.create(
            invitation_id=INVITATION_ID,
            company_id=COMPANY_ID,
            position_id=POSITION_ID,
            competency_model_version_id=UUID("00000000-0000-7000-8000-000000000010"),
            applicant_id=UUID("00000000-0000-7000-8000-000000000011"),
            applicant_email="candidate@example.com",
            applicant_display_name="합성 지원자",
            submission_requirements=DEFAULT_SUBMISSION_REQUIREMENTS,
            token_hash="a" * 64,
            expires_at=datetime(2026, 9, 1, tzinfo=UTC),
            recruiting_stage_id=REVIEW_STAGE_ID,
        )

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

    def list_invitations(
        self,
        context: TenantContext,
        position_id: UUID,
    ) -> tuple[Invitation, ...]:
        context.assert_company(COMPANY_ID)
        return (self.invitation,) if position_id == POSITION_ID else ()

    def get_invitation_for_update(
        self,
        context: TenantContext,
        invitation_id: UUID,
    ) -> Invitation:
        context.assert_company(COMPANY_ID)
        if invitation_id != INVITATION_ID:
            raise LookupError("invitation not found")
        return self.invitation

    def save_invitation(
        self,
        context: TenantContext,
        invitation: Invitation,
    ) -> Invitation:
        context.assert_company(COMPANY_ID)
        self.invitation = invitation
        return invitation


class MissingReportResolver:
    def get_invitation_review(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
    ) -> None:
        context.assert_company(COMPANY_ID)
        assert invitation_id == INVITATION_ID
        return None


def context() -> TenantContext:
    return TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.COMPANY_USER,
        actor_id=USER_ID,
        request_id=UUID(int=0),
        trace_id="stage-decision-test",
    )


def client(
    writer: DecisionWriter,
    *,
    with_report: bool = True,
) -> tuple[TestClient, SQLAlchemyReportingRepository, Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    repository = SQLAlchemyReportingRepository(session)
    if with_report:
        repository.save_report(
            context(),
            Report(
                report_id=REPORT_ID,
                company_id=COMPANY_ID,
                interview_session_id=SESSION_ID,
                invitation_id=INVITATION_ID,
                version=1,
                kind=ReportKind.AI_ORIGINAL,
                model_version="model-v1",
                prompt_version="prompt-v1",
                config_version="config-v1",
                status=ReportStatus.READY,
                summary="evidence summary",
                created_at=NOW,
            ),
        )
    runtime = create_lane_d_runtime(
        principal_provider=FakePrincipalProvider(
            company_principals={
                "company-token": CompanyPrincipal(
                    company_id=COMPANY_ID,
                    company_user_id=USER_ID,
                    identity_subject="oidc|reviewer",
                )
            }
        ),
        repository=repository,
        audit=InMemoryAuditAppender(),
        clock=FrozenClock(NOW),
        invitations=writer,
    )
    return TestClient(runtime.app), repository, session


def test_final_decision_moves_pipeline_and_records_dynamic_stage_audit() -> None:
    writer = DecisionWriter()
    http, repository, session = client(writer)

    response = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers={"Authorization": "Bearer company-token", "Idempotency-Key": "decision-1"},
        json={"recruiting_stage_id": str(STAGE_ID), "expected_pipeline_version": 3},
    )

    assert response.status_code == 201
    assert response.json()["recruiting_stage_id"] == str(STAGE_ID)
    assert response.json()["pipeline_row_version"] == 4
    assert response.json()["invitation_state"] == "reviewed"
    assert writer.moves == [(INVITATION_ID, STAGE_ID, 3)]
    assert len(writer.advances) == 1
    review = repository.list_reviews(context(), REPORT_ID)[0]
    assert review.value == {
        "recruiting_stage_id": str(STAGE_ID),
        "recruiting_stage_name": "최종 합격",
        "expected_pipeline_version": "3",
    }
    assert review.reason is None
    projection = ReportingPublic(
        repository=repository,
        deletion_service=DeletionService(repository),
    ).get_review_projection(context(), invitation_id=INVITATION_ID)
    assert projection is not None
    assert projection.human_decision_status == "최종 합격"
    session.close()


def test_stale_pipeline_version_records_no_final_decision() -> None:
    http, repository, session = client(DecisionWriter(fail_move=True))

    response = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers={"Authorization": "Bearer company-token", "Idempotency-Key": "decision-2"},
        json={"recruiting_stage_id": str(STAGE_ID), "expected_pipeline_version": 99},
    )

    assert response.status_code == 409
    assert repository.list_reviews(context(), REPORT_ID) == ()
    session.close()


def test_final_decision_without_report_returns_stable_reason_and_writes_nothing() -> None:
    writer = DecisionWriter()
    http, _repository, session = client(writer, with_report=False)

    response = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers={"Authorization": "Bearer company-token", "Idempotency-Key": "decision-3"},
        json={"recruiting_stage_id": str(STAGE_ID), "expected_pipeline_version": 3},
    )

    assert response.status_code == 409
    assert response.json() == {
        "code": "REPORT_NOT_AVAILABLE",
        "detail": "Final report is not available.",
    }
    assert writer.moves == []
    assert writer.advances == []
    session.close()


@pytest.mark.parametrize("target_stage_id", [STAGE_ID, REJECT_STAGE_ID])
def test_batch_final_stage_without_report_is_rejected_without_pipeline_write(
    target_stage_id: UUID,
) -> None:
    repository = BatchDecisionRepository()
    service = HiringService(
        repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        FrozenClock(NOW),
        object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="final report is not available"):
        service.move_applicants(
            context(),
            position_id=POSITION_ID,
            target_stage_id=target_stage_id,
            moves=(ApplicantPipelineMove(INVITATION_ID, 1),),
            invitation_reviews=MissingReportResolver(),  # type: ignore[arg-type]
            require_final_report=True,
        )

    assert repository.invitation.recruiting_stage_id == REVIEW_STAGE_ID
    assert repository.invitation.pipeline_row_version == 1


def test_legacy_fixed_decision_remains_readable_during_migration() -> None:
    _http, repository, session = client(DecisionWriter())
    ReviewService(repository).record_final_decision(
        context(),
        report_id=REPORT_ID,
        invitation_id=INVITATION_ID,
        decision=Decision.HOLD,
        reason="legacy decision reason",
        occurred_at=NOW,
    )

    projection = ReportingPublic(
        repository=repository,
        deletion_service=DeletionService(repository),
    ).get_review_projection(context(), invitation_id=INVITATION_ID)

    assert projection is not None
    assert projection.human_decision_status == "hold"
    session.close()
