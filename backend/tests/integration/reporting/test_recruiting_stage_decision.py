from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from fastapi.testclient import TestClient
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
        return StageDecision(
            invitation_id=invitation_id,
            position_id=POSITION_ID,
            recruiting_stage_id=recruiting_stage_id,
            recruiting_stage_name="최종 합격",
            pipeline_row_version=expected_pipeline_version + 1,
        )

    def authorize_invitation(
        self,
        _context: TenantContext,
        _invitation_id: UUID,
        *,
        required_state: str | frozenset[str],
    ) -> InvitationState:
        del required_state
        return InvitationState(state="completed", row_version=4)

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
        return InvitationState(state="reviewed", row_version=5)


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
