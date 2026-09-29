from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi.testclient import TestClient
from interview_evidence.company_management.application.company_service import (
    CompanyManagementPublic,
)
from interview_evidence.company_management.application.hiring_service import (
    ApplicantPipelineMove,
    HiringService,
)
from interview_evidence.company_management.repositories.postgres import (
    Base as CompanyBase,
)
from interview_evidence.company_management.repositories.postgres import (
    InvitationRow,
    PositionRow,
    RecruitingStageRow,
    SqlAlchemyCompanyRepository,
)
from interview_evidence.integration.reporting_company import ReportingHiringBoundary
from interview_evidence.reporting.api import create_lane_d_runtime
from interview_evidence.reporting.domain.report import Report, ReportKind, ReportStatus
from interview_evidence.reporting.repositories.postgres import (
    Base as ReportingBase,
)
from interview_evidence.reporting.repositories.postgres import (
    SQLAlchemyReportingRepository,
)
from interview_evidence.shared.audit import InMemoryAuditAppender
from interview_evidence.shared.ids import CommandMeta, FrozenClock
from interview_evidence.shared.security.principals import CompanyPrincipal, FakePrincipalProvider
from interview_evidence.shared.submission_materials import (
    DEFAULT_SUBMISSION_REQUIREMENTS,
    submission_requirements_to_json,
)
from interview_evidence.shared.tenant import ActorType, TenantContext
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
COMPANY_ID = UUID("00000000-0000-7000-8000-000000000101")
USER_ID = UUID("00000000-0000-7000-8000-000000000102")
POSITION_ID = UUID("00000000-0000-7000-8000-000000000103")
VERSION_ID = UUID("00000000-0000-7000-8000-000000000104")
APPLICANT_ID = UUID("00000000-0000-7000-8000-000000000105")
INVITATION_ID = UUID("00000000-0000-7000-8000-000000000106")
SESSION_ID = UUID("00000000-0000-7000-8000-000000000107")
REPORT_ID = UUID("00000000-0000-7000-8000-000000000108")
FIRST_STAGE_ID = UUID("00000000-0000-7000-8000-000000000109")
FINAL_STAGE_ID = UUID("00000000-0000-7000-8000-000000000110")


class FailingAdvanceBoundary:
    def __init__(self, boundary: ReportingHiringBoundary) -> None:
        self._boundary = boundary

    def move_to_recruiting_stage(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        recruiting_stage_id: UUID,
        expected_pipeline_version: int,
    ):
        return self._boundary.move_to_recruiting_stage(
            context,
            invitation_id,
            recruiting_stage_id=recruiting_stage_id,
            expected_pipeline_version=expected_pipeline_version,
        )

    def authorize_invitation(
        self,
        context: TenantContext,
        invitation_id: UUID,
        *,
        required_state: str | frozenset[str],
    ):
        return self._boundary.authorize_invitation(
            context,
            invitation_id,
            required_state=required_state,
        )

    def get_recruiting_stage_decision(
        self,
        context: TenantContext,
        invitation_id: UUID,
    ):
        return self._boundary.get_recruiting_stage_decision(context, invitation_id)

    def advance_invitation_state(
        self,
        _context: TenantContext,
        _invitation_id: UUID,
        *,
        from_state: str,
        to_state: str,
        meta: CommandMeta,
    ):
        del from_state, to_state, meta
        raise RuntimeError("forced failure after pipeline and review writes")


def build_runtime(*, fail_after_review: bool = False):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    CompanyBase.metadata.create_all(engine)
    ReportingBase.metadata.create_all(engine)
    session = Session(engine)
    session.add(
        PositionRow(
            company_id=COMPANY_ID,
            position_id=POSITION_ID,
            title="Backend Engineer",
            description="Evidence-based hiring",
            role_type="engineering",
            headcount=1,
            applicant_capacity=10,
            interview_capacity=10,
            interview_at=None,
            recruitment_start_at=None,
            recruitment_end_at=None,
            submission_requirements=submission_requirements_to_json(
                DEFAULT_SUBMISSION_REQUIREMENTS
            ),
            created_by=USER_ID,
            status="active",
            invitation_email_template=None,
            row_version=1,
            created_at=NOW,
        )
    )
    session.add_all(
        [
            RecruitingStageRow(
                company_id=COMPANY_ID,
                recruiting_stage_id=FIRST_STAGE_ID,
                position_id=POSITION_ID,
                name="1차 합격",
                sort_order=0,
                row_version=1,
            ),
            RecruitingStageRow(
                company_id=COMPANY_ID,
                recruiting_stage_id=FINAL_STAGE_ID,
                position_id=POSITION_ID,
                name="최종 합격",
                sort_order=1,
                row_version=1,
            ),
            InvitationRow(
                company_id=COMPANY_ID,
                invitation_id=INVITATION_ID,
                position_id=POSITION_ID,
                competency_model_version_id=VERSION_ID,
                applicant_id=APPLICANT_ID,
                applicant_email_normalized="candidate@example.test",
                applicant_display_name="Candidate",
                submission_requirements=submission_requirements_to_json(
                    DEFAULT_SUBMISSION_REQUIREMENTS
                ),
                token_hash="a" * 64,
                expires_at=NOW + timedelta(days=30),
                status="completed",
                identity_verified_at=NOW,
                last_state_actor_type="applicant",
                row_version=4,
                recruiting_stage_id=FIRST_STAGE_ID,
                pipeline_row_version=1,
            ),
        ]
    )
    reporting = SQLAlchemyReportingRepository(session)
    reporting.save_report(
        TenantContext(
            company_id=COMPANY_ID,
            actor_type=ActorType.COMPANY_USER,
            actor_id=USER_ID,
            request_id=UUID(int=0),
            trace_id="bidirectional-sql-seed",
        ),
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
    session.commit()

    company = SqlAlchemyCompanyRepository(session)
    clock = FrozenClock(NOW)
    hiring = HiringService(
        company,
        object(),  # type: ignore[arg-type]
        clock,
        object(),  # type: ignore[arg-type]
    )
    boundary = ReportingHiringBoundary(CompanyManagementPublic(company, clock), hiring)
    writer = FailingAdvanceBoundary(boundary) if fail_after_review else boundary
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
        repository=reporting,
        audit=InMemoryAuditAppender(),
        clock=clock,
        invitations=writer,
    )

    @runtime.app.middleware("http")
    async def transaction(request, call_next):
        try:
            response = await call_next(request)
            if response.status_code >= 500:
                session.rollback()
            else:
                session.commit()
            return response
        except BaseException:
            session.rollback()
            raise

    return runtime.app, session, hiring, reporting


def decision_request(http: TestClient):
    return http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers={"Authorization": "Bearer company-token", "Idempotency-Key": "decision-sql"},
        json={
            "recruiting_stage_id": str(FINAL_STAGE_ID),
            "expected_pipeline_version": 1,
        },
    )


def company_context() -> TenantContext:
    return TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.COMPANY_USER,
        actor_id=USER_ID,
        request_id=UUID(int=0),
        trace_id="bidirectional-sql-test",
    )


def test_report_and_kanban_share_the_same_sql_pipeline_assignment() -> None:
    app, session, hiring, reporting = build_runtime()
    response = decision_request(TestClient(app))

    assert response.status_code == 201
    report_written = hiring.get_applicant_recruiting_state(company_context(), INVITATION_ID)
    list_written = hiring.list_invitations(company_context(), POSITION_ID)[0]
    assert report_written.invitation.recruiting_stage_id == FINAL_STAGE_ID
    assert list_written.recruiting_stage_id == FINAL_STAGE_ID
    assert report_written.invitation.pipeline_row_version == 2

    hiring.move_applicants(
        company_context(),
        position_id=POSITION_ID,
        target_stage_id=FIRST_STAGE_ID,
        moves=(ApplicantPipelineMove(invitation_id=INVITATION_ID, expected_version=2),),
    )
    session.commit()
    report_read = hiring.get_applicant_recruiting_state(company_context(), INVITATION_ID)
    assert report_read.invitation.recruiting_stage_id == FIRST_STAGE_ID
    assert report_read.invitation.pipeline_row_version == 3
    reviews = reporting.list_reviews(company_context(), REPORT_ID)
    assert reviews[-1].value["recruiting_stage_name"] == "최종 합격"
    session.close()


def test_failure_after_review_write_rolls_back_pipeline_and_audit_together() -> None:
    app, session, hiring, reporting = build_runtime(fail_after_review=True)
    response = decision_request(TestClient(app, raise_server_exceptions=False))

    assert response.status_code == 500
    session.expire_all()
    current = hiring.get_applicant_recruiting_state(company_context(), INVITATION_ID)
    assert current.invitation.recruiting_stage_id == FIRST_STAGE_ID
    assert current.invitation.pipeline_row_version == 1
    assert current.invitation.status.value == "completed"
    assert reporting.list_reviews(company_context(), REPORT_ID) == ()
    session.close()
