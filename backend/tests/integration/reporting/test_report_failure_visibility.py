from datetime import UTC, datetime
from uuid import UUID

from fastapi.testclient import TestClient
from interview_evidence.reporting.api import create_lane_d_runtime
from interview_evidence.reporting.application.failure_service import (
    ReportGenerationFailureRecorder,
)
from interview_evidence.reporting.domain.failure import ReportGenerationFailure
from interview_evidence.reporting.repositories.postgres import (
    Base,
    SQLAlchemyReportingRepository,
)
from interview_evidence.shared.audit import InMemoryAuditAppender
from interview_evidence.shared.ids import FrozenClock
from interview_evidence.shared.messaging.outbox import OutboxEvent
from interview_evidence.shared.security.principals import CompanyPrincipal, FakePrincipalProvider
from interview_evidence.shared.tenant import ActorType, TenantContext
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

NOW = datetime(2026, 9, 29, 9, 0, tzinfo=UTC)
COMPANY_ID = UUID("00000000-0000-7000-8000-000000000101")
USER_ID = UUID("00000000-0000-7000-8000-000000000102")
SESSION_ID = UUID("00000000-0000-7000-8000-000000000103")
EVENT_ID = UUID("00000000-0000-7000-8000-000000000104")


def context() -> TenantContext:
    return TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.COMPANY_USER,
        actor_id=USER_ID,
        request_id=UUID(int=0),
        trace_id="report-failure-test",
    )


def repository() -> tuple[SQLAlchemyReportingRepository, Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    return SQLAlchemyReportingRepository(session), session


def test_terminal_failure_is_idempotent_and_exposed_as_non_retryable() -> None:
    reports, session = repository()
    failure = ReportGenerationFailure(
        company_id=COMPANY_ID,
        interview_session_id=SESSION_ID,
        source_event_id=EVENT_ID,
        last_delivery_attempt=3,
        error_code="TimeoutError",
        failed_at=NOW,
    )
    reports.save_generation_failure(context(), failure)
    reports.save_generation_failure(context(), failure)
    session.commit()
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
        repository=reports,
        audit=InMemoryAuditAppender(),
        clock=FrozenClock(NOW),
    )

    response = TestClient(runtime.app).get(
        f"/v1/interview-sessions/{SESSION_ID}/report",
        headers={"Authorization": "Bearer company-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "retryable": False,
        "message": (
            "리포트 생성에 실패했습니다. 담당자가 재처리하기 전에는 "
            "최종 채용 결정을 진행할 수 없습니다."
        ),
    }
    stored = reports.get_generation_failure_for_session(context(), SESSION_ID)
    assert stored == failure
    assert "sensitive" not in response.text
    session.close()


def test_failure_recorder_uses_only_stable_error_code_and_event_lineage() -> None:
    reports, session = repository()
    system_context = TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.SYSTEM,
        actor_id=EVENT_ID,
        request_id=EVENT_ID,
        trace_id="report-failure-worker",
    )
    event = OutboxEvent(
        outbox_event_id=EVENT_ID,
        company_id=COMPANY_ID,
        aggregate_type="interview_session",
        aggregate_id=SESSION_ID,
        aggregate_version=1,
        event_type="report.generation_requested",
        event_version=1,
        payload={"interview_session_id": str(SESSION_ID)},
        idempotency_key="report-generation-test",
        trace_id="report-failure-worker",
        occurred_at=NOW,
        delivery_attempt=3,
    )

    ReportGenerationFailureRecorder(reports, FrozenClock(NOW)).retry_exhausted(
        system_context,
        event,
        error_code="contains-sensitive-detail",
    )
    session.commit()

    stored = reports.get_generation_failure_for_session(context(), SESSION_ID)
    assert stored is not None
    assert stored.source_event_id == EVENT_ID
    assert stored.last_delivery_attempt == 3
    assert stored.error_code == "RetryableError"
    session.close()
