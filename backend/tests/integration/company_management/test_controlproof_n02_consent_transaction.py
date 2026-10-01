"""N-02 proof that consent facts share the actual HTTP transaction boundary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from interview_evidence.company_management.application.applicant_access_service import (
    ApplicantAccessService,
)
from interview_evidence.company_management.domain.applicant_access import DEFAULT_CONSENT_POLICY
from interview_evidence.company_management.repositories.postgres import (
    Base as CompanyBase,
)
from interview_evidence.company_management.repositories.postgres import (
    ConsentRecordRow,
    InvitationRow,
    InvitationStateHistoryRow,
    SqlAlchemyCompanyRepository,
)
from interview_evidence.shared.database import RequestScopedDatabase
from interview_evidence.shared.ids import FrozenClock
from interview_evidence.shared.persistence import (
    Base as SharedBase,
)
from interview_evidence.shared.persistence import (
    OutboxEventRow,
    SQLOutbox,
)
from interview_evidence.shared.security.principals import ApplicantPrincipal
from interview_evidence.shared.tenant import ActorType, TenantContext
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

COMPANY_ID = UUID("00000000-0000-7000-8000-000000000001")
COMMIT_INVITATION = UUID("00000000-0000-7000-8000-000000000010")
ROLLBACK_INVITATION = UUID("00000000-0000-7000-8000-000000000020")
COMMIT_APPLICANT = UUID("00000000-0000-7000-8000-000000000011")
ROLLBACK_APPLICANT = UUID("00000000-0000-7000-8000-000000000021")
NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _invitation(invitation_id: UUID, applicant_id: UUID, token: str) -> InvitationRow:
    return InvitationRow(
        company_id=COMPANY_ID,
        invitation_id=invitation_id,
        position_id=UUID("00000000-0000-7000-8000-000000000030"),
        competency_model_version_id=UUID("00000000-0000-7000-8000-000000000031"),
        applicant_id=applicant_id,
        applicant_email_normalized=f"{applicant_id}@example.invalid",
        applicant_display_name="ControlProof synthetic applicant",
        submission_requirements=[
            {"material_type": "resume", "required": True, "enabled": True}
        ],
        token_hash=token * 64,
        expires_at=NOW + timedelta(days=1),
        status="identity_verified",
        identity_verified_at=NOW,
        last_state_actor_type="applicant",
        row_version=1,
        recruiting_stage_id=None,
        pipeline_row_version=1,
    )


@pytest.mark.anyio
async def test_consent_record_transition_and_outbox_commit_or_rollback_together(
    tmp_path,
) -> None:
    database_path = tmp_path / "controlproof-n02.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    CompanyBase.metadata.create_all(engine)
    SharedBase.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            (
                _invitation(COMMIT_INVITATION, COMMIT_APPLICANT, "a"),
                _invitation(ROLLBACK_INVITATION, ROLLBACK_APPLICANT, "b"),
            )
        )
        session.commit()

    database = RequestScopedDatabase(
        f"sqlite+pysqlite:///{database_path}",
        engine=engine,
    )
    app = FastAPI()
    database.install_http_transaction_middleware(app)

    @app.post("/controlproof/consent/{outcome}")
    def consent(outcome: str) -> dict[str, str]:
        invitation_id, applicant_id = (
            (COMMIT_INVITATION, COMMIT_APPLICANT)
            if outcome == "commit"
            else (ROLLBACK_INVITATION, ROLLBACK_APPLICANT)
        )
        context = TenantContext(
            company_id=COMPANY_ID,
            actor_type=ActorType.APPLICANT,
            actor_id=applicant_id,
            request_id=invitation_id,
            trace_id=f"controlproof-n02-{outcome}",
        )
        principal = ApplicantPrincipal(
            company_id=COMPANY_ID,
            invitation_id=invitation_id,
            applicant_id=applicant_id,
            session_id=UUID("00000000-0000-7000-8000-000000000099"),
        )
        service = ApplicantAccessService(
            SqlAlchemyCompanyRepository(database.session),
            SQLOutbox(database.session),
            FrozenClock(NOW),
        )
        record = service.record_consent(
            context,
            principal,
            policy_version=DEFAULT_CONSENT_POLICY.policy_version,
            accepted_purposes=tuple(DEFAULT_CONSENT_POLICY.required_purposes),
            consent_content_digest=DEFAULT_CONSENT_POLICY.content_digest,
        )
        if outcome == "rollback":
            raise RuntimeError("force rollback after all consent writes")
        return {"consent_record_id": str(record.consent_record_id)}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        assert (await client.post("/controlproof/consent/commit")).status_code == 200
        with pytest.raises(RuntimeError, match="force rollback"):
            await client.post("/controlproof/consent/rollback")

    with Session(engine) as session:
        committed = session.scalar(
            select(InvitationRow).where(InvitationRow.invitation_id == COMMIT_INVITATION)
        )
        rolled_back = session.scalar(
            select(InvitationRow).where(InvitationRow.invitation_id == ROLLBACK_INVITATION)
        )
        consent_invites = set(session.scalars(select(ConsentRecordRow.invitation_id)))
        transition_invites = set(
            session.scalars(select(InvitationStateHistoryRow.invitation_id))
        )
        outbox_invites = set(
            session.scalars(
                select(OutboxEventRow.aggregate_id).where(
                    OutboxEventRow.event_type == "invitation.consent_completed"
                )
            )
        )
    assert committed.status == "consented"
    assert rolled_back.status == "identity_verified"
    assert consent_invites == {COMMIT_INVITATION}
    assert transition_invites == {COMMIT_INVITATION}
    assert outbox_invites == {COMMIT_INVITATION}
