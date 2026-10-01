"""Actual consent fault boundary rolls the entire HTTP transaction back."""

from __future__ import annotations

import json
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
from interview_evidence.runtime.controlproof_consent import ControlProofConsentFaultGuard
from interview_evidence.shared.database import RequestScopedDatabase
from interview_evidence.shared.ids import FrozenClock
from interview_evidence.shared.persistence import (
    Base as SharedBase,
)
from interview_evidence.shared.persistence import OutboxEventRow, SQLOutbox
from interview_evidence.shared.security.principals import ApplicantPrincipal
from interview_evidence.shared.tenant import ActorType, TenantContext
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

COMPANY_ID = UUID("00000000-0000-7000-8000-000000000001")
RUN_ID = UUID("00000000-0000-7000-8000-000000000201")
INVITATION_ID = UUID("00000000-0000-7000-8000-000000000202")
APPLICANT_ID = UUID("00000000-0000-7000-8000-000000000203")
REQUEST_ID = UUID("00000000-0000-7000-8000-000000000204")
SUBJECT_REF = "synthetic-consent-rollback"
NOW = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.mark.anyio
async def test_fault_after_consent_flush_returns_5xx_and_leaves_no_partial_effect(
    tmp_path,
) -> None:
    database_path = tmp_path / "controlproof-n02-rollback.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    CompanyBase.metadata.create_all(engine)
    SharedBase.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            InvitationRow(
                company_id=COMPANY_ID,
                invitation_id=INVITATION_ID,
                position_id=UUID(int=210),
                competency_model_version_id=UUID(int=211),
                applicant_id=APPLICANT_ID,
                applicant_email_normalized="rollback@example.invalid",
                applicant_display_name=SUBJECT_REF,
                submission_requirements=[
                    {"material_type": "resume", "required": True, "enabled": True}
                ],
                token_hash="a" * 64,
                expires_at=NOW + timedelta(days=1),
                status="identity_verified",
                identity_verified_at=NOW,
                last_state_actor_type="applicant",
                row_version=1,
                recruiting_stage_id=None,
                pipeline_row_version=1,
            )
        )
        session.commit()

    root = tmp_path / "faults"
    guard = ControlProofConsentFaultGuard.from_environment(
        {
            "APP_ENVIRONMENT": "test",
            "CONTROLPROOF_TEST_HOOKS_ENABLED": "true",
            "CONTROLPROOF_FAULT_ROOT": str(root),
        }
    )
    marker_now = datetime.now(UTC)
    marker = {
        "schema_version": "controlproof.whyyou-consent-fault.v1",
        "run_id": str(RUN_ID),
        "lane_id": "CONSENT_FAULT_RECOVERY",
        "subject_ref": SUBJECT_REF,
        "invitation_id": str(INVITATION_ID),
        "applicant_id": str(APPLICANT_ID),
        "fault_type": "consent_after_record_before_state_v1",
        "fault_variant": "AFTER_CONSENT_RECORD_BEFORE_STATE",
        "issued_at": (marker_now - timedelta(seconds=1)).isoformat(),
        "expires_at": (marker_now + timedelta(minutes=5)).isoformat(),
        "one_shot": True,
    }
    marker_path = root / "consent" / f"{INVITATION_ID}.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    database = RequestScopedDatabase(
        f"sqlite+pysqlite:///{database_path}", engine=engine
    )
    app = FastAPI()
    database.install_http_transaction_middleware(app)

    @app.post("/controlproof/faulted-consent")
    def faulted_consent() -> dict[str, str]:
        context = TenantContext(
            company_id=COMPANY_ID,
            actor_type=ActorType.APPLICANT,
            actor_id=APPLICANT_ID,
            request_id=REQUEST_ID,
            trace_id=(
                f"controlproof:{RUN_ID}:CONSENT_FAULT_RECOVERY:{SUBJECT_REF}"
            ),
        )
        principal = ApplicantPrincipal(
            company_id=COMPANY_ID,
            invitation_id=INVITATION_ID,
            applicant_id=APPLICANT_ID,
            session_id=UUID(int=212),
        )
        service = ApplicantAccessService(
            SqlAlchemyCompanyRepository(database.session),
            SQLOutbox(database.session),
            FrozenClock(NOW),
            consent_fault_boundary=guard,
        )
        record = service.record_consent(
            context,
            principal,
            policy_version=DEFAULT_CONSENT_POLICY.policy_version,
            accepted_purposes=tuple(DEFAULT_CONSENT_POLICY.required_purposes),
            consent_content_digest=DEFAULT_CONSENT_POLICY.content_digest,
        )
        return {"consent_record_id": str(record.consent_record_id)}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post("/controlproof/faulted-consent")
    assert response.status_code == 500

    with Session(engine) as session:
        invitation = session.scalar(
            select(InvitationRow).where(InvitationRow.invitation_id == INVITATION_ID)
        )
        consent_count = session.scalar(select(func.count()).select_from(ConsentRecordRow))
        transition_count = session.scalar(
            select(func.count()).select_from(InvitationStateHistoryRow)
        )
        event_count = session.scalar(select(func.count()).select_from(OutboxEventRow))
    assert invitation.status == "identity_verified"
    assert (consent_count, transition_count, event_count) == (0, 0, 0)
    receipt_path = root / "receipts" / f"{RUN_ID}.jsonl"
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert len(receipts) == 1
    assert receipts[0]["boundary"] == "AFTER_CONSENT_RECORD_BEFORE_INVITATION_STATE"
