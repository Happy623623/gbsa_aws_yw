from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from sqlalchemy.orm import Session

from interview_evidence.company_management.adapters.applicant_session import (
    ApplicantSessionAdapter,
)
from interview_evidence.company_management.adapters.company_auth import CompanyAuthAdapter
from interview_evidence.company_management.api.applicant_routes import create_applicant_router
from interview_evidence.company_management.api.company_routes import (
    InvitationReviewResolver,
    InvitationSessionResolver,
    create_company_router,
)
from interview_evidence.company_management.application.applicant_access_service import (
    ApplicantAccessService,
    ConsentFaultBoundary,
)
from interview_evidence.company_management.application.company_service import CompanyService
from interview_evidence.company_management.application.criteria_service import CriteriaService
from interview_evidence.company_management.application.hiring_service import HiringService
from interview_evidence.company_management.application.interviewer_service import (
    InterviewerProfileService,
)
from interview_evidence.company_management.application.invitation_template_service import (
    InvitationTemplateService,
)
from interview_evidence.company_management.repositories.postgres import (
    CompanyRepository,
    SqlAlchemyCompanyRepository,
)
from interview_evidence.company_management.workers.invitation_email import (
    InvitationEmailHandler,
)
from interview_evidence.main import create_app
from interview_evidence.shared.audit import AuditAppender
from interview_evidence.shared.aws_clients.ports import EmailSender
from interview_evidence.shared.idempotency import ResourceIdempotencyStore
from interview_evidence.shared.ids import Clock
from interview_evidence.shared.messaging.outbox import Outbox
from interview_evidence.shared.security.principals import PrincipalProvider


@dataclass(frozen=True, slots=True)
class LaneARuntime:
    app: FastAPI
    repository: CompanyRepository
    audit: AuditAppender
    sessions: ApplicantSessionAdapter
    outbox: Outbox
    email_sender: EmailSender
    company_service: CompanyService
    criteria_service: CriteriaService
    interviewer_service: InterviewerProfileService
    hiring_service: HiringService
    applicant_access_service: ApplicantAccessService
    template_service: InvitationTemplateService


def create_lane_a_runtime(
    *,
    principal_provider: PrincipalProvider,
    repository: CompanyRepository,
    audit: AuditAppender,
    clock: Clock,
    sessions: ApplicantSessionAdapter,
    outbox: Outbox,
    idempotency: ResourceIdempotencyStore,
    email_sender: EmailSender,
    applicant_access_base_url: str = "https://applicant.local/access",
    logo_base_url: str = "https://console.local",
    interview_sessions: InvitationSessionResolver | None = None,
    invitation_reviews: InvitationReviewResolver | None = None,
    consent_fault_boundary: ConsentFaultBoundary | None = None,
) -> LaneARuntime:
    active_repository = repository
    active_audit = audit
    active_clock = clock
    active_sessions = sessions
    active_outbox = outbox
    active_idempotency = idempotency
    active_email_sender = email_sender

    company_service = CompanyService(
        active_repository,
        active_clock,
        active_idempotency,
        active_outbox,
    )
    criteria_service = CriteriaService(
        active_repository,
        active_clock,
        active_idempotency,
    )
    interviewer_service = InterviewerProfileService(
        active_repository,
        active_clock,
        active_idempotency,
    )
    hiring_service = HiringService(
        active_repository,
        active_sessions,
        active_clock,
        active_idempotency,
    )
    access_service = ApplicantAccessService(
        active_repository,
        active_outbox,
        active_clock,
        consent_fault_boundary=consent_fault_boundary,
    )
    template_service = InvitationTemplateService(
        active_repository,
        active_clock,
        logo_base_url=logo_base_url,
    )
    company_router = create_company_router(
        auth=CompanyAuthAdapter(principal_provider),
        company_service=company_service,
        criteria_service=criteria_service,
        interviewer_service=interviewer_service,
        hiring_service=hiring_service,
        template_service=template_service,
        audit=active_audit,
        invitation_email=InvitationEmailHandler(active_email_sender),
        applicant_access_base_url=applicant_access_base_url,
        interview_sessions=interview_sessions,
        invitation_reviews=invitation_reviews,
    )
    applicant_router = create_applicant_router(
        sessions=active_sessions,
        access_service=access_service,
        clock=active_clock,
    )
    return LaneARuntime(
        app=create_app([company_router, applicant_router]),
        repository=active_repository,
        audit=active_audit,
        sessions=active_sessions,
        outbox=active_outbox,
        email_sender=active_email_sender,
        company_service=company_service,
        criteria_service=criteria_service,
        interviewer_service=interviewer_service,
        hiring_service=hiring_service,
        applicant_access_service=access_service,
        template_service=template_service,
    )


def create_lane_a_app(
    *,
    principal_provider: PrincipalProvider,
    repository: CompanyRepository,
    audit: AuditAppender,
    clock: Clock,
    sessions: ApplicantSessionAdapter,
    outbox: Outbox,
    idempotency: ResourceIdempotencyStore,
    email_sender: EmailSender,
    applicant_access_base_url: str = "https://applicant.local/access",
    logo_base_url: str = "https://console.local",
    interview_sessions: InvitationSessionResolver | None = None,
    invitation_reviews: InvitationReviewResolver | None = None,
    consent_fault_boundary: ConsentFaultBoundary | None = None,
) -> FastAPI:
    return create_lane_a_runtime(
        principal_provider=principal_provider,
        repository=repository,
        audit=audit,
        clock=clock,
        sessions=sessions,
        outbox=outbox,
        idempotency=idempotency,
        email_sender=email_sender,
        applicant_access_base_url=applicant_access_base_url,
        logo_base_url=logo_base_url,
        interview_sessions=interview_sessions,
        invitation_reviews=invitation_reviews,
        consent_fault_boundary=consent_fault_boundary,
    ).app


def create_sql_repository(session: Session) -> CompanyRepository:
    return SqlAlchemyCompanyRepository(session)
