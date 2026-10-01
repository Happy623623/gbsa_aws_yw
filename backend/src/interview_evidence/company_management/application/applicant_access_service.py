from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from interview_evidence.company_management.domain.applicant_access import (
    DEFAULT_CONSENT_POLICY,
    ApplicantProfile,
    ConsentPolicy,
    ConsentRecord,
    ProcessingPurpose,
)
from interview_evidence.company_management.domain.hiring import (
    Invitation,
    InvitationStateChange,
)
from interview_evidence.company_management.repositories.postgres import CompanyRepository
from interview_evidence.shared.ids import Clock, new_uuid7
from interview_evidence.shared.messaging.outbox import Outbox, OutboxEvent
from interview_evidence.shared.security.principals import ApplicantPrincipal
from interview_evidence.shared.submission_materials import SubmissionRequirement
from interview_evidence.shared.tenant import TenantContext


@dataclass(frozen=True, slots=True)
class ApplicantInvitationPreviewSnapshot:
    company_name: str
    position_title: str
    position_description: str
    role_type: str | None
    interview_at: datetime | None
    interview_duration_minutes: int
    interview_level: str
    interviewer_name: str | None
    submission_requirements: tuple[SubmissionRequirement, ...]


class ConsentFaultBoundary(Protocol):
    """Optional local/test boundary; production authorization remains unchanged."""

    def trigger_if_configured(
        self,
        *,
        invitation_id: str,
        applicant_id: str,
        request_id: str,
        trace_id: str,
    ) -> bool: ...


class ApplicantAccessService:
    def __init__(
        self,
        repository: CompanyRepository,
        outbox: Outbox,
        clock: Clock,
        *,
        consent_policy: ConsentPolicy = DEFAULT_CONSENT_POLICY,
        consent_fault_boundary: ConsentFaultBoundary | None = None,
    ) -> None:
        self._repository = repository
        self._outbox = outbox
        self._clock = clock
        self._consent_policy = consent_policy
        self._consent_fault_boundary = consent_fault_boundary

    def get_consent_policy(self) -> ConsentPolicy:
        return self._consent_policy

    def get_invitation_preview(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
        applicant_id: UUID,
    ) -> ApplicantInvitationPreviewSnapshot:
        invitation = self._repository.get_invitation(context, invitation_id)
        if invitation.applicant_id != applicant_id:
            raise PermissionError("invitation preview is outside the applicant scope")
        position = self._repository.get_position(context, invitation.position_id)
        criteria = self._repository.get_criterion_version(
            context,
            invitation.competency_model_version_id,
        )
        company = self._repository.get_company(context)
        raw_name = criteria.persona_definition.get("name")
        interviewer_name = raw_name.strip() if isinstance(raw_name, str) else None
        return ApplicantInvitationPreviewSnapshot(
            company_name=company.name,
            position_title=position.title,
            position_description=position.description,
            role_type=position.role_type,
            interview_at=position.interview_at,
            interview_duration_minutes=criteria.interview_duration_minutes,
            interview_level=criteria.interview_level.value,
            interviewer_name=interviewer_name or None,
            submission_requirements=invitation.submission_requirements,
        )

    def verify_identity(
        self,
        context: TenantContext,
        principal: ApplicantPrincipal,
        *,
        display_name: str,
    ) -> tuple[ApplicantProfile, Invitation]:
        context.assert_company(principal.company_id)
        invitation = self._repository.get_invitation(context, principal.invitation_id)
        if invitation.applicant_id != principal.applicant_id:
            raise PermissionError("applicant session is outside the invitation scope")
        if display_name.strip().casefold() != invitation.applicant_display_name.casefold():
            raise PermissionError("identity verification failed")
        profile = ApplicantProfile(
            applicant_id=principal.applicant_id,
            company_id=context.company_id,
            invitation_id=principal.invitation_id,
            display_name=display_name.strip(),
        )
        updated = invitation.transition(
            "identity_verified",
            actor_type="applicant",
            occurred_at=self._clock.now(),
            expected_version=invitation.row_version,
        )
        self._repository.save_applicant_profile(context, profile)
        self._save_transition(context, invitation, updated)
        return profile, updated

    def record_consent(
        self,
        context: TenantContext,
        principal: ApplicantPrincipal,
        *,
        policy_version: str,
        accepted_purposes: tuple[ProcessingPurpose, ...],
        consent_content_digest: str,
    ) -> ConsentRecord:
        purposes = frozenset(accepted_purposes)
        if purposes != self._consent_policy.required_purposes:
            raise ValueError("all required consent purposes must be accepted")
        if (
            policy_version != self._consent_policy.policy_version
            or consent_content_digest != self._consent_policy.content_digest
        ):
            raise ValueError("consent policy version or digest is stale")
        invitation = self._repository.get_invitation(context, principal.invitation_id)
        consent = ConsentRecord.accept(
            consent_record_id=new_uuid7(self._clock.now()),
            company_id=context.company_id,
            invitation_id=principal.invitation_id,
            policy_version=policy_version,
            purposes=tuple(purposes),
            retention_days=self._consent_policy.retention_days,
            accepted_at=self._clock.now(),
            evidence_digest=consent_content_digest,
        )
        updated = invitation.transition(
            "consented",
            actor_type="applicant",
            occurred_at=self._clock.now(),
            expected_version=invitation.row_version,
        )
        self._repository.save_consent(context, consent)
        if self._consent_fault_boundary is not None:
            self._consent_fault_boundary.trigger_if_configured(
                invitation_id=str(principal.invitation_id),
                applicant_id=str(principal.applicant_id),
                request_id=str(context.request_id),
                trace_id=context.trace_id,
            )
        self._save_transition(context, invitation, updated)
        self._outbox.append(
            OutboxEvent(
                outbox_event_id=new_uuid7(self._clock.now()),
                company_id=context.company_id,
                aggregate_type="invitation",
                aggregate_id=invitation.invitation_id,
                aggregate_version=updated.row_version,
                event_type="invitation.consent_completed",
                event_version=1,
                payload={
                    "invitation_id": str(invitation.invitation_id),
                    "applicant_id": str(principal.applicant_id),
                    "consent_record_id": str(consent.consent_record_id),
                    "purpose_codes": sorted(purpose.value for purpose in purposes),
                    "retention_days": consent.retention_days,
                },
                idempotency_key=f"consent-{consent.consent_record_id}",
                trace_id=context.trace_id,
                occurred_at=self._clock.now(),
            )
        )
        return consent

    def _save_transition(
        self,
        context: TenantContext,
        before: Invitation,
        after: Invitation,
    ) -> None:
        self._repository.save_invitation(context, after)
        self._repository.append_invitation_state_change(
            context,
            InvitationStateChange(
                invitation_state_change_id=new_uuid7(self._clock.now()),
                company_id=context.company_id,
                invitation_id=after.invitation_id,
                from_status=before.status,
                to_status=after.status,
                actor_type="applicant",
                occurred_at=self._clock.now(),
                aggregate_version=after.row_version,
            ),
        )
