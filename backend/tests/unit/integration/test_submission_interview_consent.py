"""T082 (ControlProof Spec 003): the recording boundary must require active consent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
from interview_evidence.integration.submission_interview import SubmissionInterviewBoundary
from interview_evidence.interview_engine.application.authorization import (
    InterviewAuthorizationDenied,
)
from interview_evidence.shared.security.principals import ApplicantPrincipal
from interview_evidence.shared.tenant import ActorType, TenantContext
from interview_evidence.submission_analysis.domain.strategy import StrategyStatus

COMPANY_ID = UUID("00000000-0000-7000-8000-000000000001")
INVITATION_ID = UUID("00000000-0000-7000-8000-000000000002")
APPLICANT_ID = UUID("00000000-0000-7000-8000-000000000003")
STRATEGY_ID = UUID("00000000-0000-7000-8000-000000000004")
VERSION_ID = UUID("00000000-0000-7000-8000-000000000005")


def _context() -> TenantContext:
    return TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.APPLICANT,
        actor_id=APPLICANT_ID,
        request_id=UUID("00000000-0000-7000-8000-000000000006"),
        trace_id="trace",
    )


def _principal() -> ApplicantPrincipal:
    return ApplicantPrincipal(
        company_id=COMPANY_ID,
        invitation_id=INVITATION_ID,
        applicant_id=APPLICANT_ID,
        session_id=UUID("00000000-0000-7000-8000-000000000007"),
    )


def _submission() -> Mock:
    submission = Mock()
    submission.get_strategy_snapshot.return_value = SimpleNamespace(
        status=StrategyStatus.READY,
        company_id=COMPANY_ID,
        invitation_id=INVITATION_ID,
        applicant_id=APPLICANT_ID,
        interview_strategy_id=STRATEGY_ID,
        competency_model_version_id=VERSION_ID,
    )
    submission.get_analysis_status.return_value = SimpleNamespace(
        submissions=[], strategy_ready=True, strategy_id=STRATEGY_ID
    )
    return submission


def _authorize(boundary: SubmissionInterviewBoundary):
    return boundary.authorize_start(
        _context(), _principal(), strategy_id=STRATEGY_ID, acknowledged_partial_analysis=True
    )


def test_recording_start_is_denied_without_active_recording_consent() -> None:
    """ControlProof N-02 sandbox Runs created sessions for unconsented applicants: the
    boundary checked the strategy but never the consent."""
    company = Mock()
    company.get_consent_authorization.return_value = SimpleNamespace(authorized=False)
    with pytest.raises(InterviewAuthorizationDenied, match="consent"):
        _authorize(SubmissionInterviewBoundary(_submission(), company))
    company.get_consent_authorization.assert_called_once_with(
        _context(), INVITATION_ID, required_purposes=frozenset({"recording"})
    )


def test_recording_start_fails_closed_without_a_consent_source() -> None:
    with pytest.raises(InterviewAuthorizationDenied, match="consent"):
        _authorize(SubmissionInterviewBoundary(_submission(), None))


def test_recording_start_proceeds_with_active_consent() -> None:
    company = Mock()
    company.get_consent_authorization.return_value = SimpleNamespace(authorized=True)
    authorization = _authorize(SubmissionInterviewBoundary(_submission(), company))
    assert authorization.strategy_id == STRATEGY_ID and authorization.partial_analysis is False
