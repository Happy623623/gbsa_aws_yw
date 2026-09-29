from __future__ import annotations

from uuid import UUID

from interview_evidence.reporting.domain.failure import ReportGenerationFailure
from interview_evidence.reporting.repositories.postgres import ReportingRepository
from interview_evidence.shared.ids import Clock
from interview_evidence.shared.messaging.outbox import OutboxEvent
from interview_evidence.shared.tenant import TenantContext

_SAFE_RETRY_ERROR_CODES = frozenset({"MessageRetryRequested", "TimeoutError", "ConnectionError"})


class ReportGenerationFailureRecorder:
    """Persist the terminal reporting fact without raw exceptions or queue payloads."""

    def __init__(self, repository: ReportingRepository, clock: Clock) -> None:
        self._repository = repository
        self._clock = clock

    def retry_exhausted(
        self,
        context: TenantContext,
        event: OutboxEvent,
        *,
        error_code: str,
    ) -> None:
        if event.event_type != "report.generation_requested":
            return
        session_id = UUID(str(event.payload.get("interview_session_id", event.aggregate_id)))
        self._repository.save_generation_failure(
            context,
            ReportGenerationFailure(
                company_id=context.company_id,
                interview_session_id=session_id,
                source_event_id=event.outbox_event_id,
                last_delivery_attempt=event.delivery_attempt or 1,
                error_code=(
                    error_code if error_code in _SAFE_RETRY_ERROR_CODES else "RetryableError"
                ),
                failed_at=self._clock.now(),
            ),
        )
