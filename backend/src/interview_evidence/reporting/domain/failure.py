from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ReportGenerationFailure:
    company_id: UUID
    interview_session_id: UUID
    source_event_id: UUID
    last_delivery_attempt: int
    error_code: str
    failed_at: datetime

    def __post_init__(self) -> None:
        if self.last_delivery_attempt < 1:
            raise ValueError("last delivery attempt must be positive")
        if not self.error_code or len(self.error_code) > 100:
            raise ValueError("failure error code is invalid")
