from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from interview_evidence.shared.tenant import ActorType


class ReviewType(StrEnum):
    ASSESSMENT_OVERRIDE = "assessment_override"
    REQUIREMENT_OVERRIDE = "requirement_override"
    NOTE = "note"
    BOOKMARK = "bookmark"
    FINAL_DECISION = "final_decision"


class Decision(StrEnum):
    ADVANCE = "advance"
    REJECT = "reject"
    HOLD = "hold"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True, slots=True)
class HumanReview:
    human_review_id: UUID
    company_id: UUID
    report_id: UUID
    company_user_id: UUID
    review_type: ReviewType
    target_id: UUID
    value: dict[str, str]
    reason: str | None
    created_at: datetime

    @classmethod
    def assessment_override(
        cls,
        *,
        human_review_id: UUID,
        company_id: UUID,
        report_id: UUID,
        company_user_id: UUID,
        report_item_id: UUID,
        assessment_state: str,
        reason: str,
        created_at: datetime,
    ) -> HumanReview:
        if not reason.strip():
            raise ValueError("assessment override reason is required")
        return cls(
            human_review_id=human_review_id,
            company_id=company_id,
            report_id=report_id,
            company_user_id=company_user_id,
            review_type=ReviewType.ASSESSMENT_OVERRIDE,
            target_id=report_item_id,
            value={"assessment_state": assessment_state},
            reason=reason,
            created_at=created_at,
        )

    @classmethod
    def requirement_override(
        cls,
        *,
        human_review_id: UUID,
        company_id: UUID,
        report_id: UUID,
        company_user_id: UUID,
        requirement_assessment_id: UUID,
        requirement_status: str,
        reason: str,
        created_at: datetime,
    ) -> HumanReview:
        if not reason.strip():
            raise ValueError("requirement override reason is required")
        return cls(
            human_review_id=human_review_id,
            company_id=company_id,
            report_id=report_id,
            company_user_id=company_user_id,
            review_type=ReviewType.REQUIREMENT_OVERRIDE,
            target_id=requirement_assessment_id,
            value={"requirement_status": requirement_status},
            reason=reason,
            created_at=created_at,
        )

    @classmethod
    def final_decision(
        cls,
        *,
        human_review_id: UUID,
        company_id: UUID,
        report_id: UUID,
        company_user_id: UUID,
        invitation_id: UUID,
        actor_type: ActorType,
        decision: Decision,
        reason: str,
        created_at: datetime,
    ) -> HumanReview:
        if actor_type is not ActorType.COMPANY_USER:
            raise PermissionError("final decision requires a human company user")
        if not reason.strip():
            raise ValueError("final decision reason is required")
        return cls(
            human_review_id=human_review_id,
            company_id=company_id,
            report_id=report_id,
            company_user_id=company_user_id,
            review_type=ReviewType.FINAL_DECISION,
            target_id=invitation_id,
            value={"decision": decision.value},
            reason=reason,
            created_at=created_at,
        )

    @classmethod
    def recruiting_stage_decision(
        cls,
        *,
        human_review_id: UUID,
        company_id: UUID,
        report_id: UUID,
        company_user_id: UUID,
        invitation_id: UUID,
        actor_type: ActorType,
        recruiting_stage_id: UUID,
        recruiting_stage_name: str,
        expected_pipeline_version: int,
        created_at: datetime,
    ) -> HumanReview:
        if actor_type is not ActorType.COMPANY_USER:
            raise PermissionError("final decision requires a human company user")
        normalized_name = " ".join(recruiting_stage_name.split())
        if not normalized_name:
            raise ValueError("recruiting stage name is required")
        return cls(
            human_review_id=human_review_id,
            company_id=company_id,
            report_id=report_id,
            company_user_id=company_user_id,
            review_type=ReviewType.FINAL_DECISION,
            target_id=invitation_id,
            value={
                "recruiting_stage_id": str(recruiting_stage_id),
                "recruiting_stage_name": normalized_name,
                "expected_pipeline_version": str(expected_pipeline_version),
            },
            reason=None,
            created_at=created_at,
        )
