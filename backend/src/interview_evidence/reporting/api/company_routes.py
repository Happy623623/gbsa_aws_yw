from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from interview_evidence.reporting.adapters.playback import ScopedPlaybackLocator
from interview_evidence.reporting.application.deletion_service import DeletionService
from interview_evidence.reporting.application.review_service import (
    InvitationDecisionWriter,
    InvitationStateAdvancer,
    ReviewService,
    close_invitation_review,
)
from interview_evidence.reporting.application.timeline_service import (
    QuestionRationaleProvider,
    TimelineService,
)
from interview_evidence.reporting.domain.deletion import DeletionManifest
from interview_evidence.reporting.domain.report import Report
from interview_evidence.reporting.domain.review import HumanReview, ReviewType
from interview_evidence.reporting.domain.scoring import Aggregate
from interview_evidence.reporting.repositories.postgres import (
    ReportingRepository,
    TenantScopedReportingNotFound,
)
from interview_evidence.shared.audit import AuditAppender
from interview_evidence.shared.ids import Clock
from interview_evidence.shared.security.principals import (
    CompanyPrincipal,
    PrincipalNotFoundError,
    PrincipalProvider,
)
from interview_evidence.shared.tenant import ActorType, TenantContext


class HumanAssessmentReviewCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment_state: str
    reason: str = Field(min_length=1, max_length=5000)


class HumanRequirementReviewCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_status: Literal["met", "partially_met", "not_met", "unknown"]
    reason: str = Field(min_length=1, max_length=5000)


class ReviewArtifactCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_type: Literal["note"]
    target_id: UUID
    value: str = Field(min_length=1, max_length=10_000)


class FinalDecisionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recruiting_stage_id: UUID
    expected_pipeline_version: int = Field(ge=1)


class HumanReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    human_review_id: UUID
    review_type: str
    created_by: UUID
    created_at: datetime
    value: dict[str, str]
    reason: str | None


class FinalDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    human_review: HumanReviewResponse
    invitation_id: UUID
    position_id: UUID
    recruiting_stage_id: UUID
    recruiting_stage_name: str
    pipeline_row_version: int = Field(ge=1)
    invitation_state: str


class DeletionRequestCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope_type: str
    scope_id: UUID
    reason: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True, slots=True)
class CompanyScope:
    principal: CompanyPrincipal
    context: TenantContext


@dataclass(frozen=True, slots=True)
class LaneDRuntime:
    app: FastAPI
    repository: ReportingRepository
    deletion_service: DeletionService


def _review_view(review: HumanReview) -> dict[str, object]:
    return {
        "human_review_id": review.human_review_id,
        "review_type": str(review.review_type),
        "created_by": review.company_user_id,
        "created_at": review.created_at,
        # Both were stored from the start and neither was returned, so the console could list
        # that a review happened and never what it said: a reviewer could not read back the
        # decision they recorded, nor the reason they wrote for overruling a score.
        "value": dict(review.value),
        "reason": review.reason,
    }


def _aggregate_view(aggregate: Aggregate) -> dict[str, object]:
    """One level of the score's arithmetic, in the shape the calculator renders.

    ``numerator`` and ``denominator`` travel even though the contributions imply them, because
    the calculator shows ``55.7 ÷ 0.75 = 74`` and a client re-deriving those two numbers could
    round them differently from the score sitting next to them.
    """
    return {
        "numerator": aggregate.numerator,
        "denominator": aggregate.denominator,
        "contributions": [
            {
                "key": contribution.key,
                "score": contribution.score,
                "weight": contribution.weight,
                "normalized_weight": contribution.normalized_weight,
                "contribution": contribution.contribution,
            }
            for contribution in aggregate.contributions
        ],
        # Carries weight but no score. Without these the divisor appears from nowhere.
        "exclusions": [
            {
                "key": exclusion.key,
                "weight": exclusion.weight,
                "normalized_weight": exclusion.normalized_weight,
            }
            for exclusion in aggregate.exclusions
        ],
    }


def _scoring_breakdown_view(report: Report) -> dict[str, object]:
    """The report score's arithmetic, with each excluded criterion's reason attached.

    The reason is the point of joining these here: "기준 D (25%) — 제외" tells a reviewer a
    quarter of the interview is missing from the number but not why, and the item already knows
    (``uncertainty`` and ``assessment_state``).
    """
    reasons: dict[str, dict[str, object]] = {
        str(item.criterion_id): {
            "criterion_name": item.criterion_name,
            "assessment_state": item.assessment_state.value,
            "reason": item.uncertainty,
        }
        for item in report.items
    }
    aggregate = report.criterion_aggregate
    return {
        "numerator": aggregate.numerator,
        "denominator": aggregate.denominator,
        "contributions": [
            {
                "key": contribution.key,
                "score": contribution.score,
                "weight": contribution.weight,
                "normalized_weight": contribution.normalized_weight,
                "contribution": contribution.contribution,
                **reasons.get(contribution.key, {}),
            }
            for contribution in aggregate.contributions
        ],
        "exclusions": [
            {
                "key": exclusion.key,
                "weight": exclusion.weight,
                "normalized_weight": exclusion.normalized_weight,
                **reasons.get(exclusion.key, {}),
            }
            for exclusion in aggregate.exclusions
        ],
    }


def _report_view(report: Report, reviews: tuple[HumanReview, ...]) -> dict[str, object]:
    requirement_overrides = {
        review.target_id: review
        for review in sorted(reviews, key=lambda item: item.created_at)
        if review.review_type is ReviewType.REQUIREMENT_OVERRIDE
    }
    return {
        "report_id": report.report_id,
        "report_version": report.version,
        "status": report.status.value,
        "summary": report.summary,
        "ai_original_immutable": True,
        "overall_score": report.overall_score,
        "communication_score": report.communication_score,
        "communication_scored_criteria_count": (report.communication_aggregate.scored_count),
        # Sent beside the score so a reviewer reading 82 also sees that three criteria
        # were never scored, instead of reading it as a verdict on the whole interview.
        "unscored_criteria_count": len(report.items) - len(report.scored_items),
        # The arithmetic behind `overall_score`, so the console can render it rather than assert
        # it. `denominator` is the part that cannot be inferred: 82 out of the whole interview
        # and 82 out of the 70% of it that could be judged are different claims, and only the
        # divisor distinguishes them.
        "scoring_breakdown": _scoring_breakdown_view(report),
        "items": [
            {
                "report_item_id": item.report_item_id,
                "criterion_id": item.criterion_id,
                "criterion_name": item.criterion_name,
                "assessment_state": item.assessment_state.value,
                "observation": item.observation,
                "rationale": item.rationale,
                "uncertainty": item.uncertainty,
                "follow_up_question": item.follow_up_question,
                "average_score": report.score_for(item),
                # What this criterion counted for, and the same arithmetic one level down: the
                # reviewer opening a criterion asks the same "why this number" question the
                # report score already answers.
                "criterion_weight": item.criterion_weight,
                "axis_breakdown": _aggregate_view(report.axis_aggregate_for(item)),
                "axis_assessments": [
                    {
                        "axis": axis.axis,
                        "label": axis.label,
                        "score": axis.score,
                        "rationale": axis.rationale,
                        "quoted_evidence_ids": list(axis.quoted_evidence_ids),
                        "weight": item.axis_weights.get(axis.axis),
                    }
                    for axis in item.axis_assessments
                ],
                "evidence": [
                    {
                        "evidence_id": evidence.evidence_id,
                        "answer_turn_id": evidence.answer_turn_id,
                        "transcript_segment_id": evidence.transcript_segment_id,
                        "video_start_ms": evidence.video_start_ms,
                        "video_end_ms": evidence.video_end_ms,
                        "observation": evidence.observation,
                        "rationale": evidence.rationale,
                        "sufficiency": evidence.sufficiency.value,
                    }
                    for evidence in item.evidence
                ],
            }
            for item in report.items
        ],
        "requirement_assessments": [
            {
                "requirement_assessment_id": item.requirement_assessment_id,
                "job_requirement_id": item.job_requirement_id,
                "requirement_type": item.requirement_type,
                "statement": item.statement,
                "status": item.status.value,
                "rationale": item.rationale,
                "confidence": item.confidence,
                "evidence": [
                    {
                        "evidence_id": evidence.evidence_id,
                        "source_kind": evidence.source_kind,
                        "source_type": evidence.source_type,
                        "excerpt": evidence.excerpt,
                        "locator": dict(evidence.locator),
                        "relation": evidence.relation.value,
                        "explanation": evidence.explanation,
                    }
                    for evidence in item.evidence
                ],
                "human_override": (
                    {
                        "status": requirement_overrides[item.requirement_assessment_id].value.get(
                            "requirement_status"
                        ),
                        "reason": requirement_overrides[item.requirement_assessment_id].reason,
                        "created_at": requirement_overrides[
                            item.requirement_assessment_id
                        ].created_at,
                    }
                    if item.requirement_assessment_id in requirement_overrides
                    else None
                ),
            }
            for item in report.requirement_assessments
        ],
        "human_reviews": [_review_view(review) for review in reviews],
    }


def _deletion_view(manifest: DeletionManifest) -> dict[str, object]:
    return {
        "deletion_request_id": manifest.deletion_request_id,
        "manifest_id": manifest.manifest_id,
        "status": manifest.status.value,
        "expected_targets": len(manifest.targets),
        "verified_targets": manifest.verified_targets,
        "targets": [
            {
                "target_id": target.target_id,
                "owner_lane": target.owner_lane,
                "store": target.store,
                "target_type": target.target_type,
                "status": target.status.value,
                "attempts": target.attempts,
                "verified_at": target.verified_at,
                "error_code": target.error_code,
            }
            for target in manifest.targets
        ],
    }


def create_company_router(
    *,
    principal_provider: PrincipalProvider,
    repository: ReportingRepository,
    audit: AuditAppender,
    clock: Clock,
    deletion_service: DeletionService,
    playback: ScopedPlaybackLocator,
    rationale_provider: QuestionRationaleProvider | None = None,
    invitations: InvitationDecisionWriter | InvitationStateAdvancer | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1")
    reviews = ReviewService(repository)
    timeline = TimelineService(
        repository,
        rationale_provider=rationale_provider,
    )

    def company_scope(
        request: Request,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> CompanyScope:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        try:
            principal = principal_provider.get_company_principal(
                authorization.removeprefix("Bearer ").strip()
            )
        except PrincipalNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED) from error
        request_id_text = request.headers.get("x-request-id")
        request_id = UUID(request_id_text) if request_id_text else UUID(int=0)
        return CompanyScope(
            principal=principal,
            context=TenantContext(
                company_id=principal.company_id,
                actor_type=ActorType.COMPANY_USER,
                actor_id=principal.company_user_id,
                request_id=request_id,
                trace_id=request.headers.get("x-trace-id", "trace-reporting"),
            ),
        )

    Scope = Annotated[CompanyScope, Depends(company_scope)]
    IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=8)]

    @router.get(
        "/interview-sessions/{session_id}/report",
        operation_id="getInterviewReport",
    )
    def get_report(
        session_id: UUID,
        scope: Scope,
        response: Response,
    ) -> dict[str, object]:
        report = repository.get_report_for_session(scope.context, session_id)
        if report is None:
            response.status_code = status.HTTP_202_ACCEPTED
            return {"status": "queued", "retryable": True, "message": None}
        audit.append(
            scope.context,
            action="report.view",
            resource_type="report",
            resource_id=report.report_id,
            result="allowed",
            metadata={"report_version": report.version},
        )
        return _report_view(report, repository.list_reviews(scope.context, report.report_id))

    @router.get(
        "/interview-sessions/{session_id}/timeline",
        operation_id="getInterviewTimeline",
    )
    # No query parameter: see `TimelineService.project`. A free-text filter over answer text
    # here would be recorded verbatim in the load balancer's access log.
    def get_timeline(
        session_id: UUID,
        scope: Scope,
    ) -> dict[str, object]:
        entries = timeline.project(scope.context, session_id=session_id)
        assets = repository.list_recording_assets(scope.context, session_id)
        locator = playback.create(
            scope.context,
            asset=assets[-1] if assets else None,
            now=clock.now(),
        )
        audit.append(
            scope.context,
            action="timeline.view",
            resource_type="interview_session",
            resource_id=session_id,
            result="allowed",
            metadata={"entry_count": len(entries)},
        )
        return {
            "entries": [
                {
                    "entry_id": item.entry_id,
                    "entry_type": item.entry_type,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "text": item.text,
                    "technical_failure": item.technical_failure,
                    "question_rationale": (
                        {
                            "criterion_id": (item.question_rationale.criterion_id),
                            "interview_stage": (item.question_rationale.interview_stage),
                            "verification_target_type": (
                                item.question_rationale.verification_target_type
                            ),
                            "objective": item.question_rationale.objective,
                            "question_type": (item.question_rationale.question_type),
                            "retrieval_version": (item.question_rationale.retrieval_version),
                            "generation_version": (item.question_rationale.generation_version),
                            "policy_result": (item.question_rationale.policy_result),
                            "source_references": [
                                {
                                    "source_id": source.source_id,
                                    "source_type": source.source_type,
                                    "locator": source.locator,
                                    "excerpt": source.excerpt,
                                }
                                for source in item.question_rationale.source_references
                            ],
                        }
                        if item.question_rationale is not None
                        else None
                    ),
                }
                for item in entries
            ],
            "playback": {
                "url": locator.url,
                "expires_at": locator.expires_at,
                "status": locator.status,
            },
        }

    @router.post(
        "/reports/{report_id}/items/{report_item_id}/reviews",
        status_code=201,
        operation_id="createHumanAssessmentReview",
    )
    def create_assessment_review(
        report_id: UUID,
        report_item_id: UUID,
        body: HumanAssessmentReviewCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> dict[str, object]:
        del idempotency_key
        review = reviews.override_assessment(
            scope.context,
            report_id=report_id,
            report_item_id=report_item_id,
            assessment_state=body.assessment_state,
            reason=body.reason,
            occurred_at=clock.now(),
        )
        audit.append(
            scope.context,
            action="report.assessment_review.create",
            resource_type="human_review",
            resource_id=review.human_review_id,
            result="created",
            metadata={"report_id": str(report_id), "report_item_id": str(report_item_id)},
        )
        return _review_view(review)

    @router.post(
        "/reports/{report_id}/requirements/{requirement_assessment_id}/reviews",
        status_code=201,
        operation_id="createHumanRequirementReview",
    )
    def create_requirement_review(
        report_id: UUID,
        requirement_assessment_id: UUID,
        body: HumanRequirementReviewCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> dict[str, object]:
        del idempotency_key
        try:
            review = reviews.override_requirement(
                scope.context,
                report_id=report_id,
                requirement_assessment_id=requirement_assessment_id,
                requirement_status=body.requirement_status,
                reason=body.reason,
                occurred_at=clock.now(),
            )
        except LookupError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        audit.append(
            scope.context,
            action="report.requirement_review.create",
            resource_type="human_review",
            resource_id=review.human_review_id,
            result="created",
            metadata={
                "report_id": str(report_id),
                "requirement_assessment_id": str(requirement_assessment_id),
            },
        )
        return _review_view(review)

    @router.post(
        "/interview-sessions/{session_id}/review-artifacts",
        status_code=201,
        operation_id="createReviewArtifact",
    )
    def create_review_artifact(
        session_id: UUID,
        body: ReviewArtifactCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> dict[str, object]:
        del idempotency_key
        report = repository.get_report_for_session(scope.context, session_id)
        if report is None:
            raise HTTPException(status_code=404)
        review = reviews.create_artifact(
            scope.context,
            report_id=report.report_id,
            target_id=body.target_id,
            review_type=ReviewType.NOTE,
            value=body.value,
            occurred_at=clock.now(),
        )
        audit.append(
            scope.context,
            action="review_artifact.create",
            resource_type="human_review",
            resource_id=review.human_review_id,
            result="created",
            metadata={"review_type": body.review_type, "target_id": str(body.target_id)},
        )
        return _review_view(review)

    @router.post(
        "/invitations/{invitation_id}/final-decisions",
        status_code=201,
        response_model=FinalDecisionResponse,
        operation_id="recordHumanFinalDecision",
    )
    def record_final_decision(
        invitation_id: UUID,
        body: FinalDecisionCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> dict[str, object] | JSONResponse:
        del idempotency_key
        report = repository.get_report_for_invitation(scope.context, invitation_id)
        if report is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "code": "REPORT_NOT_AVAILABLE",
                    "detail": "Final report is not available.",
                },
            )
        if invitations is None or not hasattr(invitations, "move_to_recruiting_stage"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="recruiting stage decision writer is unavailable",
            )
        occurred_at = clock.now()
        decision_writer = cast(InvitationDecisionWriter, invitations)
        try:
            stage = decision_writer.move_to_recruiting_stage(
                scope.context,
                invitation_id,
                recruiting_stage_id=body.recruiting_stage_id,
                expected_pipeline_version=body.expected_pipeline_version,
            )
        except ValueError as error:
            # The one-applicant move validates its stage and optimistic-lock version before
            # persisting. A stale report therefore remains a conflict with no partial write.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        review = reviews.record_recruiting_stage_decision(
            scope.context,
            report_id=report.report_id,
            invitation_id=invitation_id,
            recruiting_stage_id=stage.recruiting_stage_id,
            recruiting_stage_name=stage.recruiting_stage_name,
            occurred_at=occurred_at,
        )
        # Do not translate failures after the pipeline write to a 4xx response: the shared
        # request middleware rolls back on an exception/5xx, keeping these three writes atomic.
        invitation_state = close_invitation_review(
            decision_writer,
            scope.context,
            invitation_id=invitation_id,
            occurred_at=occurred_at,
        )
        audit.append(
            scope.context,
            action="final_decision.create",
            resource_type="human_review",
            resource_id=review.human_review_id,
            result="created",
            metadata={
                "invitation_id": str(invitation_id),
                "recruiting_stage_id": str(stage.recruiting_stage_id),
                "recruiting_stage_name": stage.recruiting_stage_name,
                "invitation_state": invitation_state or "unchanged",
            },
        )
        return {
            "human_review": _review_view(review),
            "invitation_id": stage.invitation_id,
            "position_id": stage.position_id,
            "recruiting_stage_id": stage.recruiting_stage_id,
            "recruiting_stage_name": stage.recruiting_stage_name,
            "pipeline_row_version": stage.pipeline_row_version,
            "invitation_state": invitation_state,
        }

    @router.post(
        "/privacy/deletion-requests",
        status_code=202,
        operation_id="createDeletionRequest",
    )
    def create_deletion_request(
        body: DeletionRequestCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> dict[str, object]:
        del idempotency_key
        try:
            _, manifest = deletion_service.request(
                scope.context,
                scope_type=body.scope_type,
                scope_id=body.scope_id,
                reason=body.reason,
                policy_snapshot={
                    "source": "manual_request",
                    "scope_type": body.scope_type,
                },
                occurred_at=clock.now(),
            )
        except LookupError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="deletion scope not found",
            ) from error
        audit.append(
            scope.context,
            action="privacy_deletion.request",
            resource_type="deletion_manifest",
            resource_id=manifest.manifest_id,
            result="accepted",
            metadata={
                "scope_type": body.scope_type,
                "scope_id": str(body.scope_id),
                "expected_targets": len(manifest.targets),
            },
        )
        return _deletion_view(manifest)

    @router.get(
        "/privacy/deletion-requests/{deletion_request_id}",
        operation_id="getDeletionRequest",
    )
    def get_deletion_request(
        deletion_request_id: UUID,
        scope: Scope,
    ) -> dict[str, object]:
        _, manifest = repository.get_deletion(scope.context, deletion_request_id)
        audit.append(
            scope.context,
            action="privacy_deletion.status_view",
            resource_type="deletion_manifest",
            resource_id=manifest.manifest_id,
            result="allowed",
            metadata={
                "verified_targets": manifest.verified_targets,
                "expected_targets": len(manifest.targets),
            },
        )
        return _deletion_view(manifest)

    return router


def create_lane_d_runtime(
    *,
    principal_provider: PrincipalProvider,
    repository: ReportingRepository,
    audit: AuditAppender,
    clock: Clock,
    deletion_service: DeletionService | None = None,
    rationale_provider: QuestionRationaleProvider | None = None,
    invitations: InvitationStateAdvancer | None = None,
) -> LaneDRuntime:
    active_repository = repository
    app = FastAPI(title="Interview Evidence Reporting")

    @app.exception_handler(TenantScopedReportingNotFound)
    async def reporting_not_found(
        _request: Request,
        _error: TenantScopedReportingNotFound,
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "resource not found"})

    service = deletion_service or DeletionService(active_repository)
    app.include_router(
        create_company_router(
            principal_provider=principal_provider,
            repository=active_repository,
            audit=audit,
            clock=clock,
            deletion_service=service,
            playback=ScopedPlaybackLocator(),
            rationale_provider=rationale_provider,
            invitations=invitations,
        )
    )
    return LaneDRuntime(
        app=app,
        repository=active_repository,
        deletion_service=service,
    )


def create_lane_d_app(
    *,
    principal_provider: PrincipalProvider,
    repository: ReportingRepository,
    audit: AuditAppender,
    clock: Clock,
    rationale_provider: QuestionRationaleProvider | None = None,
) -> FastAPI:
    return create_lane_d_runtime(
        principal_provider=principal_provider,
        repository=repository,
        audit=audit,
        clock=clock,
        rationale_provider=rationale_provider,
    ).app
