from datetime import date, datetime
from typing import Annotated, Literal, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from interview_evidence.company_management.adapters.company_auth import CompanyAuthAdapter
from interview_evidence.company_management.application.company_service import CompanyService
from interview_evidence.company_management.application.criteria_service import CriteriaService
from interview_evidence.company_management.application.hiring_service import (
    ApplicantInvitationInput,
    ApplicantPipelineMove,
    HiringService,
)
from interview_evidence.company_management.application.interviewer_service import (
    InterviewerProfileService,
)
from interview_evidence.company_management.application.invitation_template_service import (
    InvitationTemplateService,
    LogoTooLargeError,
    UnsupportedLogoTypeError,
)
from interview_evidence.company_management.domain.company import (
    InterviewerProfile,
    InterviewerTone,
    Position,
    PositionStatus,
    StalePositionVersionError,
)
from interview_evidence.company_management.domain.criteria import (
    CompetencyModelVersion,
    PublishedVersionImmutableError,
    StaleCriterionVersionError,
)
from interview_evidence.company_management.domain.hiring import (
    Invitation,
    InvitationStateError,
    RecruitingStage,
)
from interview_evidence.company_management.repositories.postgres import (
    TenantScopedResourceNotFound,
)
from interview_evidence.company_management.workers.invitation_email import (
    InvitationEmailCommand,
    InvitationEmailHandler,
    format_deadline,
)
from interview_evidence.shared.audit import AuditAppender
from interview_evidence.shared.email_templates import (
    MAX_GUIDE_LINES,
    InvitationEmailTemplate,
)
from interview_evidence.shared.interview_level import (
    DEFAULT_INTERVIEW_LEVEL,
    InterviewLevel,
)
from interview_evidence.shared.security.principals import (
    CompanyPrincipal,
    PrincipalNotFoundError,
)
from interview_evidence.shared.submission_materials import (
    DEFAULT_SUBMISSION_REQUIREMENTS,
    SubmissionRequirement,
)
from interview_evidence.shared.tenant import TenantContext


class CompanyUserView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_user_id: UUID
    company_id: UUID
    email: str
    status: str


class PositionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=20_000)
    role_type: str | None = Field(default=None, max_length=100)
    headcount: int | None = Field(default=None, ge=1, le=10_000)
    applicant_capacity: int | None = Field(default=None, ge=1, le=100_000)
    interview_capacity: int | None = Field(default=None, ge=1, le=400)
    interview_at: datetime | None = None
    recruitment_start_at: date | None = None
    recruitment_end_at: date | None = None
    submission_requirements: tuple[SubmissionRequirement, ...] = DEFAULT_SUBMISSION_REQUIREMENTS

    @field_validator("interview_at")
    @classmethod
    def interview_time_must_include_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("interview_at must include a timezone")
        return value


class PositionView(PositionCreate):
    # Historical rows may exceed the current guaranteed reservation ceiling.
    interview_capacity: int | None = Field(default=None, ge=1, le=10_000)
    position_id: UUID
    status: str
    row_version: int
    created_at: datetime


class PositionUpdate(PositionCreate):
    status: PositionStatus


class PositionPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PositionView]
    next_cursor: str | None = None


class InterviewerProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    tone: InterviewerTone
    voice_id: str = Field(min_length=1, max_length=100)


class InterviewerProfileView(InterviewerProfileCreate):
    interviewer_profile_id: UUID
    row_version: int
    created_at: datetime


class InterviewerProfilePage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[InterviewerProfileView]
    next_cursor: str | None = None


class InterviewerPersonaDefinitionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    tone: InterviewerTone
    voice_id: str = Field(min_length=1, max_length=100)


class EvaluationCriterionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[A-Z0-9_-]{2,40}$")
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    weight: float = Field(ge=0)
    verification_guide: "CriterionVerificationGuideInput"
    abstain_guidance: str = Field(min_length=1)
    common_questions: tuple[str, ...] = ()
    required: bool


class JobRequirementInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_type: str = Field(pattern=r"^(required|preferred)$")
    statement: str = Field(min_length=1, max_length=4000)
    priority: int = Field(ge=1, le=5)
    criterion_code: str = Field(pattern=r"^[A-Z0-9_-]{2,40}$")


class CriterionVerificationGuideInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observable_dimensions: tuple[str, ...] = Field(min_length=1, max_length=12)
    strong_answer_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    weak_answer_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    follow_up_directions: tuple[str, ...] = Field(min_length=1, max_length=8)
    max_follow_ups: int = Field(ge=0, le=3)
    time_budget_seconds: int = Field(ge=60, le=1800)


class CompetencyModelVersionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_requirements: tuple[JobRequirementInput, ...] = Field(min_length=1, max_length=50)
    criteria: tuple[EvaluationCriterionInput, ...] = Field(min_length=1)
    prohibited_topics: tuple[str, ...]
    interview_duration_minutes: Literal[30]
    interview_level: InterviewLevel = DEFAULT_INTERVIEW_LEVEL
    #: Declared here because `model_config` forbids extras: the console posts this field, and
    #: without it the whole publish request is rejected as `extra_forbidden` rather than the
    #: weights merely being dropped. The five sliders are the only way a company sets them.
    #:
    #: Deliberately unvalidated beyond its type, unlike the criterion total below.
    #: `CompetencyModelVersion` refuses an unknown key, a partial mapping, a negative weight and
    #: a total that is not 100, and the route turns that ValueError into a 422 carrying the
    #: reason. Four rules restated here would be four more places to drift, and a drifted weight
    #: rule produces a wrong score rather than an error.
    axis_weights: dict[str, float] = Field(default_factory=dict)
    persona_definition: InterviewerPersonaDefinitionInput | None = None

    @model_validator(mode="after")
    def criterion_weights_total_one_hundred(self) -> "CompetencyModelVersionCreate":
        if abs(sum(criterion.weight for criterion in self.criteria) - 100) > 0.001:
            raise ValueError("criterion weights must total 100")
        return self


class CompetencyModelVersionView(CompetencyModelVersionCreate):
    job_requirements: tuple[JobRequirementInput, ...] = Field(max_length=50)
    competency_model_version_id: UUID
    position_id: UUID
    version_number: int
    status: str
    row_version: int
    published_at: datetime | None

    @model_validator(mode="after")
    def criterion_weights_total_one_hundred(self) -> "CompetencyModelVersionView":
        """Overrides the request rule to nothing: a response describes what *is* stored.

        This class inherits the field shape from the request model, which is convenient and was
        also a defect -- the "weights total 100" rule came with it and ran while *serialising*.
        Nothing enforced a total before ``m_013``, so versions stored with any total are the
        normal case, and refusing to serialise them turned the criteria list into a 500 rather
        than a list. Reading them is safe: ``scoring.aggregate`` divides by whatever the weights
        total, so 30/25/20 already scores as the 40%/33%/27% the recruiter set.

        The rule itself is not weakened. It still rejects a submitted version here, and
        ``CompetencyModelVersion.create`` rejects one that reaches the domain by any other
        route, so nothing new can be stored with a total that is not 100.
        """
        return self


class CompetencyModelVersionPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CompetencyModelVersionView]
    next_cursor: str | None = None


class ApplicantInvitationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    display_name: str = Field(min_length=1, max_length=200)


class InvitationBatchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicants: tuple[ApplicantInvitationRequest, ...] = Field(min_length=1, max_length=1000)
    expires_at: datetime
    delivery_method: Literal["email", "manual_link"] = "email"


class InvitationView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invitation_id: UUID
    position_id: UUID
    competency_model_version_id: UUID
    applicant_email: str
    applicant_display_name: str | None = None
    status: str
    expires_at: datetime
    row_version: int
    analysis_status: str | None = None
    interview_status: str | None = None
    report_status: str | None = None
    interview_session_id: UUID | None = None
    #: The weighted score from the report, so a position's applicant list can rank on it.
    #: None until a report exists, and None when nothing in it could be scored -- never zero,
    #: which would sort an applicant we could not assess below one who answered badly.
    overall_score: int | None = None
    #: How many of the position's criteria the score covers. Sent with the score because two
    #: applicants whose interviews reached different criteria do not have comparable numbers,
    #: and a ranked column has to be able to say so.
    scored_criteria_count: int | None = None
    total_criteria_count: int | None = None
    recruiting_stage_id: UUID | None = None
    pipeline_row_version: int = 1


class RecruitingStageView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recruiting_stage_id: UUID
    position_id: UUID
    name: str
    sort_order: int
    row_version: int


class RecruitingStagePage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RecruitingStageView]


class RecruitingStageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=40)


class RecruitingStageUpdate(RecruitingStageCreate):
    pass


class RecruitingStageReorder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordered_stage_ids: tuple[UUID, ...] = Field(min_length=1, max_length=20)


class RecruitingStageDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replacement_stage_id: UUID


class ApplicantPipelineMoveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invitation_id: UUID
    expected_version: int = Field(ge=1)


class ApplicantPipelineMoveBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_stage_id: UUID
    applicants: tuple[ApplicantPipelineMoveInput, ...] = Field(
        min_length=1,
        max_length=1000,
    )


class ApplicantPipelineAssignmentView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invitation_id: UUID
    recruiting_stage_id: UUID
    pipeline_row_version: int


class ApplicantPipelineAssignmentPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ApplicantPipelineAssignmentView]


class ApplicantRecruitingStateView(BaseModel):
    """Live pipeline state used by reports, lists, and the kanban board."""

    model_config = ConfigDict(extra="forbid")

    invitation_id: UUID
    position_id: UUID
    recruiting_stage_id: UUID
    pipeline_row_version: int
    stages: list[RecruitingStageView]


class InvitationSessionSnapshot(Protocol):
    @property
    def interview_session_id(self) -> UUID: ...

    @property
    def state(self) -> str: ...


class InvitationSessionResolver(Protocol):
    def find_session_for_invitation(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
    ) -> InvitationSessionSnapshot | None: ...


class InvitationReviewSnapshot(Protocol):
    @property
    def report_status(self) -> str: ...

    @property
    def overall_score(self) -> int | None: ...

    @property
    def scored_criteria_count(self) -> int: ...

    @property
    def total_criteria_count(self) -> int: ...


class InvitationReviewResolver(Protocol):
    def get_invitation_review(
        self,
        context: TenantContext,
        *,
        invitation_id: UUID,
    ) -> InvitationReviewSnapshot | None: ...


class InvitationPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[InvitationView]
    next_cursor: str | None = None


class InvitationAccessLinkView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invitation_id: UUID
    applicant_email: str
    applicant_display_name: str | None = None
    access_url: str
    expires_at: datetime


class InvitationBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_count: int
    rejected_count: int
    invitations: list[InvitationView]
    access_links: list[InvitationAccessLinkView]


class InvitationEmailTemplateInput(BaseModel):
    """The editable template as the console sends it.

    ``logo_url`` is deliberately absent: the server derives it from the uploaded logo so
    a client cannot point every invitation at a host it controls.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=200)
    headline: str = Field(min_length=1, max_length=200)
    intro: str = Field(min_length=1, max_length=2_000)
    guides: tuple[str, ...] = Field(default=(), max_length=MAX_GUIDE_LINES)
    cta_label: str = Field(min_length=1, max_length=40)
    outro: str = Field(default="", max_length=1_000)
    footer: str = Field(default="", max_length=300)
    brand_color: str = Field(default="#5966ce", pattern=r"^#[0-9a-fA-F]{6}$")
    use_applicant_name: bool = True
    emphasize_deadline: bool = True
    show_security_notice: bool = True

    def to_template(self) -> InvitationEmailTemplate:
        return InvitationEmailTemplate.model_validate(self.model_dump())


class InvitationEmailTemplateView(InvitationEmailTemplateInput):
    logo_url: str | None = None
    #: False when this position inherits the company-wide template.
    is_position_override: bool = False


class InvitationEmailPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    html_body: str


class CompanyLogoView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logo_url: str
    content_type: str
    byte_size: int


class CompanyRequestScope(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    principal: CompanyPrincipal
    context: TenantContext


def create_company_router(
    *,
    auth: CompanyAuthAdapter,
    company_service: CompanyService,
    criteria_service: CriteriaService,
    interviewer_service: InterviewerProfileService,
    hiring_service: HiringService,
    template_service: InvitationTemplateService,
    audit: AuditAppender,
    invitation_email: InvitationEmailHandler | None = None,
    applicant_access_base_url: str = "https://applicant.local/access",
    interview_sessions: InvitationSessionResolver | None = None,
    invitation_reviews: InvitationReviewResolver | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def company_scope(
        request: Request,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> CompanyRequestScope:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        request_id = _optional_uuid(request.headers.get("x-request-id"))
        try:
            principal, context = auth.authenticate(
                authorization.removeprefix("Bearer ").strip(),
                request_id=request_id,
                trace_id=request.headers.get("x-trace-id"),
            )
        except PrincipalNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED) from error
        company_service.get_current_user(context, principal)
        return CompanyRequestScope(principal=principal, context=context)

    Scope = Annotated[CompanyRequestScope, Depends(company_scope)]
    IdempotencyKey = Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=16, max_length=128),
    ]
    IfMatchVersion = Annotated[
        int,
        Header(alias="If-Match-Version", ge=1),
    ]

    @router.get("/me", response_model=CompanyUserView, operation_id="getCurrentCompanyUser")
    def get_current_company_user(scope: Scope) -> CompanyUserView:
        snapshot = company_service.get_current_user(scope.context, scope.principal)
        return CompanyUserView.model_validate(snapshot.model_dump())

    @router.get(
        "/positions",
        response_model=PositionPage,
        operation_id="listPositions",
    )
    def list_positions(
        scope: Scope,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> PositionPage:
        del cursor
        positions = company_service.list_positions(scope.context)[:limit]
        return PositionPage(items=[_position_view(position) for position in positions])

    @router.post(
        "/positions",
        response_model=PositionView,
        status_code=status.HTTP_201_CREATED,
        operation_id="createPosition",
    )
    def create_position(
        body: PositionCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> PositionView:
        position = company_service.create_position(
            scope.context,
            scope.principal,
            title=body.title,
            description=body.description,
            role_type=body.role_type,
            headcount=body.headcount,
            applicant_capacity=body.applicant_capacity,
            interview_capacity=body.interview_capacity,
            interview_at=body.interview_at,
            recruitment_start_at=body.recruitment_start_at,
            recruitment_end_at=body.recruitment_end_at,
            submission_requirements=body.submission_requirements,
            idempotency_key=idempotency_key,
        )
        audit.append(
            scope.context,
            action="position.created",
            resource_type="position",
            resource_id=position.position_id,
            result="success",
            metadata={"row_version": position.row_version},
        )
        return _position_view(position)

    @router.get(
        "/positions/{position_id}",
        response_model=PositionView,
        operation_id="getPosition",
    )
    def get_position(position_id: UUID, scope: Scope) -> PositionView:
        try:
            position = company_service.get_position(scope.context, position_id)
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return _position_view(position)

    @router.patch(
        "/positions/{position_id}",
        response_model=PositionView,
        operation_id="updatePosition",
    )
    def update_position(
        position_id: UUID,
        body: PositionUpdate,
        scope: Scope,
        if_match_version: IfMatchVersion,
    ) -> PositionView:
        try:
            position = company_service.update_position(
                scope.context,
                position_id=position_id,
                expected_version=if_match_version,
                title=body.title,
                description=body.description,
                role_type=body.role_type,
                headcount=body.headcount,
                applicant_capacity=body.applicant_capacity,
                interview_capacity=body.interview_capacity,
                interview_at=body.interview_at,
                recruitment_start_at=body.recruitment_start_at,
                recruitment_end_at=body.recruitment_end_at,
                submission_requirements=body.submission_requirements,
                status=body.status,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except StalePositionVersionError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            ) from error
        audit.append(
            scope.context,
            action="position.updated",
            resource_type="position",
            resource_id=position.position_id,
            result="success",
            metadata={
                "row_version": position.row_version,
                "status": position.status.value,
            },
        )
        return _position_view(position)

    @router.get(
        "/interviewer-profiles",
        response_model=InterviewerProfilePage,
        operation_id="listInterviewerProfiles",
    )
    def list_interviewer_profiles(
        scope: Scope,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> InterviewerProfilePage:
        del cursor
        profiles = interviewer_service.list(scope.context)[:limit]
        return InterviewerProfilePage(
            items=[_interviewer_profile_view(profile) for profile in profiles]
        )

    @router.post(
        "/interviewer-profiles",
        response_model=InterviewerProfileView,
        status_code=status.HTTP_201_CREATED,
        operation_id="createInterviewerProfile",
    )
    def create_interviewer_profile(
        body: InterviewerProfileCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> InterviewerProfileView:
        profile = interviewer_service.create(
            scope.context,
            name=body.name,
            tone=body.tone,
            voice_id=body.voice_id,
            idempotency_key=idempotency_key,
        )
        audit.append(
            scope.context,
            action="interviewer_profile.created",
            resource_type="interviewer_profile",
            resource_id=profile.interviewer_profile_id,
            result="success",
            metadata={"row_version": profile.row_version, "tone": profile.tone.value},
        )
        return _interviewer_profile_view(profile)

    @router.post(
        "/positions/{position_id}/competency-model-versions",
        response_model=CompetencyModelVersionView,
        status_code=status.HTTP_201_CREATED,
        operation_id="createCompetencyModelVersion",
    )
    def create_competency_model_version(
        position_id: UUID,
        body: CompetencyModelVersionCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> CompetencyModelVersionView:
        try:
            version = criteria_service.create_version(
                scope.context,
                position_id=position_id,
                job_requirements=tuple(item.model_dump() for item in body.job_requirements),
                criteria=tuple(item.model_dump() for item in body.criteria),
                prohibited_topics=body.prohibited_topics,
                interview_duration_minutes=body.interview_duration_minutes,
                interview_level=body.interview_level,
                axis_weights=body.axis_weights,
                persona_definition=(
                    body.persona_definition.model_dump(mode="json")
                    if body.persona_definition is not None
                    else None
                ),
                idempotency_key=idempotency_key,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=_domain_error_detail(error),
            ) from error
        audit.append(
            scope.context,
            action="criterion_version.created",
            resource_type="competency_model_version",
            resource_id=version.competency_model_version_id,
            result="success",
            metadata={"row_version": version.row_version},
        )
        return _criterion_view(version)

    @router.get(
        "/positions/{position_id}/competency-model-versions",
        response_model=CompetencyModelVersionPage,
        operation_id="listCompetencyModelVersions",
    )
    def list_competency_model_versions(
        position_id: UUID,
        scope: Scope,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> CompetencyModelVersionPage:
        del cursor
        try:
            versions = criteria_service.list_versions(scope.context, position_id)[:limit]
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return CompetencyModelVersionPage(items=[_criterion_view(version) for version in versions])

    @router.post(
        "/competency-model-versions/{version_id}/publish",
        response_model=CompetencyModelVersionView,
        operation_id="publishCompetencyModelVersion",
    )
    def publish_competency_model_version(
        version_id: UUID,
        scope: Scope,
        idempotency_key: IdempotencyKey,
        if_match_version: IfMatchVersion,
    ) -> CompetencyModelVersionView:
        del idempotency_key
        try:
            version = criteria_service.publish_version(
                scope.context,
                version_id=version_id,
                expected_version=if_match_version,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except (StaleCriterionVersionError, PublishedVersionImmutableError) as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        audit.append(
            scope.context,
            action="criterion_version.published",
            resource_type="competency_model_version",
            resource_id=version.competency_model_version_id,
            result="success",
            metadata={"row_version": version.row_version},
        )
        return _criterion_view(version)

    @router.get(
        "/recruiting-stages",
        response_model=RecruitingStagePage,
        operation_id="listRecruitingStages",
    )
    def list_recruiting_stages(
        scope: Scope,
        position_id: Annotated[UUID | None, Query()] = None,
    ) -> RecruitingStagePage:
        try:
            stages = hiring_service.list_recruiting_stages(
                scope.context,
                position_id,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return RecruitingStagePage(items=[_recruiting_stage_view(stage) for stage in stages])

    @router.get(
        "/invitations/{invitation_id}/recruiting-state",
        response_model=ApplicantRecruitingStateView,
        operation_id="getApplicantRecruitingState",
    )
    def get_applicant_recruiting_state(
        invitation_id: UUID,
        scope: Scope,
    ) -> ApplicantRecruitingStateView:
        try:
            state = hiring_service.get_applicant_recruiting_state(
                scope.context,
                invitation_id,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        stage_id = state.invitation.recruiting_stage_id
        if stage_id is None:  # Guard the HTTP contract even if a custom repository is broken.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="applicant recruiting stage is not initialized",
            )
        return ApplicantRecruitingStateView(
            invitation_id=state.invitation.invitation_id,
            position_id=state.invitation.position_id,
            recruiting_stage_id=stage_id,
            pipeline_row_version=state.invitation.pipeline_row_version,
            stages=[_recruiting_stage_view(stage) for stage in state.stages],
        )

    @router.post(
        "/positions/{position_id}/recruiting-stages",
        response_model=RecruitingStageView,
        status_code=status.HTTP_201_CREATED,
        operation_id="createRecruitingStage",
    )
    def create_recruiting_stage(
        position_id: UUID,
        body: RecruitingStageCreate,
        scope: Scope,
    ) -> RecruitingStageView:
        try:
            stage = hiring_service.create_recruiting_stage(
                scope.context,
                position_id=position_id,
                name=body.name,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        audit.append(
            scope.context,
            action="recruiting_stage.created",
            resource_type="recruiting_stage",
            resource_id=stage.recruiting_stage_id,
            result="success",
            metadata={"position_id": str(position_id)},
        )
        return _recruiting_stage_view(stage)

    @router.patch(
        "/positions/{position_id}/recruiting-stages/{stage_id}",
        response_model=RecruitingStageView,
        operation_id="updateRecruitingStage",
    )
    def update_recruiting_stage(
        position_id: UUID,
        stage_id: UUID,
        body: RecruitingStageUpdate,
        scope: Scope,
        if_match_version: IfMatchVersion,
    ) -> RecruitingStageView:
        try:
            stage = hiring_service.rename_recruiting_stage(
                scope.context,
                position_id=position_id,
                stage_id=stage_id,
                name=body.name,
                expected_version=if_match_version,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except InvitationStateError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        audit.append(
            scope.context,
            action="recruiting_stage.updated",
            resource_type="recruiting_stage",
            resource_id=stage_id,
            result="success",
            metadata={"position_id": str(position_id), "row_version": stage.row_version},
        )
        return _recruiting_stage_view(stage)

    @router.post(
        "/positions/{position_id}/recruiting-stages/reorder",
        response_model=RecruitingStagePage,
        operation_id="reorderRecruitingStages",
    )
    def reorder_recruiting_stages(
        position_id: UUID,
        body: RecruitingStageReorder,
        scope: Scope,
    ) -> RecruitingStagePage:
        try:
            stages = hiring_service.reorder_recruiting_stages(
                scope.context,
                position_id=position_id,
                ordered_stage_ids=body.ordered_stage_ids,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        audit.append(
            scope.context,
            action="recruiting_stage.reordered",
            resource_type="position",
            resource_id=position_id,
            result="success",
            metadata={"stage_count": len(stages)},
        )
        return RecruitingStagePage(items=[_recruiting_stage_view(stage) for stage in stages])

    @router.post(
        "/positions/{position_id}/recruiting-stages/{stage_id}/delete",
        response_model=RecruitingStagePage,
        operation_id="deleteRecruitingStage",
    )
    def delete_recruiting_stage(
        position_id: UUID,
        stage_id: UUID,
        body: RecruitingStageDelete,
        scope: Scope,
    ) -> RecruitingStagePage:
        try:
            stages = hiring_service.delete_recruiting_stage(
                scope.context,
                position_id=position_id,
                stage_id=stage_id,
                replacement_stage_id=body.replacement_stage_id,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        audit.append(
            scope.context,
            action="recruiting_stage.deleted",
            resource_type="recruiting_stage",
            resource_id=stage_id,
            result="success",
            metadata={"position_id": str(position_id)},
        )
        return RecruitingStagePage(items=[_recruiting_stage_view(stage) for stage in stages])

    @router.patch(
        "/positions/{position_id}/invitations/recruiting-stage",
        response_model=ApplicantPipelineAssignmentPage,
        operation_id="moveApplicantsToRecruitingStage",
    )
    def move_applicants_to_recruiting_stage(
        position_id: UUID,
        body: ApplicantPipelineMoveBatch,
        scope: Scope,
    ) -> ApplicantPipelineAssignmentPage:
        try:
            invitations = hiring_service.move_applicants(
                scope.context,
                position_id=position_id,
                target_stage_id=body.target_stage_id,
                moves=tuple(
                    ApplicantPipelineMove(
                        invitation_id=item.invitation_id,
                        expected_version=item.expected_version,
                    )
                    for item in body.applicants
                ),
                invitation_reviews=invitation_reviews,
                require_final_report=True,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except InvitationStateError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        audit.append(
            scope.context,
            action="applicant_pipeline.moved",
            resource_type="position",
            resource_id=position_id,
            result="success",
            metadata={
                "target_stage_id": str(body.target_stage_id),
                "applicant_count": len(invitations),
            },
        )
        return ApplicantPipelineAssignmentPage(
            items=[
                ApplicantPipelineAssignmentView(
                    invitation_id=invitation.invitation_id,
                    recruiting_stage_id=body.target_stage_id,
                    pipeline_row_version=invitation.pipeline_row_version,
                )
                for invitation in invitations
            ]
        )

    @router.get(
        "/positions/{position_id}/invitations",
        response_model=InvitationPage,
        operation_id="listInvitations",
    )
    def list_invitations(
        position_id: UUID,
        scope: Scope,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> InvitationPage:
        try:
            offset = int(cursor or "0")
            if offset < 0:
                raise ValueError
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid invitation cursor",
            ) from error
        try:
            invitations = hiring_service.list_invitations(scope.context, position_id)
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        page = invitations[offset : offset + limit]
        next_offset = offset + len(page)
        return InvitationPage(
            items=[
                _invitation_view(
                    invitation,
                    interview_sessions=interview_sessions,
                    invitation_reviews=invitation_reviews,
                    context=scope.context,
                )
                for invitation in page
            ],
            next_cursor=str(next_offset) if next_offset < len(invitations) else None,
        )

    @router.post(
        "/positions/{position_id}/invitations",
        response_model=InvitationBatchResult,
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="createInvitations",
    )
    def create_invitations(
        position_id: UUID,
        body: InvitationBatchCreate,
        scope: Scope,
        idempotency_key: IdempotencyKey,
    ) -> InvitationBatchResult:
        del idempotency_key
        try:
            issuances = hiring_service.issue_invitations(
                scope.context,
                position_id=position_id,
                applicants=tuple(
                    ApplicantInvitationInput(
                        email=applicant.email,
                        display_name=applicant.display_name,
                    )
                    for applicant in body.applicants
                ),
                expires_at=body.expires_at,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        audit.append(
            scope.context,
            action="invitation.batch_created",
            resource_type="position",
            resource_id=position_id,
            result="success",
            metadata={
                "accepted_count": len(issuances),
                "rejected_count": 0,
                "delivery_method": body.delivery_method,
            },
        )
        undeliverable: set[UUID] = set()
        link_issuances = list(issuances) if body.delivery_method == "manual_link" else []
        if body.delivery_method == "email" and invitation_email is not None:
            position = company_service.get_position(scope.context, position_id)
            template = template_service.resolve_for_sending(scope.context, position_id)
            company_name = template_service.company_name(scope.context)
            for issuance in issuances:
                try:
                    invitation_email.handle(
                        scope.context,
                        InvitationEmailCommand(
                            invitation_id=issuance.invitation.invitation_id,
                            applicant_ref=issuance.invitation.applicant_id,
                            company_name=company_name,
                            position_title=position.title,
                            deadline_text=format_deadline(issuance.invitation.expires_at),
                            template=template,
                            recipient_address=issuance.invitation.applicant_email,
                            invitation_url=_invitation_access_url(
                                applicant_access_base_url,
                                issuance.token.raw_token,
                            ),
                            applicant_display_name=issuance.invitation.applicant_display_name,
                        ),
                    )
                except Exception:  # noqa: BLE001 -- see below
                    # One recipient's rejection is not the batch's failure.
                    #
                    # This send is synchronous and inside the request, and the HTTP
                    # transaction middleware rolls back on any status >= 500. So an
                    # exception escaping here discarded every invitation in the batch --
                    # including the ones whose mail had already left. The applicant holds a
                    # link, and the row the token resolves against no longer exists.
                    #
                    # SES makes that the ordinary case rather than a rare one: in sandbox it
                    # rejects any unverified recipient, so one address nobody confirmed took
                    # down the whole batch. The exception type cannot be narrowed usefully --
                    # `AwsSesEmailSender` wraps everything botocore raises in
                    # `AwsAdapterError`, but other adapter failures are handled the same way.
                    #
                    # The invitation is kept. Its token is valid, the reviewer can see who
                    # was not reached, and resending is a second call to this endpoint --
                    # whereas a dropped invitation cannot be recovered from anywhere.
                    undeliverable.add(issuance.invitation.invitation_id)
                    link_issuances.append(issuance)
                    audit.append(
                        scope.context,
                        action="invitation.email_failed",
                        resource_type="invitation",
                        resource_id=issuance.invitation.invitation_id,
                        result="failure",
                        # No address and no token: this record is written on a path that
                        # exists because a delivery was refused, and the refused address is
                        # exactly what must not be copied into an audit row.
                        metadata={},
                    )
        if body.delivery_method == "manual_link":
            audit.append(
                scope.context,
                action="invitation.manual_links_created",
                resource_type="position",
                resource_id=position_id,
                result="success",
                metadata={"link_count": len(link_issuances)},
            )
        return InvitationBatchResult(
            # What the reviewer is told. `accepted_count` counts invitations that exist and
            # can be used; `rejected_count` counts those whose mail did not go out, which is
            # the number the console already renders and which was hardcoded to 0 -- so a
            # partly-delivered batch used to report as a complete success.
            accepted_count=len(issuances) - len(undeliverable),
            rejected_count=len(undeliverable),
            invitations=[
                _invitation_view(
                    issuance.invitation,
                    interview_sessions=interview_sessions,
                    invitation_reviews=invitation_reviews,
                    context=scope.context,
                )
                for issuance in issuances
            ],
            access_links=[
                InvitationAccessLinkView(
                    invitation_id=issuance.invitation.invitation_id,
                    applicant_email=issuance.invitation.applicant_email,
                    applicant_display_name=issuance.invitation.applicant_display_name,
                    access_url=_invitation_access_url(
                        applicant_access_base_url,
                        issuance.token.raw_token,
                    ),
                    expires_at=issuance.invitation.expires_at,
                )
                for issuance in link_issuances
            ],
        )

    @router.get(
        "/invitation-email-template",
        response_model=InvitationEmailTemplateView,
        operation_id="getInvitationEmailTemplate",
    )
    def get_invitation_email_template(scope: Scope) -> InvitationEmailTemplateView:
        try:
            template = template_service.get_company_template(scope.context)
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return _template_view(template, is_position_override=False)

    @router.put(
        "/invitation-email-template",
        response_model=InvitationEmailTemplateView,
        operation_id="replaceInvitationEmailTemplate",
    )
    def replace_invitation_email_template(
        body: InvitationEmailTemplateInput,
        scope: Scope,
    ) -> InvitationEmailTemplateView:
        try:
            template = template_service.save_company_template(
                scope.context,
                body.to_template(),
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        audit.append(
            scope.context,
            action="invitation_email_template.updated",
            resource_type="company",
            resource_id=scope.principal.company_id,
            result="success",
            metadata={"scope": "company"},
        )
        return _template_view(template, is_position_override=False)

    @router.delete(
        "/invitation-email-template",
        response_model=InvitationEmailTemplateView,
        operation_id="deleteInvitationEmailTemplate",
    )
    def delete_invitation_email_template(scope: Scope) -> InvitationEmailTemplateView:
        """Drop the company's edits so it tracks the platform default copy again.

        The console needs this to offer "revert to default" without shipping its own
        copy of the Korean wording, which would drift from the renderer.
        """
        try:
            template = template_service.clear_company_template(scope.context)
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        audit.append(
            scope.context,
            action="invitation_email_template.reverted",
            resource_type="company",
            resource_id=scope.principal.company_id,
            result="success",
            metadata={"scope": "company"},
        )
        return _template_view(template, is_position_override=False)

    @router.get(
        "/positions/{position_id}/invitation-email-template",
        response_model=InvitationEmailTemplateView,
        operation_id="getPositionInvitationEmailTemplate",
    )
    def get_position_invitation_email_template(
        position_id: UUID,
        scope: Scope,
    ) -> InvitationEmailTemplateView:
        try:
            resolved = template_service.get_position_template(scope.context, position_id)
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return _template_view(
            resolved.template,
            is_position_override=resolved.is_position_override,
        )

    @router.put(
        "/positions/{position_id}/invitation-email-template",
        response_model=InvitationEmailTemplateView,
        operation_id="replacePositionInvitationEmailTemplate",
    )
    def replace_position_invitation_email_template(
        position_id: UUID,
        body: InvitationEmailTemplateInput,
        scope: Scope,
    ) -> InvitationEmailTemplateView:
        try:
            resolved = template_service.save_position_template(
                scope.context,
                position_id,
                body.to_template(),
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        audit.append(
            scope.context,
            action="invitation_email_template.updated",
            resource_type="position",
            resource_id=position_id,
            result="success",
            metadata={"scope": "position"},
        )
        return _template_view(
            resolved.template,
            is_position_override=resolved.is_position_override,
        )

    @router.delete(
        "/positions/{position_id}/invitation-email-template",
        response_model=InvitationEmailTemplateView,
        operation_id="deletePositionInvitationEmailTemplate",
    )
    def delete_position_invitation_email_template(
        position_id: UUID,
        scope: Scope,
    ) -> InvitationEmailTemplateView:
        """Drop the position override so the position inherits the company template."""
        try:
            resolved = template_service.save_position_template(scope.context, position_id, None)
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        audit.append(
            scope.context,
            action="invitation_email_template.reverted",
            resource_type="position",
            resource_id=position_id,
            result="success",
            metadata={"scope": "position"},
        )
        return _template_view(
            resolved.template,
            is_position_override=resolved.is_position_override,
        )

    @router.post(
        "/invitation-email-template/preview",
        response_model=InvitationEmailPreview,
        operation_id="previewInvitationEmailTemplate",
    )
    def preview_invitation_email_template(
        body: InvitationEmailTemplateInput,
        scope: Scope,
    ) -> InvitationEmailPreview:
        """Render unsaved edits against sample data so the console can show a live preview."""
        rendered = template_service.preview(scope.context, body.to_template())
        return InvitationEmailPreview(subject=rendered.subject, html_body=rendered.html_body)

    @router.put(
        "/invitation-email-template/logo",
        response_model=CompanyLogoView,
        operation_id="replaceCompanyLogo",
    )
    async def replace_company_logo(request: Request, scope: Scope) -> CompanyLogoView:
        content_type = request.headers.get("content-type", "")
        body = await request.body()
        try:
            logo = template_service.upload_logo(
                scope.context,
                content=body,
                content_type=content_type,
            )
        except TenantScopedResourceNotFound as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except LogoTooLargeError as error:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(error),
            ) from error
        except UnsupportedLogoTypeError as error:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        audit.append(
            scope.context,
            action="company_logo.replaced",
            resource_type="company",
            resource_id=logo.company_id,
            result="success",
            metadata={"byte_size": logo.byte_size, "content_type": logo.content_type},
        )
        return CompanyLogoView(
            logo_url=template_service.logo_url(logo.company_id),
            content_type=logo.content_type,
            byte_size=logo.byte_size,
        )

    @router.delete(
        "/invitation-email-template/logo",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="deleteCompanyLogo",
    )
    def delete_company_logo(scope: Scope) -> None:
        template_service.delete_logo(scope.context)
        audit.append(
            scope.context,
            action="company_logo.deleted",
            resource_type="company",
            resource_id=scope.principal.company_id,
            result="success",
            metadata={},
        )

    @router.get(
        "/public/companies/{company_id}/logo",
        response_class=Response,
        operation_id="getPublicCompanyLogo",
    )
    def get_public_company_logo(company_id: UUID) -> Response:
        """Serve the logo to unauthenticated mail clients rendering an invitation.

        A recipient's mail client fetches remote images with no credentials, so this
        route cannot require a tenant context. It exposes only the logo a company chose
        to embed in its outbound email, and nothing else about the tenant.
        """
        logo = template_service.find_public_logo(company_id)
        if logo is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return Response(
            content=logo.content,
            media_type=logo.content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "ETag": f'"{logo.sha256}"',
            },
        )

    return router


def _template_view(
    template: InvitationEmailTemplate,
    *,
    is_position_override: bool,
) -> InvitationEmailTemplateView:
    return InvitationEmailTemplateView(
        **template.model_dump(exclude={"logo_url"}),
        logo_url=template.logo_url,
        is_position_override=is_position_override,
    )


def _optional_uuid(value: str | None) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _position_view(position: Position) -> PositionView:
    return PositionView(
        position_id=position.position_id,
        title=position.title,
        description=position.description,
        role_type=position.role_type,
        headcount=position.headcount,
        applicant_capacity=position.applicant_capacity,
        interview_capacity=position.interview_capacity,
        interview_at=position.interview_at,
        recruitment_start_at=position.recruitment_start_at,
        recruitment_end_at=position.recruitment_end_at,
        submission_requirements=position.submission_requirements,
        status=position.status.value,
        row_version=position.row_version,
        created_at=position.created_at,
    )


def _invitation_access_url(base_url: str, raw_token: str) -> str:
    return f"{base_url.rstrip('/')}/{raw_token}"


def _interviewer_profile_view(profile: InterviewerProfile) -> InterviewerProfileView:
    return InterviewerProfileView(
        interviewer_profile_id=profile.interviewer_profile_id,
        name=profile.name,
        tone=profile.tone,
        voice_id=profile.voice_id,
        row_version=profile.row_version,
        created_at=profile.created_at,
    )


def _recruiting_stage_view(stage: RecruitingStage) -> RecruitingStageView:
    return RecruitingStageView(
        recruiting_stage_id=stage.recruiting_stage_id,
        position_id=stage.position_id,
        name=stage.name,
        sort_order=stage.sort_order,
        row_version=stage.row_version,
    )


def _domain_error_detail(error: ValueError) -> str:
    """The reason a domain rule refused a request, without pydantic's frame around it.

    ``CompetencyModelVersion`` rejects a bad weight mapping from a ``model_validator``, so what
    reaches the route is a ``ValidationError`` whose ``str()`` is four lines of type tags, an
    input dump and a documentation URL. ``companyClient`` hands ``detail`` straight to the
    console, and the recruiter needs "axis weights must total 100, got 50" -- the sentence the
    validator wrote -- not the envelope it travelled in.

    A plain ``ValueError`` is already that sentence and passes through untouched.
    """
    if not isinstance(error, ValidationError):
        return str(error)
    reasons = [
        str(item["msg"]).removeprefix("Value error, ") for item in error.errors(include_url=False)
    ]
    return "; ".join(reasons) or str(error)


def _persona_view(
    persona_definition: dict[str, object],
) -> InterviewerPersonaDefinitionInput | None:
    """A recruiter-defined interviewer persona, or ``None`` for the system-managed default.

    ``CompetencyModelVersion`` defaults ``persona_definition`` to
    ``{"mode": "system_managed", "tone": "neutral", "voice_id": "Seoyeon"}``, which is not an
    ``InterviewerPersonaDefinition``: it has no ``name``, ``neutral`` is not one of the four
    tones, and ``mode`` is an extra key. Passing it through raised a ValidationError while
    building the *response*, so every version published without a persona -- both of the ones
    on the workstation this was found on -- turned the criteria list into a 500.

    ``None`` rather than an invented persona, because that is what the shape means and what the
    console already does with it: ``toCompanyPersona`` in ``routeAdapters.tsx`` returns
    ``undefined`` for exactly these three reasons, and ``persona_definition`` is not in the
    contract's required list.

    Swallowing the error is safe here because a recruiter's persona cannot arrive malformed:
    ``InterviewerPersonaDefinitionInput`` validates it on the way in, so anything that fails on
    the way out was written by the domain default rather than by a company.
    """
    try:
        return InterviewerPersonaDefinitionInput.model_validate(persona_definition)
    except ValidationError:
        return None


def _criterion_view(version: CompetencyModelVersion) -> CompetencyModelVersionView:
    return CompetencyModelVersionView(
        competency_model_version_id=version.competency_model_version_id,
        position_id=version.position_id,
        version_number=version.version_number,
        job_requirements=[
            JobRequirementInput.model_validate(
                requirement.model_dump(exclude={"job_requirement_id"})
            )
            for requirement in version.job_requirements
        ],
        criteria=[
            EvaluationCriterionInput.model_validate(criterion.model_dump(exclude={"criterion_id"}))
            for criterion in version.criteria
        ],
        prohibited_topics=version.prohibited_topics,
        interview_duration_minutes=version.interview_duration_minutes,
        interview_level=version.interview_level,
        # Read back so the wizard reopens on the weights that were saved rather than on the
        # equal split an empty mapping means. Without this the sliders silently reset to 20
        # each on the second visit, which reads as the company having chosen that.
        axis_weights=version.axis_weights,
        persona_definition=_persona_view(version.persona_definition),
        status=version.status.value,
        row_version=version.row_version,
        published_at=version.published_at,
    )


def _invitation_view(
    invitation: Invitation,
    *,
    interview_sessions: InvitationSessionResolver | None = None,
    invitation_reviews: InvitationReviewResolver | None = None,
    context: TenantContext | None = None,
) -> InvitationView:
    session = (
        interview_sessions.find_session_for_invitation(
            context,
            invitation_id=invitation.invitation_id,
        )
        if interview_sessions is not None and context is not None
        else None
    )
    # Absent until Lane D has a report for the invitation, which is what the console reads
    # to decide whether the analysis report can be opened at all.
    review = (
        invitation_reviews.get_invitation_review(
            context,
            invitation_id=invitation.invitation_id,
        )
        if invitation_reviews is not None and context is not None
        else None
    )
    return InvitationView(
        invitation_id=invitation.invitation_id,
        position_id=invitation.position_id,
        competency_model_version_id=invitation.competency_model_version_id,
        applicant_email=invitation.applicant_email,
        applicant_display_name=invitation.applicant_display_name,
        status=invitation.status.value,
        expires_at=invitation.expires_at,
        row_version=invitation.row_version,
        interview_status=session.state if session is not None else None,
        report_status=review.report_status if review is not None else None,
        interview_session_id=(session.interview_session_id if session is not None else None),
        overall_score=review.overall_score if review is not None else None,
        # None rather than 0 when there is no report: 0 of 0 would render as a coverage figure
        # for an interview that has not happened.
        scored_criteria_count=(review.scored_criteria_count if review is not None else None),
        total_criteria_count=(review.total_criteria_count if review is not None else None),
        recruiting_stage_id=invitation.recruiting_stage_id,
        pipeline_row_version=invitation.pipeline_row_version,
    )
