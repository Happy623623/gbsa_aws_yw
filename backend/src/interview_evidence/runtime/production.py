from __future__ import annotations

from collections.abc import Mapping
from typing import cast
from urllib.parse import urlsplit

from interview_evidence.company_management.adapters.applicant_session import (
    ApplicantSessionAdapter,
)
from interview_evidence.company_management.adapters.company_auth import CompanyAuthAdapter
from interview_evidence.company_management.api import (
    create_lane_a_runtime,
)
from interview_evidence.company_management.api import (
    create_sql_repository as create_company_repository,
)
from interview_evidence.company_management.api.applicant_routes import (
    create_applicant_router as create_company_applicant_router,
)
from interview_evidence.company_management.api.company_routes import (
    create_company_router as create_hiring_router,
)
from interview_evidence.company_management.application.company_service import (
    CompanyManagementPublic,
)
from interview_evidence.company_management.application.deletion_targets import (
    CompanyDeletionTargets,
    CompanyTargetDeleter,
)
from interview_evidence.company_management.workers.invitation_email import (
    InvitationEmailHandler,
)
from interview_evidence.integration.company_submission import (
    CompanySubmissionAuthorization,
)
from interview_evidence.integration.interview_reporting import (
    InterviewReportingBoundary,
)
from interview_evidence.integration.privacy_deletion import PrivacyDeletionBoundary
from interview_evidence.integration.production_deletion import (
    ObjectTargetVerifier,
    ProductionInterviewTargetDeleter,
    ProductionSubmissionTargetDeleter,
    RelationalTargetVerifier,
    SearchTargetVerifier,
)
from interview_evidence.integration.reporting_company import (
    ReportingCompanyBoundary,
    ReportingHiringBoundary,
)
from interview_evidence.integration.submission_interview import (
    SubmissionInterviewBoundary,
)
from interview_evidence.interview_engine.adapters.recent_context import (
    PostgresRecentContext,
    RecentContextPort,
)
from interview_evidence.interview_engine.api import (
    create_lane_c_runtime,
)
from interview_evidence.interview_engine.api import (
    create_sql_repository as create_interview_repository,
)
from interview_evidence.interview_engine.api.applicant_routes import (
    create_applicant_interview_router,
)
from interview_evidence.interview_engine.api.streaming_speech import WebSocketSpeechRuntime
from interview_evidence.interview_engine.api.websocket import (
    create_interview_websocket_router,
)
from interview_evidence.interview_engine.application.deletion_targets import (
    InterviewDeletionTargets,
)
from interview_evidence.interview_engine.application.idempotency import (
    SqlAlchemyIdempotencyStore,
)
from interview_evidence.interview_engine.application.public import InterviewEnginePublic
from interview_evidence.main import Runtime, create_app
from interview_evidence.recruiting_assistant.api import create_assistant_router
from interview_evidence.recruiting_assistant.application import (
    AssistantAnswerService,
    AssistantSearchService,
    ReportSearchProjector,
)
from interview_evidence.recruiting_assistant.repository import (
    SQLAlchemyAssistantDocumentRepository,
)
from interview_evidence.reporting.adapters.playback import (
    RecordingPresigner,
    ScopedPlaybackLocator,
)
from interview_evidence.reporting.api import (
    create_lane_d_runtime,
)
from interview_evidence.reporting.api import (
    create_sql_repository as create_reporting_repository,
)
from interview_evidence.reporting.api.company_routes import (
    create_company_router as create_reporting_router,
)
from interview_evidence.reporting.application.assessment_service import CriterionAssessor
from interview_evidence.reporting.application.deletion_service import DeletionService
from interview_evidence.reporting.application.evidence_service import EvidenceService
from interview_evidence.reporting.application.public import ReportingPublic
from interview_evidence.reporting.application.requirement_assessment import (
    RequirementAssessor,
)
from interview_evidence.reporting.application.transcript_service import TranscriptService
from interview_evidence.runtime.controlproof_model_substitute import (
    controlproof_health,
    validate_controlproof_test_controls,
)
from interview_evidence.runtime.email import create_local_email_sender
from interview_evidence.runtime.speech import create_speech_runtime_dependencies
from interview_evidence.shared.aws_clients.ports import (
    AIModel,
    ConsumableQueue,
    EmailSender,
    ObjectStorage,
    SpeechToText,
    TextEmbedder,
    TextToSpeech,
)
from interview_evidence.shared.aws_clients.task_protection import create_task_protection
from interview_evidence.shared.database import RequestScopedDatabase
from interview_evidence.shared.ids import SystemClock
from interview_evidence.shared.operations import (
    DependencyReadiness,
    MetricRecorder,
    NullMetricRecorder,
    ReadinessChecker,
)
from interview_evidence.shared.persistence import (
    SQLApplicantSessionStore,
    SQLAuditAppender,
    SQLCommandIdempotencyStore,
    SQLOutbox,
    SQLUploadIntentStore,
)
from interview_evidence.shared.security.local import resolve_company_principal_provider
from interview_evidence.shared.security.principals import (
    ApplicantPrincipal,
    CompanyPrincipal,
    PrincipalProvider,
)
from interview_evidence.submission_analysis.adapters.postgres_hybrid import (
    PostgresHybridSearchIndex,
)
from interview_evidence.submission_analysis.adapters.search import (
    AnalysisDebugSearch,
    SearchIndex,
)
from interview_evidence.submission_analysis.api import (
    create_lane_b_runtime,
)
from interview_evidence.submission_analysis.api import (
    create_sql_repository as create_submission_repository,
)
from interview_evidence.submission_analysis.api.applicant_routes import (
    create_applicant_submission_router,
)
from interview_evidence.submission_analysis.api.company_routes import (
    create_company_submission_router,
)
from interview_evidence.submission_analysis.application.deletion_targets import (
    SubmissionDeletionTargets,
)
from interview_evidence.submission_analysis.application.public import (
    SubmissionAnalysisPublic,
)
from interview_evidence.submission_analysis.application.retrieval import (
    HybridRetrievalConfig,
    HybridRetriever,
)
from interview_evidence.workers.reporting.media import (
    FfmpegWebMRemuxer,
    MediaObjectStore,
    MediaPostProcessor,
    RecordingAssembler,
)
from interview_evidence.workers.reporting.report import ReportGenerator


def create_production_runtime(
    environment: Mapping[str, str],
    *,
    principal_provider: PrincipalProvider | None = None,
    object_storage: ObjectStorage | None = None,
    media_storage: ObjectStorage | None = None,
    email_sender: EmailSender | None = None,
    recent_context: RecentContextPort | None = None,
    search_index: SearchIndex | None = None,
    database: RequestScopedDatabase | None = None,
    metrics: MetricRecorder | None = None,
    readiness: ReadinessChecker | None = None,
    queues: Mapping[str, ConsumableQueue] | None = None,
    model: AIModel | None = None,
    embedder: TextEmbedder | None = None,
    speech_to_text: SpeechToText | None = None,
    text_to_speech: TextToSpeech | None = None,
) -> Runtime:
    validate_controlproof_test_controls(environment)
    streaming_speech = create_speech_runtime_dependencies(environment)
    email_sender = email_sender or create_local_email_sender(environment)
    applicant_access_base_url = _applicant_access_base_url(environment)
    logo_base_url = _logo_base_url(environment, applicant_access_base_url)
    aws = None
    if (
        principal_provider is None
        or object_storage is None
        or email_sender is None
        or model is None
        or embedder is None
        or speech_to_text is None
        or text_to_speech is None
    ):
        from interview_evidence.runtime.aws import create_aws_runtime_dependencies

        aws = create_aws_runtime_dependencies(environment)
        if principal_provider is None:
            principal_provider = resolve_company_principal_provider(
                environment,
                default=aws.principal_provider,
            )
        object_storage = object_storage or aws.object_storage
        media_storage = media_storage or aws.media_storage
        email_sender = email_sender or aws.email_sender
        search_index = search_index or aws.search_index
        model = model or aws.model
        embedder = embedder or aws.embedder
        if speech_to_text is None and streaming_speech.stt_provider == "aws_legacy":
            speech_to_text = aws.speech_to_text
        if text_to_speech is None and streaming_speech.tts_provider == "aws_legacy":
            text_to_speech = aws.text_to_speech
        database_url = aws.database_url
    else:
        database_url = environment.get("DATABASE_URL", "").strip()
    media_storage = media_storage or object_storage
    if database is None:
        if not database_url:
            raise RuntimeError("production DATABASE_URL is required")
        database = RequestScopedDatabase(database_url)
    if principal_provider is None:
        raise RuntimeError("production principal provider is required")
    active_principal_provider = principal_provider
    active_metrics = metrics or (aws.metrics if aws is not None else NullMetricRecorder())

    clock = SystemClock()
    session = database.session
    search_index = search_index or PostgresHybridSearchIndex(session)
    audit = SQLAuditAppender(session)
    outbox = SQLOutbox(session)
    resource_idempotency = SQLCommandIdempotencyStore(session)
    interview_idempotency = SqlAlchemyIdempotencyStore(session)
    interview_repository = create_interview_repository(session)
    recent_context = recent_context or PostgresRecentContext(interview_repository)
    applicant_sessions = ApplicantSessionAdapter(
        clock=clock,
        store=SQLApplicantSessionStore(session),
    )

    lane_a = create_lane_a_runtime(
        principal_provider=active_principal_provider,
        repository=create_company_repository(session),
        audit=audit,
        clock=clock,
        sessions=applicant_sessions,
        outbox=outbox,
        idempotency=resource_idempotency,
        email_sender=email_sender,
        applicant_access_base_url=applicant_access_base_url,
        logo_base_url=logo_base_url,
    )

    class RuntimePrincipalProvider:
        def get_company_principal(self, credential: str) -> CompanyPrincipal:
            return active_principal_provider.get_company_principal(credential)

        def get_applicant_principal(self, credential: str) -> ApplicantPrincipal:
            return applicant_sessions.get_applicant_principal(credential)

    principals = RuntimePrincipalProvider()
    company_public = CompanyManagementPublic(
        lane_a.repository,
        clock,
        deletion_targets=CompanyDeletionTargets(
            lane_a.repository,
            lane_a.outbox,
            clock,
        ),
        target_deleter=CompanyTargetDeleter(
            lane_a.repository,
            audit,
        ),
    )
    company_submission = CompanySubmissionAuthorization(company_public)
    lane_b = create_lane_b_runtime(
        principal_provider=principals,
        authorization=company_submission,
        repository=create_submission_repository(session),
        object_storage=object_storage,
        audit=audit,
        clock=clock,
        outbox=outbox,
        idempotency=resource_idempotency,
        upload_intents=SQLUploadIntentStore(session),
    )
    submission_public = SubmissionAnalysisPublic(
        repository=lane_b.repository,
        retriever=HybridRetriever(search_index, HybridRetrievalConfig()),
        deletion_targets=SubmissionDeletionTargets(lane_b.repository),
        target_deleter=ProductionSubmissionTargetDeleter(
            repository=cast(RelationalTargetVerifier, lane_b.repository),
            object_storage=cast(ObjectTargetVerifier, object_storage),
            search_index=cast(SearchTargetVerifier, search_index),
            metrics=active_metrics,
        ),
    )
    submission_interview = SubmissionInterviewBoundary(submission_public, company_public)
    lane_c = create_lane_c_runtime(
        principal_provider=principals,
        authorization=submission_interview,
        repository=interview_repository,
        object_storage=media_storage,
        audit=audit,
        clock=clock,
        hot_view=recent_context,
        outbox=outbox,
        idempotency=interview_idempotency,
        plan_provider=submission_interview,
        retrieval_provider=submission_interview,
        model=model,
        text_embedder=embedder,
        speech_to_text=speech_to_text,
        text_to_speech=text_to_speech,
        websocket_speech=WebSocketSpeechRuntime(
            speech_to_text=streaming_speech.streaming_speech_to_text,
            text_to_speech=streaming_speech.streaming_text_to_speech,
            recognition_language_code=environment.get("GCP_STT_LANGUAGE_CODE", "ko-KR").strip(),
            recognition_model=environment.get("GCP_STT_MODEL", "latest_long").strip(),
            final_result_timeout_seconds=float(
                environment.get("GCP_STT_FINAL_TIMEOUT_SECONDS", "8")
            ),
        ),
        allow_automated_answers=_automated_interviews_enabled(environment),
    )
    interview_public = InterviewEnginePublic(
        repository=lane_c.repository,
        deletion_targets=InterviewDeletionTargets(lane_c.repository),
        target_deleter=ProductionInterviewTargetDeleter(
            repository=cast(RelationalTargetVerifier, lane_c.repository),
            object_storage=cast(ObjectTargetVerifier, media_storage),
            metrics=active_metrics,
        ),
    )
    base_lane_d = create_lane_d_runtime(
        principal_provider=principals,
        repository=create_reporting_repository(session),
        audit=audit,
        clock=clock,
        idempotency=resource_idempotency,
    )
    assistant_documents = SQLAlchemyAssistantDocumentRepository(session)
    assistant_projector = ReportSearchProjector(assistant_documents, embedder)
    assistant_search = AssistantSearchService(
        assistant_documents,
        embedder,
        minimum_score=float(environment.get("ASSISTANT_MIN_RELEVANCE_SCORE", "0")),
    )
    assistant_answers = AssistantAnswerService(assistant_search, model)
    base_reporting_public = ReportingPublic(
        repository=base_lane_d.repository,
        deletion_service=base_lane_d.deletion_service,
        assistant_documents=assistant_documents,
    )
    privacy_deletion = PrivacyDeletionBoundary(
        company=company_public,
        submission=submission_public,
        interview=interview_public,
        reporting=base_reporting_public,
        clock=clock,
        object_storage=media_storage,
        metrics=active_metrics,
    )
    deletion_service = DeletionService(
        base_lane_d.repository,
        enumerators=(privacy_deletion.enumerate,),
        executors={
            "A": privacy_deletion.execute_company,
            "B": privacy_deletion.execute_submission,
            "C": privacy_deletion.execute_interview,
            "D": privacy_deletion.execute_reporting,
        },
        outbox=outbox,
    )
    reporting_hiring = ReportingHiringBoundary(company_public, lane_a.hiring_service)
    lane_d = create_lane_d_runtime(
        principal_provider=principals,
        repository=base_lane_d.repository,
        audit=audit,
        clock=clock,
        deletion_service=deletion_service,
        rationale_provider=interview_public,
        invitations=reporting_hiring,
        idempotency=resource_idempotency,
    )
    media_processor = MediaPostProcessor(lane_d.repository)
    interview_reporting = InterviewReportingBoundary(
        interview=interview_public,
        transcript_service=TranscriptService(lane_d.repository),
        media_processor=media_processor,
        assembler=RecordingAssembler(
            cast(MediaObjectStore, media_storage),
            FfmpegWebMRemuxer(),
        ),
    )
    reporting_public = ReportingPublic(
        repository=lane_d.repository,
        deletion_service=lane_d.deletion_service,
        assistant_documents=assistant_documents,
    )
    reporting_company = ReportingCompanyBoundary(reporting_public)

    if readiness is None:
        probes = {"database": database.healthcheck}
        for name, dependency in {
            "object_storage": object_storage,
            "media_storage": media_storage,
            "recent_context": recent_context,
            "search": search_index,
        }.items():
            probe = getattr(dependency, "healthcheck", None)
            if callable(probe):
                probes[name] = probe
        active_queues = queues or (aws.queues if aws is not None else {})
        if active_queues:
            for queue_name, queue in active_queues.items():
                probes[f"{queue_name}_queue"] = queue.healthcheck
        readiness = DependencyReadiness(probes)
    active_queues = queues or (aws.queues if aws is not None else {})

    root = create_app(
        [
            create_hiring_router(
                auth=CompanyAuthAdapter(active_principal_provider),
                company_service=lane_a.company_service,
                criteria_service=lane_a.criteria_service,
                interviewer_service=lane_a.interviewer_service,
                hiring_service=lane_a.hiring_service,
                template_service=lane_a.template_service,
                audit=audit,
                invitation_email=InvitationEmailHandler(lane_a.email_sender),
                applicant_access_base_url=applicant_access_base_url,
                interview_sessions=interview_public,
                invitation_reviews=reporting_company,
            ),
            create_company_applicant_router(
                sessions=lane_a.sessions,
                access_service=lane_a.applicant_access_service,
            ),
            create_applicant_submission_router(
                principal_provider=principals,
                authorization=company_submission,
                service=lane_b.service,
                audit=audit,
                debug_repository=(
                    lane_b.repository
                    if environment.get("APP_ENVIRONMENT", "").strip().casefold() == "local"
                    else None
                ),
                debug_search=(
                    cast(AnalysisDebugSearch, search_index)
                    if environment.get("APP_ENVIRONMENT", "").strip().casefold() == "local"
                    and hasattr(search_index, "list_debug_documents")
                    else None
                ),
            ),
            create_company_submission_router(
                principal_provider=principals,
                repository=lane_b.repository,
                presigner=cast(RecordingPresigner, object_storage),
            ),
            create_applicant_interview_router(
                principal_provider=principals,
                service=lane_c.service,
                audit=audit,
            ),
            create_interview_websocket_router(
                principal_provider=principals,
                handler=lane_c.stream_handler,
                database=database,
                speech=lane_c.websocket_speech,
                task_protection=create_task_protection(
                    agent_uri=environment.get("ECS_AGENT_URI"),
                    service="api",
                    metrics=active_metrics,
                ),
            ),
            create_reporting_router(
                principal_provider=principals,
                repository=lane_d.repository,
                audit=audit,
                clock=clock,
                deletion_service=lane_d.deletion_service,
                invitations=reporting_hiring,
                # The presigner comes from the media bucket's own storage adapter, so the
                # URL the console puts in `<video src>` is a signed read of the assembled
                # recording rather than a placeholder host that resolves nowhere.
                playback=ScopedPlaybackLocator(presigner=cast(RecordingPresigner, media_storage)),
                # This router, not `lane_d.app`, is what serves the timeline. Without the
                # provider here the response carries no question rationale, so the console
                # shows each question with nothing behind it -- as if the AI made it up.
                rationale_provider=interview_public,
            ),
            create_assistant_router(
                principal_provider=principals,
                company_service=lane_a.company_service,
                search_service=assistant_search,
                answer_service=assistant_answers,
                audit=audit,
                clock=clock,
                pipeline_reader=lane_a.hiring_service,
            ),
        ],
        readiness=readiness,
    )
    root.exception_handlers.update(lane_d.app.exception_handlers)
    if environment.get("APP_ENVIRONMENT", "").strip().casefold() in {"local", "test"}:
        root.add_api_route(
            "/internal/controlproof/health",
            lambda: controlproof_health(environment),
            methods=["GET"],
            include_in_schema=False,
        )
    database.install_http_transaction_middleware(root)
    return Runtime(
        app=root,
        lanes={
            "company_management": lane_a,
            "submission_analysis": lane_b,
            "interview_engine": lane_c,
            "reporting": lane_d,
        },
        boundaries={
            "company_management": company_public,
            "submission_analysis": submission_public,
            "interview_engine": interview_public,
            "reporting": reporting_public,
            "company_submission": company_submission,
            "submission_interview": submission_interview,
            "interview_reporting": interview_reporting,
            "reporting_company": reporting_company,
            "recruiting_assistant": assistant_search,
        },
        worker_handlers={
            "invitation_email": InvitationEmailHandler(lane_a.email_sender),
            "media_postprocess": media_processor,
            "report_generation": ReportGenerator(
                lane_d.repository,
                EvidenceService(lane_d.repository),
                CriterionAssessor(
                    model,
                    metrics=active_metrics,
                    require_scores=True,
                ),
                RequirementAssessor(
                    model,
                    require_assessment=True,
                ),
            ),
            "privacy_deletion": lane_d.deletion_service,
        },
        resources={
            "audit": audit,
            "clock": clock,
            "database": database,
            "outbox": outbox,
            "object_storage": object_storage,
            "search_index": search_index,
            "text_embedder": embedder,
            "assistant_documents": assistant_documents,
            "assistant_projector": assistant_projector,
            "assistant_answers": assistant_answers,
            "privacy_deletion": privacy_deletion,
            "metrics": active_metrics,
            "readiness": readiness,
            "queues": active_queues,
        },
    )


def _applicant_access_base_url(environment: Mapping[str, str]) -> str:
    value = environment.get("APPLICANT_ACCESS_BASE_URL", "").strip()
    if not value.startswith(("https://", "http://")) or "?" in value or "#" in value:
        raise RuntimeError("valid APPLICANT_ACCESS_BASE_URL is required")
    return value.rstrip("/")


def _logo_base_url(environment: Mapping[str, str], applicant_access_base_url: str) -> str:
    """Resolve the origin a recipient's mail client uses to fetch the company logo.

    Defaults to the origin of the applicant access URL, which is already public and
    already fronts ``/v1/*``, so a deployment gets working logos without new settings.
    """
    value = environment.get("PUBLIC_ASSET_BASE_URL", "").strip()
    if value:
        if not value.startswith(("https://", "http://")) or "?" in value or "#" in value:
            raise RuntimeError("PUBLIC_ASSET_BASE_URL must be an absolute URL without a query")
        return value.rstrip("/")
    parsed = urlsplit(applicant_access_base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _automated_interviews_enabled(environment: Mapping[str, str]) -> bool:
    return environment.get("APP_ENVIRONMENT", "").strip().casefold() == "local" or (
        environment.get("AUTOMATED_INTERVIEW_ENABLED", "").strip().casefold()
        in {"1", "true", "yes", "on"}
    )
