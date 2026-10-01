"""Live interview HTTP and WebSocket protocol routes."""

from dataclasses import dataclass
from typing import cast

from fastapi import FastAPI
from sqlalchemy.orm import Session

from interview_evidence.interview_engine.adapters.polly import SpeechSynthesisAdapter
from interview_evidence.interview_engine.adapters.recent_context import RecentContextPort
from interview_evidence.interview_engine.adapters.retrieval_client import (
    RetrievalClient,
    SubmissionRetrieval,
)
from interview_evidence.interview_engine.api.applicant_routes import (
    create_applicant_interview_router,
)
from interview_evidence.interview_engine.api.live_handlers import LiveInterviewHandler
from interview_evidence.interview_engine.api.streaming_speech import WebSocketSpeechRuntime
from interview_evidence.interview_engine.api.websocket import (
    ProtocolStreamHandler,
    create_interview_websocket_router,
)
from interview_evidence.interview_engine.application.authorization import (
    InterviewAuthorizationPort,
)
from interview_evidence.interview_engine.application.automated_answer_generator import (
    AutomatedAnswerGenerator,
)
from interview_evidence.interview_engine.application.checkpoints import CheckpointService
from interview_evidence.interview_engine.application.context_builder import ContextBuilder
from interview_evidence.interview_engine.application.context_reconciliation import (
    ContextReconciler,
)
from interview_evidence.interview_engine.application.idempotency import IdempotencyStore
from interview_evidence.interview_engine.application.interview_plan import (
    InterviewPlanProvider,
)
from interview_evidence.interview_engine.application.interview_service import InterviewService
from interview_evidence.interview_engine.application.question_generator import QuestionGenerator
from interview_evidence.interview_engine.application.question_policy import QuestionPolicy
from interview_evidence.interview_engine.application.recording_service import (
    RecordingService,
    StorageRecordingVerifier,
    VerifiableObjectStorage,
)
from interview_evidence.interview_engine.application.recovery_service import RecoveryService
from interview_evidence.interview_engine.application.session_service import (
    ProcessingObserverPort,
    SessionApplicationService,
)
from interview_evidence.interview_engine.repositories.postgres import (
    InterviewRepository,
    SqlAlchemyInterviewRepository,
)
from interview_evidence.main import create_app
from interview_evidence.shared.audit import AuditAppender
from interview_evidence.shared.aws_clients.ports import (
    AIModel,
    ObjectStorage,
    SpeechToText,
    TextEmbedder,
    TextToSpeech,
)
from interview_evidence.shared.ids import Clock
from interview_evidence.shared.messaging.outbox import Outbox
from interview_evidence.shared.security.principals import PrincipalProvider


@dataclass(frozen=True, slots=True)
class LaneCRuntime:
    app: FastAPI
    repository: InterviewRepository
    service: SessionApplicationService
    idempotency: IdempotencyStore
    hot_view: RecentContextPort
    audit: AuditAppender
    outbox: Outbox
    stream_handler: ProtocolStreamHandler
    websocket_speech: WebSocketSpeechRuntime


def create_lane_c_runtime(
    *,
    principal_provider: PrincipalProvider,
    authorization: InterviewAuthorizationPort,
    repository: InterviewRepository,
    object_storage: ObjectStorage,
    audit: AuditAppender,
    clock: Clock,
    idempotency: IdempotencyStore,
    hot_view: RecentContextPort,
    outbox: Outbox,
    plan_provider: InterviewPlanProvider | None = None,
    retrieval_provider: SubmissionRetrieval | None = None,
    model: AIModel | None = None,
    text_embedder: TextEmbedder | None = None,
    speech_to_text: SpeechToText | None = None,
    text_to_speech: TextToSpeech | None = None,
    websocket_speech: WebSocketSpeechRuntime | None = None,
    allow_automated_answers: bool = False,
    processing_observer: ProcessingObserverPort | None = None,
) -> LaneCRuntime:
    active_repository = repository
    active_storage = object_storage
    active_audit = audit
    active_clock = clock
    active_idempotency = idempotency
    active_hot_view = hot_view
    active_outbox = outbox
    active_websocket_speech = websocket_speech or WebSocketSpeechRuntime()
    checkpoints = CheckpointService(active_repository, active_outbox)
    reconciler = ContextReconciler(active_repository, active_hot_view)
    service = SessionApplicationService(
        repository=active_repository,
        authorization=authorization,
        idempotency=active_idempotency,
        checkpoints=checkpoints,
        reconciler=reconciler,
        # Without the repository, idempotency store and verifier a confirmed upload
        # raises instead of recording the chunk, which left every session with no
        # verified recording and therefore no timeline video and no report.
        recording=RecordingService(
            active_storage,
            repository=active_repository,
            idempotency=active_idempotency,
            verifier=StorageRecordingVerifier(cast(VerifiableObjectStorage, active_storage)),
        ),
        clock=active_clock,
        processing_observer=processing_observer,
    )
    stream_handler = ProtocolStreamHandler(session_service=service)
    core_live_dependencies = (
        plan_provider,
        retrieval_provider,
        model,
    )
    live_requested = any(dependency is not None for dependency in core_live_dependencies) or any(
        dependency is not None
        for dependency in (
            speech_to_text,
            text_to_speech,
            active_websocket_speech.speech_to_text,
            active_websocket_speech.text_to_speech,
        )
    )
    if live_requested:
        if not all(dependency is not None for dependency in core_live_dependencies):
            raise ValueError("plan, retrieval and model must be configured for live interviews")
        speech = SpeechSynthesisAdapter(text_to_speech)
        recovery = RecoveryService(
            repository=active_repository,
            idempotency=active_idempotency,
            checkpoints=checkpoints,
            reconciler=reconciler,
        )
        retrieval = RetrievalClient(
            cast(SubmissionRetrieval, retrieval_provider),
            embedder=text_embedder,
        )
        interview_service = InterviewService(
            repository=active_repository,
            idempotency=active_idempotency,
            recovery=recovery,
            checkpoints=checkpoints,
            context_builder=ContextBuilder(token_budget=2400),
            retrieval=retrieval,
            generator=QuestionGenerator(cast(AIModel, model)),
            policy=QuestionPolicy(),
            speech=speech,
            outbox=active_outbox,
        )
        live_handler = LiveInterviewHandler(
            repository=active_repository,
            session_service=service,
            interview_service=interview_service,
            plan_provider=cast(InterviewPlanProvider, plan_provider),
            speech_to_text=speech_to_text,
            speech=speech,
            idempotency=active_idempotency,
            checkpoints=checkpoints,
            clock=active_clock,
            automated_answer_generator=(
                AutomatedAnswerGenerator(
                    repository=active_repository,
                    retrieval=retrieval,
                    model=cast(AIModel, model),
                )
                if allow_automated_answers
                else None
            ),
        )
        stream_handler = ProtocolStreamHandler(
            session_service=service,
            start_handler=live_handler,
            answer_handler=live_handler,
            audio_handler=live_handler,
            automated_answer_handler=(live_handler if allow_automated_answers else None),
        )
    router = create_applicant_interview_router(
        principal_provider=principal_provider,
        service=service,
        audit=active_audit,
    )
    websocket_router = create_interview_websocket_router(
        principal_provider=principal_provider,
        handler=stream_handler,
        speech=active_websocket_speech,
    )
    return LaneCRuntime(
        app=create_app([router, websocket_router]),
        repository=active_repository,
        service=service,
        idempotency=active_idempotency,
        hot_view=active_hot_view,
        audit=active_audit,
        outbox=active_outbox,
        stream_handler=stream_handler,
        websocket_speech=active_websocket_speech,
    )


def create_lane_c_app(
    *,
    principal_provider: PrincipalProvider,
    authorization: InterviewAuthorizationPort,
    repository: InterviewRepository,
    object_storage: ObjectStorage,
    audit: AuditAppender,
    clock: Clock,
    idempotency: IdempotencyStore,
    hot_view: RecentContextPort,
    outbox: Outbox,
    plan_provider: InterviewPlanProvider | None = None,
    retrieval_provider: SubmissionRetrieval | None = None,
    model: AIModel | None = None,
    text_embedder: TextEmbedder | None = None,
    speech_to_text: SpeechToText | None = None,
    text_to_speech: TextToSpeech | None = None,
    websocket_speech: WebSocketSpeechRuntime | None = None,
    allow_automated_answers: bool = False,
) -> FastAPI:
    return create_lane_c_runtime(
        principal_provider=principal_provider,
        authorization=authorization,
        repository=repository,
        object_storage=object_storage,
        audit=audit,
        clock=clock,
        idempotency=idempotency,
        hot_view=hot_view,
        outbox=outbox,
        plan_provider=plan_provider,
        retrieval_provider=retrieval_provider,
        model=model,
        text_embedder=text_embedder,
        speech_to_text=speech_to_text,
        text_to_speech=text_to_speech,
        websocket_speech=websocket_speech,
        allow_automated_answers=allow_automated_answers,
    ).app


def create_sql_repository(session: Session) -> InterviewRepository:
    return SqlAlchemyInterviewRepository(session)
