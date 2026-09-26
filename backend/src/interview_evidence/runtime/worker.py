from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol, cast
from uuid import UUID

from interview_evidence.capacity_management.event_handler import (
    PositionCapacityChangedHandler,
)
from interview_evidence.capacity_management.planner import (
    CapacityPlanner,
    CapacityPlannerConfig,
)
from interview_evidence.capacity_management.repository import (
    SqlAlchemyCapacityRepository,
)
from interview_evidence.capacity_management.scaling import (
    AwsEcsScheduledScaling,
    InMemoryScheduledScaling,
)
from interview_evidence.company_management.application.company_service import (
    CompanyManagementPublic,
)
from interview_evidence.integration.company_analysis import CompanyAnalysisAxisProvider
from interview_evidence.integration.interview_reporting import InterviewReportingBoundary
from interview_evidence.interview_engine.application.public import InterviewEnginePublic
from interview_evidence.recruiting_assistant.application import ReportSearchProjector
from interview_evidence.reporting.api import LaneDRuntime
from interview_evidence.reporting.application.assessment_service import CriterionAssessor
from interview_evidence.reporting.application.deletion_service import DeletionService
from interview_evidence.reporting.application.evidence_service import EvidenceService
from interview_evidence.reporting.application.requirement_assessment import (
    RequirementAssessor,
    RequirementEvidenceCandidate,
)
from interview_evidence.reporting.domain.timeline import TranscriptSegment
from interview_evidence.runtime.controlproof_faults import ControlProofReportingFaultGuard
from interview_evidence.runtime.controlproof_model_substitute import (
    resolve_controlproof_model,
    validate_controlproof_test_controls,
)
from interview_evidence.runtime.document_ai import create_document_extractor
from interview_evidence.shared.aws_clients.ports import (
    ConsumableQueue,
    InMemoryQueue,
    TextEmbedder,
)
from interview_evidence.shared.aws_clients.task_protection import (
    TaskProtection,
    create_task_protection,
)
from interview_evidence.shared.database import RequestScopedDatabase
from interview_evidence.shared.ids import Clock, CommandMeta, SystemClock, new_uuid7
from interview_evidence.shared.messaging.outbox import InMemoryOutbox, Outbox, OutboxEvent
from interview_evidence.shared.messaging.worker import (
    EventHandler,
    InMemoryProcessedMessageStore,
    MessageConsumer,
    OutboxDispatcher,
    ProcessedMessageStore,
)
from interview_evidence.shared.operations import MetricRecorder, NullMetricRecorder
from interview_evidence.shared.persistence import SQLProcessedMessageStore
from interview_evidence.shared.tenant import TenantContext
from interview_evidence.shared.tracing import configure_worker_tracing
from interview_evidence.submission_analysis.adapters.search import SearchIndex
from interview_evidence.submission_analysis.api import LaneBRuntime
from interview_evidence.submission_analysis.application.public import (
    SubmissionAnalysisPublic,
)
from interview_evidence.workers.analysis.event_handler import (
    AnalysisCompletedEventHandler,
    AnalysisRequestedEventHandler,
)
from interview_evidence.workers.analysis.git_fetch import (
    BoundedGitFetcher,
    GitFetchLimits,
    GitHubPublicTransport,
)
from interview_evidence.workers.analysis.handlers import AnalysisJobHandler
from interview_evidence.workers.analysis.pipeline import SubmissionAnalysisPipeline
from interview_evidence.workers.reporting.report import (
    CriterionAnswerInput,
    CriterionInput,
    ReportGenerator,
    RequirementInput,
)

EVENT_QUEUE_ROUTING = {
    "system.parity_probe": "analysis",
    "invitation.consent_completed": "analysis",
    "submission.analysis_requested": "analysis",
    "submission.analysis_completed": "analysis",
    "strategy.ready": "analysis",
    "interview.turn_finalized": "reporting",
    "interview.session_paused": "reporting",
    "interview.completed": "reporting",
    "media.postprocess_requested": "media",
    "report.generation_requested": "reporting",
    "report.ready": "reporting",
    "deletion.requested": "deletion",
    "deletion.target_requested": "deletion",
    "deletion.target_verified": "deletion",
    "retention.expired": "deletion",
    "position.capacity_changed": "capacity",
    "capacity.reconcile_requested": "capacity",
}


class ParityProbeEventHandler:
    def __call__(self, context: TenantContext, event: OutboxEvent) -> object:
        context.assert_company(event.company_id)
        probe_id = UUID(str(event.payload["probe_id"]))
        if event.aggregate_type != "system_parity" or probe_id != event.aggregate_id:
            raise ValueError("invalid parity probe event")
        return {"probe_id": str(probe_id), "status": "processed"}


@dataclass(slots=True)
class WorkerRuntime:
    dispatcher: OutboxDispatcher
    consumers: tuple[MessageConsumer, ...]
    database: RequestScopedDatabase | None = None

    def run_once(self) -> int:
        if self.database is None:
            return self._run_without_transaction()
        completed = self._run_in_transaction(self.dispatcher.dispatch_once)
        for consumer in self.consumers:
            completed += self._run_consumer(consumer)
        return completed

    def _run_consumer(self, consumer: MessageConsumer) -> int:
        if self.database is None:
            return consumer.consume_once(max_messages=1)
        token = self.database.begin_scope()
        try:
            return consumer.consume_once(
                max_messages=1,
                commit=self.database.session.commit,
                rollback=self.database.session.rollback,
            )
        finally:
            self.database.end_scope(token)

    def _run_in_transaction(self, operation: Callable[[], int]) -> int:
        if self.database is None:
            return operation()
        token = self.database.begin_scope()
        try:
            completed = operation()
            self.database.session.commit()
            return completed
        except BaseException:
            self.database.session.rollback()
            raise
        finally:
            self.database.end_scope(token)

    def _run_without_transaction(self) -> int:
        completed = self.dispatcher.dispatch_once()
        for consumer in self.consumers:
            completed += consumer.consume_once(max_messages=1)
        return completed


class MediaRequestedEventHandler:
    def __init__(
        self,
        interview: InterviewEnginePublic,
        reporting: InterviewReportingBoundary,
        outbox: Outbox,
        clock: Clock,
    ) -> None:
        self._interview = interview
        self._reporting = reporting
        self._outbox = outbox
        self._clock = clock

    def __call__(self, context: TenantContext, event: OutboxEvent) -> object:
        session_id = UUID(str(event.payload["interview_session_id"]))
        turns = self._interview.list_final_turns(context, session_id=session_id)
        chunks = self._interview.resolve_recording_chunks(context, session_id=session_id)
        if not turns or not chunks:
            raise TimeoutError("completed session projection is not ready")
        projection = self._reporting.project_completed_session(
            context,
            session_id=session_id,
            occurred_at=self._clock.now(),
        )
        occurred_at = self._clock.now()
        self._outbox.append(
            OutboxEvent(
                outbox_event_id=new_uuid7(occurred_at),
                company_id=context.company_id,
                aggregate_type="interview_session",
                aggregate_id=session_id,
                aggregate_version=event.aggregate_version,
                event_type="report.generation_requested",
                event_version=1,
                payload={
                    "interview_session_id": str(session_id),
                    "report_version": "report-v1",
                    "competency_model_version_id": str(projection.competency_model_version_id),
                },
                idempotency_key=f"report-generation-{session_id}",
                trace_id=context.trace_id,
                occurred_at=occurred_at,
            )
        )
        return projection


class InterviewCompletedEventHandler:
    def __init__(
        self,
        outbox: Outbox,
        clock: Clock,
        company: CompanyManagementPublic | None = None,
    ) -> None:
        self._outbox = outbox
        self._clock = clock
        self._company = company

    def __call__(self, context: TenantContext, event: OutboxEvent) -> object:
        session_id = UUID(str(event.payload["interview_session_id"]))
        occurred_at = self._clock.now()
        self._close_invitation(context, event)
        requested = OutboxEvent(
            outbox_event_id=new_uuid7(occurred_at),
            company_id=context.company_id,
            aggregate_type="interview_session",
            aggregate_id=session_id,
            aggregate_version=event.aggregate_version,
            event_type="media.postprocess_requested",
            event_version=1,
            payload={
                "interview_session_id": str(session_id),
                "ordered_chunk_set_id": f"session-{session_id}-verified",
                "output_profile_version": "hls-v1",
            },
            idempotency_key=f"media-postprocess-{session_id}",
            trace_id=context.trace_id,
            occurred_at=occurred_at,
        )
        self._outbox.append(requested)
        return requested

    def _close_invitation(self, context: TenantContext, event: OutboxEvent) -> None:
        """Move the finished interview's invitation to `completed`, i.e. awaiting review.

        Nothing did this before, so an applicant who finished their interview stayed at whatever
        state the analysis pipeline had left them in. The console counts "검토 대기" as
        `status == "completed"` and "검토 완료" as `reviewed`, so both stayed at zero however many
        interviews were run, and the reviewer had no way to tell which applicants were waiting.

        `interviewing` is passed through rather than skipped: the domain only allows
        `ready → interviewing → completed`, and going straight to `completed` would be rejected.
        A live interview never announced itself, so this is the first point at which the invitation
        can be told the interview happened at all.
        """
        if self._company is None:
            return
        raw_invitation_id = event.payload.get("invitation_id")
        if not isinstance(raw_invitation_id, str):
            return
        invitation_id = UUID(raw_invitation_id)
        # Every state the interview could legitimately be left in, including the two this method
        # produces: the event is delivered at least once and re-running it must be a no-op.
        authorization = self._company.authorize_invitation(
            context,
            invitation_id,
            required_state=frozenset(
                {"ready", "interviewing", "interrupted", "completed", "reviewed"}
            ),
        )
        if authorization.state in {"completed", "reviewed"}:
            return
        if authorization.state not in {"ready", "interviewing", "interrupted"}:
            # An unexpected state is left alone rather than raised: raising would requeue the
            # event forever, and the media post-processing this handler also requests is not
            # conditional on the invitation.
            return
        row_version = authorization.row_version
        if authorization.state == "ready":
            row_version = self._company.advance_invitation_state(
                context,
                invitation_id,
                from_state="ready",
                to_state="interviewing",
                meta=CommandMeta.create(
                    f"interview-started-{invitation_id}",
                    expected_version=row_version,
                    clock=self._clock,
                ),
            ).row_version
        self._company.advance_invitation_state(
            context,
            invitation_id,
            from_state="interviewing" if authorization.state != "interrupted" else "interrupted",
            to_state="completed",
            meta=CommandMeta.create(
                f"interview-completed-{invitation_id}",
                expected_version=row_version,
                clock=self._clock,
            ),
        )


class ReportRequestedEventHandler:
    def __init__(
        self,
        *,
        company: CompanyManagementPublic,
        interview: InterviewEnginePublic,
        reporting: LaneDRuntime,
        generator: ReportGenerator,
        clock: Clock,
        assistant_projector: ReportSearchProjector | None = None,
        submission: SubmissionAnalysisPublic | None = None,
        embedder: TextEmbedder | None = None,
        controlproof_fault_guard: ControlProofReportingFaultGuard | None = None,
    ) -> None:
        self._company = company
        self._interview = interview
        self._reporting = reporting
        self._generator = generator
        self._clock = clock
        self._assistant_projector = assistant_projector
        self._submission = submission
        self._embedder = embedder
        self._controlproof_fault_guard = controlproof_fault_guard

    def __call__(self, context: TenantContext, event: OutboxEvent) -> object:
        session_id = UUID(str(event.payload["interview_session_id"]))
        if self._controlproof_fault_guard is not None:
            # This must remain before every report read/write or external model call.
            self._controlproof_fault_guard.before_report_side_effect(event)
        snapshot = self._interview.get_session_snapshot(context, session_id=session_id)
        criterion = self._company.get_criterion_version(
            context,
            snapshot.competency_model_version_id,
        )
        subject = self._company.get_recruiting_assistant_subject(
            context,
            snapshot.invitation_id,
        )
        existing = self._reporting.repository.get_report_for_session(context, session_id)
        if existing is not None:
            if self._assistant_projector is not None:
                self._assistant_projector.project(
                    context,
                    position_id=subject.position_id,
                    position_title=subject.position_title,
                    applicant_id=subject.applicant_id,
                    applicant_display_name=subject.applicant_display_name,
                    report=existing,
                )
            return existing
        final_turns = self._interview.list_final_turns(context, session_id=session_id)
        turns = tuple(turn for turn in final_turns if turn.speaker.value == "applicant")
        transcripts = {
            segment.turn_id: segment
            for segment in self._reporting.repository.list_transcripts(context, session_id)
        }
        answers_by_criterion = _criterion_answers_by_criterion(
            final_turns,
            self._interview.list_question_rationales(context, session_id=session_id),
            transcripts,
        )
        recording = next(
            (
                asset
                for asset in reversed(
                    self._reporting.repository.list_recording_assets(context, session_id)
                )
                if asset.asset_type == "final_video"
            ),
            None,
        )
        if recording is None or not turns:
            raise TimeoutError("report inputs are not ready")
        inputs = tuple(
            CriterionInput(
                criterion_id=criterion_item.criterion_id,
                criterion_name=criterion_item.name,
                criterion_text=criterion_item.description,
                observation=(
                    "면접 질문에 대한 지원자의 답변들을 종합한 결과"
                    if answers_by_criterion.get(criterion_item.criterion_id)
                    else "이 기준을 확인할 답변이 면접에서 나오지 않았습니다"
                ),
                answers=answers_by_criterion.get(criterion_item.criterion_id, ()),
                weight=criterion_item.weight,
            )
            for criterion_item in criterion.criteria
        )
        requirements = self._requirement_inputs(
            context,
            applicant_id=subject.applicant_id,
            invitation_id=snapshot.invitation_id,
            competency_model_version_id=snapshot.competency_model_version_id,
            requirements=criterion.job_requirements,
        )
        report = self._generator.generate(
            context,
            session_id=session_id,
            invitation_id=snapshot.invitation_id,
            competency_model_version_id=snapshot.competency_model_version_id,
            criteria=inputs,
            requirements=requirements,
            recording=recording,
            events=self._reporting.repository.list_session_events(context, session_id),
            occurred_at=self._clock.now(),
            interview_level=criterion.interview_level,
            axis_weights=criterion.axis_weights,
        )
        if self._assistant_projector is not None:
            self._assistant_projector.project(
                context,
                position_id=subject.position_id,
                position_title=subject.position_title,
                applicant_id=subject.applicant_id,
                applicant_display_name=subject.applicant_display_name,
                report=report,
            )
        return report

    def _requirement_inputs(
        self,
        context: TenantContext,
        *,
        applicant_id: UUID,
        invitation_id: UUID,
        competency_model_version_id: UUID,
        requirements: Sequence[_RequirementLike],
    ) -> tuple[RequirementInput, ...]:
        return tuple(
            RequirementInput(
                job_requirement_id=UUID(str(requirement.job_requirement_id)),
                requirement_type=str(requirement.requirement_type),
                statement=str(requirement.statement),
                material_evidence=self._requirement_material_evidence(
                    context,
                    applicant_id=applicant_id,
                    invitation_id=invitation_id,
                    competency_model_version_id=competency_model_version_id,
                    statement=str(requirement.statement),
                ),
            )
            for requirement in requirements
        )

    def _requirement_material_evidence(
        self,
        context: TenantContext,
        *,
        applicant_id: UUID,
        invitation_id: UUID,
        competency_model_version_id: UUID,
        statement: str,
    ) -> tuple[RequirementEvidenceCandidate, ...]:
        if self._submission is None or self._embedder is None:
            return ()
        results = self._submission.retrieve_context(
            context,
            applicant_id=applicant_id,
            invitation_id=invitation_id,
            competency_model_version_id=competency_model_version_id,
            query=statement,
            query_vector=self._embedder.embed(context, statement, dimensions=1024),
            criterion_id=None,
            config_version="requirement-assessment-retrieval-v1",
            limit=6,
            embedding_model=self._embedder.model_id,
            embedding_version=self._embedder.embedding_version,
        )
        return tuple(
            RequirementEvidenceCandidate(
                evidence_id=result.source_id,
                source_kind="submission",
                source_type=(result.material_type or result.source_type),
                excerpt=result.excerpt,
                locator=result.locator,
            )
            for result in results
            if result.score > 0
        )


class _TurnLike(Protocol):
    """The three fields pairing needs, so this helper is testable without a live Lane C."""

    @property
    def turn_id(self) -> UUID: ...

    @property
    def speaker(self) -> _SpeakerLike: ...

    @property
    def text(self) -> str | None: ...


class _RequirementLike(Protocol):
    @property
    def job_requirement_id(self) -> UUID: ...

    @property
    def requirement_type(self) -> str: ...

    @property
    def statement(self) -> str: ...


class _SpeakerLike(Protocol):
    @property
    def value(self) -> str: ...


class _RationaleLike(Protocol):
    @property
    def question_turn_id(self) -> UUID: ...

    @property
    def criterion_id(self) -> UUID: ...

    @property
    def interview_stage(self) -> str: ...


def _criterion_answers_by_criterion(
    turns: Sequence[_TurnLike],
    rationales: Sequence[_RationaleLike],
    transcripts: Mapping[UUID, TranscriptSegment],
) -> dict[UUID, tuple[CriterionAnswerInput, ...]]:
    """Group every answered scoring question, removing only repeated question-answer pairs."""
    rationale_by_turn = {item.question_turn_id: item for item in rationales}
    grouped: dict[UUID, list[CriterionAnswerInput]] = {}
    question: _TurnLike | None = None
    for turn in turns:
        if turn.speaker.value == "interviewer":
            question = turn
            continue
        if turn.speaker.value != "applicant" or question is None:
            continue
        transcript = transcripts.get(turn.turn_id)
        rationale = rationale_by_turn.get(question.turn_id)
        if transcript is None:
            question = None
            continue
        if rationale_by_turn and rationale is None:
            # The current flow deliberately leaves the greeting/warm-up without a scoring
            # rationale. It can steer retrieval but must not silently become the whole report.
            question = None
            continue
        criterion_id = (
            rationale.criterion_id
            if rationale is not None
            else getattr(question, "target_criterion_id", None)
        )
        if criterion_id is None:
            question = None
            continue
        candidate = CriterionAnswerInput(
            question=question.text or "",
            answer_turn_id=turn.turn_id,
            transcript=transcript,
            video_start_ms=transcript.session_start_ms,
            video_end_ms=transcript.session_end_ms,
            interview_stage=(rationale.interview_stage if rationale is not None else "unknown"),
        )
        collected = grouped.setdefault(criterion_id, [])
        if not any(_same_question_and_answer(candidate, seen) for seen in collected):
            collected.append(candidate)
        question = None
    return {criterion_id: tuple(answers) for criterion_id, answers in grouped.items()}


def _same_question_and_answer(
    left: CriterionAnswerInput,
    right: CriterionAnswerInput,
) -> bool:
    return (
        _text_similarity(left.question, right.question) >= 0.92
        and _text_similarity(
            left.transcript.text,
            right.transcript.text,
        )
        >= 0.88
    )


def _text_similarity(left: str, right: str) -> float:
    normalized_left = "".join(character.casefold() for character in left if character.isalnum())
    normalized_right = "".join(character.casefold() for character in right if character.isalnum())
    if not normalized_left or not normalized_right:
        return float(normalized_left == normalized_right)
    return SequenceMatcher(None, normalized_left, normalized_right, autojunk=False).ratio()


class DeletionRequestedEventHandler:
    def __init__(self, service: DeletionService, clock: Clock) -> None:
        self._service = service
        self._clock = clock

    def __call__(self, context: TenantContext, event: OutboxEvent) -> object:
        if event.event_type == "retention.expired":
            return self._service.consume_retention_expired(
                context,
                invitation_id=UUID(str(event.payload["invitation_id"])),
                policy_snapshot=dict(event.payload["policy_snapshot"]),
                occurred_at=self._clock.now(),
            )
        manifest = self._service.execute(
            context,
            request_id=UUID(str(event.payload["deletion_request_id"])),
            occurred_at=self._clock.now(),
        )
        if not manifest.is_settled:
            raise TimeoutError("deletion targets remain unverified")
        return manifest


def create_worker_runtime(
    *,
    outbox: Outbox,
    queues: Mapping[str, ConsumableQueue],
    processed: ProcessedMessageStore,
    handlers: Mapping[str, EventHandler],
    clock: Clock,
    database: RequestScopedDatabase | None = None,
    metrics: MetricRecorder | None = None,
    task_protection: TaskProtection | None = None,
) -> WorkerRuntime:
    active_metrics = metrics or NullMetricRecorder()
    consumers = tuple(
        MessageConsumer(
            consumer_name=f"{queue_name}-worker",
            queue_name=queue_name,
            queue=queue,
            processed=processed,
            handlers={
                event_type: handlers[event_type]
                for event_type, routed_queue in EVENT_QUEUE_ROUTING.items()
                if routed_queue == queue_name and event_type in handlers
            },
            clock=clock,
            metrics=active_metrics,
            task_protection=task_protection,
        )
        for queue_name, queue in queues.items()
    )
    return WorkerRuntime(
        dispatcher=OutboxDispatcher(
            outbox=outbox,
            queues=queues,
            routing=EVENT_QUEUE_ROUTING,
            metrics=active_metrics,
        ),
        consumers=consumers,
        database=database,
    )


def create_production_worker_runtime(environment: Mapping[str, str]) -> WorkerRuntime:
    from interview_evidence.runtime.aws import create_aws_runtime_dependencies
    from interview_evidence.runtime.production import create_production_runtime

    validate_controlproof_test_controls(environment)
    fault_guard = ControlProofReportingFaultGuard.from_environment(environment)
    aws = create_aws_runtime_dependencies(environment)
    report_model = resolve_controlproof_model(environment, aws.model)
    database = RequestScopedDatabase(aws.database_url)
    runtime = create_production_runtime(
        environment,
        principal_provider=aws.principal_provider,
        object_storage=aws.object_storage,
        media_storage=aws.media_storage,
        email_sender=aws.email_sender,
        search_index=aws.search_index,
        database=database,
        metrics=aws.metrics,
        queues=aws.queues,
        # Keep the fixed fixture scoped to report assessors below; unrelated API AI calls
        # continue to use the normal dependency and cannot accidentally receive report JSON.
        model=aws.model,
        speech_to_text=aws.speech_to_text,
        text_to_speech=aws.text_to_speech,
    )
    search_index = cast(SearchIndex, runtime.resources["search_index"])
    outbox = cast(Outbox, runtime.resources["outbox"])
    clock = cast(Clock, runtime.resources["clock"])
    metrics = cast(MetricRecorder, runtime.resources["metrics"])
    lane_b = runtime.lanes["submission_analysis"]
    company = runtime.boundaries["company_management"]
    interview = runtime.boundaries["interview_engine"]
    reporting_boundary = runtime.boundaries["interview_reporting"]
    submission = runtime.boundaries["submission_analysis"]
    assistant_projector = runtime.resources["assistant_projector"]
    lane_d = runtime.lanes["reporting"]
    if not isinstance(lane_b, LaneBRuntime):
        raise TypeError("production analysis runtime is invalid")
    if not isinstance(company, CompanyManagementPublic):
        raise TypeError("production company boundary is invalid")
    if not isinstance(interview, InterviewEnginePublic):
        raise TypeError("production interview boundary is invalid")
    if not isinstance(reporting_boundary, InterviewReportingBoundary):
        raise TypeError("production reporting projection is invalid")
    if not isinstance(submission, SubmissionAnalysisPublic):
        raise TypeError("production submission boundary is invalid")
    if not isinstance(lane_d, LaneDRuntime):
        raise TypeError("production reporting runtime is invalid")
    if not isinstance(assistant_projector, ReportSearchProjector):
        raise TypeError("production assistant projector is invalid")
    deletion_service = lane_d.deletion_service
    document_extractor = create_document_extractor(
        environment,
        object_storage=aws.object_storage,
    )
    analysis_pipeline = SubmissionAnalysisPipeline(
        repository=lane_b.repository,
        extractor=document_extractor,
        search_index=search_index,
        text_embedder=aws.embedder,
        strategy_model=aws.model,
        axis_provider=CompanyAnalysisAxisProvider(company),
        outbox=outbox,
        clock=clock,
        git_fetcher=BoundedGitFetcher(
            GitHubPublicTransport(token=environment.get("GITHUB_TOKEN")),
            GitFetchLimits(
                max_analyzed_commits=int(environment.get("GIT_MAX_ANALYZED_COMMITS", "20")),
            ),
        ),
    )
    analysis_handler = AnalysisJobHandler(
        analysis_pipeline,
        outbox,
        clock,
        max_attempts=3,
    )
    capacity_repository = SqlAlchemyCapacityRepository(database.session)
    capacity_scaling = (
        InMemoryScheduledScaling()
        if environment.get("APP_ENVIRONMENT", "").strip().casefold()
        in {"local", "local-production", "test"}
        else AwsEcsScheduledScaling(
            aws.application_auto_scaling,
            cluster_name=_required_worker_setting(environment, "ECS_CLUSTER_NAME"),
            api_service_name=_required_worker_setting(
                environment,
                "ECS_API_SERVICE_NAME",
            ),
            worker_service_name=_required_worker_setting(
                environment,
                "ECS_WORKER_SERVICE_NAME",
            ),
        )
    )
    capacity_planner = CapacityPlanner(
        CapacityPlannerConfig(
            api_baseline_tasks=int(environment.get("CAPACITY_API_BASELINE_TASKS", "2")),
            api_max_tasks=int(environment.get("CAPACITY_API_MAX_TASKS", "20")),
            api_safe_sessions_per_task=int(
                environment.get("CAPACITY_API_SAFE_SESSIONS_PER_TASK", "25")
            ),
            worker_baseline_tasks=int(environment.get("CAPACITY_WORKER_BASELINE_TASKS", "1")),
            worker_max_tasks=int(environment.get("CAPACITY_WORKER_MAX_TASKS", "30")),
            worker_safe_completions_per_task=int(
                environment.get("CAPACITY_WORKER_SAFE_COMPLETIONS_PER_TASK", "25")
            ),
        )
    )
    capacity_handler = PositionCapacityChangedHandler(
        capacity_repository,
        capacity_planner,
        capacity_scaling,
        clock,
        metrics,
    )
    return create_worker_runtime(
        outbox=outbox,
        queues=aws.queues,
        processed=SQLProcessedMessageStore(database.session),
        handlers={
            "submission.analysis_requested": AnalysisRequestedEventHandler(
                lane_b,
                analysis_handler,
                company,
            ),
            "submission.analysis_completed": AnalysisCompletedEventHandler(
                lane_b,
                analysis_pipeline,
                company,
            ),
            "interview.completed": InterviewCompletedEventHandler(
                outbox,
                clock,
                company,
            ),
            "media.postprocess_requested": MediaRequestedEventHandler(
                interview,
                reporting_boundary,
                outbox,
                clock,
            ),
            "report.generation_requested": ReportRequestedEventHandler(
                company=company,
                interview=interview,
                reporting=lane_d,
                generator=ReportGenerator(
                    lane_d.repository,
                    EvidenceService(lane_d.repository),
                    CriterionAssessor(
                        report_model,
                        metrics=metrics,
                        require_scores=True,
                    ),
                    RequirementAssessor(
                        report_model,
                        require_assessment=True,
                    ),
                ),
                clock=clock,
                assistant_projector=assistant_projector,
                submission=submission,
                embedder=aws.embedder,
                controlproof_fault_guard=fault_guard,
            ),
            "deletion.requested": DeletionRequestedEventHandler(
                deletion_service,
                clock,
            ),
            "retention.expired": DeletionRequestedEventHandler(
                deletion_service,
                clock,
            ),
            "position.capacity_changed": capacity_handler,
            "capacity.reconcile_requested": capacity_handler,
        },
        clock=clock,
        database=database,
        metrics=metrics,
        task_protection=create_task_protection(
            agent_uri=environment.get("ECS_AGENT_URI"),
            service="worker",
            metrics=metrics,
        ),
    )


def create_local_worker_runtime() -> WorkerRuntime:
    queues = {
        name: InMemoryQueue() for name in ("analysis", "media", "reporting", "deletion", "capacity")
    }
    return create_worker_runtime(
        outbox=InMemoryOutbox(),
        queues=queues,
        processed=InMemoryProcessedMessageStore(),
        handlers={"system.parity_probe": ParityProbeEventHandler()},
        clock=SystemClock(),
    )


def _required_worker_setting(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required worker setting is missing: {name}")
    return value


def create_environment_worker_runtime(
    environment: Mapping[str, str] | None = None,
) -> WorkerRuntime:
    active_environment = dict(os.environ if environment is None else environment)
    # A no-op unless the task definition set an OTLP endpoint, which it does only when the ADOT
    # sidecar is running. Installed here rather than in `worker.main` so that every entry point
    # into a worker runtime gets it.
    configure_worker_tracing(active_environment)
    runtime_mode = active_environment.get("WORKER_RUNTIME_MODE", "production").strip().casefold()
    if runtime_mode == "in-memory":
        return create_local_worker_runtime()
    if runtime_mode != "production":
        raise RuntimeError("WORKER_RUNTIME_MODE must be production or in-memory")
    return create_production_worker_runtime(active_environment)
