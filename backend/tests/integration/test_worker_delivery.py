from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from interview_evidence.runtime.worker import (
    EVENT_QUEUE_ROUTING,
    InterviewCompletedEventHandler,
    ParityProbeEventHandler,
    WorkerRuntime,
    create_environment_worker_runtime,
)
from interview_evidence.shared.aws_clients.ports import InMemoryQueue
from interview_evidence.shared.ids import FrozenClock
from interview_evidence.shared.messaging.outbox import (
    InMemoryOutbox,
    OutboxEvent,
    ProcessedMessage,
)
from interview_evidence.shared.messaging.worker import (
    InMemoryProcessedMessageStore,
    MessageConsumer,
    OutboxDispatcher,
    _retry_delay_seconds,
)
from interview_evidence.shared.operations import InMemoryMetricRecorder
from interview_evidence.shared.tenant import ActorType, TenantContext

import scripts.run_workers as worker_launcher
from scripts.run_workers import _write_session_manifest

NOW = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)
COMPANY_ID = UUID("00000000-0000-7000-8000-000000000101")
ACTOR_ID = UUID("00000000-0000-7000-8000-000000000102")
EVENT_ID = UUID("00000000-0000-7000-8000-000000000103")
AGGREGATE_ID = UUID("00000000-0000-7000-8000-000000000104")


class RecordingTaskProtection:
    def __init__(self) -> None:
        self.acquired: list[UUID] = []
        self.released: list[UUID] = []

    def acquire(self, workload_id: UUID) -> bool:
        self.acquired.append(workload_id)
        return True

    def release(self, workload_id: UUID) -> bool:
        self.released.append(workload_id)
        return True


def _context() -> TenantContext:
    return TenantContext(
        company_id=COMPANY_ID,
        actor_type=ActorType.SYSTEM,
        actor_id=ACTOR_ID,
        request_id=EVENT_ID,
        trace_id="worker-trace",
    )


def _event() -> OutboxEvent:
    return OutboxEvent(
        outbox_event_id=EVENT_ID,
        company_id=COMPANY_ID,
        aggregate_type="submission",
        aggregate_id=AGGREGATE_ID,
        aggregate_version=1,
        event_type="submission.analysis_requested",
        event_version=1,
        payload={
            "submission_id": str(AGGREGATE_ID),
            "analysis_version": 1,
            "source_type": "pdf",
            "source_object_id": str(AGGREGATE_ID),
        },
        idempotency_key="analysis-request-0001",
        trace_id="worker-trace",
        occurred_at=NOW,
    )


def test_outbox_dispatch_marks_published_only_after_queue_accepts() -> None:
    outbox = InMemoryOutbox()
    queue = InMemoryQueue()
    outbox.append(_event())

    dispatched = OutboxDispatcher(
        outbox=outbox,
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    ).dispatch_once()

    assert dispatched == 1
    assert outbox.pending() == ()
    delivery = queue.receive(max_messages=1)[0]
    assert delivery.event_id == EVENT_ID
    assert delivery.company_id == COMPANY_ID
    assert delivery.payload["submission_id"] == str(AGGREGATE_ID)


def test_consumer_records_success_and_suppresses_duplicate_delivery() -> None:
    queue = InMemoryQueue()
    processed = InMemoryProcessedMessageStore()
    calls: list[UUID] = []
    task_protection = RecordingTaskProtection()

    def handle(context: TenantContext, event: OutboxEvent) -> str:
        context.assert_company(event.company_id)
        calls.append(event.outbox_event_id)
        return "ready"

    dispatcher = OutboxDispatcher(
        outbox=InMemoryOutbox(),
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    )
    dispatcher.outbox.append(_event())
    dispatcher.dispatch_once()
    queue.redeliver_all()

    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=processed,
        handlers={"submission.analysis_requested": handle},
        clock=FrozenClock(NOW),
        task_protection=task_protection,
    )

    assert consumer.consume_once(max_messages=10) == 2
    assert calls == [EVENT_ID]
    assert task_protection.acquired == [EVENT_ID]
    assert task_protection.released == [EVENT_ID]
    assert queue.receive(max_messages=10) == ()
    assert processed.contains(
        consumer_name="analysis-worker",
        event_id=EVENT_ID,
        event_version=1,
    )


def test_consumer_requeues_retryable_failure_without_recording_success() -> None:
    queue = InMemoryQueue()
    processed = InMemoryProcessedMessageStore()
    attempts = 0

    def handle(_context: TenantContext, _event: OutboxEvent) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("dependency timeout")
        return "ready"

    dispatcher = OutboxDispatcher(
        outbox=InMemoryOutbox(),
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    )
    dispatcher.outbox.append(_event())
    dispatcher.dispatch_once()
    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=processed,
        handlers={"submission.analysis_requested": handle},
        clock=FrozenClock(NOW),
    )

    assert consumer.consume_once(max_messages=1) == 0
    assert not processed.contains(
        consumer_name="analysis-worker",
        event_id=EVENT_ID,
        event_version=1,
    )
    assert consumer.consume_once(max_messages=1) == 1
    assert attempts == 2
    assert processed.get(
        consumer_name="analysis-worker",
        event_id=EVENT_ID,
        event_version=1,
    ) == ProcessedMessage(
        consumer_name="analysis-worker",
        event_id=EVENT_ID,
        event_version=1,
        idempotency_key="analysis-request-0001",
        first_processed_at=NOW,
        outcome_digest=processed.get(
            consumer_name="analysis-worker",
            event_id=EVENT_ID,
            event_version=1,
        ).outcome_digest,
    )


def test_consumer_persists_terminal_failure_only_on_configured_last_attempt() -> None:
    queue = InMemoryQueue()
    outbox = InMemoryOutbox()
    event = _event().model_copy(
        update={
            "aggregate_type": "interview_session",
            "aggregate_id": AGGREGATE_ID,
            "event_type": "report.generation_requested",
            "payload": {"interview_session_id": str(AGGREGATE_ID)},
        }
    )
    outbox.append(event)
    OutboxDispatcher(
        outbox=outbox,
        queues={"reporting": queue},
        routing={"report.generation_requested": "reporting"},
    ).dispatch_once()

    # Put the same source delivery at attempt 3 without invoking a handler for attempts 1 and 2.
    for _ in range(2):
        delivery = queue.receive(max_messages=1)[0]
        queue.retry(delivery.receipt_handle)

    failures: list[tuple[UUID, int, str]] = []
    operations: list[str] = []

    class Observer:
        def retry_exhausted(
            self,
            _context: TenantContext,
            exhausted: OutboxEvent,
            *,
            error_code: str,
        ) -> None:
            failures.append(
                (exhausted.outbox_event_id, exhausted.delivery_attempt or 0, error_code)
            )
            operations.append("failure")

    consumer = MessageConsumer(
        consumer_name="reporting-worker",
        queue=queue,
        processed=InMemoryProcessedMessageStore(),
        handlers={
            "report.generation_requested": lambda _context, _event: (_ for _ in ()).throw(
                TimeoutError("sensitive dependency detail")
            )
        },
        clock=FrozenClock(NOW),
        max_receive_count=3,
        retry_exhaustion_observer=Observer(),
    )

    assert consumer.consume_once(
        max_messages=1,
        rollback=lambda: operations.append("rollback"),
        commit=lambda: operations.append("commit"),
    ) == 0
    assert failures == [(EVENT_ID, 3, "TimeoutError")]
    assert operations == ["rollback", "failure", "commit"]


def test_consumer_requeues_retrying_outcome_without_recording_success() -> None:
    retry_delays: list[int] = []

    class RetryQueue(InMemoryQueue):
        def retry(self, receipt_handle: str, *, delay_seconds: int = 0) -> None:
            retry_delays.append(delay_seconds)
            super().retry(receipt_handle, delay_seconds=delay_seconds)

    queue = RetryQueue()
    processed = InMemoryProcessedMessageStore()
    dispatcher = OutboxDispatcher(
        outbox=InMemoryOutbox(),
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    )
    dispatcher.outbox.append(_event())
    dispatcher.dispatch_once()
    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=processed,
        handlers={
            "submission.analysis_requested": lambda _context, _event: SimpleNamespace(
                status="retrying"
            )
        },
        clock=FrozenClock(NOW),
    )

    assert consumer.consume_once(max_messages=1) == 0
    assert not processed.contains(
        consumer_name="analysis-worker",
        event_id=EVENT_ID,
        event_version=1,
    )
    assert queue.receive(max_messages=1)[0].receive_count == 2
    assert retry_delays == [5]


@pytest.mark.parametrize(
    ("receive_count", "expected_delay"),
    ((1, 15), (2, 45), (3, 120), (4, 300), (5, 600), (6, 900), (12, 900)),
)
def test_report_generation_uses_a_long_bounded_retry_schedule(
    receive_count: int,
    expected_delay: int,
) -> None:
    assert _retry_delay_seconds("report.generation_requested", receive_count) == expected_delay


def test_other_workflows_keep_the_existing_short_retry_schedule() -> None:
    assert _retry_delay_seconds("submission.analysis_requested", 1) == 5
    assert _retry_delay_seconds("submission.analysis_requested", 5) == 60
    assert _retry_delay_seconds("submission.analysis_requested", 12) == 60


def test_consumer_commits_database_before_acknowledging_sqs() -> None:
    operations: list[str] = []

    class OrderedQueue(InMemoryQueue):
        def acknowledge(self, receipt_handle: str) -> None:
            operations.append("ack")
            super().acknowledge(receipt_handle)

    queue = OrderedQueue()
    dispatcher = OutboxDispatcher(
        outbox=InMemoryOutbox(),
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    )
    dispatcher.outbox.append(_event())
    dispatcher.dispatch_once()
    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=InMemoryProcessedMessageStore(),
        handlers={"submission.analysis_requested": lambda _context, _event: "ready"},
        clock=FrozenClock(NOW),
    )

    assert (
        consumer.consume_once(
            max_messages=1,
            commit=lambda: operations.append("commit"),
            rollback=lambda: operations.append("rollback"),
        )
        == 1
    )
    assert operations == ["commit", "ack"]


def test_consumer_orders_handler_processed_commit_hook_then_ack() -> None:
    operations: list[str] = []

    class OrderedQueue(InMemoryQueue):
        def acknowledge(self, receipt_handle: str) -> None:
            operations.append("ack")
            super().acknowledge(receipt_handle)

    class OrderedProcessed(InMemoryProcessedMessageStore):
        def record(self, message: ProcessedMessage) -> None:
            operations.append("processed")
            super().record(message)

    class Observer:
        def after_commit_before_ack(self, event, *, consumer_name):
            operations.append("after-hook")
            return False

        def duplicate_acknowledged(self, event, *, consumer_name):
            operations.append("duplicate-receipt")

    queue = OrderedQueue()
    outbox = InMemoryOutbox()
    outbox.append(_event())
    OutboxDispatcher(
        outbox=outbox,
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    ).dispatch_once()
    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=OrderedProcessed(),
        handlers={
            "submission.analysis_requested": lambda _context, _event: operations.append(
                "handler"
            )
        },
        clock=FrozenClock(NOW),
        delivery_observer=Observer(),
    )

    assert consumer.consume_once(
        max_messages=1,
        commit=lambda: operations.append("commit"),
    ) == 1
    assert operations == ["handler", "processed", "commit", "after-hook", "ack"]


def test_after_commit_ack_omission_redelivers_via_processed_short_circuit() -> None:
    calls: list[UUID] = []
    observations: list[str] = []

    class VisibilityQueue(InMemoryQueue):
        def expire_visibility(self) -> None:
            deliveries = tuple(self._inflight.values())
            self._inflight.clear()
            self._available.extend(
                replace(
                    delivery,
                    receipt_handle=str(uuid4()),
                    receive_count=delivery.receive_count + 1,
                )
                for delivery in deliveries
            )

    class OneShotObserver:
        def __init__(self) -> None:
            self.consumed = False

        def after_commit_before_ack(self, event, *, consumer_name):
            if self.consumed:
                return False
            self.consumed = True
            observations.append("boundary")
            return True

        def duplicate_acknowledged(self, event, *, consumer_name):
            observations.append(f"duplicate-ack:{event.delivery_attempt}")

    queue = VisibilityQueue()
    outbox = InMemoryOutbox()
    outbox.append(_event())
    OutboxDispatcher(
        outbox=outbox,
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    ).dispatch_once()
    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=InMemoryProcessedMessageStore(),
        handlers={
            "submission.analysis_requested": lambda _context, event: calls.append(
                event.outbox_event_id
            )
        },
        clock=FrozenClock(NOW),
        delivery_observer=OneShotObserver(),
    )

    assert consumer.consume_once(max_messages=1) == 0
    assert calls == [EVENT_ID]
    assert queue.approximate_depth() == 1

    queue.expire_visibility()
    assert consumer.consume_once(max_messages=1) == 1
    assert calls == [EVENT_ID]
    assert observations == ["boundary", "duplicate-ack:2"]
    assert queue.approximate_depth() == 0


def test_consumer_extends_visibility_while_handler_is_running() -> None:
    extensions: list[int] = []

    class HeartbeatQueue(InMemoryQueue):
        visibility_timeout_seconds = 3

        def extend_visibility(self, receipt_handle: str, timeout_seconds: int) -> None:
            super().extend_visibility(receipt_handle, timeout_seconds)
            extensions.append(timeout_seconds)

    queue = HeartbeatQueue()
    dispatcher = OutboxDispatcher(
        outbox=InMemoryOutbox(),
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    )
    dispatcher.outbox.append(_event())
    dispatcher.dispatch_once()

    def slow_handler(_context: TenantContext, _event: OutboxEvent) -> str:
        time.sleep(1.1)
        return "ready"

    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=InMemoryProcessedMessageStore(),
        handlers={"submission.analysis_requested": slow_handler},
        clock=FrozenClock(NOW),
    )

    assert consumer.consume_once(max_messages=1) == 1
    assert extensions == [3]


def test_consumer_does_not_acknowledge_when_database_commit_fails() -> None:
    operations: list[str] = []

    class OrderedQueue(InMemoryQueue):
        def acknowledge(self, receipt_handle: str) -> None:
            operations.append("ack")
            super().acknowledge(receipt_handle)

    queue = OrderedQueue()
    dispatcher = OutboxDispatcher(
        outbox=InMemoryOutbox(),
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
    )
    dispatcher.outbox.append(_event())
    dispatcher.dispatch_once()
    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue=queue,
        processed=InMemoryProcessedMessageStore(),
        handlers={"submission.analysis_requested": lambda _context, _event: "ready"},
        clock=FrozenClock(NOW),
    )

    def fail_commit() -> None:
        operations.append("commit")
        raise RuntimeError("database commit failed")

    with pytest.raises(RuntimeError, match="database commit failed"):
        consumer.consume_once(
            max_messages=1,
            commit=fail_commit,
            rollback=lambda: operations.append("rollback"),
        )

    assert operations == ["commit", "rollback"]
    assert queue.approximate_depth() == 1


def test_local_worker_runtime_executes_a_cycle_without_cloud_dependencies() -> None:
    runtime = create_environment_worker_runtime(
        {
            "APP_ENVIRONMENT": "local",
            "WORKER_RUNTIME_MODE": "in-memory",
        }
    )

    assert runtime.run_once() == 0


def test_worker_cycle_attests_its_own_pid_and_isolation_profile(tmp_path) -> None:
    runtime = create_environment_worker_runtime(
        {"APP_ENVIRONMENT": "test", "WORKER_RUNTIME_MODE": "in-memory"}
    )
    session_id = str(uuid4())
    runtime.controlproof_attestation = (tmp_path, session_id, "a" * 64, 1234, 0)
    assert runtime.run_once() == 0
    path = tmp_path / "worker-attestations" / f"{session_id}-{os.getpid()}.json"
    proof = json.loads(path.read_text(encoding="utf-8"))
    assert proof["worker_pid"] == os.getpid()
    assert proof["launcher_pid"] == 1234
    assert proof["worker_slot"] == 0
    assert proof["ai_isolation_digest"] == "a" * 64


def test_failed_worker_cycle_does_not_attest(tmp_path) -> None:
    session_id = str(uuid4())

    def fail_dispatch() -> int:
        raise RuntimeError("cycle failed")

    runtime = WorkerRuntime(
        dispatcher=SimpleNamespace(dispatch_once=fail_dispatch),
        consumers=(),
        controlproof_attestation=(tmp_path, session_id, "a" * 64, 1234, 0),
    )
    with pytest.raises(RuntimeError, match="cycle failed"):
        runtime.run_once()
    assert not (tmp_path / "worker-attestations").exists()


def test_unsafe_controlproof_worker_refuses_before_aws_dependency_creation(
    monkeypatch,
) -> None:
    from interview_evidence.runtime import aws

    constructed = []
    monkeypatch.setattr(
        aws,
        "create_aws_runtime_dependencies",
        lambda _environment: constructed.append(True),
    )
    with pytest.raises(RuntimeError, match="AI_PROVIDER"):
        create_environment_worker_runtime(
            {
                "APP_ENVIRONMENT": "test",
                "WORKER_RUNTIME_MODE": "production",
                "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED": "true",
            }
        )
    assert constructed == []


def test_worker_launcher_manifest_names_every_child_pid(tmp_path) -> None:
    session_id = str(uuid4())
    path = _write_session_manifest(
        tmp_path,
        session_id=session_id,
        isolation_digest="b" * 64,
        worker_pids=[111, 222, 333],
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["session_id"] == session_id
    assert manifest["expected_worker_count"] == 3
    assert manifest["worker_pids"] == [111, 222, 333]
    assert manifest["ai_isolation_digest"] == "b" * 64


def test_worker_launcher_uses_attested_interpreter_pids_not_wrapper_pids(tmp_path) -> None:
    session_id = str(uuid4())
    directory = tmp_path / "worker-attestations"
    directory.mkdir()
    for slot, pid in ((0, 111), (1, 222)):
        (directory / f"{session_id}-{pid}.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "launcher_pid": os.getpid(),
                    "worker_slot": slot,
                    "worker_pid": pid,
                    "ai_isolation_digest": "b" * 64,
                    "heartbeat_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )
    assert worker_launcher._attested_worker_pids(
        tmp_path, session_id=session_id, isolation_digest="b" * 64, concurrency=2
    ) == [111, 222]
    assert worker_launcher._attested_worker_pids(
        tmp_path, session_id=session_id, isolation_digest="b" * 64, concurrency=3
    ) is None


def test_controlproof_worker_pool_rejects_second_launcher(tmp_path) -> None:
    lock_path = tmp_path / "worker-pool.lock"
    with worker_launcher._exclusive_pool_lock(lock_path), pytest.raises(
        RuntimeError, match="already active"
    ), worker_launcher._exclusive_pool_lock(lock_path):
        pass


def test_partial_worker_startup_cleans_up_first_child(tmp_path, monkeypatch) -> None:
    started = []
    cleaned = []

    class Process:
        pid = 9876

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            cleaned.append("terminate")

        def kill(self):
            cleaned.append("kill")

    def popen(_args, **_kwargs):
        if started:
            raise OSError("second child failed")
        started.append(Process())
        return started[0]

    monkeypatch.setattr(worker_launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(
        worker_launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: cleaned.append("taskkill"),
    )
    monkeypatch.setattr(worker_launcher.signal, "signal", lambda *_args: None)
    with pytest.raises(OSError, match="second child failed"):
        worker_launcher._run_pool(
            2,
            root=tmp_path,
            session_id=str(uuid4()),
            isolation_digest="c" * 64,
        )
    assert cleaned == (["taskkill"] if os.name == "nt" else ["terminate"])
    assert not (tmp_path / "worker-session.json").exists()


def test_worker_records_queue_depth_handler_latency_and_retry_outcome() -> None:
    queue = InMemoryQueue()
    metrics = InMemoryMetricRecorder()
    dispatcher = OutboxDispatcher(
        outbox=InMemoryOutbox(),
        queues={"analysis": queue},
        routing={"submission.analysis_requested": "analysis"},
        metrics=metrics,
    )
    dispatcher.outbox.append(_event())
    dispatcher.dispatch_once()

    def fail_once(_context: TenantContext, _event: OutboxEvent) -> str:
        raise TimeoutError("temporary dependency failure")

    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue_name="analysis",
        queue=queue,
        processed=InMemoryProcessedMessageStore(),
        handlers={"submission.analysis_requested": fail_once},
        clock=FrozenClock(NOW),
        metrics=metrics,
    )
    assert consumer.consume_once(max_messages=1) == 0

    names = [record.name for record in metrics.records]
    assert "queue_depth" in names
    assert "pipeline_stage_latency_ms" in names
    assert any(
        record.name == "worker_delivery" and record.dimensions["outcome"] == "retrying"
        for record in metrics.records
    )


def test_parity_probe_uses_real_worker_delivery_contract() -> None:
    probe_event = _event().model_copy(
        update={
            "aggregate_type": "system_parity",
            "event_type": "system.parity_probe",
            "payload": {"probe_id": str(AGGREGATE_ID)},
            "idempotency_key": "system-parity-probe-0001",
        }
    )
    outbox = InMemoryOutbox()
    queue = InMemoryQueue()
    processed = InMemoryProcessedMessageStore()
    outbox.append(probe_event)

    dispatcher = OutboxDispatcher(
        outbox=outbox,
        queues={"analysis": queue},
        routing=EVENT_QUEUE_ROUTING,
    )
    consumer = MessageConsumer(
        consumer_name="analysis-worker",
        queue_name="analysis",
        queue=queue,
        processed=processed,
        handlers={"system.parity_probe": ParityProbeEventHandler()},
        clock=FrozenClock(NOW),
    )

    assert dispatcher.dispatch_once() == 1
    assert consumer.consume_once(max_messages=1) == 1
    assert processed.contains(
        consumer_name="analysis-worker",
        event_id=EVENT_ID,
        event_version=1,
    )
    assert EVENT_QUEUE_ROUTING["system.parity_probe"] == "analysis"


def test_interview_completion_requests_media_postprocessing() -> None:
    outbox = InMemoryOutbox()
    completed = _event().model_copy(
        update={
            "aggregate_type": "interview_session",
            "event_type": "interview.completed",
            "payload": {
                "interview_session_id": str(AGGREGATE_ID),
                "invitation_id": str(UUID("00000000-0000-7000-8000-000000000105")),
                "last_turn_id": str(UUID("00000000-0000-7000-8000-000000000106")),
                "completed_at": NOW.isoformat(),
                "media_status": "pending",
            },
        }
    )

    requested = InterviewCompletedEventHandler(
        outbox,
        FrozenClock(NOW),
    )(_context(), completed)

    assert requested.event_type == "media.postprocess_requested"
    assert requested.payload["interview_session_id"] == str(AGGREGATE_ID)
    assert outbox.pending() == (requested,)
