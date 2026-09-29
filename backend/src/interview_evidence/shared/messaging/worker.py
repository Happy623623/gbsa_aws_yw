from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from functools import partial
from threading import Event, Thread
from typing import Protocol
from uuid import UUID

from interview_evidence.shared.aws_clients.ports import ConsumableQueue, EventQueue
from interview_evidence.shared.aws_clients.task_protection import (
    NullTaskProtection,
    TaskProtection,
)
from interview_evidence.shared.ids import Clock
from interview_evidence.shared.messaging.outbox import (
    Outbox,
    OutboxEvent,
    ProcessedMessage,
)
from interview_evidence.shared.operations import MetricRecorder, NullMetricRecorder
from interview_evidence.shared.tenant import ActorType, TenantContext

EventHandler = Callable[[TenantContext, OutboxEvent], object]
TransactionCallback = Callable[[], None]


class MessageRetryRequested(RuntimeError):
    """A handled delivery that must be retried without being marked processed."""


class ProcessedMessageStore(Protocol):
    def contains(
        self,
        *,
        consumer_name: str,
        event_id: UUID,
        event_version: int,
    ) -> bool: ...

    def get(
        self,
        *,
        consumer_name: str,
        event_id: UUID,
        event_version: int,
    ) -> ProcessedMessage: ...

    def record(self, message: ProcessedMessage) -> None: ...


class DeliveryLifecycleObserver(Protocol):
    """Optional local/test observer around the durable-commit/SQS-ack boundary."""

    def after_commit_before_ack(
        self,
        event: OutboxEvent,
        *,
        consumer_name: str,
    ) -> bool:
        """Return true only when this delivery's acknowledge must be omitted."""

    def duplicate_acknowledged(
        self,
        event: OutboxEvent,
        *,
        consumer_name: str,
    ) -> None: ...


class NullDeliveryLifecycleObserver:
    def after_commit_before_ack(
        self,
        event: OutboxEvent,
        *,
        consumer_name: str,
    ) -> bool:
        return False

    def duplicate_acknowledged(
        self,
        event: OutboxEvent,
        *,
        consumer_name: str,
    ) -> None:
        return None


class RetryExhaustionObserver(Protocol):
    """Persist a sanitized terminal fact after the last source processing attempt."""

    def retry_exhausted(
        self,
        context: TenantContext,
        event: OutboxEvent,
        *,
        error_code: str,
    ) -> None: ...


class InMemoryProcessedMessageStore:
    def __init__(self) -> None:
        self._messages: dict[tuple[str, UUID, int], ProcessedMessage] = {}

    def contains(
        self,
        *,
        consumer_name: str,
        event_id: UUID,
        event_version: int,
    ) -> bool:
        return (consumer_name, event_id, event_version) in self._messages

    def get(
        self,
        *,
        consumer_name: str,
        event_id: UUID,
        event_version: int,
    ) -> ProcessedMessage:
        return self._messages[(consumer_name, event_id, event_version)]

    def record(self, message: ProcessedMessage) -> None:
        self._messages.setdefault(
            (message.consumer_name, message.event_id, message.event_version),
            message,
        )


class OutboxDispatcher:
    def __init__(
        self,
        *,
        outbox: Outbox,
        queues: Mapping[str, EventQueue],
        routing: Mapping[str, str],
        metrics: MetricRecorder | None = None,
    ) -> None:
        self.outbox = outbox
        self._queues = dict(queues)
        self._routing = dict(routing)
        self._metrics = metrics or NullMetricRecorder()

    def dispatch_once(self) -> int:
        published = 0
        # Ask only for what this dispatcher can route. An unroutable type is skipped below and
        # so stays pending for good; left in the query it competes for the same page of rows and,
        # once there are a page of them, hides every routable event queued behind it.
        for event in self.outbox.pending(event_types=self._routing.keys()):
            queue_name = self._routing.get(event.event_type)
            if queue_name is None:
                continue
            queue = self._queues[queue_name]
            queue.publish(
                TenantContext(
                    company_id=event.company_id,
                    actor_type=ActorType.SYSTEM,
                    actor_id=event.outbox_event_id,
                    request_id=event.outbox_event_id,
                    trace_id=event.trace_id,
                ),
                event.event_type,
                {
                    "event_id": str(event.outbox_event_id),
                    "event_version": event.event_version,
                    "idempotency_key": event.idempotency_key,
                    "occurred_at": event.occurred_at.isoformat(),
                    "aggregate_type": event.aggregate_type,
                    "aggregate_id": str(event.aggregate_id),
                    "aggregate_version": event.aggregate_version,
                    "payload": event.payload,
                },
            )
            self.outbox.mark_published(event.outbox_event_id)
            self._record_queue_depth(queue_name, queue)
            published += 1
        return published

    def _record_queue_depth(self, queue_name: str, queue: EventQueue) -> None:
        depth_reader = getattr(queue, "approximate_depth", None)
        if not callable(depth_reader):
            return
        self._metrics.record(
            "queue_depth",
            float(depth_reader()),
            unit="Count",
            dimensions={"queue": queue_name},
        )


class MessageConsumer:
    def __init__(
        self,
        *,
        consumer_name: str,
        queue: ConsumableQueue,
        processed: ProcessedMessageStore,
        handlers: Mapping[str, EventHandler],
        clock: Clock,
        queue_name: str | None = None,
        metrics: MetricRecorder | None = None,
        task_protection: TaskProtection | None = None,
        delivery_observer: DeliveryLifecycleObserver | None = None,
        max_receive_count: int | None = None,
        retry_exhaustion_observer: RetryExhaustionObserver | None = None,
    ) -> None:
        if max_receive_count is not None and max_receive_count < 1:
            raise ValueError("max receive count must be positive")
        self._consumer_name = consumer_name
        self._queue = queue
        self._processed = processed
        self._handlers = dict(handlers)
        self._clock = clock
        self._queue_name = queue_name or consumer_name.removesuffix("-worker")
        self._metrics = metrics or NullMetricRecorder()
        self._task_protection = task_protection or NullTaskProtection()
        self._delivery_observer = delivery_observer or NullDeliveryLifecycleObserver()
        self._max_receive_count = max_receive_count
        self._retry_exhaustion_observer = retry_exhaustion_observer

    def consume_once(
        self,
        *,
        max_messages: int,
        commit: TransactionCallback | None = None,
        rollback: TransactionCallback | None = None,
    ) -> int:
        commit_transaction = commit or (lambda: None)
        rollback_transaction = rollback or (lambda: None)
        completed = 0
        deliveries = self._queue.receive(max_messages=max_messages)
        self._record_queue_depth()
        for delivery in deliveries:
            event = OutboxEvent(
                outbox_event_id=delivery.event_id,
                company_id=delivery.company_id,
                aggregate_type=delivery.aggregate_type,
                aggregate_id=delivery.aggregate_id,
                aggregate_version=delivery.aggregate_version,
                event_type=delivery.event_type,
                event_version=delivery.event_version,
                payload=dict(delivery.payload),
                idempotency_key=delivery.idempotency_key,
                trace_id=delivery.trace_id,
                occurred_at=delivery.occurred_at,
                delivery_attempt=delivery.receive_count,
            )
            if self._processed.contains(
                consumer_name=self._consumer_name,
                event_id=delivery.event_id,
                event_version=delivery.event_version,
            ):
                self._queue.acknowledge(delivery.receipt_handle)
                self._delivery_observer.duplicate_acknowledged(
                    event,
                    consumer_name=self._consumer_name,
                )
                self._record_delivery("duplicate")
                completed += 1
                continue
            handler = self._handlers.get(delivery.event_type)
            if handler is None:
                self._queue.acknowledge(delivery.receipt_handle)
                self._record_delivery("ignored")
                completed += 1
                continue
            context = TenantContext(
                company_id=delivery.company_id,
                actor_type=ActorType.SYSTEM,
                actor_id=delivery.event_id,
                request_id=delivery.event_id,
                trace_id=delivery.trace_id,
            )
            started_at = time.perf_counter()
            try:
                self._task_protection.acquire(delivery.event_id)
                try:
                    outcome = self._handle_with_heartbeat(
                        delivery.receipt_handle,
                        partial(handler, context, event),
                    )
                finally:
                    self._task_protection.release(delivery.event_id)
                if _requests_retry(outcome):
                    raise MessageRetryRequested("handler requested retry")
            except (MessageRetryRequested, TimeoutError, ConnectionError) as error:
                rollback_transaction()
                if (
                    self._retry_exhaustion_observer is not None
                    and self._max_receive_count is not None
                    and delivery.receive_count >= self._max_receive_count
                ):
                    try:
                        self._retry_exhaustion_observer.retry_exhausted(
                            context,
                            event,
                            error_code=type(error).__name__,
                        )
                        commit_transaction()
                    except BaseException:
                        rollback_transaction()
                        raise
                self._record_latency(delivery.event_type, delivery.event_version, started_at)
                self._record_delivery("retrying")
                self._queue.retry(
                    delivery.receipt_handle,
                    delay_seconds=_retry_delay_seconds(
                        delivery.event_type,
                        delivery.receive_count,
                    ),
                )
                continue
            except BaseException:
                rollback_transaction()
                raise
            self._record_latency(delivery.event_type, delivery.event_version, started_at)
            digest = hashlib.sha256(repr(outcome).encode("utf-8")).hexdigest()
            self._processed.record(
                ProcessedMessage(
                    consumer_name=self._consumer_name,
                    event_id=delivery.event_id,
                    event_version=delivery.event_version,
                    idempotency_key=delivery.idempotency_key,
                    first_processed_at=self._clock.now(),
                    outcome_digest=digest,
                )
            )
            try:
                commit_transaction()
            except BaseException:
                rollback_transaction()
                raise
            if self._delivery_observer.after_commit_before_ack(
                event,
                consumer_name=self._consumer_name,
            ):
                self._record_delivery("committed_ack_omitted")
                continue
            self._queue.acknowledge(delivery.receipt_handle)
            self._record_delivery("completed")
            completed += 1
        return completed

    def _handle_with_heartbeat(
        self,
        receipt_handle: str,
        handler: Callable[[], object],
    ) -> object:
        extend = getattr(self._queue, "extend_visibility", None)
        visibility_timeout = int(getattr(self._queue, "visibility_timeout_seconds", 0))
        if not callable(extend) or visibility_timeout < 3:
            return handler()
        stopped = Event()
        failures: list[BaseException] = []

        def heartbeat() -> None:
            interval = max(1, visibility_timeout // 3)
            while not stopped.wait(interval):
                try:
                    extend(receipt_handle, visibility_timeout)
                except BaseException as error:
                    failures.append(error)
                    stopped.set()

        thread = Thread(target=heartbeat, name=f"{self._queue_name}-visibility", daemon=True)
        thread.start()
        try:
            outcome = handler()
        finally:
            stopped.set()
            thread.join(timeout=1)
        if failures:
            raise ConnectionError("queue visibility heartbeat failed") from failures[0]
        return outcome

    def _record_queue_depth(self) -> None:
        depth_reader = getattr(self._queue, "approximate_depth", None)
        if not callable(depth_reader):
            return
        self._metrics.record(
            "queue_depth",
            float(depth_reader()),
            unit="Count",
            dimensions={"queue": self._queue_name},
        )

    def _record_latency(
        self,
        event_type: str,
        event_version: int,
        started_at: float,
    ) -> None:
        self._metrics.record(
            "pipeline_stage_latency_ms",
            (time.perf_counter() - started_at) * 1000,
            unit="Milliseconds",
            dimensions={
                "stage": event_type,
                "config_version": f"event-v{event_version}",
            },
        )

    def _record_delivery(self, outcome: str) -> None:
        self._metrics.record(
            "worker_delivery",
            1,
            unit="Count",
            dimensions={
                "queue": self._queue_name,
                "outcome": outcome,
            },
        )


_REPORT_GENERATION_RETRY_DELAYS_SECONDS = (15, 45, 120, 300, 600, 900)


def _retry_delay_seconds(event_type: str, receive_count: int) -> int:
    if event_type == "report.generation_requested":
        index = min(max(0, receive_count - 1), len(_REPORT_GENERATION_RETRY_DELAYS_SECONDS) - 1)
        return _REPORT_GENERATION_RETRY_DELAYS_SECONDS[index]
    exponent = min(max(0, receive_count - 1), 4)
    return min(5 * (1 << exponent), 60)


def _requests_retry(outcome: object) -> bool:
    status = getattr(outcome, "status", None)
    return getattr(status, "value", status) == "retrying"
