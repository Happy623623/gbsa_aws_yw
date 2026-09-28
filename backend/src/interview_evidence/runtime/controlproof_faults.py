"""Local/test-only reporting fault hook used by ControlProof H-03."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from interview_evidence.shared.messaging.outbox import OutboxEvent

LOGGER = logging.getLogger(__name__)
FAULT_SCHEMA = "controlproof.whyyou-fault.v1"
FAULT_TYPE = "reporting_handler_timeout_v1"
AFTER_FAULT_TYPE = "reporting_after_commit_drop_ack_v1"
RECEIPT_SCHEMA = "controlproof.whyyou-fault-receipt.v2"
DUPLICATE_ACK_SCHEMA = "controlproof.whyyou-duplicate-ack.v1"
BEFORE_FAULT_VARIANT = "BEFORE_RESULT_DURABLE"
BEFORE_BOUNDARY = "BEFORE_REPORT_SIDE_EFFECT"
AFTER_FAULT_VARIANT = "AFTER_RESULT_DURABLE_BEFORE_COMPLETION"
AFTER_BOUNDARY = "AFTER_DB_COMMIT_BEFORE_SQS_ACK"
ALLOWED_ENVIRONMENTS = frozenset({"local", "test"})


def _enabled(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "false").strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ControlProofReportingFaultGuard:
    root: Path
    enabled: bool

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> ControlProofReportingFaultGuard:
        active = dict(os.environ if environment is None else environment)
        enabled = _enabled(active, "CONTROLPROOF_TEST_HOOKS_ENABLED")
        profile = active.get("APP_ENVIRONMENT", "").strip().casefold()
        if enabled and profile not in ALLOWED_ENVIRONMENTS:
            raise RuntimeError("ControlProof test fault hooks are forbidden outside local/test")
        root_value = active.get("CONTROLPROOF_FAULT_ROOT", "").strip()
        if enabled and not root_value:
            raise RuntimeError("CONTROLPROOF_FAULT_ROOT is required when test hooks are enabled")
        return cls(root=Path(root_value).resolve() if root_value else Path.cwd(), enabled=enabled)

    def before_report_side_effect(self, event: OutboxEvent) -> None:
        """Persist a correlated trigger receipt, then enter the existing retry path."""
        if not self.enabled or event.event_type != "report.generation_requested":
            return
        raw_session_id = event.payload.get("interview_session_id")
        try:
            session_id = UUID(str(raw_session_id))
        except ValueError:
            LOGGER.warning(
                "CONTROLPROOF_FAULT_MARKER_REJECTED",
                extra={"reason": "invalid_session"},
            )
            return
        marker = self._read_marker(session_id, expected_fault_type=FAULT_TYPE)
        if marker is None:
            return
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "run_id": marker["run_id"],
            "session_id": str(session_id),
            "outbox_event_id": str(event.outbox_event_id),
            "event_version": event.event_version,
            "delivery_attempt": event.delivery_attempt,
            "fault_type": FAULT_TYPE,
            "fault_variant": BEFORE_FAULT_VARIANT,
            "boundary": BEFORE_BOUNDARY,
            "triggered_at": datetime.now(UTC).isoformat(),
            "one_shot_consumed": False,
        }
        receipt_path = self.root / "receipts" / f"{marker['run_id']}.jsonl"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with receipt_path.open("ab") as stream:
            stream.write(payload + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        LOGGER.warning("CONTROLPROOF_FAULT_TRIGGERED", extra=receipt)
        raise TimeoutError("ControlProof local/test reporting fault triggered")

    def after_commit_before_ack(
        self,
        event: OutboxEvent,
        *,
        consumer_name: str,
    ) -> bool:
        """Consume one AFTER marker after commit and request one omitted ack."""
        marker = self._matching_after_marker(event, consumer_name=consumer_name)
        if marker is None:
            return False
        consumed = self._consumed_path(marker["run_id"], marker["interview_session_id"])
        consumed.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(consumed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(b"consumed\n")
                stream.flush()
                os.fsync(stream.fileno())
            receipt = {
                "schema_version": RECEIPT_SCHEMA,
                "run_id": marker["run_id"],
                "session_id": marker["interview_session_id"],
                "outbox_event_id": str(event.outbox_event_id),
                "event_version": event.event_version,
                "delivery_attempt": event.delivery_attempt,
                "fault_type": AFTER_FAULT_TYPE,
                "fault_variant": AFTER_FAULT_VARIANT,
                "boundary": AFTER_BOUNDARY,
                "triggered_at": datetime.now(UTC).isoformat(),
                "one_shot_consumed": True,
            }
            self._append_receipt(marker["run_id"], receipt)
        except BaseException:
            consumed.unlink(missing_ok=True)
            raise
        LOGGER.warning("CONTROLPROOF_AFTER_COMMIT_ACK_OMITTED", extra=receipt)
        return True

    def duplicate_acknowledged(
        self,
        event: OutboxEvent,
        *,
        consumer_name: str,
    ) -> None:
        """Persist sanitized proof that redelivery used the processed-message branch."""
        marker = self._matching_after_marker(event, consumer_name=consumer_name)
        if marker is None:
            return
        consumed = self._consumed_path(marker["run_id"], marker["interview_session_id"])
        if not consumed.exists():
            return
        receipt = {
            "schema_version": DUPLICATE_ACK_SCHEMA,
            "run_id": marker["run_id"],
            "session_id": marker["interview_session_id"],
            "outbox_event_id": str(event.outbox_event_id),
            "event_version": event.event_version,
            "delivery_attempt": event.delivery_attempt,
            "consumer_name": consumer_name,
            "handler_skipped": True,
            "acknowledged": True,
            "observed_at": datetime.now(UTC).isoformat(),
        }
        self._append_receipt(marker["run_id"], receipt)
        LOGGER.warning("CONTROLPROOF_DUPLICATE_ACKNOWLEDGED", extra=receipt)

    def _matching_after_marker(
        self,
        event: OutboxEvent,
        *,
        consumer_name: str,
    ) -> dict[str, str] | None:
        if (
            not self.enabled
            or event.event_type != "report.generation_requested"
            or consumer_name != "reporting-worker"
        ):
            return None
        try:
            session_id = UUID(str(event.payload.get("interview_session_id")))
        except ValueError:
            return None
        return self._read_marker(session_id, expected_fault_type=AFTER_FAULT_TYPE)

    def _append_receipt(self, run_id: str, receipt: dict[str, object]) -> None:
        receipt_path = self.root / "receipts" / f"{run_id}.jsonl"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with receipt_path.open("ab") as stream:
            stream.write(payload + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _consumed_path(self, run_id: str, session_id: str) -> Path:
        return self.root / "consumed" / f"{run_id}-{session_id}.after"

    def _read_marker(
        self,
        session_id: UUID,
        *,
        expected_fault_type: str,
    ) -> dict[str, str] | None:
        path = self.root / "reporting" / f"{session_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            if value.get("schema_version") != FAULT_SCHEMA:
                return None
            run_id = str(UUID(str(value["run_id"])))
            marker_session = str(UUID(str(value["interview_session_id"])))
            if marker_session != str(session_id) or value.get("fault_type") != expected_fault_type:
                return None
            expected_variant = (
                BEFORE_FAULT_VARIANT
                if expected_fault_type == FAULT_TYPE
                else AFTER_FAULT_VARIANT
            )
            if value.get("fault_variant", expected_variant) != expected_variant:
                return None
            if expected_fault_type == AFTER_FAULT_TYPE and value.get("one_shot") is not True:
                return None
            issued_at = datetime.fromisoformat(str(value["issued_at"]))
            expires_at = datetime.fromisoformat(str(value["expires_at"]))
            now = datetime.now(UTC)
            if issued_at.tzinfo is None or expires_at.tzinfo is None:
                return None
            if not issued_at <= now < expires_at or (expires_at - issued_at).total_seconds() > 600:
                return None
        except (KeyError, ValueError, TypeError):
            return None
        return {"run_id": run_id, "interview_session_id": marker_session}
