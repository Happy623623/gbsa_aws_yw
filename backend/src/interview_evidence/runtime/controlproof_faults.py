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
        marker = self._read_marker(session_id)
        if marker is None:
            return
        receipt = {
            "run_id": marker["run_id"],
            "session_id": str(session_id),
            "outbox_event_id": str(event.outbox_event_id),
            "delivery_attempt": event.delivery_attempt,
            "fault_type": FAULT_TYPE,
            "triggered_at": datetime.now(UTC).isoformat(),
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

    def _read_marker(self, session_id: UUID) -> dict[str, str] | None:
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
            if marker_session != str(session_id) or value.get("fault_type") != FAULT_TYPE:
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
