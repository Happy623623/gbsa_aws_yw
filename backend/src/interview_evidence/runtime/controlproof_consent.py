"""Disabled-by-default local/test primitives for ControlProof N-02.

These primitives expose facts and a bounded one-shot fault boundary. They do not add
or change the product's consent authorization rules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

LOGGER = logging.getLogger(__name__)
ALLOWED_ENVIRONMENTS = frozenset({"local", "test"})
FAULT_SCHEMA = "controlproof.whyyou-consent-fault.v1"
FAULT_RECEIPT_SCHEMA = "controlproof.whyyou-consent-fault-receipt.v1"
PROCESSING_RECEIPT_SCHEMA = "controlproof.whyyou-processing-receipt.v1"
FAULT_TYPE = "consent_after_record_before_state_v1"
FAULT_VARIANT = "AFTER_CONSENT_RECORD_BEFORE_STATE"
FAULT_BOUNDARY = "AFTER_CONSENT_RECORD_BEFORE_INVITATION_STATE"
FAULT_LANE = "CONSENT_FAULT_RECOVERY"
LANES = frozenset(
    {
        "PRISTINE_BASELINE",
        "DOCUMENT_BYPASS",
        "RECORDING_BOUNDARY_PROBE",
        "ASSESSMENT_BOUNDARY_PROBE",
        "NORMAL_ORDER",
        FAULT_LANE,
    }
)
PATHS = frozenset({"DOCUMENT_ANALYSIS", "RECORDING", "AI_ASSESSMENT"})
BOUNDARIES = frozenset(
    {
        "ANALYSIS_HANDLER_ENTERED",
        "INTERVIEW_SESSION_CREATED",
        "INTERVIEW_SESSION_STARTED",
        "RECORDING_CONFIRMED",
        "REPORT_HANDLER_ENTERED",
        "REPORT_ASSESSMENT_STARTED",
        "REPORT_ASSESSMENT_REFUSED",
    }
)


class ControlProofConsentFaultTriggered(RuntimeError):
    """Requests transaction rollback after a durable trigger receipt exists."""


def _enabled(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "false").strip().casefold() in {"1", "true", "yes", "on"}


def _root(environment: Mapping[str, str], name: str, *, enabled: bool) -> Path:
    raw = environment.get(name, "").strip()
    if enabled and not raw:
        raise RuntimeError(f"{name} is required when the ControlProof control is enabled")
    return Path(raw).resolve() if raw else Path.cwd()


def _guard_environment(environment: Mapping[str, str], *, enabled: bool) -> None:
    profile = environment.get("APP_ENVIRONMENT", "").strip().casefold()
    if enabled and profile not in ALLOWED_ENVIRONMENTS:
        raise RuntimeError("ControlProof N-02 controls are forbidden outside local/test")


def _aware(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


@dataclass(frozen=True, slots=True)
class ControlProofConsentFaultGuard:
    root: Path
    enabled: bool

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> ControlProofConsentFaultGuard:
        active = dict(os.environ if environment is None else environment)
        enabled = _enabled(active, "CONTROLPROOF_TEST_HOOKS_ENABLED")
        _guard_environment(active, enabled=enabled)
        return cls(
            root=_root(active, "CONTROLPROOF_FAULT_ROOT", enabled=enabled),
            enabled=enabled,
        )

    def evaluate(
        self,
        *,
        invitation_id: str,
        applicant_id: str,
        now: datetime | None = None,
    ) -> dict[str, str] | None:
        if not self.enabled:
            return None
        try:
            invitation = str(UUID(invitation_id))
            applicant = str(UUID(applicant_id))
        except (ValueError, TypeError):
            return None
        marker_path = self.root / "consent" / f"{invitation}.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            run_id = str(UUID(str(marker["run_id"])))
            marker_invitation = str(UUID(str(marker["invitation_id"])))
            marker_applicant = str(UUID(str(marker["applicant_id"])))
            issued_at = _aware(marker["issued_at"])
            expires_at = _aware(marker["expires_at"])
            active_now = now or datetime.now(UTC)
            valid = (
                marker.get("schema_version") == FAULT_SCHEMA
                and marker.get("lane_id") == FAULT_LANE
                and marker.get("fault_type") == FAULT_TYPE
                and marker.get("fault_variant") == FAULT_VARIANT
                and marker.get("one_shot") is True
                and marker_invitation == invitation
                and marker_applicant == applicant
                and issued_at <= active_now < expires_at
                and 0 < (expires_at - issued_at).total_seconds() <= 600
                and bool(str(marker.get("subject_ref", "")).strip())
            )
            if not valid:
                return None
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return {
            "run_id": run_id,
            "lane_id": FAULT_LANE,
            "subject_ref": str(marker["subject_ref"]),
            "invitation_id": invitation,
            "applicant_id": applicant,
        }

    def trigger_if_configured(
        self,
        *,
        invitation_id: str,
        applicant_id: str,
        request_id: str,
        trace_id: str,
    ) -> bool:
        marker = self.evaluate(invitation_id=invitation_id, applicant_id=applicant_id)
        if marker is None:
            return False
        try:
            request = str(UUID(request_id))
            trace_parts = trace_id.split(":", 3)
            if len(trace_parts) != 4 or trace_parts[0] != "controlproof":
                return False
            trace_run_id = str(UUID(trace_parts[1]))
            trace_lane = trace_parts[2]
            trace_subject = trace_parts[3]
        except (ValueError, TypeError):
            return False
        if (
            trace_run_id != marker["run_id"]
            or trace_lane != marker["lane_id"]
            or trace_subject != marker["subject_ref"]
        ):
            return False
        consumed = self.root / "consumed" / (
            f"{marker['run_id']}-{marker['invitation_id']}.consent"
        )
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
                "schema_version": FAULT_RECEIPT_SCHEMA,
                "receipt_id": str(uuid4()),
                **marker,
                "request_id": request,
                "fault_type": FAULT_TYPE,
                "fault_variant": FAULT_VARIANT,
                "boundary": FAULT_BOUNDARY,
                "triggered_at": datetime.now(UTC).isoformat(),
                "one_shot_consumed": True,
            }
            self._append_receipt(marker["run_id"], receipt)
        except OSError:
            consumed.unlink(missing_ok=True)
            LOGGER.warning("CONTROLPROOF_CONSENT_FAULT_RECEIPT_FAILED")
            return False
        raise ControlProofConsentFaultTriggered(
            "ControlProof local/test consent fault triggered"
        )

    def _append_receipt(self, run_id: str, receipt: dict[str, Any]) -> None:
        path = self.root / "receipts" / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        with path.open("ab") as stream:
            stream.write(payload + b"\n")
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(frozen=True, slots=True)
class ControlProofProcessingObserver:
    root: Path
    enabled: bool

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> ControlProofProcessingObserver:
        active = dict(os.environ if environment is None else environment)
        enabled = _enabled(active, "CONTROLPROOF_OBSERVER_ENABLED")
        _guard_environment(active, enabled=enabled)
        return cls(
            root=_root(active, "CONTROLPROOF_OBSERVER_ROOT", enabled=enabled),
            enabled=enabled,
        )

    def record(
        self,
        *,
        trace_id: str,
        path_id: str,
        boundary: str,
        subject_ref: str = "",
        request_or_event_id: str = "",
    ) -> bool:
        if not self.enabled:
            return False
        parts = trace_id.split(":", 3)
        try:
            if len(parts) != 4 or parts[0] != "controlproof":
                return False
            run_id = str(UUID(parts[1]))
            lane_id = parts[2]
            request_identity = str(UUID(request_or_event_id))
            if lane_id not in LANES or path_id not in PATHS or boundary not in BOUNDARIES:
                return False
            if not subject_ref.strip() or any(char in subject_ref for char in "@/\\"):
                return False
        except (ValueError, TypeError):
            return False
        receipt = {
            "schema_version": PROCESSING_RECEIPT_SCHEMA,
            "receipt_id": str(uuid4()),
            "run_id": run_id,
            "lane_id": lane_id,
            "subject_ref": subject_ref,
            "path_id": path_id,
            "boundary": boundary,
            "request_or_event_id": request_identity,
            "trace_id_digest": hashlib.sha256(trace_id.encode()).hexdigest(),
            "observed_at": datetime.now(UTC).isoformat(),
        }
        path = self.root / "receipts" / f"{run_id}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
            with path.open("ab") as stream:
                stream.write(payload + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            LOGGER.warning("CONTROLPROOF_PROCESSING_OBSERVER_WRITE_FAILED")
            return False
        return True
