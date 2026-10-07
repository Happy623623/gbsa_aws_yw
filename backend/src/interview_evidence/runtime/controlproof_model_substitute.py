"""Fixed local/test AI dependency for deterministic ControlProof report runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from interview_evidence.shared.tenant import TenantContext, require_tenant_context

FIXTURE_ID = "h03-report-v1"
FIXTURE_SEED = "controlproof:h03-report-v1"
FIXTURE_DIGEST = hashlib.sha256(FIXTURE_SEED.encode("utf-8")).hexdigest()
#: Spec 004 (E-01/E-02) fixture: citation modes and scores chosen by a marker in the criterion text.
SPEC004_FIXTURE_ID = "spec004-report-v1"
SPEC004_FIXTURE_SEED = "controlproof:spec004-report-v1"
SPEC004_FIXTURE_DIGEST = hashlib.sha256(SPEC004_FIXTURE_SEED.encode("utf-8")).hexdigest()
_FIXTURE_DIGESTS = {FIXTURE_ID: FIXTURE_DIGEST, SPEC004_FIXTURE_ID: SPEC004_FIXTURE_DIGEST}
ALLOWED_ENVIRONMENTS = frozenset({"local", "test"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_AI_ENDPOINTS = (
    "BEDROCK_RUNTIME_ENDPOINT_URL",
    "TRANSCRIBE_ENDPOINT_URL",
    "POLLY_ENDPOINT_URL",
    "GCP_DOCUMENT_AI_API_ENDPOINT",
)


def _enabled(environment: Mapping[str, str]) -> bool:
    return environment.get("CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _requested_fixture(environment: Mapping[str, str]) -> str:
    requested = environment.get("CONTROLPROOF_MODEL_FIXTURE_ID", FIXTURE_ID).strip()
    if requested not in _FIXTURE_DIGESTS:
        raise RuntimeError("unsupported ControlProof model fixture ID")
    return requested


def validate_controlproof_test_controls(environment: Mapping[str, str]) -> None:
    profile = environment.get("APP_ENVIRONMENT", "").strip().casefold()
    if (
        _enabled(environment)
        or environment.get("CONTROLPROOF_TEST_HOOKS_ENABLED", "false").strip().casefold()
        in {"1", "true", "yes", "on"}
    ) and profile not in ALLOWED_ENVIRONMENTS:
        raise RuntimeError("ControlProof test controls are forbidden outside local/test")
    if _enabled(environment):
        controlproof_ai_isolation_digest(environment)


def controlproof_ai_isolation_digest(environment: Mapping[str, str]) -> str:
    """Fingerprint the only supported N-02 local AI route without exposing endpoints."""
    if environment.get("APP_ENVIRONMENT", "").strip().casefold() not in ALLOWED_ENVIRONMENTS:
        raise RuntimeError("ControlProof AI isolation requires local/test")
    if environment.get("CONTROLPROOF_EXTERNAL_AI_ALLOWED", "false").strip().casefold() != "false":
        raise RuntimeError("ControlProof external AI must be disabled")
    required = {
        "AI_PROVIDER": "aws",
        "EMBEDDING_PROVIDER": "aws",
        "STT_PROVIDER": "disabled",
        "TTS_PROVIDER": "text_only",
    }
    for name, expected in required.items():
        if environment.get(name, "").strip().casefold() != expected:
            raise RuntimeError(f"ControlProof AI isolation requires {name}={expected}")
    routes = {}
    for name in _AI_ENDPOINTS:
        value = environment.get(name, "").strip()
        parsed = urlsplit(value if name != "GCP_DOCUMENT_AI_API_ENDPOINT" else f"http://{value}")
        try:
            port = parsed.port
        except ValueError as error:
            raise RuntimeError(f"ControlProof AI endpoint is invalid: {name}") from error
        if (
            parsed.scheme != "http"
            or parsed.hostname not in _LOOPBACK_HOSTS
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise RuntimeError(f"ControlProof AI endpoint must be loopback: {name}")
        routes[name] = f"{parsed.hostname}:{port}"
    fixture_id = _requested_fixture(environment)
    payload = {
        "contract": "controlproof.n02-ai-isolation.v1",
        "fixture_id": fixture_id,
        "fixture_digest": _FIXTURE_DIGESTS[fixture_id],
        "providers": required,
        "routes": routes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ControlProofFixedModel:
    fixture_id = FIXTURE_ID
    fixture_digest = FIXTURE_DIGEST

    def generate(
        self,
        context: TenantContext,
        model_input: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        require_tenant_context(context)
        payload = _task_payload(model_input)
        task = payload.get("task")
        if task == "assess_interview_criterion":
            evidence_ids = [item["evidence_id"] for item in payload.get("provided_answers", [])]
            evidence = evidence_ids[:1]
            return {
                "criterion_id": payload["criterion"]["criterion_id"],
                "assessment_state": "confirmed" if evidence else "insufficient_evidence",
                "axis_scores": [
                    {
                        "axis": axis["key"],
                        "score": 72 if evidence else None,
                        "rationale": (
                            "고정된 합성 답변 근거를 사용하는 ControlProof 시험 fixture입니다."
                        ),
                        "quoted_evidence_ids": evidence,
                    }
                    for axis in payload.get("axes", [])
                ],
                "summary": "고정 fixture가 생성한 결정론적 합성 평가입니다.",
                "follow_up_question": None,
            }
        if task == "assess_job_requirement":
            candidates = payload.get("evidence_candidates", [])
            return {
                "signals": [
                    {
                        "evidence_id": item["evidence_id"],
                        "relation": "partially_supports",
                        "explanation": "고정 fixture의 합성 근거입니다.",
                    }
                    for item in candidates[:1]
                ]
            }
        raise ValueError(f"ControlProof fixed model does not support task: {task}")


_SPEC004_MARKER = re.compile(r"^\[controlproof-spec004(?P<body>\s[^\]]*)?\]")
_SPEC004_MODES = frozenset({"VALID", "EMPTY", "NONEXISTENT", "OTHER_APPLICANT", "OTHER_CRITERION"})
_SPEC004_ARGUMENT_MODES = frozenset({"NONEXISTENT", "OTHER_APPLICANT", "OTHER_CRITERION"})
_SPEC004_DEFAULT_SCORE = 72
_SPEC004_MEMORY_LIMIT = 256
#: Same wording as the h03 fixture, so an unmarked criterion produces byte-identical output.
_SPEC004_RATIONALE = "고정된 합성 답변 근거를 사용하는 ControlProof 시험 fixture입니다."


def _spec004_marker(text: str) -> tuple[str, str | None, int | None, str]:
    """Read ``[controlproof-spec004 mode=M arg=A score=N]`` as (mode, arg, score, status).

    A text without the marker is the h03 behaviour (`DEFAULT`, `EMITTED`); a malformed marker is
    also answered with h03 behaviour but reported as `MARKER_INVALID`, so a typo in a seeded
    criterion never surfaces as a model failure inside the target's worker.
    """
    match = _SPEC004_MARKER.match(text)
    if match is None:
        status = "MARKER_INVALID" if text.startswith("[controlproof-spec004") else "EMITTED"
        return "DEFAULT", None, None, status
    fields: dict[str, str] = {}
    for token in (match.group("body") or "").split():
        key, separator, value = token.partition("=")
        if not separator or key not in {"mode", "arg", "score"} or key in fields:
            return "DEFAULT", None, None, "MARKER_INVALID"
        fields[key] = value
    mode, argument = fields.get("mode"), fields.get("arg")
    if mode not in _SPEC004_MODES or (argument is not None) != (mode in _SPEC004_ARGUMENT_MODES):
        return "DEFAULT", None, None, "MARKER_INVALID"
    score: int | None = None
    try:
        if argument is not None:
            argument = str(UUID(argument))
        if "score" in fields:
            score = int(fields["score"])
    except ValueError:
        return "DEFAULT", None, None, "MARKER_INVALID"
    if score is not None and not 0 <= score <= 100:
        return "DEFAULT", None, None, "MARKER_INVALID"
    return mode, argument, score, "EMITTED"


def _uuid7_millis(value: str) -> int | None:
    try:
        parsed = UUID(value)
    except ValueError:
        return None
    return parsed.int >> 80 if parsed.version == 7 else None


class ControlProofSpec004Model:
    """Deterministic Spec 004 substitute: the criterion text picks what the model cites.

    ``OTHER_CRITERION`` reuses the Evidence ID this instance was given for an earlier criterion,
    scoped by company and report request (the worker's Outbox event ID). A UUIDv7 millisecond
    match is an additional stale-retry check, not report identity. Another company's or another
    request's memory is never used; ``OTHER_APPLICANT`` cites only the marker's explicit ID.
    """

    fixture_id = SPEC004_FIXTURE_ID
    fixture_digest = SPEC004_FIXTURE_DIGEST

    def __init__(self, observer_root: Path | None = None) -> None:
        self._observer_root = observer_root
        self._h03 = ControlProofFixedModel()
        self._provided: OrderedDict[tuple[UUID, UUID, str], str] = OrderedDict()
        self._lock = threading.Lock()

    def generate(
        self,
        context: TenantContext,
        model_input: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        require_tenant_context(context)
        scope = (context.company_id, context.request_id)
        payload = _task_payload(model_input)
        if payload.get("task") != "assess_interview_criterion":
            return self._h03.generate(context, model_input)
        criterion_id = str(payload["criterion"]["criterion_id"])
        provided = [str(item["evidence_id"]) for item in payload.get("provided_answers", [])]
        mode, argument, score, status = _spec004_marker(str(payload["criterion"].get("text", "")))
        if mode == "OTHER_CRITERION" and argument == criterion_id:
            mode, argument, score, status = "DEFAULT", None, None, "MARKER_INVALID"
        quoted: list[str] = []
        emitted_score: int | None = None
        if provided:
            quoted, status = self._citation(scope, mode, argument, provided, status)
            if status != "MODE_SOURCE_MISSING":
                emitted_score = _SPEC004_DEFAULT_SCORE if score is None else score
            self._remember(scope, criterion_id, provided[0])
        self._write_receipt(criterion_id, mode, status, provided, quoted, emitted_score)
        return {
            "criterion_id": payload["criterion"]["criterion_id"],
            "assessment_state": "confirmed" if quoted else "insufficient_evidence",
            "axis_scores": [
                {
                    "axis": axis["key"],
                    "score": emitted_score,
                    "rationale": _SPEC004_RATIONALE,
                    "quoted_evidence_ids": list(quoted),
                }
                for axis in payload.get("axes", [])
            ],
            "summary": "고정 fixture가 생성한 결정론적 합성 평가입니다.",
            "follow_up_question": None,
        }

    def _citation(
        self,
        scope: tuple[UUID, UUID],
        mode: str,
        argument: str | None,
        provided: list[str],
        status: str,
    ) -> tuple[list[str], str]:
        if mode in {"DEFAULT", "VALID"}:
            return provided[:1], status
        if mode == "EMPTY":
            return [], status
        if mode in {"NONEXISTENT", "OTHER_APPLICANT"}:
            return [str(argument)], status
        with self._lock:
            remembered = self._provided.get((*scope, str(argument)))
        current = _uuid7_millis(provided[0])
        if remembered is None or current is None or _uuid7_millis(remembered) != current:
            return [], "MODE_SOURCE_MISSING"
        return [remembered], status

    def _remember(self, scope: tuple[UUID, UUID], criterion_id: str, evidence_id: str) -> None:
        key = (*scope, criterion_id)
        with self._lock:
            self._provided[key] = evidence_id
            self._provided.move_to_end(key)
            while len(self._provided) > _SPEC004_MEMORY_LIMIT:
                self._provided.popitem(last=False)

    def _write_receipt(
        self,
        criterion_id: str,
        mode: str,
        status: str,
        provided: list[str],
        quoted: list[str],
        score: int | None,
    ) -> None:
        if self._observer_root is None:
            return
        receipt_id = str(uuid4())
        receipt = {
            "schema_version": "controlproof.spec004-model-emission.v1",
            "receipt_id": receipt_id,
            "fixture_id": SPEC004_FIXTURE_ID,
            "criterion_id": criterion_id,
            "mode": mode,
            "mode_status": status,
            "provided_evidence_ids": provided,
            "emitted_quoted_ids": quoted,
            "emitted_score": score,
            "emitted_at": datetime.now(UTC).isoformat(),
        }
        directory = self._observer_root / "model"
        temporary = directory / f"{receipt_id}.tmp"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(receipt, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / f"{receipt_id}.json")
        except OSError:
            # An observer receipt must never fail the target's report generation.
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)


class ControlProofFixedEmbedder:
    """Deterministic local vector source for ControlProof-only report projection."""

    model_id = "controlproof-fixed-embedding-v1"
    embedding_version = "controlproof-h03-v1"

    def embed(
        self,
        context: TenantContext,
        text: str,
        *,
        dimensions: int = 1024,
    ) -> tuple[float, ...]:
        require_tenant_context(context)
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        digest = hashlib.sha256(text.encode()).digest()
        values = tuple((digest[index % len(digest)] - 127.5) / 127.5 for index in range(dimensions))
        magnitude = math.sqrt(sum(value * value for value in values))
        return tuple(value / magnitude for value in values)


def resolve_controlproof_model(
    environment: Mapping[str, str],
    fallback: Any,
) -> Any:
    validate_controlproof_test_controls(environment)
    if not _enabled(environment):
        return fallback
    if _requested_fixture(environment) == SPEC004_FIXTURE_ID:
        observer_root = environment.get("CONTROLPROOF_OBSERVER_ROOT", "").strip()
        return ControlProofSpec004Model(Path(observer_root) if observer_root else None)
    return ControlProofFixedModel()


def resolve_controlproof_embedder(
    environment: Mapping[str, str],
    fallback: Any,
) -> Any:
    validate_controlproof_test_controls(environment)
    if not _enabled(environment):
        return fallback
    _requested_fixture(environment)
    return ControlProofFixedEmbedder()


def controlproof_health(environment: Mapping[str, str]) -> dict[str, Any]:
    validate_controlproof_test_controls(environment)
    model_enabled = _enabled(environment)
    fixture_id = _requested_fixture(environment) if model_enabled else None
    hooks_enabled = environment.get(
        "CONTROLPROOF_TEST_HOOKS_ENABLED", "false"
    ).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "fault_hooks_enabled": hooks_enabled,
        "fault_root_digest": _fault_root_digest(environment) if hooks_enabled else None,
        "model_substitute_enabled": model_enabled,
        "fixture_id": fixture_id,
        "fixture_digest": _FIXTURE_DIGESTS[fixture_id] if fixture_id else None,
        "external_ai_isolated": model_enabled,
        "ai_isolation_digest": (
            controlproof_ai_isolation_digest(environment) if model_enabled else None
        ),
    }


def _fault_root_digest(environment: Mapping[str, str]) -> str | None:
    raw = environment.get("CONTROLPROOF_FAULT_ROOT", "").strip()
    if not raw:
        return None
    normalized = Path(raw).resolve().as_posix().casefold().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _task_payload(model_input: Mapping[str, Any]) -> dict[str, Any]:
    messages = model_input.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("model input has no messages")
    content = messages[0].get("content")
    if not isinstance(content, list):
        raise ValueError("model input content is invalid")
    text = "".join(
        item.get("text", "")
        for item in content
        if isinstance(item, Mapping) and item.get("type", "text") == "text"
    )
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("model task payload must be an object")
    return deepcopy(decoded)
