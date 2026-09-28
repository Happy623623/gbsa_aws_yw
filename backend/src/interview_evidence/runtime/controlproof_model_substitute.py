"""Fixed local/test AI dependency for deterministic ControlProof report runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from interview_evidence.shared.tenant import TenantContext, require_tenant_context

FIXTURE_ID = "h03-report-v1"
FIXTURE_SEED = "controlproof:h03-report-v1"
FIXTURE_DIGEST = hashlib.sha256(FIXTURE_SEED.encode("utf-8")).hexdigest()
ALLOWED_ENVIRONMENTS = frozenset({"local", "test"})


def _enabled(environment: Mapping[str, str]) -> bool:
    return environment.get("CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def validate_controlproof_test_controls(environment: Mapping[str, str]) -> None:
    profile = environment.get("APP_ENVIRONMENT", "").strip().casefold()
    if (
        _enabled(environment)
        or environment.get("CONTROLPROOF_TEST_HOOKS_ENABLED", "false").strip().casefold()
        in {"1", "true", "yes", "on"}
    ) and profile not in ALLOWED_ENVIRONMENTS:
        raise RuntimeError("ControlProof test controls are forbidden outside local/test")


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
    requested = environment.get("CONTROLPROOF_MODEL_FIXTURE_ID", FIXTURE_ID).strip()
    if requested != FIXTURE_ID:
        raise RuntimeError("unsupported ControlProof model fixture ID")
    return ControlProofFixedModel()


def resolve_controlproof_embedder(
    environment: Mapping[str, str],
    fallback: Any,
) -> Any:
    validate_controlproof_test_controls(environment)
    if not _enabled(environment):
        return fallback
    requested = environment.get("CONTROLPROOF_MODEL_FIXTURE_ID", FIXTURE_ID).strip()
    if requested != FIXTURE_ID:
        raise RuntimeError("unsupported ControlProof model fixture ID")
    return ControlProofFixedEmbedder()


def controlproof_health(environment: Mapping[str, str]) -> dict[str, Any]:
    validate_controlproof_test_controls(environment)
    model_enabled = _enabled(environment)
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
        "fixture_id": FIXTURE_ID if model_enabled else None,
        "fixture_digest": FIXTURE_DIGEST if model_enabled else None,
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
