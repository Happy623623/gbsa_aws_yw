from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol, TypeVar
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    delete,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    Mapped,
    Session,
    mapped_column,
)

from interview_evidence.reporting.domain.deletion import (
    DeletionManifest,
    DeletionRequest,
    DeletionTarget,
    TargetStatus,
)
from interview_evidence.reporting.domain.failure import ReportGenerationFailure
from interview_evidence.reporting.domain.report import (
    AssessmentState,
    AxisAssessment,
    Evidence,
    Report,
    ReportItem,
    ReportKind,
    ReportStatus,
    RequirementAssessment,
    RequirementAssessmentStatus,
    RequirementEvidence,
    RequirementEvidenceRelation,
    Sufficiency,
)
from interview_evidence.reporting.domain.review import HumanReview, ReviewType
from interview_evidence.reporting.domain.timeline import (
    RecordingAsset,
    RecordingStatus,
    SessionEvent,
    TranscriptSegment,
)
from interview_evidence.shared.tenant import TenantContext, require_tenant_context


class TenantScopedReportingNotFound(LookupError):
    """Raised without revealing another tenant's reporting resources."""


class TenantOwned(Protocol):
    @property
    def company_id(self) -> UUID: ...


TenantOwnedT = TypeVar("TenantOwnedT", bound=TenantOwned)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _scoring_inputs(report: Report) -> dict[str, object]:
    """Record the arithmetic that produced this report's score.

    The score a reviewer reads is protected by ``report_items.criterion_weight`` and
    ``axis_weights``, not by this: those are frozen per item and the display path re-derives
    from them, so a company adjusting its weights next month cannot change a number someone
    already acted on. What this adds is the arithmetic *as it ran* -- which re-derivation cannot
    reproduce if the arithmetic changes -- and an answer to "why 74?" that survives the version
    it came from being superseded or deleted.

    ``numerator`` and ``denominator`` are kept even though they are recoverable from the
    contributions, because they are what the calculator renders (``55.7 ÷ 0.75 = 74``).

    No reader yet. Deliberately written before there is one: the values are only useful if they
    were captured at generation time, and a report is immutable, so a column added later would
    be empty for every report that already exists.
    """
    aggregate = report.criterion_aggregate
    communication = report.communication_aggregate
    return {
        "criteria": [
            {
                "criterion_id": contribution.key,
                "score": contribution.score,
                "weight": contribution.weight,
                "normalized_weight": contribution.normalized_weight,
                "contribution": contribution.contribution,
            }
            for contribution in aggregate.contributions
        ],
        # Excluded criteria carry weight but no score. Recorded here so the calculator can show
        # the shortfall against 1.0 with a reason next to it, rather than a divisor that appears
        # from nowhere.
        "excluded": [
            {
                "criterion_id": exclusion.key,
                "weight": exclusion.weight,
                "normalized_weight": exclusion.normalized_weight,
            }
            for exclusion in aggregate.exclusions
        ],
        "numerator": aggregate.numerator,
        "denominator": aggregate.denominator,
        "axis_weights": {
            str(item.criterion_id): dict(item.axis_weights)
            for item in report.items
            if item.axis_weights
        },
        "communication": {
            "score": communication.score,
            "numerator": communication.numerator,
            "denominator": communication.denominator,
            "scored_criteria": [item.key for item in communication.contributions],
            "unscored_criteria": [item.key for item in communication.exclusions],
        },
    }


def _stored_axes(axes: tuple[AxisAssessment, ...]) -> list[dict[str, object]]:
    return [
        {
            "axis": axis.axis,
            "label": axis.label,
            "score": axis.score,
            "rationale": axis.rationale,
            "quoted_evidence_ids": [str(value) for value in axis.quoted_evidence_ids],
        }
        for axis in axes
    ]


def _restored_axes(stored: object) -> tuple[AxisAssessment, ...]:
    """Rebuild the axis scores, skipping rows this build cannot read.

    A stored axis that no longer parses -- an axis key retired since it was written, a
    truncated row -- is dropped rather than raised on, because a reviewer needs the report
    itself more than they need any one score, and the report is an immutable original we
    cannot rewrite to fix.
    """
    if not isinstance(stored, list):
        return ()
    restored: list[AxisAssessment] = []
    for entry in stored:
        if not isinstance(entry, Mapping):
            continue
        raw_score = entry.get("score")
        cited = entry.get("quoted_evidence_ids")
        try:
            restored.append(
                AxisAssessment(
                    axis=str(entry["axis"]),
                    label=str(entry.get("label", entry["axis"])),
                    score=None if raw_score is None else int(str(raw_score)),
                    rationale=str(entry["rationale"]),
                    quoted_evidence_ids=tuple(UUID(str(value)) for value in cited)
                    if isinstance(cited, list)
                    else (),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(restored)


def _stored_requirement_assessments(
    assessments: tuple[RequirementAssessment, ...],
) -> list[dict[str, object]]:
    return [
        {
            "requirement_assessment_id": str(item.requirement_assessment_id),
            "job_requirement_id": str(item.job_requirement_id),
            "requirement_type": item.requirement_type,
            "statement": item.statement,
            "status": item.status.value,
            "rationale": item.rationale,
            "confidence": item.confidence,
            "evidence": [
                {
                    "evidence_id": str(evidence.evidence_id),
                    "source_kind": evidence.source_kind,
                    "source_type": evidence.source_type,
                    "excerpt": evidence.excerpt,
                    "locator": dict(evidence.locator),
                    "relation": evidence.relation.value,
                    "explanation": evidence.explanation,
                }
                for evidence in item.evidence
            ],
        }
        for item in assessments
    ]


def _restored_requirement_assessments(
    stored: object,
) -> tuple[RequirementAssessment, ...]:
    if not isinstance(stored, list):
        return ()
    restored: list[RequirementAssessment] = []
    for entry in stored:
        if not isinstance(entry, Mapping):
            continue
        raw_evidence = entry.get("evidence")
        evidence: list[RequirementEvidence] = []
        if isinstance(raw_evidence, list):
            for raw in raw_evidence:
                if not isinstance(raw, Mapping):
                    continue
                try:
                    evidence.append(
                        RequirementEvidence(
                            evidence_id=UUID(str(raw["evidence_id"])),
                            source_kind=str(raw["source_kind"]),
                            source_type=str(raw["source_type"]),
                            excerpt=str(raw["excerpt"]),
                            locator=(
                                dict(raw["locator"])
                                if isinstance(raw.get("locator"), Mapping)
                                else {}
                            ),
                            relation=RequirementEvidenceRelation(str(raw["relation"])),
                            explanation=str(raw["explanation"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        try:
            restored.append(
                RequirementAssessment(
                    requirement_assessment_id=UUID(str(entry["requirement_assessment_id"])),
                    job_requirement_id=UUID(str(entry["job_requirement_id"])),
                    requirement_type=str(entry["requirement_type"]),
                    statement=str(entry["statement"]),
                    status=RequirementAssessmentStatus(str(entry["status"])),
                    rationale=str(entry["rationale"]),
                    confidence=float(entry["confidence"]),
                    evidence=tuple(evidence),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(restored)


class Base(DeclarativeBase):
    pass


class TranscriptSegmentRow(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint("company_id", "turn_id", "version", name="uq_transcript_segment_version"),
        Index("ix_transcript_segments_session", "company_id", "interview_session_id"),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    transcript_segment_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    interview_session_id: Mapped[UUID] = mapped_column(Uuid)
    turn_id: Mapped[UUID] = mapped_column(Uuid)
    speaker: Mapped[str] = mapped_column(String(30))
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    session_start_ms: Mapped[int] = mapped_column(Integer)
    session_end_ms: Mapped[int] = mapped_column(Integer)
    source_audio_key: Mapped[str] = mapped_column(String(2048))
    version: Mapped[int] = mapped_column(Integer)
    corrected_by: Mapped[UUID | None] = mapped_column(Uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RecordingAssetRow(Base):
    __tablename__ = "recording_assets"
    __table_args__ = (Index("ix_recording_assets_session", "company_id", "interview_session_id"),)

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    recording_asset_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    interview_session_id: Mapped[UUID] = mapped_column(Uuid)
    asset_type: Mapped[str] = mapped_column(String(30))
    object_key: Mapped[str] = mapped_column(String(2048))
    content_hash: Mapped[str] = mapped_column(String(64))
    duration_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30))
    missing_ranges: Mapped[list[list[int]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SessionEventRow(Base):
    __tablename__ = "session_events"
    __table_args__ = (Index("ix_session_events_session", "company_id", "interview_session_id"),)

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    session_event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    interview_session_id: Mapped[UUID] = mapped_column(Uuid)
    event_type: Mapped[str] = mapped_column(String(100))
    session_start_ms: Mapped[int] = mapped_column(Integer)
    session_end_ms: Mapped[int] = mapped_column(Integer)
    technical_failure: Mapped[bool] = mapped_column(Boolean)
    details: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ReportRow(Base):
    __tablename__ = "reports"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "interview_session_id", "version", name="uq_report_session_version"
        ),
        Index("ix_reports_invitation", "company_id", "invitation_id"),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    report_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    interview_session_id: Mapped[UUID] = mapped_column(Uuid)
    invitation_id: Mapped[UUID] = mapped_column(Uuid)
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(30))
    model_version: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(100))
    config_version: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(30))
    summary: Mapped[str] = mapped_column(Text)
    #: The weighted score, denormalised onto the row so the applicant list can sort and filter
    #: on it in one query. It is derived from the items, but recomputing it per row would make
    #: the invitation list load every report's items to order a column.
    #:
    #: Nullable because a report where nothing could be scored has no score -- never zero,
    #: which would read as "every answer was wrong".
    overall_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: The arithmetic this report was produced with: the weights, the numerator, the divisor and
    #: what was excluded. Frozen, for the same reason the report itself is immutable -- a company
    #: adjusting its weights next month must not silently change a score a reviewer already
    #: acted on, and "why 74?" has to stay answerable from the report alone.
    scoring_inputs: Mapped[dict[str, object]] = mapped_column(JSON, server_default="{}")
    requirement_assessments: Mapped[list[dict[str, object]]] = mapped_column(
        JSON,
        server_default="[]",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ReportGenerationFailureRow(Base):
    __tablename__ = "report_generation_failures"
    __table_args__ = (
        Index(
            "ix_report_generation_failures_session",
            "company_id",
            "interview_session_id",
            "failed_at",
        ),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    source_event_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    interview_session_id: Mapped[UUID] = mapped_column(Uuid)
    last_delivery_attempt: Mapped[int] = mapped_column(Integer)
    error_code: Mapped[str] = mapped_column(String(100))
    failed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ReportItemRow(Base):
    __tablename__ = "report_items"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "report_id"],
            ["reports.company_id", "reports.report_id"],
            name="fk_report_items_company_id_reports",
        ),
        Index("ix_report_items_report", "company_id", "report_id"),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    report_item_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    report_id: Mapped[UUID] = mapped_column(Uuid)
    criterion_id: Mapped[UUID] = mapped_column(Uuid)
    criterion_name: Mapped[str] = mapped_column(String(200), server_default="")
    competency_model_version_id: Mapped[UUID] = mapped_column(Uuid)
    assessment_state: Mapped[str] = mapped_column(String(40))
    observation: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    sufficiency: Mapped[str] = mapped_column(String(30))
    uncertainty: Mapped[str] = mapped_column(Text)
    follow_up_question: Mapped[str | None] = mapped_column(Text)
    #: The model's per-axis scores, stored as one JSON array rather than a child table:
    #: they are read and written with their item and never queried across reports.
    axis_assessments: Mapped[list[dict[str, object]]] = mapped_column(JSON, server_default="[]")
    #: What this criterion counted for, snapshotted from the published version. 1.0 for rows
    #: written before weights were applied, which with every row at 1.0 reproduces the plain
    #: mean those reports were scored with.
    criterion_weight: Mapped[float] = mapped_column(Float, server_default="1.0")
    #: What each axis counted for within this criterion. Empty means equal weight, on the same
    #: reasoning.
    axis_weights: Mapped[dict[str, float]] = mapped_column(JSON, server_default="{}")


class EvidenceRow(Base):
    __tablename__ = "evidence"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "report_item_id"],
            ["report_items.company_id", "report_items.report_item_id"],
            name="fk_evidence_company_id_report_items",
        ),
        Index("ix_evidence_report_item", "company_id", "report_item_id"),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    evidence_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    report_item_id: Mapped[UUID] = mapped_column(Uuid)
    criterion_id: Mapped[UUID] = mapped_column(Uuid)
    competency_model_version_id: Mapped[UUID] = mapped_column(Uuid)
    answer_turn_id: Mapped[UUID] = mapped_column(Uuid)
    transcript_segment_id: Mapped[UUID] = mapped_column(Uuid)
    video_start_ms: Mapped[int] = mapped_column(Integer)
    video_end_ms: Mapped[int] = mapped_column(Integer)
    observation: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    sufficiency: Mapped[str] = mapped_column(String(30))
    generation_version: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class HumanReviewRow(Base):
    __tablename__ = "human_reviews"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "report_id"],
            ["reports.company_id", "reports.report_id"],
            name="fk_human_reviews_company_id_reports",
        ),
        Index("ix_human_reviews_report", "company_id", "report_id"),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    human_review_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    report_id: Mapped[UUID] = mapped_column(Uuid)
    company_user_id: Mapped[UUID] = mapped_column(Uuid)
    review_type: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[UUID] = mapped_column(Uuid)
    value: Mapped[dict[str, str]] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DeletionRequestRow(Base):
    __tablename__ = "deletion_requests"

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    deletion_request_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(30))
    scope_id: Mapped[UUID] = mapped_column(Uuid)
    reason: Mapped[str] = mapped_column(Text)
    requester_type: Mapped[str] = mapped_column(String(30))
    requester_id: Mapped[UUID] = mapped_column(Uuid)
    policy_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DeletionManifestRow(Base):
    __tablename__ = "deletion_manifests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "deletion_request_id"],
            ["deletion_requests.company_id", "deletion_requests.deletion_request_id"],
            name="fk_deletion_manifests_company_id_deletion_requests",
        ),
        Index("ix_deletion_manifests_request", "company_id", "deletion_request_id"),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    manifest_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    deletion_request_id: Mapped[UUID] = mapped_column(Uuid)
    manifest_version: Mapped[int] = mapped_column(Integer)


class DeletionTargetRow(Base):
    __tablename__ = "deletion_targets"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "manifest_id"],
            ["deletion_manifests.company_id", "deletion_manifests.manifest_id"],
            name="fk_deletion_targets_company_id_deletion_manifests",
        ),
        Index("ix_deletion_targets_manifest", "company_id", "manifest_id"),
    )

    company_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    target_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    manifest_id: Mapped[UUID] = mapped_column(Uuid)
    owner_lane: Mapped[str] = mapped_column(String(1))
    store: Mapped[str] = mapped_column(String(30))
    target_type: Mapped[str] = mapped_column(String(100))
    resource_id: Mapped[str] = mapped_column(String(2048))
    status: Mapped[str] = mapped_column(String(30))
    attempts: Mapped[int] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(100))
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReportingRepository(Protocol):
    def save_transcript(
        self, context: TenantContext, segment: TranscriptSegment
    ) -> TranscriptSegment: ...
    def list_transcripts(
        self, context: TenantContext, session_id: UUID
    ) -> tuple[TranscriptSegment, ...]: ...
    def save_recording_asset(
        self, context: TenantContext, asset: RecordingAsset
    ) -> RecordingAsset: ...
    def list_recording_assets(
        self, context: TenantContext, session_id: UUID
    ) -> tuple[RecordingAsset, ...]: ...
    def save_session_event(self, context: TenantContext, event: SessionEvent) -> SessionEvent: ...
    def list_session_events(
        self, context: TenantContext, session_id: UUID
    ) -> tuple[SessionEvent, ...]: ...
    def save_report(self, context: TenantContext, report: Report) -> Report: ...
    def get_report(self, context: TenantContext, report_id: UUID) -> Report: ...
    def get_report_for_session(self, context: TenantContext, session_id: UUID) -> Report | None: ...
    def get_report_for_invitation(
        self, context: TenantContext, invitation_id: UUID
    ) -> Report | None: ...
    def save_generation_failure(
        self,
        context: TenantContext,
        failure: ReportGenerationFailure,
    ) -> ReportGenerationFailure: ...
    def get_generation_failure_for_session(
        self,
        context: TenantContext,
        session_id: UUID,
    ) -> ReportGenerationFailure | None: ...
    def save_review(self, context: TenantContext, review: HumanReview) -> HumanReview: ...
    def get_review(self, context: TenantContext, review_id: UUID) -> HumanReview: ...
    def list_reviews(self, context: TenantContext, report_id: UUID) -> tuple[HumanReview, ...]: ...
    def save_deletion(
        self,
        context: TenantContext,
        request: DeletionRequest,
        manifest: DeletionManifest,
    ) -> DeletionManifest: ...
    def get_deletion(
        self, context: TenantContext, request_id: UUID
    ) -> tuple[DeletionRequest, DeletionManifest]: ...
    def update_deletion_manifest(
        self,
        context: TenantContext,
        request: DeletionRequest,
        manifest: DeletionManifest,
    ) -> DeletionManifest: ...
    def delete_and_verify_target(
        self,
        context: TenantContext,
        *,
        store: str,
        target_type: str,
        resource_id: str,
    ) -> bool: ...


class SQLAlchemyReportingRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _tenant(context: TenantContext) -> UUID:
        return require_tenant_context(context).company_id

    def save_transcript(
        self, context: TenantContext, segment: TranscriptSegment
    ) -> TranscriptSegment:
        context.assert_company(segment.company_id)
        self._session.add(
            TranscriptSegmentRow(
                transcript_segment_id=segment.transcript_segment_id,
                company_id=segment.company_id,
                interview_session_id=segment.interview_session_id,
                turn_id=segment.turn_id,
                speaker=segment.speaker,
                text=segment.text,
                confidence=segment.confidence,
                session_start_ms=segment.session_start_ms,
                session_end_ms=segment.session_end_ms,
                source_audio_key=segment.source_audio_key,
                version=segment.version,
                corrected_by=segment.corrected_by,
                created_at=segment.created_at,
            )
        )
        self._session.flush()
        return segment

    def list_transcripts(
        self, context: TenantContext, session_id: UUID
    ) -> tuple[TranscriptSegment, ...]:
        rows = self._session.scalars(
            select(TranscriptSegmentRow)
            .where(
                TranscriptSegmentRow.company_id == self._tenant(context),
                TranscriptSegmentRow.interview_session_id == session_id,
            )
            .order_by(TranscriptSegmentRow.session_start_ms, TranscriptSegmentRow.version)
        )
        return tuple(
            TranscriptSegment(
                transcript_segment_id=row.transcript_segment_id,
                company_id=row.company_id,
                interview_session_id=row.interview_session_id,
                turn_id=row.turn_id,
                speaker=row.speaker,
                text=row.text,
                confidence=row.confidence,
                session_start_ms=row.session_start_ms,
                session_end_ms=row.session_end_ms,
                source_audio_key=row.source_audio_key,
                version=row.version,
                corrected_by=row.corrected_by,
                created_at=_aware(row.created_at),
            )
            for row in rows
        )

    def save_recording_asset(self, context: TenantContext, asset: RecordingAsset) -> RecordingAsset:
        context.assert_company(asset.company_id)
        self._session.add(
            RecordingAssetRow(
                recording_asset_id=asset.recording_asset_id,
                company_id=asset.company_id,
                interview_session_id=asset.interview_session_id,
                asset_type=asset.asset_type,
                object_key=asset.object_key,
                content_hash=asset.content_hash,
                duration_ms=asset.duration_ms,
                status=asset.status.value,
                missing_ranges=[list(item) for item in asset.missing_ranges],
                created_at=asset.created_at,
            )
        )
        self._session.flush()
        return asset

    def list_recording_assets(
        self, context: TenantContext, session_id: UUID
    ) -> tuple[RecordingAsset, ...]:
        rows = self._session.scalars(
            select(RecordingAssetRow).where(
                RecordingAssetRow.company_id == self._tenant(context),
                RecordingAssetRow.interview_session_id == session_id,
            )
        )
        return tuple(
            RecordingAsset(
                recording_asset_id=row.recording_asset_id,
                company_id=row.company_id,
                interview_session_id=row.interview_session_id,
                asset_type=row.asset_type,
                object_key=row.object_key,
                content_hash=row.content_hash,
                duration_ms=row.duration_ms,
                status=RecordingStatus(row.status),
                missing_ranges=tuple((item[0], item[1]) for item in row.missing_ranges),
                created_at=_aware(row.created_at),
            )
            for row in rows
        )

    def save_session_event(self, context: TenantContext, event: SessionEvent) -> SessionEvent:
        context.assert_company(event.company_id)
        self._session.add(
            SessionEventRow(
                session_event_id=event.session_event_id,
                company_id=event.company_id,
                interview_session_id=event.interview_session_id,
                event_type=event.event_type,
                session_start_ms=event.session_start_ms,
                session_end_ms=event.session_end_ms,
                technical_failure=event.technical_failure,
                details=event.details,
                created_at=event.created_at,
            )
        )
        self._session.flush()
        return event

    def list_session_events(
        self, context: TenantContext, session_id: UUID
    ) -> tuple[SessionEvent, ...]:
        rows = self._session.scalars(
            select(SessionEventRow)
            .where(
                SessionEventRow.company_id == self._tenant(context),
                SessionEventRow.interview_session_id == session_id,
            )
            .order_by(SessionEventRow.session_start_ms)
        )
        return tuple(
            SessionEvent(
                session_event_id=row.session_event_id,
                company_id=row.company_id,
                interview_session_id=row.interview_session_id,
                event_type=row.event_type,
                session_start_ms=row.session_start_ms,
                session_end_ms=row.session_end_ms,
                technical_failure=row.technical_failure,
                details=row.details,
                created_at=_aware(row.created_at),
            )
            for row in rows
        )

    def save_report(self, context: TenantContext, report: Report) -> Report:
        context.assert_company(report.company_id)
        if self._session.scalar(
            select(ReportRow).where(
                ReportRow.company_id == self._tenant(context),
                ReportRow.report_id == report.report_id,
            )
        ):
            raise ValueError("AI original report is immutable")
        self._session.add(
            ReportRow(
                report_id=report.report_id,
                company_id=report.company_id,
                interview_session_id=report.interview_session_id,
                invitation_id=report.invitation_id,
                version=report.version,
                kind=report.kind.value,
                model_version=report.model_version,
                prompt_version=report.prompt_version,
                config_version=report.config_version,
                status=report.status.value,
                summary=report.summary,
                # Recorded here from the weights the items carry. Note that the read path does
                # *not* use it: `_report_view` and the invitation projection both call
                # `Report.overall_score`, which re-derives the figure from
                # `report_items.criterion_weight`. That is safe -- those weights are frozen per
                # item, so a company re-weighting its criteria cannot move a stored report's
                # score, which is the guarantee this track set out to make.
                #
                # What the column and `scoring_inputs` add is a record of the arithmetic as it
                # ran, which re-derivation cannot reproduce if the arithmetic itself ever
                # changes (rounding, normalisation). Nothing reads them yet; making the stored
                # value authoritative on read is a separate change, and it needs a test that
                # compares the two rather than trusting that they agree.
                overall_score=report.overall_score,
                scoring_inputs=_scoring_inputs(report),
                requirement_assessments=_stored_requirement_assessments(
                    report.requirement_assessments
                ),
                created_at=report.created_at,
            )
        )
        # Flushed per foreign key level, parents first. No relationship() links these
        # mappers, so a single flush would have SQLAlchemy order the inserts by mapper
        # sort key -- alphabetically -- which puts evidence before report_items and
        # report_items before reports, the exact reverse of what the constraints allow.
        self._session.flush()
        for item in report.items:
            self._session.add(
                ReportItemRow(
                    report_item_id=item.report_item_id,
                    company_id=item.company_id,
                    report_id=item.report_id,
                    criterion_id=item.criterion_id,
                    criterion_name=item.criterion_name,
                    competency_model_version_id=item.competency_model_version_id,
                    assessment_state=item.assessment_state.value,
                    observation=item.observation,
                    rationale=item.rationale,
                    sufficiency=item.sufficiency,
                    uncertainty=item.uncertainty,
                    follow_up_question=item.follow_up_question,
                    axis_assessments=_stored_axes(item.axis_assessments),
                    criterion_weight=item.criterion_weight,
                    axis_weights=dict(item.axis_weights),
                )
            )
        self._session.flush()
        for item in report.items:
            for evidence in item.evidence:
                self._session.add(
                    EvidenceRow(
                        evidence_id=evidence.evidence_id,
                        company_id=evidence.company_id,
                        report_item_id=evidence.report_item_id,
                        criterion_id=evidence.criterion_id,
                        competency_model_version_id=evidence.competency_model_version_id,
                        answer_turn_id=evidence.answer_turn_id,
                        transcript_segment_id=evidence.transcript_segment_id,
                        video_start_ms=evidence.video_start_ms,
                        video_end_ms=evidence.video_end_ms,
                        observation=evidence.observation,
                        rationale=evidence.rationale,
                        sufficiency=evidence.sufficiency.value,
                        generation_version=evidence.generation_version,
                        created_at=evidence.created_at,
                    )
                )
        self._session.flush()
        return report

    def _report_from_row(self, row: ReportRow) -> Report:
        item_rows = tuple(
            self._session.scalars(
                select(ReportItemRow).where(
                    ReportItemRow.company_id == row.company_id,
                    ReportItemRow.report_id == row.report_id,
                )
            )
        )
        evidence_by_item: dict[UUID, list[Evidence]] = {
            item.report_item_id: [] for item in item_rows
        }
        # One query for the whole report: an item-by-item fetch made the most-viewed
        # screen issue N+1 round trips, each of them tenant-wide.
        evidence_rows = self._session.scalars(
            select(EvidenceRow).where(
                EvidenceRow.company_id == row.company_id,
                EvidenceRow.report_item_id.in_(tuple(evidence_by_item)),
            )
        )
        for value in evidence_rows:
            evidence_by_item[value.report_item_id].append(
                Evidence(
                    evidence_id=value.evidence_id,
                    company_id=value.company_id,
                    report_item_id=value.report_item_id,
                    criterion_id=value.criterion_id,
                    competency_model_version_id=value.competency_model_version_id,
                    answer_turn_id=value.answer_turn_id,
                    transcript_segment_id=value.transcript_segment_id,
                    video_start_ms=value.video_start_ms,
                    video_end_ms=value.video_end_ms,
                    observation=value.observation,
                    rationale=value.rationale,
                    sufficiency=Sufficiency(value.sufficiency),
                    generation_version=value.generation_version,
                    created_at=_aware(value.created_at),
                )
            )
        items = [
            ReportItem(
                report_item_id=item.report_item_id,
                company_id=item.company_id,
                report_id=item.report_id,
                criterion_id=item.criterion_id,
                criterion_name=item.criterion_name,
                competency_model_version_id=item.competency_model_version_id,
                assessment_state=AssessmentState(item.assessment_state),
                observation=item.observation,
                rationale=item.rationale,
                sufficiency=item.sufficiency,
                uncertainty=item.uncertainty,
                evidence=tuple(evidence_by_item[item.report_item_id]),
                follow_up_question=item.follow_up_question,
                axis_assessments=_restored_axes(item.axis_assessments),
                # A row written before weights existed holds the column default, and a legacy
                # row can hold SQL NULL. Both mean "not weighted", which the domain reads as
                # equal -- the arithmetic those reports were actually scored with.
                criterion_weight=(
                    item.criterion_weight if item.criterion_weight is not None else 1.0
                ),
                axis_weights=dict(item.axis_weights or {}),
            )
            for item in item_rows
        ]
        return Report(
            report_id=row.report_id,
            company_id=row.company_id,
            interview_session_id=row.interview_session_id,
            invitation_id=row.invitation_id,
            version=row.version,
            kind=ReportKind(row.kind),
            model_version=row.model_version,
            prompt_version=row.prompt_version,
            config_version=row.config_version,
            status=ReportStatus(row.status),
            summary=row.summary,
            created_at=_aware(row.created_at),
            items=tuple(items),
            requirement_assessments=_restored_requirement_assessments(row.requirement_assessments),
        )

    def get_report(self, context: TenantContext, report_id: UUID) -> Report:
        row = self._session.scalar(
            select(ReportRow).where(
                ReportRow.company_id == self._tenant(context),
                ReportRow.report_id == report_id,
            )
        )
        if row is None:
            raise TenantScopedReportingNotFound("reporting resource not found")
        return self._report_from_row(row)

    def _latest_report(
        self,
        context: TenantContext,
        column: InstrumentedAttribute[UUID],
        value: UUID,
    ) -> Report | None:
        row = self._session.scalar(
            select(ReportRow)
            .where(ReportRow.company_id == self._tenant(context), column == value)
            .order_by(ReportRow.version.desc())
        )
        return None if row is None else self._report_from_row(row)

    def get_report_for_session(self, context: TenantContext, session_id: UUID) -> Report | None:
        return self._latest_report(context, ReportRow.interview_session_id, session_id)

    def get_report_for_invitation(
        self, context: TenantContext, invitation_id: UUID
    ) -> Report | None:
        return self._latest_report(context, ReportRow.invitation_id, invitation_id)

    def save_generation_failure(
        self,
        context: TenantContext,
        failure: ReportGenerationFailure,
    ) -> ReportGenerationFailure:
        context.assert_company(failure.company_id)
        row = self._session.scalar(
            select(ReportGenerationFailureRow).where(
                ReportGenerationFailureRow.company_id == self._tenant(context),
                ReportGenerationFailureRow.source_event_id == failure.source_event_id,
            )
        )
        if row is None:
            row = ReportGenerationFailureRow(
                company_id=failure.company_id,
                source_event_id=failure.source_event_id,
                interview_session_id=failure.interview_session_id,
                last_delivery_attempt=failure.last_delivery_attempt,
                error_code=failure.error_code,
                failed_at=failure.failed_at,
            )
            self._session.add(row)
        else:
            row.interview_session_id = failure.interview_session_id
            row.last_delivery_attempt = max(
                row.last_delivery_attempt,
                failure.last_delivery_attempt,
            )
            row.error_code = failure.error_code
            row.failed_at = failure.failed_at
        self._session.flush()
        return failure

    def get_generation_failure_for_session(
        self,
        context: TenantContext,
        session_id: UUID,
    ) -> ReportGenerationFailure | None:
        row = self._session.scalar(
            select(ReportGenerationFailureRow)
            .where(
                ReportGenerationFailureRow.company_id == self._tenant(context),
                ReportGenerationFailureRow.interview_session_id == session_id,
            )
            .order_by(ReportGenerationFailureRow.failed_at.desc())
        )
        if row is None:
            return None
        return ReportGenerationFailure(
            company_id=row.company_id,
            interview_session_id=row.interview_session_id,
            source_event_id=row.source_event_id,
            last_delivery_attempt=row.last_delivery_attempt,
            error_code=row.error_code,
            failed_at=_aware(row.failed_at),
        )

    def save_review(self, context: TenantContext, review: HumanReview) -> HumanReview:
        context.assert_company(review.company_id)
        self.get_report(context, review.report_id)
        self._session.add(
            HumanReviewRow(
                human_review_id=review.human_review_id,
                company_id=review.company_id,
                report_id=review.report_id,
                company_user_id=review.company_user_id,
                review_type=review.review_type.value,
                target_id=review.target_id,
                value=review.value,
                reason=review.reason,
                created_at=review.created_at,
            )
        )
        self._session.flush()
        return review

    def get_review(self, context: TenantContext, review_id: UUID) -> HumanReview:
        row = self._session.scalar(
            select(HumanReviewRow).where(
                HumanReviewRow.company_id == self._tenant(context),
                HumanReviewRow.human_review_id == review_id,
            )
        )
        if row is None:
            raise TenantScopedReportingNotFound("human review not found")
        return HumanReview(
            human_review_id=row.human_review_id,
            company_id=row.company_id,
            report_id=row.report_id,
            company_user_id=row.company_user_id,
            review_type=ReviewType(row.review_type),
            target_id=row.target_id,
            value=row.value,
            reason=row.reason,
            created_at=_aware(row.created_at),
        )

    def list_reviews(self, context: TenantContext, report_id: UUID) -> tuple[HumanReview, ...]:
        self.get_report(context, report_id)
        rows = self._session.scalars(
            select(HumanReviewRow)
            .where(
                HumanReviewRow.company_id == self._tenant(context),
                HumanReviewRow.report_id == report_id,
            )
            .order_by(HumanReviewRow.created_at)
        )
        return tuple(
            HumanReview(
                human_review_id=row.human_review_id,
                company_id=row.company_id,
                report_id=row.report_id,
                company_user_id=row.company_user_id,
                review_type=ReviewType(row.review_type),
                target_id=row.target_id,
                value=row.value,
                reason=row.reason,
                created_at=_aware(row.created_at),
            )
            for row in rows
        )

    def save_deletion(
        self,
        context: TenantContext,
        request: DeletionRequest,
        manifest: DeletionManifest,
    ) -> DeletionManifest:
        context.assert_company(request.company_id)
        self._session.add(
            DeletionRequestRow(
                deletion_request_id=request.deletion_request_id,
                company_id=request.company_id,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
                reason=request.reason,
                requester_type=request.requester_type,
                requester_id=request.requester_id,
                policy_snapshot=request.policy_snapshot,
                requested_at=_aware(request.requested_at),
            )
        )
        # Flushed per foreign key level for the reason given in ``save_report``: mapper
        # sort order puts deletion_manifests ahead of the deletion_requests they point at.
        self._session.flush()
        self._session.add(
            DeletionManifestRow(
                manifest_id=manifest.manifest_id,
                company_id=request.company_id,
                deletion_request_id=request.deletion_request_id,
                manifest_version=manifest.manifest_version,
            )
        )
        self._session.flush()
        for target in manifest.targets:
            self._session.add(
                DeletionTargetRow(
                    target_id=target.target_id,
                    company_id=request.company_id,
                    manifest_id=manifest.manifest_id,
                    owner_lane=target.owner_lane,
                    store=target.store,
                    target_type=target.target_type,
                    resource_id=target.resource_id,
                    status=target.status.value,
                    attempts=target.attempts,
                    error_code=target.error_code,
                    verified_at=target.verified_at,
                )
            )
        self._session.flush()
        return manifest

    def get_deletion(
        self, context: TenantContext, request_id: UUID
    ) -> tuple[DeletionRequest, DeletionManifest]:
        request = self._session.scalar(
            select(DeletionRequestRow).where(
                DeletionRequestRow.company_id == self._tenant(context),
                DeletionRequestRow.deletion_request_id == request_id,
            )
        )
        if request is None:
            raise TenantScopedReportingNotFound("deletion request not found")
        manifest = self._session.scalar(
            select(DeletionManifestRow).where(
                DeletionManifestRow.company_id == self._tenant(context),
                DeletionManifestRow.deletion_request_id == request_id,
            )
        )
        if manifest is None:
            raise TenantScopedReportingNotFound("deletion manifest not found")
        target_rows = self._session.scalars(
            select(DeletionTargetRow).where(
                DeletionTargetRow.company_id == self._tenant(context),
                DeletionTargetRow.manifest_id == manifest.manifest_id,
            )
        )
        return (
            DeletionRequest(
                deletion_request_id=request.deletion_request_id,
                company_id=request.company_id,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
                reason=request.reason,
                requester_type=request.requester_type,
                requester_id=request.requester_id,
                policy_snapshot=request.policy_snapshot,
                requested_at=request.requested_at,
            ),
            DeletionManifest(
                manifest_id=manifest.manifest_id,
                deletion_request_id=manifest.deletion_request_id,
                manifest_version=manifest.manifest_version,
                targets=tuple(
                    DeletionTarget(
                        target_id=row.target_id,
                        owner_lane=row.owner_lane,
                        store=row.store,
                        target_type=row.target_type,
                        resource_id=row.resource_id,
                        status=TargetStatus(row.status),
                        attempts=row.attempts,
                        error_code=row.error_code,
                        verified_at=(
                            _aware(row.verified_at) if row.verified_at is not None else None
                        ),
                    )
                    for row in target_rows
                ),
            ),
        )

    def update_deletion_manifest(
        self,
        context: TenantContext,
        request: DeletionRequest,
        manifest: DeletionManifest,
    ) -> DeletionManifest:
        context.assert_company(request.company_id)
        existing = {
            row.target_id: row
            for row in self._session.scalars(
                select(DeletionTargetRow).where(
                    DeletionTargetRow.company_id == self._tenant(context),
                    DeletionTargetRow.manifest_id == manifest.manifest_id,
                )
            )
        }
        for target in manifest.targets:
            row = existing.get(target.target_id)
            if row is None:
                raise TenantScopedReportingNotFound("deletion target not found")
            row.status = target.status.value
            row.attempts = target.attempts
            row.error_code = target.error_code
            row.verified_at = target.verified_at
        self._session.flush()
        return manifest

    def delete_and_verify_target(
        self,
        context: TenantContext,
        *,
        store: str,
        target_type: str,
        resource_id: str,
    ) -> bool:
        if store != "aurora":
            return False
        row_map: dict[
            str,
            tuple[
                type[Base],
                InstrumentedAttribute[UUID],
                InstrumentedAttribute[UUID],
            ],
        ] = {
            "transcript_segment": (
                TranscriptSegmentRow,
                TranscriptSegmentRow.company_id,
                TranscriptSegmentRow.transcript_segment_id,
            ),
            "recording_asset": (
                RecordingAssetRow,
                RecordingAssetRow.company_id,
                RecordingAssetRow.recording_asset_id,
            ),
            "session_event": (
                SessionEventRow,
                SessionEventRow.company_id,
                SessionEventRow.session_event_id,
            ),
            "report": (ReportRow, ReportRow.company_id, ReportRow.report_id),
            "report_item": (
                ReportItemRow,
                ReportItemRow.company_id,
                ReportItemRow.report_item_id,
            ),
            "evidence": (EvidenceRow, EvidenceRow.company_id, EvidenceRow.evidence_id),
            "human_review": (
                HumanReviewRow,
                HumanReviewRow.company_id,
                HumanReviewRow.human_review_id,
            ),
        }
        row = row_map.get(target_type)
        if row is None:
            raise ValueError("unsupported reporting deletion target")
        target_id = UUID(resource_id)
        predicate = (
            row[1] == self._tenant(context),
            row[2] == target_id,
        )
        self._session.execute(delete(row[0]).where(*predicate))
        self._session.flush()
        return self._session.scalar(select(row[2]).where(*predicate)) is None
