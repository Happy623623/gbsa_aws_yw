"""Persist terminal report-generation failures separately from immutable reports."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "m_021_report_generation_failures"
down_revision: str = "m_020_requirement_assessments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "report_generation_failures",
        sa.Column("company_id", sa.Uuid(), nullable=False),
        sa.Column("source_event_id", sa.Uuid(), nullable=False),
        sa.Column("interview_session_id", sa.Uuid(), nullable=False),
        sa.Column("last_delivery_attempt", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(100), nullable=False),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("company_id", "source_event_id"),
    )
    op.create_index(
        "ix_report_generation_failures_session",
        "report_generation_failures",
        ["company_id", "interview_session_id", "failed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_report_generation_failures_session",
        table_name="report_generation_failures",
    )
    op.drop_table("report_generation_failures")
