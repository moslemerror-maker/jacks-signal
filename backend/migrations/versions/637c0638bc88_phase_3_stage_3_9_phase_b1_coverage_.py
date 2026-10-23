from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '637c0638bc88'
down_revision: Union[str, Sequence[str], None] = '0e7ff71c22f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'corporate_action_coverage_assessments',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('instrument_id', sa.UUID(), nullable=False),
        sa.Column('window_start', sa.Date(), nullable=False),
        sa.Column('window_end', sa.Date(), nullable=False),
        sa.Column(
            'status',
            sa.Enum('COMPLETE', 'PARTIAL', 'UNKNOWN', 'CONFLICTED', name='corporateactioncoveragestatus'),
            nullable=False,
        ),
        sa.Column('reasoning', sa.String(length=2048), nullable=False),
        sa.Column('sources_considered', postgresql.JSONB(), nullable=False),
        sa.Column('assessed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('assessed_by', sa.String(length=128), nullable=False),
        sa.CheckConstraint('window_end >= window_start', name='ck_coverage_assessment_window_order'),
        sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.execute("""
        CREATE FUNCTION prevent_coverage_assessment_update()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'corporate_action_coverage_assessments rows are immutable - re-assess by inserting a new row: id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_coverage_assessment_immutable
        BEFORE UPDATE ON corporate_action_coverage_assessments
        FOR EACH ROW EXECUTE FUNCTION prevent_coverage_assessment_update();
    """)
    op.execute("""
        CREATE FUNCTION prevent_coverage_assessment_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'corporate_action_coverage_assessments rows cannot be deleted (immutable research provenance): id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_coverage_assessment_no_delete
        BEFORE DELETE ON corporate_action_coverage_assessments
        FOR EACH ROW EXECUTE FUNCTION prevent_coverage_assessment_delete();
    """)


def downgrade() -> None:
    op.drop_table('corporate_action_coverage_assessments')
    op.execute("DROP FUNCTION IF EXISTS prevent_coverage_assessment_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_coverage_assessment_delete()")
    sa.Enum(name='corporateactioncoveragestatus').drop(op.get_bind(), checkfirst=True)
