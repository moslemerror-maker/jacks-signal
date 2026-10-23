from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '4843c50996f1'
down_revision: Union[str, Sequence[str], None] = '637c0638bc88'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COMPONENT_TRIGGER_FN_V1 = """
    CREATE OR REPLACE FUNCTION prevent_corporate_action_component_content_update()
    RETURNS TRIGGER AS $$
    BEGIN
        IF NEW.event_id IS DISTINCT FROM OLD.event_id
           OR NEW.component_order IS DISTINCT FROM OLD.component_order
           OR NEW.component_type IS DISTINCT FROM OLD.component_type
           OR NEW.raw_text_fragment IS DISTINCT FROM OLD.raw_text_fragment
           OR NEW.face_value_from IS DISTINCT FROM OLD.face_value_from
           OR NEW.face_value_to IS DISTINCT FROM OLD.face_value_to
           OR NEW.ratio_new IS DISTINCT FROM OLD.ratio_new
           OR NEW.ratio_existing IS DISTINCT FROM OLD.ratio_existing
           OR NEW.cash_amount_per_share IS DISTINCT FROM OLD.cash_amount_per_share
           OR NEW.currency IS DISTINCT FROM OLD.currency
           OR NEW.created_at IS DISTINCT FROM OLD.created_at
        THEN
            RAISE EXCEPTION USING MESSAGE =
                'corporate_action_components is immutable except verification_status/verification_notes: cannot modify id=' || OLD.id::text;
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
"""

_COMPONENT_TRIGGER_FN_V2 = _COMPONENT_TRIGGER_FN_V1.replace(
    "OR NEW.created_at IS DISTINCT FROM OLD.created_at",
    "OR NEW.created_at IS DISTINCT FROM OLD.created_at\n"
    "           OR NEW.parse_revision IS DISTINCT FROM OLD.parse_revision\n"
    "           OR NEW.parser_version IS DISTINCT FROM OLD.parser_version",
)


def upgrade() -> None:
    op.create_table(
        'corporate_action_verification_log',
        sa.Column('seq', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('target_kind', sa.String(length=24), nullable=False),
        sa.Column('target_id', sa.UUID(), nullable=False),
        sa.Column('action', sa.String(length=32), nullable=False),
        sa.Column('from_status', sa.String(length=24), nullable=True),
        sa.Column('to_status', sa.String(length=24), nullable=True),
        sa.Column('from_notes', sa.Text(), nullable=True),
        sa.Column('to_notes', sa.Text(), nullable=True),
        sa.Column('payload', postgresql.JSONB(), nullable=True),
        sa.Column('actor', sa.String(length=128), nullable=False),
        sa.Column('logged_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('seq', name='uq_verification_log_seq'),
        sa.CheckConstraint(
            "target_kind IN ('EVENT', 'COMPONENT', 'COVERAGE_ASSESSMENT')",
            name='ck_verification_log_target_kind',
        ),
        sa.CheckConstraint(
            "action IN ('BASELINE', 'STATUS_CHANGE', 'CORRECTION', 'REPARSE', 'ANNOTATION')",
            name='ck_verification_log_action',
        ),
    )
    op.create_index(
        'ix_verification_log_target', 'corporate_action_verification_log',
        ['target_kind', 'target_id', 'seq'],
    )
    op.execute("""
        CREATE FUNCTION prevent_verification_log_update()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'corporate_action_verification_log is append-only: cannot modify seq=' || OLD.seq::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_verification_log_immutable
        BEFORE UPDATE ON corporate_action_verification_log
        FOR EACH ROW EXECUTE FUNCTION prevent_verification_log_update();
    """)
    op.execute("""
        CREATE FUNCTION prevent_verification_log_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'corporate_action_verification_log rows cannot be deleted (audit trail): seq=' || OLD.seq::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_verification_log_no_delete
        BEFORE DELETE ON corporate_action_verification_log
        FOR EACH ROW EXECUTE FUNCTION prevent_verification_log_delete();
    """)

    op.add_column(
        'corporate_action_components',
        sa.Column('parse_revision', sa.Integer(), server_default='1', nullable=False),
    )
    op.add_column(
        'corporate_action_components',
        sa.Column('parser_version', sa.String(length=64), server_default='stage3.9-phase-b-v1', nullable=False),
    )
    op.create_check_constraint(
        'ck_corporate_action_component_parse_revision', 'corporate_action_components',
        'parse_revision >= 1',
    )
    op.drop_constraint('uq_corporate_action_component_order', 'corporate_action_components', type_='unique')
    op.create_unique_constraint(
        'uq_corporate_action_component_revision_order', 'corporate_action_components',
        ['event_id', 'parse_revision', 'component_order'],
    )
    op.execute(_COMPONENT_TRIGGER_FN_V2)


def downgrade() -> None:
    bind = op.get_bind()
    reparsed = bind.execute(
        sa.text("SELECT count(*) FROM corporate_action_components WHERE parse_revision > 1")
    ).scalar()
    logged = bind.execute(sa.text("SELECT count(*) FROM corporate_action_verification_log")).scalar()
    if reparsed or logged:
        raise RuntimeError(
            f"refusing to downgrade: {reparsed} re-parsed component row(s) and {logged} "
            "verification-log row(s) exist and would be destroyed (this migration's tables/"
            "columns hold audit-trail data that cannot be reconstructed). See docs/32 - "
            "resolve deliberately, do not force."
        )

    op.execute(_COMPONENT_TRIGGER_FN_V1)
    op.drop_constraint('uq_corporate_action_component_revision_order', 'corporate_action_components', type_='unique')
    op.create_unique_constraint(
        'uq_corporate_action_component_order', 'corporate_action_components',
        ['event_id', 'component_order'],
    )
    op.drop_constraint('ck_corporate_action_component_parse_revision', 'corporate_action_components', type_='check')
    op.drop_column('corporate_action_components', 'parser_version')
    op.drop_column('corporate_action_components', 'parse_revision')

    op.drop_index('ix_verification_log_target', table_name='corporate_action_verification_log')
    op.drop_table('corporate_action_verification_log')
    op.execute("DROP FUNCTION IF EXISTS prevent_verification_log_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_verification_log_delete()")
