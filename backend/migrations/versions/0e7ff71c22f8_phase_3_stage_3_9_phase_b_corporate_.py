from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '0e7ff71c22f8'
down_revision: Union[str, Sequence[str], None] = '2991fb8cb505'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'corporate_action_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('instrument_id', sa.UUID(), nullable=False),
        sa.Column('source_symbol', sa.String(length=32), nullable=False),
        sa.Column('ex_date', sa.Date(), nullable=False),
        sa.Column('record_date', sa.Date(), nullable=True),
        sa.Column('source_isin', sa.String(length=32), nullable=True),
        sa.Column('raw_subject_text', sa.String(length=512), nullable=False),
        sa.Column('raw_face_val', sa.String(length=32), nullable=True),
        sa.Column('source_provider', sa.String(length=128), nullable=False),
        sa.Column('source_url', sa.String(length=512), nullable=False),
        sa.Column('source_snapshot_path', sa.String(length=512), nullable=False),
        sa.Column('source_snapshot_sha256', sa.String(length=64), nullable=False),
        sa.Column('source_fetched_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('announcement_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'first_observed_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.Column(
            'verification_status',
            sa.Enum('UNVERIFIED', 'MACHINE_PARSED', 'MANUALLY_VERIFIED', 'DISPUTED',
                    name='corporateactionverificationstatus'),
            nullable=False,
        ),
        sa.Column('verification_notes', sa.String(length=2048), nullable=True),
        sa.Column('supersedes', sa.UUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id']),
        sa.ForeignKeyConstraint(['supersedes'], ['corporate_action_events.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'instrument_id', 'ex_date', 'raw_subject_text',
            name='uq_corporate_action_event_identity',
        ),
    )
    op.create_table(
        'corporate_action_components',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('event_id', sa.UUID(), nullable=False),
        sa.Column('component_order', sa.Integer(), nullable=False),
        sa.Column(
            'component_type',
            sa.Enum('SPLIT', 'BONUS', 'RIGHTS', 'DIVIDEND', 'MERGER', 'DEMERGER', 'OTHER',
                    name='corporateactioncomponenttype'),
            nullable=False,
        ),
        sa.Column('raw_text_fragment', sa.String(length=512), nullable=False),
        sa.Column('face_value_from', sa.Float(), nullable=True),
        sa.Column('face_value_to', sa.Float(), nullable=True),
        sa.Column('ratio_new', sa.Integer(), nullable=True),
        sa.Column('ratio_existing', sa.Integer(), nullable=True),
        sa.Column('cash_amount_per_share', sa.Float(), nullable=True),
        sa.Column('currency', sa.String(length=8), nullable=True),
        sa.Column(
            'verification_status',
            sa.Enum('UNVERIFIED', 'MACHINE_PARSED', 'MANUALLY_VERIFIED', 'DISPUTED',
                    name='corporateactionverificationstatus'),
            nullable=False,
        ),
        sa.Column('verification_notes', sa.String(length=2048), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['event_id'], ['corporate_action_events.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('event_id', 'component_order', name='uq_corporate_action_component_order'),
    )

    op.execute("""
        CREATE FUNCTION prevent_corporate_action_event_content_update()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.instrument_id IS DISTINCT FROM OLD.instrument_id
               OR NEW.source_symbol IS DISTINCT FROM OLD.source_symbol
               OR NEW.ex_date IS DISTINCT FROM OLD.ex_date
               OR NEW.record_date IS DISTINCT FROM OLD.record_date
               OR NEW.source_isin IS DISTINCT FROM OLD.source_isin
               OR NEW.raw_subject_text IS DISTINCT FROM OLD.raw_subject_text
               OR NEW.raw_face_val IS DISTINCT FROM OLD.raw_face_val
               OR NEW.source_provider IS DISTINCT FROM OLD.source_provider
               OR NEW.source_url IS DISTINCT FROM OLD.source_url
               OR NEW.source_snapshot_path IS DISTINCT FROM OLD.source_snapshot_path
               OR NEW.source_snapshot_sha256 IS DISTINCT FROM OLD.source_snapshot_sha256
               OR NEW.source_fetched_at IS DISTINCT FROM OLD.source_fetched_at
               OR NEW.announcement_date IS DISTINCT FROM OLD.announcement_date
               OR NEW.first_observed_at IS DISTINCT FROM OLD.first_observed_at
               OR NEW.supersedes IS DISTINCT FROM OLD.supersedes
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR NEW.created_by IS DISTINCT FROM OLD.created_by
            THEN
                RAISE EXCEPTION USING MESSAGE =
                    'corporate_action_events is immutable except verification_status/verification_notes: cannot modify id=' || OLD.id::text;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_corporate_action_events_immutable
        BEFORE UPDATE ON corporate_action_events
        FOR EACH ROW EXECUTE FUNCTION prevent_corporate_action_event_content_update();
    """)
    op.execute("""
        CREATE FUNCTION prevent_corporate_action_event_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'corporate_action_events rows cannot be deleted (immutable research provenance): id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_corporate_action_events_no_delete
        BEFORE DELETE ON corporate_action_events
        FOR EACH ROW EXECUTE FUNCTION prevent_corporate_action_event_delete();
    """)

    op.execute("""
        CREATE FUNCTION prevent_corporate_action_component_content_update()
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
    """)
    op.execute("""
        CREATE TRIGGER trg_corporate_action_components_immutable
        BEFORE UPDATE ON corporate_action_components
        FOR EACH ROW EXECUTE FUNCTION prevent_corporate_action_component_content_update();
    """)
    op.execute("""
        CREATE FUNCTION prevent_corporate_action_component_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'corporate_action_components rows cannot be deleted (immutable research provenance): id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_corporate_action_components_no_delete
        BEFORE DELETE ON corporate_action_components
        FOR EACH ROW EXECUTE FUNCTION prevent_corporate_action_component_delete();
    """)


def downgrade() -> None:
    op.drop_table('corporate_action_components')
    op.drop_table('corporate_action_events')

    op.execute("DROP FUNCTION IF EXISTS prevent_corporate_action_event_content_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_corporate_action_event_delete()")
    op.execute("DROP FUNCTION IF EXISTS prevent_corporate_action_component_content_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_corporate_action_component_delete()")

    sa.Enum(name='corporateactionverificationstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='corporateactioncomponenttype').drop(op.get_bind(), checkfirst=True)
