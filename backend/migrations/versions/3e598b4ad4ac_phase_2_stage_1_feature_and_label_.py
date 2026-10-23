from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '3e598b4ad4ac'
down_revision: Union[str, Sequence[str], None] = 'b701f3c8a7df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('feature_definitions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('feature_group', sa.Enum('PRICE', 'TREND', 'MOMENTUM', 'VOLUME', 'STRUCTURE', 'CONTEXT', 'DERIVATIVES', name='featuregroup'), nullable=False),
    sa.Column('description', sa.String(length=512), nullable=False),
    sa.Column('formula_ref', sa.String(length=256), nullable=False),
    sa.Column('input_dependencies', sa.String(length=256), nullable=False),
    sa.Column('lookback_window', sa.Integer(), nullable=True),
    sa.Column('availability_lag', sa.Integer(), nullable=False),
    sa.Column('timeframe', sa.String(length=8), nullable=False),
    sa.Column('status', sa.Enum('ACTIVE', 'DEPRECATED', 'RETIRED', name='definitionstatus'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_by', sa.String(length=128), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'version', name='uq_feature_definition_name_version')
    )
    op.create_table('label_definitions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('description', sa.String(length=512), nullable=False),
    sa.Column('formula_ref', sa.String(length=256), nullable=False),
    sa.Column('horizon_days', sa.Integer(), nullable=False),
    sa.Column('target_stop_definition', sa.String(length=256), nullable=True),
    sa.Column('same_bar_resolution_rule', sa.String(length=256), nullable=True),
    sa.Column('timeframe', sa.String(length=8), nullable=False),
    sa.Column('status', sa.Enum('ACTIVE', 'DEPRECATED', 'RETIRED', name='definitionstatus'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_by', sa.String(length=128), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'version', name='uq_label_definition_name_version')
    )
    op.create_table('feature_values',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('instrument_id', sa.UUID(), nullable=False),
    sa.Column('feature_definition_id', sa.UUID(), nullable=False),
    sa.Column('computed_for_date', sa.Date(), nullable=False),
    sa.Column('timeframe', sa.String(length=8), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.Column('available_at', sa.Date(), nullable=False),
    sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('available_at >= computed_for_date', name='ck_feature_value_available_at_order'),
    sa.ForeignKeyConstraint(['feature_definition_id'], ['feature_definitions.id'], ),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('instrument_id', 'feature_definition_id', 'computed_for_date', 'timeframe', name='uq_feature_value_identity')
    )
    op.create_table('label_values',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('instrument_id', sa.UUID(), nullable=False),
    sa.Column('label_definition_id', sa.UUID(), nullable=False),
    sa.Column('entry_date', sa.Date(), nullable=False),
    sa.Column('horizon_end_date', sa.Date(), nullable=False),
    sa.Column('timeframe', sa.String(length=8), nullable=False),
    sa.Column('realized_return', sa.Float(), nullable=True),
    sa.Column('hit_target', sa.Boolean(), nullable=True),
    sa.Column('hit_stop', sa.Boolean(), nullable=True),
    sa.Column('mfe', sa.Float(), nullable=True),
    sa.Column('mae', sa.Float(), nullable=True),
    sa.Column('holding_time_days', sa.Integer(), nullable=True),
    sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('horizon_end_date >= entry_date', name='ck_label_value_horizon_order'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['label_definition_id'], ['label_definitions.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('instrument_id', 'label_definition_id', 'entry_date', 'timeframe', name='uq_label_value_identity')
    )

    op.execute("""
        CREATE FUNCTION prevent_feature_definition_content_update()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.name IS DISTINCT FROM OLD.name
               OR NEW.version IS DISTINCT FROM OLD.version
               OR NEW.feature_group IS DISTINCT FROM OLD.feature_group
               OR NEW.description IS DISTINCT FROM OLD.description
               OR NEW.formula_ref IS DISTINCT FROM OLD.formula_ref
               OR NEW.input_dependencies IS DISTINCT FROM OLD.input_dependencies
               OR NEW.lookback_window IS DISTINCT FROM OLD.lookback_window
               OR NEW.availability_lag IS DISTINCT FROM OLD.availability_lag
               OR NEW.timeframe IS DISTINCT FROM OLD.timeframe
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR NEW.created_by IS DISTINCT FROM OLD.created_by
            THEN
                RAISE EXCEPTION USING MESSAGE =
                    'feature_definitions is immutable except status: cannot modify id=' || OLD.id::text || ' (only UPDATE of status is permitted)';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_feature_definitions_immutable
        BEFORE UPDATE ON feature_definitions
        FOR EACH ROW EXECUTE FUNCTION prevent_feature_definition_content_update();
    """)
    op.execute("""
        CREATE FUNCTION prevent_feature_definition_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'feature_definitions rows cannot be deleted (immutable research provenance): id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_feature_definitions_no_delete
        BEFORE DELETE ON feature_definitions
        FOR EACH ROW EXECUTE FUNCTION prevent_feature_definition_delete();
    """)

    op.execute("""
        CREATE FUNCTION prevent_label_definition_content_update()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.name IS DISTINCT FROM OLD.name
               OR NEW.version IS DISTINCT FROM OLD.version
               OR NEW.description IS DISTINCT FROM OLD.description
               OR NEW.formula_ref IS DISTINCT FROM OLD.formula_ref
               OR NEW.horizon_days IS DISTINCT FROM OLD.horizon_days
               OR NEW.target_stop_definition IS DISTINCT FROM OLD.target_stop_definition
               OR NEW.same_bar_resolution_rule IS DISTINCT FROM OLD.same_bar_resolution_rule
               OR NEW.timeframe IS DISTINCT FROM OLD.timeframe
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR NEW.created_by IS DISTINCT FROM OLD.created_by
            THEN
                RAISE EXCEPTION USING MESSAGE =
                    'label_definitions is immutable except status: cannot modify id=' || OLD.id::text || ' (only UPDATE of status is permitted)';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_label_definitions_immutable
        BEFORE UPDATE ON label_definitions
        FOR EACH ROW EXECUTE FUNCTION prevent_label_definition_content_update();
    """)
    op.execute("""
        CREATE FUNCTION prevent_label_definition_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'label_definitions rows cannot be deleted (immutable research provenance): id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_label_definitions_no_delete
        BEFORE DELETE ON label_definitions
        FOR EACH ROW EXECUTE FUNCTION prevent_label_definition_delete();
    """)


def downgrade() -> None:
    op.drop_table('label_values')
    op.drop_table('feature_values')
    op.drop_table('label_definitions')
    op.drop_table('feature_definitions')

    op.execute("DROP FUNCTION IF EXISTS prevent_feature_definition_content_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_feature_definition_delete()")
    op.execute("DROP FUNCTION IF EXISTS prevent_label_definition_content_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_label_definition_delete()")

    sa.Enum(name='featuregroup').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='definitionstatus').drop(op.get_bind(), checkfirst=True)
