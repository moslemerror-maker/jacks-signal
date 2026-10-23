from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '104bb336068a'
down_revision: Union[str, Sequence[str], None] = '7f08a0c262e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('experiment_definitions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('hypothesis', sa.String(length=1024), nullable=False),
    sa.Column('rule_feature_definition_id', sa.UUID(), nullable=False),
    sa.Column('rule_description', sa.String(length=512), nullable=False),
    sa.Column('cost_model_description', sa.String(length=1024), nullable=False),
    sa.Column('status', postgresql.ENUM('ACTIVE', 'DEPRECATED', 'RETIRED', name='definitionstatus', create_type=False), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_by', sa.String(length=128), nullable=False),
    sa.ForeignKeyConstraint(['rule_feature_definition_id'], ['feature_definitions.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'version', name='uq_experiment_definition_name_version')
    )
    op.create_table('experiment_runs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('experiment_definition_id', sa.UUID(), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('rule_feature_definition_id', sa.UUID(), nullable=False),
    sa.Column('git_commit', sa.String(length=40), nullable=False),
    sa.Column('git_dirty', sa.Boolean(), nullable=False),
    sa.Column('dependency_versions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('configuration', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('cost_assumptions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('train_start_date', sa.Date(), nullable=False),
    sa.Column('train_end_date', sa.Date(), nullable=False),
    sa.Column('validation_start_date', sa.Date(), nullable=False),
    sa.Column('validation_end_date', sa.Date(), nullable=False),
    sa.Column('test_start_date', sa.Date(), nullable=False),
    sa.Column('test_end_date', sa.Date(), nullable=False),
    sa.Column('walk_forward_scheme', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('executed_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('status', sa.Enum('COMPLETED', 'FAILED', name='runstatus'), nullable=False),
    sa.Column('failure_reason', sa.String(length=1024), nullable=True),
    sa.Column('conclusion', postgresql.ENUM('ACCEPT', 'REJECT', 'NEEDS_MORE_TESTING', name='experimentconclusion', create_type=False), nullable=True),
    sa.Column('conclusion_rationale', sa.String(length=1024), nullable=True),
    sa.Column('checksum', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_by', sa.String(length=128), nullable=False),
    sa.CheckConstraint("(status = 'FAILED' AND conclusion IS NULL) OR (status = 'COMPLETED' AND conclusion IS NOT NULL)", name='ck_experiment_run_conclusion_matches_status'),
    sa.CheckConstraint('test_end_date >= test_start_date', name='ck_experiment_run_test_order'),
    sa.CheckConstraint('test_start_date > validation_end_date', name='ck_experiment_run_validation_before_test'),
    sa.CheckConstraint('train_end_date >= train_start_date', name='ck_experiment_run_train_order'),
    sa.CheckConstraint('validation_end_date >= validation_start_date', name='ck_experiment_run_validation_order'),
    sa.CheckConstraint('validation_start_date > train_end_date', name='ck_experiment_run_train_before_validation'),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['experiment_definition_id'], ['experiment_definitions.id'], ),
    sa.ForeignKeyConstraint(['rule_feature_definition_id'], ['feature_definitions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('experiment_metrics',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('split', sa.Enum('TRAIN', 'VALIDATION', 'TEST', name='metricsplit'), nullable=False),
    sa.Column('category', sa.Enum('DESCRIPTIVE', 'HYPOTHESIS_TEST', 'HYPOTHETICAL_SIGNAL', 'REALIZABLE_STRATEGY', name='metriccategory'), nullable=False),
    sa.Column('metric_name', sa.String(length=64), nullable=False),
    sa.Column('metric_value', sa.Float(), nullable=True),
    sa.Column('sample_size', sa.Integer(), nullable=True),
    sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['experiment_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_id', 'split', 'category', 'metric_name', name='uq_experiment_metric_identity')
    )

    op.execute("""
        CREATE FUNCTION prevent_experiment_definition_content_update()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.name IS DISTINCT FROM OLD.name
                OR NEW.version IS DISTINCT FROM OLD.version
                OR NEW.hypothesis IS DISTINCT FROM OLD.hypothesis
                OR NEW.rule_feature_definition_id IS DISTINCT FROM OLD.rule_feature_definition_id
                OR NEW.rule_description IS DISTINCT FROM OLD.rule_description
                OR NEW.cost_model_description IS DISTINCT FROM OLD.cost_model_description
                OR NEW.created_by IS DISTINCT FROM OLD.created_by
            THEN
                RAISE EXCEPTION USING MESSAGE =
                    'experiment_definitions is immutable except status: cannot '
                    'modify id=' || OLD.id::text ||
                    ' (only UPDATE of status is permitted)';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE FUNCTION prevent_experiment_definition_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'experiment_definitions rows cannot be deleted (id=' ||
                OLD.id::text || ') - deprecate via status instead';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_experiment_definitions_content_immutable
        BEFORE UPDATE ON experiment_definitions
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_definition_content_update();
    """)
    op.execute("""
        CREATE TRIGGER trg_experiment_definitions_no_delete
        BEFORE DELETE ON experiment_definitions
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_definition_delete();
    """)

    op.execute("""
        CREATE FUNCTION prevent_experiment_run_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'experiment_runs rows are immutable once written (never UPDATE '
                'or DELETE - a changed input of any kind means running a new '
                'experiment)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_experiment_runs_no_update
        BEFORE UPDATE ON experiment_runs
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_run_mutation();
    """)
    op.execute("""
        CREATE TRIGGER trg_experiment_runs_no_delete
        BEFORE DELETE ON experiment_runs
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_run_mutation();
    """)

    op.execute("""
        CREATE FUNCTION prevent_experiment_metric_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'experiment_metrics rows are immutable once written (never '
                'UPDATE or DELETE - a changed metric means running a new '
                'experiment)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_experiment_metrics_no_update
        BEFORE UPDATE ON experiment_metrics
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_metric_mutation();
    """)
    op.execute("""
        CREATE TRIGGER trg_experiment_metrics_no_delete
        BEFORE DELETE ON experiment_metrics
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_metric_mutation();
    """)


def downgrade() -> None:
    op.drop_table('experiment_metrics')
    op.drop_table('experiment_runs')
    op.drop_table('experiment_definitions')

    op.execute("DROP FUNCTION IF EXISTS prevent_experiment_definition_content_update() CASCADE")
    op.execute("DROP FUNCTION IF EXISTS prevent_experiment_definition_delete() CASCADE")
    op.execute("DROP FUNCTION IF EXISTS prevent_experiment_run_mutation() CASCADE")
    op.execute("DROP FUNCTION IF EXISTS prevent_experiment_metric_mutation() CASCADE")

    sa.Enum(name='runstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='metricsplit').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='metriccategory').drop(op.get_bind(), checkfirst=True)
