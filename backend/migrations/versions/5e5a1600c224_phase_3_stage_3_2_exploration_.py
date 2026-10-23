from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '5e5a1600c224'
down_revision: Union[str, Sequence[str], None] = '104bb336068a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DEFAULT_FAMILY_ID = "b1f6b2e2-6c1a-4a6e-9c1e-1a2b3c4d5e6f"

_LEGACY_CONFIRMATION_RUN_ID = "051ea8fa-a422-4c01-85c5-291d1e2b07f9"
_LEGACY_EXPLORATION_RUN_IDS = (
    "2e5f9417-d3b6-4063-b84d-6580d01fbde9",
    "de3b91a3-34f2-4176-8447-d9bbdb90fed9",
)


def upgrade() -> None:
    op.create_table(
        'hypothesis_families',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('description', sa.String(length=1024), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.String(length=128), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )
    op.execute("""
        CREATE FUNCTION prevent_hypothesis_family_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'hypothesis_families rows are immutable once written (never '
                'UPDATE or DELETE - start a new family for a genuinely '
                'reframed research question)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_hypothesis_families_no_update
        BEFORE UPDATE ON hypothesis_families
        FOR EACH ROW EXECUTE FUNCTION prevent_hypothesis_family_mutation();
    """)
    op.execute("""
        CREATE TRIGGER trg_hypothesis_families_no_delete
        BEFORE DELETE ON hypothesis_families
        FOR EACH ROW EXECUTE FUNCTION prevent_hypothesis_family_mutation();
    """)

    op.add_column(
        'experiment_definitions',
        sa.Column('hypothesis_family_id', sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        'fk_experiment_definitions_hypothesis_family_id',
        'experiment_definitions', 'hypothesis_families',
        ['hypothesis_family_id'], ['id'],
    )

    op.execute(f"""
        INSERT INTO hypothesis_families (id, name, description, created_at, created_by)
        VALUES (
            '{_DEFAULT_FAMILY_ID}',
            'stage5-6-trend-following-baseline',
            'Retroactively-assigned family for the Stage 5/6 close_vs_sma20 '
            'trend-following baseline experiment definition(s), created '
            'before hypothesis families existed as a concept. Backfilled by '
            'the Stage 3.2 migration.',
            now(),
            'stage3.2-migration-backfill'
        )
        ON CONFLICT (name) DO NOTHING;
    """)
    op.execute(f"""
        UPDATE experiment_definitions
        SET hypothesis_family_id = '{_DEFAULT_FAMILY_ID}'
        WHERE hypothesis_family_id IS NULL;
    """)
    op.alter_column('experiment_definitions', 'hypothesis_family_id', nullable=False)

    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_experiment_definition_content_update()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.name IS DISTINCT FROM OLD.name
                OR NEW.version IS DISTINCT FROM OLD.version
                OR NEW.hypothesis_family_id IS DISTINCT FROM OLD.hypothesis_family_id
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

    op.execute("CREATE TYPE runpurpose AS ENUM ('EXPLORATION', 'CONFIRMATION')")
    op.add_column(
        'experiment_runs',
        sa.Column(
            'run_purpose',
            postgresql.ENUM('EXPLORATION', 'CONFIRMATION', name='runpurpose', create_type=False),
            nullable=True,
        ),
    )
    op.alter_column('experiment_runs', 'validation_start_date', existing_type=sa.DATE(), nullable=True)
    op.alter_column('experiment_runs', 'validation_end_date', existing_type=sa.DATE(), nullable=True)
    op.alter_column('experiment_runs', 'test_start_date', existing_type=sa.DATE(), nullable=True)
    op.alter_column('experiment_runs', 'test_end_date', existing_type=sa.DATE(), nullable=True)

    op.execute("ALTER TABLE experiment_runs DISABLE TRIGGER trg_experiment_runs_no_update;")
    op.execute(f"""
        UPDATE experiment_runs SET run_purpose = 'CONFIRMATION'
        WHERE id = '{_LEGACY_CONFIRMATION_RUN_ID}';
    """)
    op.execute(f"""
        UPDATE experiment_runs SET run_purpose = 'EXPLORATION'
        WHERE id IN ('{_LEGACY_EXPLORATION_RUN_IDS[0]}', '{_LEGACY_EXPLORATION_RUN_IDS[1]}');
    """)
    op.execute("""
        UPDATE experiment_runs SET run_purpose = 'CONFIRMATION' WHERE run_purpose IS NULL;
    """)
    op.execute("ALTER TABLE experiment_runs ENABLE TRIGGER trg_experiment_runs_no_update;")

    op.alter_column('experiment_runs', 'run_purpose', nullable=False)

    op.drop_constraint('ck_experiment_run_conclusion_matches_status', 'experiment_runs', type_='check')
    op.create_check_constraint(
        'ck_experiment_run_conclusion_matches_status',
        'experiment_runs',
        "(status = 'FAILED' AND conclusion IS NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'CONFIRMATION' AND conclusion IS NOT NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'EXPLORATION')",
    )
    op.create_check_constraint(
        'ck_experiment_run_confirmation_requires_periods',
        'experiment_runs',
        "run_purpose <> 'CONFIRMATION' OR "
        "(validation_start_date IS NOT NULL AND validation_end_date IS NOT NULL "
        "AND test_start_date IS NOT NULL AND test_end_date IS NOT NULL)",
    )

    op.create_index(
        'uq_experiment_run_one_confirmation_per_definition_dataset',
        'experiment_runs',
        ['experiment_definition_id', 'dataset_id'],
        unique=True,
        postgresql_where=sa.text("run_purpose = 'CONFIRMATION'"),
    )

    op.execute("""
        CREATE FUNCTION prevent_exploration_run_confirmation_metrics()
        RETURNS TRIGGER AS $$
        DECLARE
            purpose text;
        BEGIN
            SELECT run_purpose::text INTO purpose
            FROM experiment_runs WHERE id = NEW.run_id;

            IF purpose = 'EXPLORATION' AND NEW.split IN ('VALIDATION', 'TEST') THEN
                RAISE EXCEPTION USING MESSAGE =
                    'exploration runs (run_id=' || NEW.run_id::text || ') must not '
                    'emit VALIDATION/TEST confirmation metrics - only TRAIN '
                    'diagnostics are permitted';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_experiment_metrics_no_exploration_confirmation_metrics
        BEFORE INSERT ON experiment_metrics
        FOR EACH ROW EXECUTE FUNCTION prevent_exploration_run_confirmation_metrics();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_experiment_metrics_no_exploration_confirmation_metrics ON experiment_metrics;")
    op.execute("DROP FUNCTION IF EXISTS prevent_exploration_run_confirmation_metrics() CASCADE")

    op.drop_index(
        'uq_experiment_run_one_confirmation_per_definition_dataset',
        table_name='experiment_runs',
        postgresql_where=sa.text("run_purpose = 'CONFIRMATION'"),
    )
    op.drop_constraint('ck_experiment_run_confirmation_requires_periods', 'experiment_runs', type_='check')
    op.drop_constraint('ck_experiment_run_conclusion_matches_status', 'experiment_runs', type_='check')
    op.create_check_constraint(
        'ck_experiment_run_conclusion_matches_status',
        'experiment_runs',
        "(status = 'FAILED' AND conclusion IS NULL) OR "
        "(status = 'COMPLETED' AND conclusion IS NOT NULL)",
    )

    op.drop_column('experiment_runs', 'run_purpose')
    op.alter_column('experiment_runs', 'test_end_date', existing_type=sa.DATE(), nullable=False)
    op.alter_column('experiment_runs', 'test_start_date', existing_type=sa.DATE(), nullable=False)
    op.alter_column('experiment_runs', 'validation_end_date', existing_type=sa.DATE(), nullable=False)
    op.alter_column('experiment_runs', 'validation_start_date', existing_type=sa.DATE(), nullable=False)

    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_experiment_definition_content_update()
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

    op.drop_constraint('fk_experiment_definitions_hypothesis_family_id', 'experiment_definitions', type_='foreignkey')
    op.drop_column('experiment_definitions', 'hypothesis_family_id')

    op.execute("DROP TRIGGER IF EXISTS trg_hypothesis_families_no_update ON hypothesis_families;")
    op.execute("DROP TRIGGER IF EXISTS trg_hypothesis_families_no_delete ON hypothesis_families;")
    op.execute("DROP FUNCTION IF EXISTS prevent_hypothesis_family_mutation() CASCADE")
    op.drop_table('hypothesis_families')

    sa.Enum(name='runpurpose').drop(op.get_bind(), checkfirst=True)
