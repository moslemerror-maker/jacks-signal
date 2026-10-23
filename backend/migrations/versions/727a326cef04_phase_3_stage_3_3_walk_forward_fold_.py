from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '727a326cef04'
down_revision: Union[str, Sequence[str], None] = 'a6470afd3b13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'experiment_metrics',
        sa.Column('fold_index', sa.Integer(), nullable=False, server_default='0'),
    )
    op.alter_column('experiment_metrics', 'fold_index', server_default=None)

    op.drop_constraint('uq_experiment_metric_identity', 'experiment_metrics', type_='unique')
    op.create_unique_constraint(
        'uq_experiment_metric_identity',
        'experiment_metrics',
        ['run_id', 'split', 'category', 'metric_name', 'fold_index'],
    )

    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_exploration_run_confirmation_metrics()
        RETURNS TRIGGER AS $$
        DECLARE
            purpose text;
        BEGIN
            SELECT run_purpose::text INTO purpose
            FROM experiment_runs WHERE id = NEW.run_id;

            IF purpose = 'EXPLORATION' AND NEW.split IN ('VALIDATION', 'TEST')
                AND NEW.fold_index = 0
            THEN
                RAISE EXCEPTION USING MESSAGE =
                    'exploration runs (run_id=' || NEW.run_id::text || ') must not '
                    'emit VALIDATION/TEST confirmation metrics at fold_index=0 - '
                    'only TRAIN diagnostics are permitted for a non-walk-forward '
                    'exploration run; a walk-forward evaluation (fold_index <> 0) '
                    'may emit per-fold VALIDATION/TEST diagnostic metrics, which '
                    'are not a confirmatory claim';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_exploration_run_confirmation_metrics()
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

    op.drop_constraint('uq_experiment_metric_identity', 'experiment_metrics', type_='unique')
    op.create_unique_constraint(
        'uq_experiment_metric_identity',
        'experiment_metrics',
        ['run_id', 'split', 'category', 'metric_name'],
    )
    op.drop_column('experiment_metrics', 'fold_index')
