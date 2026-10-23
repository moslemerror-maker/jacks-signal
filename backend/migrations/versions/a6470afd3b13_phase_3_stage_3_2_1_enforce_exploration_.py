from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a6470afd3b13'
down_revision: Union[str, Sequence[str], None] = '5e5a1600c224'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_HISTORICAL_EXPLORATION_RUN_WITH_STALE_CONCLUSION = "2e5f9417-d3b6-4063-b84d-6580d01fbde9"
_HISTORICAL_ORIGINAL_CONCLUSION = "NEEDS_MORE_TESTING"
_HISTORICAL_ORIGINAL_CONCLUSION_RATIONALE = (
    "mean net realized_return disagrees in sign between validation (0.003445) "
    "and test (-0.000767)."
)


def upgrade() -> None:
    op.execute("ALTER TABLE experiment_runs DISABLE TRIGGER trg_experiment_runs_no_update;")
    op.execute(f"""
        UPDATE experiment_runs
        SET conclusion = NULL, conclusion_rationale = NULL
        WHERE id = '{_HISTORICAL_EXPLORATION_RUN_WITH_STALE_CONCLUSION}'
          AND run_purpose = 'EXPLORATION';
    """)
    op.execute("ALTER TABLE experiment_runs ENABLE TRIGGER trg_experiment_runs_no_update;")

    op.drop_constraint('ck_experiment_run_conclusion_matches_status', 'experiment_runs', type_='check')
    op.create_check_constraint(
        'ck_experiment_run_conclusion_matches_status',
        'experiment_runs',
        "(status = 'FAILED' AND conclusion IS NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'CONFIRMATION' AND conclusion IS NOT NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'EXPLORATION' AND conclusion IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint('ck_experiment_run_conclusion_matches_status', 'experiment_runs', type_='check')
    op.create_check_constraint(
        'ck_experiment_run_conclusion_matches_status',
        'experiment_runs',
        "(status = 'FAILED' AND conclusion IS NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'CONFIRMATION' AND conclusion IS NOT NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'EXPLORATION')",
    )

    op.execute("ALTER TABLE experiment_runs DISABLE TRIGGER trg_experiment_runs_no_update;")
    op.execute(f"""
        UPDATE experiment_runs
        SET conclusion = '{_HISTORICAL_ORIGINAL_CONCLUSION}',
            conclusion_rationale = '{_HISTORICAL_ORIGINAL_CONCLUSION_RATIONALE}'
        WHERE id = '{_HISTORICAL_EXPLORATION_RUN_WITH_STALE_CONCLUSION}'
          AND run_purpose = 'EXPLORATION';
    """)
    op.execute("ALTER TABLE experiment_runs ENABLE TRIGGER trg_experiment_runs_no_update;")
