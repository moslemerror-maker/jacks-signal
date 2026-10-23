from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '2991fb8cb505'
down_revision: Union[str, Sequence[str], None] = '727a326cef04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'experiment_runs',
        sa.Column('confirmation_report', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.drop_constraint('ck_experiment_run_conclusion_matches_status', 'experiment_runs', type_='check')
    op.create_check_constraint(
        'ck_experiment_run_conclusion_matches_status',
        'experiment_runs',
        "(status = 'FAILED' AND conclusion IS NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'CONFIRMATION' "
        "AND (conclusion IS NOT NULL OR confirmation_report IS NOT NULL)) OR "
        "(status = 'COMPLETED' AND run_purpose = 'EXPLORATION' AND conclusion IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint('ck_experiment_run_conclusion_matches_status', 'experiment_runs', type_='check')
    op.create_check_constraint(
        'ck_experiment_run_conclusion_matches_status',
        'experiment_runs',
        "(status = 'FAILED' AND conclusion IS NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'CONFIRMATION' AND conclusion IS NOT NULL) OR "
        "(status = 'COMPLETED' AND run_purpose = 'EXPLORATION' AND conclusion IS NULL)",
    )

    op.drop_column('experiment_runs', 'confirmation_report')
