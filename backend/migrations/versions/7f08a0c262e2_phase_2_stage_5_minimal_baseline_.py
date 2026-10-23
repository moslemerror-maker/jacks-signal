from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '7f08a0c262e2'
down_revision: Union[str, Sequence[str], None] = 'fd78230b734d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('experiments',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('hypothesis', sa.String(length=1024), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('rule_feature_definition_id', sa.UUID(), nullable=False),
    sa.Column('rule_description', sa.String(length=512), nullable=False),
    sa.Column('cost_bps_per_trade', sa.Float(), nullable=False),
    sa.Column('cost_model_description', sa.String(length=1024), nullable=False),
    sa.Column('train_start_date', sa.Date(), nullable=False),
    sa.Column('train_end_date', sa.Date(), nullable=False),
    sa.Column('validation_start_date', sa.Date(), nullable=False),
    sa.Column('validation_end_date', sa.Date(), nullable=False),
    sa.Column('test_start_date', sa.Date(), nullable=False),
    sa.Column('test_end_date', sa.Date(), nullable=False),
    sa.Column('results', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('conclusion', sa.Enum('ACCEPT', 'REJECT', 'NEEDS_MORE_TESTING', name='experimentconclusion'), nullable=False),
    sa.Column('conclusion_rationale', sa.String(length=1024), nullable=False),
    sa.Column('checksum', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_by', sa.String(length=128), nullable=False),
    sa.CheckConstraint('test_end_date >= test_start_date', name='ck_experiment_test_order'),
    sa.CheckConstraint('test_start_date > validation_end_date', name='ck_experiment_validation_before_test'),
    sa.CheckConstraint('train_end_date >= train_start_date', name='ck_experiment_train_order'),
    sa.CheckConstraint('validation_end_date >= validation_start_date', name='ck_experiment_validation_order'),
    sa.CheckConstraint('validation_start_date > train_end_date', name='ck_experiment_train_before_validation'),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['rule_feature_definition_id'], ['feature_definitions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    op.execute("""
        CREATE FUNCTION prevent_experiment_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'experiments rows are immutable once written (never UPDATE or '
                'DELETE - a changed rule, cost assumption, split, or dataset '
                'means running a new experiment)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_experiments_no_update
        BEFORE UPDATE ON experiments
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_mutation();
    """)
    op.execute("""
        CREATE TRIGGER trg_experiments_no_delete
        BEFORE DELETE ON experiments
        FOR EACH ROW EXECUTE FUNCTION prevent_experiment_mutation();
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS prevent_experiment_mutation() CASCADE")

    op.drop_table('experiments')
    sa.Enum(name='experimentconclusion').drop(op.get_bind(), checkfirst=True)
