from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'fd78230b734d'
down_revision: Union[str, Sequence[str], None] = 'a297f8fb115d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('datasets',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('label_definition_id', sa.UUID(), nullable=False),
    sa.Column('start_date', sa.Date(), nullable=False),
    sa.Column('end_date', sa.Date(), nullable=False),
    sa.Column('timeframe', sa.String(length=8), nullable=False),
    sa.Column('known_as_of', sa.Date(), nullable=False),
    sa.Column('universe_caveat', sa.String(length=1024), nullable=False),
    sa.Column('source_data_watermark', sa.DateTime(timezone=True), nullable=False),
    sa.Column('row_count', sa.Integer(), nullable=False),
    sa.Column('rows_with_any_missing_feature', sa.Integer(), nullable=False),
    sa.Column('rows_with_unresolved_label', sa.Integer(), nullable=False),
    sa.Column('checksum', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_by', sa.String(length=128), nullable=False),
    sa.CheckConstraint('end_date >= start_date', name='ck_dataset_date_range_order'),
    sa.ForeignKeyConstraint(['label_definition_id'], ['label_definitions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('dataset_feature_definitions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('feature_definition_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['feature_definition_id'], ['feature_definitions.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id', 'feature_definition_id', name='uq_dataset_feature_definition')
    )
    op.create_table('dataset_instruments',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('instrument_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id', 'instrument_id', name='uq_dataset_instrument')
    )
    op.create_table('dataset_rows',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('row_index', sa.Integer(), nullable=False),
    sa.Column('instrument_id', sa.UUID(), nullable=False),
    sa.Column('entry_date', sa.Date(), nullable=False),
    sa.Column('features', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('label_resolved', sa.Boolean(), nullable=False),
    sa.Column('realized_return', sa.Float(), nullable=True),
    sa.Column('hit_target', sa.Boolean(), nullable=True),
    sa.Column('hit_stop', sa.Boolean(), nullable=True),
    sa.Column('mfe', sa.Float(), nullable=True),
    sa.Column('mae', sa.Float(), nullable=True),
    sa.Column('holding_time_days', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id', 'instrument_id', 'entry_date', name='uq_dataset_row_identity'),
    sa.UniqueConstraint('dataset_id', 'row_index', name='uq_dataset_row_index')
    )

    op.execute("""
        CREATE FUNCTION prevent_dataset_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'dataset tables are immutable once written (never UPDATE or '
                'DELETE - a changed binding or recomputed content means '
                'building a new dataset)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    for table in (
        "datasets",
        "dataset_instruments",
        "dataset_feature_definitions",
        "dataset_rows",
    ):
        op.execute(f"""
            CREATE TRIGGER trg_{table}_no_update
            BEFORE UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION prevent_dataset_mutation();
        """)
        op.execute(f"""
            CREATE TRIGGER trg_{table}_no_delete
            BEFORE DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION prevent_dataset_mutation();
        """)


def downgrade() -> None:
    op.drop_table('dataset_rows')
    op.drop_table('dataset_instruments')
    op.drop_table('dataset_feature_definitions')
    op.drop_table('datasets')

    op.execute("DROP FUNCTION IF EXISTS prevent_dataset_mutation() CASCADE")
