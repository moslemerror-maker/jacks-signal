from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b701f3c8a7df'
down_revision: Union[str, Sequence[str], None] = '390e2ae4313f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('instrument_isins',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('instrument_id', sa.UUID(), nullable=False),
    sa.Column('isin', sa.String(length=12), nullable=False),
    sa.Column('effective_from', sa.Date(), nullable=False),
    sa.Column('effective_to', sa.Date(), nullable=True),
    sa.CheckConstraint('effective_to IS NULL OR effective_to >= effective_from', name='ck_instrument_isin_period_order'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('isin')
    )
    op.drop_constraint(op.f('instruments_isin_key'), 'instruments', type_='unique')
    op.drop_column('instruments', 'isin')


def downgrade() -> None:
    op.add_column('instruments', sa.Column('isin', sa.VARCHAR(length=12), autoincrement=False, nullable=True))
    op.create_unique_constraint(op.f('instruments_isin_key'), 'instruments', ['isin'], postgresql_nulls_not_distinct=False)
    op.drop_table('instrument_isins')
