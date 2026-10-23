from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9c903932e04b'
down_revision: Union[str, Sequence[str], None] = '461a08f5235b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'stock_collection_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('trading_date', sa.Date(), nullable=False),
        sa.Column('severity', sa.String(length=16), nullable=False),
        sa.Column('category', sa.String(length=48), nullable=False),
        sa.Column('symbols_expected', sa.Integer(), nullable=True),
        sa.Column('symbols_succeeded', sa.Integer(), nullable=True),
        sa.Column('symbols_failed', sa.Integer(), nullable=True),
        sa.Column('latest_completed_minute', sa.DateTime(timezone=True), nullable=True),
        sa.Column('bars_persisted', sa.Integer(), nullable=True),
        sa.Column('detail', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_stock_collection_event_trading_date', 'stock_collection_events', ['trading_date'])
    op.create_index('ix_stock_collection_event_category', 'stock_collection_events', ['category'])


def downgrade() -> None:
    op.drop_index('ix_stock_collection_event_category', table_name='stock_collection_events')
    op.drop_index('ix_stock_collection_event_trading_date', table_name='stock_collection_events')
    op.drop_table('stock_collection_events')
