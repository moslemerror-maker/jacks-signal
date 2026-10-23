from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '461a08f5235b'
down_revision: Union[str, Sequence[str], None] = 'b9db0402f02e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'intraday_equity_bars',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('provider', sa.String(length=16), nullable=False),
        sa.Column('exchange_segment', sa.String(length=16), nullable=False),
        sa.Column('security_id', sa.Integer(), nullable=False),
        sa.Column('trading_symbol', sa.String(length=32), nullable=False),
        sa.Column('timeframe', sa.String(length=8), nullable=False),
        sa.Column('bar_start_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('trading_date', sa.Date(), nullable=False),
        sa.Column('raw_timestamp_epoch', sa.BigInteger(), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=True),
        sa.Column('source_provider', sa.String(length=16), nullable=False),
        sa.Column('source_dataset', sa.String(length=64), nullable=False),
        sa.Column('source_key', sa.String(length=128), nullable=False),
        sa.Column('source_sha256', sa.String(length=64), nullable=False),
        sa.Column('normalizer_version', sa.String(length=64), nullable=False),
        sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('provider', 'exchange_segment', 'security_id', 'timeframe', 'bar_start_utc', name='uq_intraday_equity_bar_identity'),
        sa.CheckConstraint('high >= low', name='ck_intraday_equity_bar_high_ge_low'),
        sa.CheckConstraint('open BETWEEN low AND high', name='ck_intraday_equity_bar_open_in_range'),
        sa.CheckConstraint('close BETWEEN low AND high', name='ck_intraday_equity_bar_close_in_range'),
        sa.CheckConstraint('open > 0 AND high > 0 AND low > 0 AND close > 0', name='ck_intraday_equity_bar_positive_price'),
        sa.CheckConstraint('volume IS NULL OR volume >= 0', name='ck_intraday_equity_bar_volume_nonneg'),
    )
    op.create_index('ix_intraday_equity_bar_security_date', 'intraday_equity_bars', ['security_id', 'trading_date'])


def downgrade() -> None:
    op.drop_index('ix_intraday_equity_bar_security_date', table_name='intraday_equity_bars')
    op.drop_table('intraday_equity_bars')
