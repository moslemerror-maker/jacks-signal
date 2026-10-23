from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '4e5c0126d839'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('instruments',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('isin', sa.String(length=12), nullable=True),
    sa.Column('exchange', sa.String(length=16), nullable=False),
    sa.Column('asset_type', sa.Enum('EQUITY', 'INDEX', 'FUTURE', 'OPTION', name='assettype'), nullable=False),
    sa.Column('sector', sa.String(length=64), nullable=True),
    sa.Column('underlying_instrument_id', sa.UUID(), nullable=True),
    sa.Column('expiry_date', sa.Date(), nullable=True),
    sa.Column('strike_price', sa.Float(), nullable=True),
    sa.Column('option_type', sa.Enum('CE', 'PE', name='optiontype'), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['underlying_instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('isin')
    )
    op.create_table('instrument_symbols',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('instrument_id', sa.UUID(), nullable=False),
    sa.Column('symbol', sa.String(length=32), nullable=False),
    sa.Column('effective_from', sa.Date(), nullable=False),
    sa.Column('effective_to', sa.Date(), nullable=True),
    sa.CheckConstraint('effective_to IS NULL OR effective_to >= effective_from', name='ck_instrument_symbol_period_order'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('instrument_id', 'effective_from', name='uq_instrument_symbol_period')
    )
    op.create_table('market_bars',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('instrument_id', sa.UUID(), nullable=False),
    sa.Column('trade_date', sa.Date(), nullable=False),
    sa.Column('series', sa.String(length=8), nullable=False),
    sa.Column('timeframe', sa.String(length=8), nullable=False),
    sa.Column('open', sa.Float(), nullable=False),
    sa.Column('high', sa.Float(), nullable=False),
    sa.Column('low', sa.Float(), nullable=False),
    sa.Column('close', sa.Float(), nullable=False),
    sa.Column('last_price', sa.Float(), nullable=True),
    sa.Column('prev_close', sa.Float(), nullable=True),
    sa.Column('volume', sa.Float(), nullable=True),
    sa.Column('turnover', sa.Float(), nullable=True),
    sa.Column('total_trades', sa.Integer(), nullable=True),
    sa.Column('open_interest', sa.Float(), nullable=True),
    sa.Column('source_format', sa.Enum('NSE_LEGACY_CSV', 'NSE_UDIFF', name='sourceformat'), nullable=False),
    sa.Column('source_file', sa.String(length=128), nullable=False),
    sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('close BETWEEN low AND high', name='ck_market_bar_close_in_range'),
    sa.CheckConstraint('high >= low', name='ck_market_bar_high_ge_low'),
    sa.CheckConstraint('open BETWEEN low AND high', name='ck_market_bar_open_in_range'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('instrument_id', 'trade_date', 'series', 'timeframe', name='uq_market_bar_identity')
    )


def downgrade() -> None:
    op.drop_table('market_bars')
    op.drop_table('instrument_symbols')
    op.drop_table('instruments')

    sa.Enum(name='sourceformat').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='optiontype').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='assettype').drop(op.get_bind(), checkfirst=True)
