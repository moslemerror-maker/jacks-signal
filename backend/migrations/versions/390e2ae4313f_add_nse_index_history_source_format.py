from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '390e2ae4313f'
down_revision: Union[str, Sequence[str], None] = '4e5c0126d839'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE sourceformat ADD VALUE 'NSE_INDEX_HISTORY'")


def downgrade() -> None:
    op.execute("ALTER TYPE sourceformat RENAME TO sourceformat_old")
    op.execute("CREATE TYPE sourceformat AS ENUM ('NSE_LEGACY_CSV', 'NSE_UDIFF')")
    op.execute(
        "ALTER TABLE market_bars ALTER COLUMN source_format "
        "TYPE sourceformat USING source_format::text::sourceformat"
    )
    op.execute("DROP TYPE sourceformat_old")
