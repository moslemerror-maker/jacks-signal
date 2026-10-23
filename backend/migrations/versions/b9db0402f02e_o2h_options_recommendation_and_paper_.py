from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'b9db0402f02e'
down_revision: Union[str, Sequence[str], None] = '0ec974920efd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ("option_recommendations", "option_paper_trade_events", "option_session_events")


def upgrade() -> None:
    op.create_table(
        'option_recommendations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('decision_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('underlying', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('shape', sa.String(length=32), nullable=True),
        sa.Column('data_quality', sa.String(length=48), nullable=False),
        sa.Column('model_version', sa.String(length=64), nullable=True),
        sa.Column('strategy_version', sa.String(length=64), nullable=True),
        sa.Column('source', sa.String(length=32), nullable=False),
        sa.Column('payload', postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("status IN ('TRADE', 'WAIT', 'NO_TRADE')", name='ck_option_recommendations_status'),
    )
    op.create_index('ix_option_recommendations_decision_ts', 'option_recommendations', ['underlying', 'decision_timestamp'])
    op.create_index('ix_option_recommendations_status', 'option_recommendations', ['status'])

    op.create_table(
        'option_paper_trade_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('recommendation_id', sa.UUID(), nullable=False),
        sa.Column('event_type', sa.String(length=16), nullable=False),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('payload', postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['recommendation_id'], ['option_recommendations.id'], name='fk_option_paper_trade_events_recommendation'),
        sa.CheckConstraint("event_type IN ('PAPER_ENTRY', 'OBSERVATION', 'EXIT')", name='ck_option_paper_trade_events_type'),
    )
    op.create_index('ix_option_paper_trade_events_recommendation', 'option_paper_trade_events', ['recommendation_id', 'event_timestamp'])

    op.create_table(
        'option_session_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('severity', sa.String(length=16), nullable=False),
        sa.Column('category', sa.String(length=48), nullable=False),
        sa.Column('detail', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("severity IN ('INFO', 'WARNING', 'ERROR')", name='ck_option_session_events_severity'),
    )
    op.create_index('ix_option_session_events_ts', 'option_session_events', ['event_timestamp'])

    op.execute("""
        CREATE FUNCTION prevent_option_table_update()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                TG_TABLE_NAME || ' is immutable: cannot modify id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE FUNCTION prevent_option_table_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                TG_TABLE_NAME || ' rows cannot be deleted (immutable/append-only): id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    for t in _TABLES:
        op.execute(f"""
            CREATE TRIGGER trg_{t}_no_update
            BEFORE UPDATE ON {t}
            FOR EACH ROW EXECUTE FUNCTION prevent_option_table_update();
        """)
        op.execute(f"""
            CREATE TRIGGER trg_{t}_no_delete
            BEFORE DELETE ON {t}
            FOR EACH ROW EXECUTE FUNCTION prevent_option_table_delete();
        """)


def downgrade() -> None:
    bind = op.get_bind()
    counts = {t: bind.execute(sa.text(f'SELECT count(*) FROM "{t}"')).scalar() for t in _TABLES}
    if any(counts.values()):
        raise RuntimeError(
            f"refusing to downgrade: rows exist in {({t: n for t, n in counts.items() if n})} - these tables hold "
            "immutable/append-only decision and event data that cannot be reconstructed. See docs/32 - resolve "
            "deliberately, do not force."
        )

    for t in _TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{t}_no_update ON {t}")
        op.execute(f"DROP TRIGGER IF EXISTS trg_{t}_no_delete ON {t}")
    op.execute("DROP FUNCTION IF EXISTS prevent_option_table_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_option_table_delete()")

    op.drop_index('ix_option_session_events_ts', table_name='option_session_events')
    op.drop_table('option_session_events')
    op.drop_index('ix_option_paper_trade_events_recommendation', table_name='option_paper_trade_events')
    op.drop_table('option_paper_trade_events')
    op.drop_index('ix_option_recommendations_status', table_name='option_recommendations')
    op.drop_index('ix_option_recommendations_decision_ts', table_name='option_recommendations')
    op.drop_table('option_recommendations')
