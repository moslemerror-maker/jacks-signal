from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1f3c9e7d2b4'
down_revision: Union[str, Sequence[str], None] = '9c903932e04b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ("option_shadow_observations", "option_shadow_forward_observations")


def upgrade() -> None:
    op.create_table(
        'option_shadow_observations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('recommendation_id', sa.UUID(), nullable=False),
        sa.Column('research_phase_id', sa.String(length=64), nullable=False),
        sa.Column('hypothesis_id', sa.String(length=64), nullable=True),
        sa.Column('data_version', sa.String(length=64), nullable=False),
        sa.Column('feature_version', sa.String(length=64), nullable=False),
        sa.Column('cost_model_version', sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['recommendation_id'], ['option_recommendations.id'], name='fk_option_shadow_observations_recommendation'),
    )
    op.create_index('ix_option_shadow_observations_recommendation', 'option_shadow_observations', ['recommendation_id'])
    op.create_index('ix_option_shadow_observations_phase', 'option_shadow_observations', ['research_phase_id'])

    op.create_table(
        'option_shadow_forward_observations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('shadow_observation_id', sa.UUID(), nullable=False),
        sa.Column('forward_recommendation_id', sa.UUID(), nullable=False),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('elapsed_seconds', sa.Float(), nullable=False),
        sa.Column('quote_status', sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['shadow_observation_id'], ['option_shadow_observations.id'], name='fk_option_shadow_forward_observations_shadow'),
        sa.ForeignKeyConstraint(['forward_recommendation_id'], ['option_recommendations.id'], name='fk_option_shadow_forward_observations_recommendation'),
        sa.CheckConstraint("quote_status IN ('QUOTE_AVAILABLE', 'QUOTE_MISSING')", name='ck_option_shadow_forward_observations_quote_status'),
        sa.CheckConstraint("elapsed_seconds > 0", name='ck_option_shadow_forward_observations_elapsed_positive'),
    )
    op.create_index('ix_option_shadow_forward_observations_shadow', 'option_shadow_forward_observations', ['shadow_observation_id'])

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
            "immutable/append-only research data that cannot be reconstructed. See docs/32 - resolve deliberately, do not force."
        )

    for t in _TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{t}_no_update ON {t}")
        op.execute(f"DROP TRIGGER IF EXISTS trg_{t}_no_delete ON {t}")

    op.drop_index('ix_option_shadow_forward_observations_shadow', table_name='option_shadow_forward_observations')
    op.drop_table('option_shadow_forward_observations')
    op.drop_index('ix_option_shadow_observations_phase', table_name='option_shadow_observations')
    op.drop_index('ix_option_shadow_observations_recommendation', table_name='option_shadow_observations')
    op.drop_table('option_shadow_observations')
