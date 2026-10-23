from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'fdd64860370f'
down_revision: Union[str, Sequence[str], None] = '4843c50996f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_REQUIRED_CHECKS_SQL = (
    "'source_integrity_ok', 'two_independent_providers', 'window_covered', 'freshness_ok', "
    "'boundary_semantics_verified', 'partition_consistent', 'truncation_evidence_clear', "
    "'reconciliation_clean', 'revisions_resolved', 'snapshot_diffs_clear', 'blockers_clear'"
)
_ZERO_COUNT_PATHS_SQL = (
    "'{reconciliation,conflict_count}', '{reconciliation,one_sided_count}', "
    "'{reconciliation,unclassified_count}', '{reconciliation,duplicate_price_affecting_count}', "
    "'{reconciliation,ambiguous_same_date_count}', '{reconciliation,unparseable_row_count}', "
    "'{revisions,blocking_count}', '{diagnostics,jump_scan,blocking_count}', "
    "'{diagnostics,isin_transitions,uncorroborated_count}'"
)


def upgrade() -> None:
    op.add_column('corporate_action_coverage_assessments', sa.Column('evidence', postgresql.JSONB(), nullable=True))
    op.add_column('corporate_action_coverage_assessments', sa.Column('evidence_sha256', sa.String(length=64), nullable=True))
    op.add_column('corporate_action_coverage_assessments', sa.Column('derived_by', sa.String(length=96), nullable=True))
    op.create_check_constraint(
        'ck_coverage_evidence_columns_together', 'corporate_action_coverage_assessments',
        '(evidence IS NULL) = (evidence_sha256 IS NULL) AND (evidence IS NULL) = (derived_by IS NULL)',
    )
    op.create_check_constraint(
        'ck_coverage_complete_requires_derivation', 'corporate_action_coverage_assessments',
        "status <> 'COMPLETE' OR (evidence IS NOT NULL AND evidence_sha256 IS NOT NULL AND derived_by IS NOT NULL)",
    )

    op.execute(f"""
        CREATE FUNCTION enforce_derived_coverage_assessment()
        RETURNS TRIGGER AS $$
        DECLARE
            k text;
            path_text text;
            n int;
        BEGIN
            IF NEW.evidence IS NOT NULL THEN
                IF NEW.derived_by NOT LIKE 'coverage-derivation/%' THEN
                    RAISE EXCEPTION USING MESSAGE = 'coverage assessment derived_by must be a coverage-derivation engine id, got ' || coalesce(NEW.derived_by, 'NULL');
                END IF;
                IF (NEW.evidence ->> 'verdict') IS DISTINCT FROM NEW.status::text THEN
                    RAISE EXCEPTION USING MESSAGE = 'coverage status ' || NEW.status::text || ' does not equal the evidence-derived verdict ' || coalesce(NEW.evidence ->> 'verdict', 'NULL') || ' - status cannot be asserted independently of its evidence';
                END IF;
                IF (NEW.evidence ->> 'schema') IS DISTINCT FROM 'coverage-evidence/1' THEN
                    RAISE EXCEPTION USING MESSAGE = 'unrecognised coverage evidence schema';
                END IF;
            END IF;

            IF NEW.status = 'COMPLETE' THEN
                IF NEW.evidence IS NULL THEN
                    RAISE EXCEPTION USING MESSAGE = 'a COMPLETE coverage assessment requires an evidence bundle';
                END IF;
                FOREACH k IN ARRAY ARRAY[{_REQUIRED_CHECKS_SQL}] LOOP
                    IF (NEW.evidence -> 'checks' ->> k) IS DISTINCT FROM 'true' THEN
                        RAISE EXCEPTION USING MESSAGE = 'COMPLETE coverage rejected: evidence check ' || k || ' is not recorded as passed';
                    END IF;
                END LOOP;
                FOREACH path_text IN ARRAY ARRAY[{_ZERO_COUNT_PATHS_SQL}] LOOP
                    n := coalesce((NEW.evidence #>> path_text::text[])::int, -1);
                    IF n <> 0 THEN
                        RAISE EXCEPTION USING MESSAGE = 'COMPLETE coverage rejected: ' || path_text || ' must be recorded as 0, found ' || n::text;
                    END IF;
                END LOOP;
                IF NOT jsonb_path_exists(NEW.evidence, '$.sources[*] ? (@.provider == "NSE" && @.whole != null)')
                   OR NOT jsonb_path_exists(NEW.evidence, '$.sources[*] ? (@.provider == "BSE" && @.whole != null)') THEN
                    RAISE EXCEPTION USING MESSAGE = 'COMPLETE coverage rejected: both NSE and BSE must have a whole-window snapshot recorded';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_coverage_assessment_derived
        BEFORE INSERT ON corporate_action_coverage_assessments
        FOR EACH ROW EXECUTE FUNCTION enforce_derived_coverage_assessment();
    """)

    op.create_table(
        'corporate_action_unresolved_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('instrument_id', sa.UUID(), nullable=False),
        sa.Column('assessment_id', sa.UUID(), nullable=False),
        sa.Column('event_id', sa.UUID(), nullable=True),
        sa.Column('ex_date', sa.Date(), nullable=True),
        sa.Column('action_class', sa.String(length=24), nullable=False),
        sa.Column('reason_code', sa.String(length=48), nullable=False),
        sa.Column('severity', sa.String(length=24), nullable=False),
        sa.Column('detail', postgresql.JSONB(), nullable=False),
        sa.Column('registered_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id']),
        sa.ForeignKeyConstraint(['assessment_id'], ['corporate_action_coverage_assessments.id']),
        sa.ForeignKeyConstraint(['event_id'], ['corporate_action_events.id']),
        sa.CheckConstraint(
            "severity IN ('BLOCKS_COMPLETE', 'BLOCKS_ADJUSTMENT', 'INFO')",
            name='ck_unresolved_event_severity',
        ),
    )
    op.create_index(
        'ix_unresolved_events_instrument_assessment', 'corporate_action_unresolved_events',
        ['instrument_id', 'assessment_id'],
    )
    op.execute("""
        CREATE FUNCTION prevent_unresolved_event_update()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE = 'corporate_action_unresolved_events is append-only: cannot modify id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_unresolved_events_immutable
        BEFORE UPDATE ON corporate_action_unresolved_events
        FOR EACH ROW EXECUTE FUNCTION prevent_unresolved_event_update();
    """)
    op.execute("""
        CREATE FUNCTION prevent_unresolved_event_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE = 'corporate_action_unresolved_events rows cannot be deleted (audit trail): id=' || OLD.id::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_unresolved_events_no_delete
        BEFORE DELETE ON corporate_action_unresolved_events
        FOR EACH ROW EXECUTE FUNCTION prevent_unresolved_event_delete();
    """)


def downgrade() -> None:
    bind = op.get_bind()
    registered = bind.execute(sa.text("SELECT count(*) FROM corporate_action_unresolved_events")).scalar()
    derived = bind.execute(
        sa.text("SELECT count(*) FROM corporate_action_coverage_assessments WHERE evidence IS NOT NULL")
    ).scalar()
    if registered or derived:
        raise RuntimeError(
            f"refusing to downgrade: {registered} unresolved-event register row(s) and {derived} "
            "evidence-bearing coverage assessment(s) exist and would be destroyed (audit data that "
            "cannot be rebuilt). See docs/32 - resolve deliberately, do not force."
        )
    op.drop_index('ix_unresolved_events_instrument_assessment', table_name='corporate_action_unresolved_events')
    op.drop_table('corporate_action_unresolved_events')
    op.execute("DROP FUNCTION IF EXISTS prevent_unresolved_event_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_unresolved_event_delete()")

    op.execute("DROP TRIGGER IF EXISTS trg_coverage_assessment_derived ON corporate_action_coverage_assessments")
    op.execute("DROP FUNCTION IF EXISTS enforce_derived_coverage_assessment()")
    op.drop_constraint('ck_coverage_complete_requires_derivation', 'corporate_action_coverage_assessments', type_='check')
    op.drop_constraint('ck_coverage_evidence_columns_together', 'corporate_action_coverage_assessments', type_='check')
    op.drop_column('corporate_action_coverage_assessments', 'derived_by')
    op.drop_column('corporate_action_coverage_assessments', 'evidence_sha256')
    op.drop_column('corporate_action_coverage_assessments', 'evidence')
