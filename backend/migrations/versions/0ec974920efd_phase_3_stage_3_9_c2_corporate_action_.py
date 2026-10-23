from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '0ec974920efd'
down_revision: Union[str, Sequence[str], None] = 'fdd64860370f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_RESERVED_ATTESTER_PATTERN = (
    "^[[:space:]]*(system|auto|automation|automated|automatic|machine|assistant|"
    "model|llm|gpt|bot|script|stage|unspecified|unknown|none|null|root|admin|cron|batch|tool|agent|pytest|"
    "test|tests|human)([^a-zA-Z]|$)"
)


def upgrade() -> None:
    op.create_table(
        'corporate_action_term_verifications',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('seq', sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column('component_id', sa.UUID(), nullable=False),
        sa.Column('action', sa.String(length=16), nullable=False),
        sa.Column('action_kind', sa.String(length=16), nullable=False),
        sa.Column('term_a', sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column('term_b', sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column('tier', sa.SmallInteger(), nullable=False),
        sa.Column('evidence', postgresql.JSONB(), nullable=False),
        sa.Column('evidence_sha256', sa.String(length=64), nullable=False),
        sa.Column('attested_by', sa.String(length=128), nullable=False),
        sa.Column('attestation_note', sa.Text(), nullable=False),
        sa.Column('attested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('pit_status', sa.String(length=24), nullable=False, server_default='NOT_ESTABLISHED'),
        sa.Column('revokes_id', sa.UUID(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('seq', name='uq_term_verification_seq'),
        sa.ForeignKeyConstraint(['component_id'], ['corporate_action_components.id'], name='fk_term_verification_component'),
        sa.ForeignKeyConstraint(['revokes_id'], ['corporate_action_term_verifications.id'], name='fk_term_verification_revokes'),
        sa.CheckConstraint("action IN ('VERIFIED', 'REVOKED')", name='ck_term_verification_action'),
        sa.CheckConstraint("action_kind IN ('SPLIT', 'BONUS')", name='ck_term_verification_kind'),
        sa.CheckConstraint('term_a > 0 AND term_b > 0', name='ck_term_verification_terms_positive'),
        sa.CheckConstraint(
            "action_kind <> 'BONUS' OR (term_a = trunc(term_a) AND term_b = trunc(term_b))",
            name='ck_term_verification_bonus_whole',
        ),
        sa.CheckConstraint("action_kind <> 'SPLIT' OR term_a <> term_b", name='ck_term_verification_split_not_noop'),
        sa.CheckConstraint('tier IN (1, 2)', name='ck_term_verification_tier'),
        sa.CheckConstraint("evidence_sha256 ~ '^[0-9a-f]{64}$'", name='ck_term_verification_sha256'),
        sa.CheckConstraint("pit_status = 'NOT_ESTABLISHED'", name='ck_term_verification_pit_status'),
        sa.CheckConstraint("length(btrim(attestation_note)) > 0", name='ck_term_verification_note'),
        sa.CheckConstraint(
            "(action = 'VERIFIED' AND revokes_id IS NULL) OR (action = 'REVOKED' AND revokes_id IS NOT NULL)",
            name='ck_term_verification_revokes_shape',
        ),
        sa.CheckConstraint(
            "length(btrim(attested_by)) >= 3 AND attested_by ~ '[A-Za-z]' "
            f"AND attested_by !~* '{_RESERVED_ATTESTER_PATTERN}'",
            name='ck_term_verification_attested_by_human',
        ),
    )
    op.create_index(
        'ix_term_verification_component_seq', 'corporate_action_term_verifications', ['component_id', 'seq']
    )

    op.execute("""
        CREATE FUNCTION prevent_term_verification_change()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE = 'corporate_action_term_verifications is append-only: ' || TG_OP || ' is not permitted (id=' || OLD.id::text || ')';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_term_verification_append_only
        BEFORE UPDATE OR DELETE ON corporate_action_term_verifications
        FOR EACH ROW EXECUTE FUNCTION prevent_term_verification_change();
    """)

    op.execute("""
        CREATE FUNCTION enforce_term_verification_insert()
        RETURNS TRIGGER AS $$
        DECLARE
            c RECORD;
            cur_rev int;
            v RECORD;
        BEGIN
            SELECT id, event_id, component_type::text AS ctype, parse_revision, face_value_from, face_value_to,
                   ratio_new, ratio_existing
              INTO c FROM corporate_action_components WHERE id = NEW.component_id;
            IF NOT FOUND THEN
                RAISE EXCEPTION USING MESSAGE = 'term verification references an unknown component ' || NEW.component_id::text;
            END IF;
            IF c.ctype IS DISTINCT FROM NEW.action_kind THEN
                RAISE EXCEPTION USING MESSAGE = 'term verification kind ' || NEW.action_kind || ' does not match the component type ' || c.ctype;
            END IF;

            IF NEW.action = 'VERIFIED' THEN
                SELECT max(parse_revision) INTO cur_rev FROM corporate_action_components WHERE event_id = c.event_id;
                IF c.parse_revision <> cur_rev THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: component is parse revision ' || c.parse_revision::text || ' but the event is now at revision ' || cur_rev::text || ' (a re-parse creates a new component that must be verified on its own)';
                END IF;
                IF (c.ctype = 'SPLIT' AND (NEW.term_a IS DISTINCT FROM round(c.face_value_from::numeric, 6) OR NEW.term_b IS DISTINCT FROM round(c.face_value_to::numeric, 6)))
                   OR (c.ctype = 'BONUS' AND (NEW.term_a IS DISTINCT FROM c.ratio_new::numeric OR NEW.term_b IS DISTINCT FROM c.ratio_existing::numeric)) THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: attested terms do not equal the component''s own stored terms';
                END IF;
                IF EXISTS (
                    SELECT 1 FROM corporate_action_term_verifications x
                    WHERE x.component_id = NEW.component_id AND x.action = 'VERIFIED'
                      AND NOT EXISTS (SELECT 1 FROM corporate_action_term_verifications r WHERE r.action = 'REVOKED' AND r.revokes_id = x.id)
                ) THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: this component already has an active verification (revoke it first)';
                END IF;
                IF (NEW.evidence ->> 'schema') IS DISTINCT FROM 'corporate-action-term-evidence-sheet/1' THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: evidence is not a corporate-action-term-evidence-sheet/1';
                END IF;
                IF (NEW.evidence #>> '{component,component_id}') IS DISTINCT FROM NEW.component_id::text THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: the evidence sheet is for a different component';
                END IF;
                IF (NEW.evidence ->> 'pit_status') IS DISTINCT FROM 'NOT_ESTABLISHED' THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: evidence must state pit_status NOT_ESTABLISHED';
                END IF;
                IF (NEW.evidence #>> '{readiness,result}') NOT IN ('READY_TIER1', 'READY_TIER2')
                   OR (NEW.evidence #>> '{readiness,verifiable}') IS DISTINCT FROM 'true'
                   OR coalesce(jsonb_array_length(NEW.evidence #> '{readiness,blockers}'), -1) <> 0 THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: the evidence sheet is not READY with zero blockers';
                END IF;
                IF (NEW.evidence #>> '{readiness,tier_required}') IS DISTINCT FROM NEW.tier::text THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: tier does not equal the evidence sheet''s required tier';
                END IF;
                IF (NEW.evidence #>> '{terms,kind}') IS DISTINCT FROM NEW.action_kind
                   OR (NEW.evidence #>> '{terms,term_a}')::numeric IS DISTINCT FROM NEW.term_a
                   OR (NEW.evidence #>> '{terms,term_b}')::numeric IS DISTINCT FROM NEW.term_b THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: the evidence sheet''s terms differ from the attested terms';
                END IF;
                IF NEW.tier = 2 AND (
                    (NEW.evidence #>> '{primary_source,valid}') IS DISTINCT FROM 'true'
                    OR (NEW.evidence #>> '{primary_source,sha256}') !~ '^[0-9a-f]{64}$'
                ) THEN
                    RAISE EXCEPTION USING MESSAGE = 'term verification refused: Tier 2 requires a valid, hashed primary-source document';
                END IF;
            ELSE
                SELECT * INTO v FROM corporate_action_term_verifications WHERE id = NEW.revokes_id;
                IF NOT FOUND OR v.action <> 'VERIFIED' THEN
                    RAISE EXCEPTION USING MESSAGE = 'a revocation must reference a VERIFIED record';
                END IF;
                IF v.component_id <> NEW.component_id OR v.action_kind <> NEW.action_kind
                   OR v.term_a <> NEW.term_a OR v.term_b <> NEW.term_b THEN
                    RAISE EXCEPTION USING MESSAGE = 'a revocation must carry the component, kind and exact terms of the record it revokes';
                END IF;
                IF EXISTS (SELECT 1 FROM corporate_action_term_verifications r WHERE r.action = 'REVOKED' AND r.revokes_id = NEW.revokes_id) THEN
                    RAISE EXCEPTION USING MESSAGE = 'that verification has already been revoked';
                END IF;
                IF length(btrim(coalesce(NEW.evidence #>> '{revocation,reason}', ''))) = 0 THEN
                    RAISE EXCEPTION USING MESSAGE = 'a revocation must state a reason';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_term_verification_insert_guard
        BEFORE INSERT ON corporate_action_term_verifications
        FOR EACH ROW EXECUTE FUNCTION enforce_term_verification_insert();
    """)

    op.execute("""
        CREATE FUNCTION enforce_component_manual_verification()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.verification_status = 'MANUALLY_VERIFIED' AND OLD.verification_status IS DISTINCT FROM 'MANUALLY_VERIFIED' THEN
                IF NOT EXISTS (
                    SELECT 1 FROM corporate_action_term_verifications v
                    WHERE v.component_id = NEW.id
                      AND v.action = 'VERIFIED'
                      AND v.action_kind = NEW.component_type::text
                      AND NOT EXISTS (SELECT 1 FROM corporate_action_term_verifications r WHERE r.action = 'REVOKED' AND r.revokes_id = v.id)
                      AND (
                            (NEW.component_type::text = 'SPLIT' AND v.term_a = round(NEW.face_value_from::numeric, 6) AND v.term_b = round(NEW.face_value_to::numeric, 6))
                         OR (NEW.component_type::text = 'BONUS' AND v.term_a = NEW.ratio_new::numeric AND v.term_b = NEW.ratio_existing::numeric)
                          )
                ) THEN
                    RAISE EXCEPTION USING MESSAGE = 'MANUALLY_VERIFIED requires an active C2 term verification record matching this component id, kind and exact terms (component ' || NEW.id::text || ')';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_component_manual_verification_guard
        BEFORE UPDATE OF verification_status ON corporate_action_components
        FOR EACH ROW EXECUTE FUNCTION enforce_component_manual_verification();
    """)


def downgrade() -> None:
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT count(*) FROM corporate_action_term_verifications")).scalar()
    if rows:
        raise RuntimeError(
            f"refusing to downgrade: {rows} term-verification record(s) exist. They are the append-only audit "
            "trail of human attestations and cannot be rebuilt. See docs/32 - resolve deliberately, do not force."
        )
    op.execute("DROP TRIGGER IF EXISTS trg_component_manual_verification_guard ON corporate_action_components")
    op.execute("DROP FUNCTION IF EXISTS enforce_component_manual_verification()")
    op.execute("DROP TRIGGER IF EXISTS trg_term_verification_insert_guard ON corporate_action_term_verifications")
    op.execute("DROP FUNCTION IF EXISTS enforce_term_verification_insert()")
    op.execute("DROP TRIGGER IF EXISTS trg_term_verification_append_only ON corporate_action_term_verifications")
    op.execute("DROP FUNCTION IF EXISTS prevent_term_verification_change()")
    op.drop_index('ix_term_verification_component_seq', table_name='corporate_action_term_verifications')
    op.drop_table('corporate_action_term_verifications')
