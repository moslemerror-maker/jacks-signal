from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a297f8fb115d'
down_revision: Union[str, Sequence[str], None] = '3e598b4ad4ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE FUNCTION prevent_feature_value_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'feature_values rows are immutable once written (never UPDATE or '
                'DELETE - a changed value means the definition needs a new version)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_feature_values_no_update
        BEFORE UPDATE ON feature_values
        FOR EACH ROW EXECUTE FUNCTION prevent_feature_value_mutation();
    """)
    op.execute("""
        CREATE TRIGGER trg_feature_values_no_delete
        BEFORE DELETE ON feature_values
        FOR EACH ROW EXECUTE FUNCTION prevent_feature_value_mutation();
    """)

    op.execute("""
        CREATE FUNCTION prevent_label_value_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE =
                'label_values rows are immutable once written (never UPDATE or '
                'DELETE - a changed outcome means the definition needs a new version)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_label_values_no_update
        BEFORE UPDATE ON label_values
        FOR EACH ROW EXECUTE FUNCTION prevent_label_value_mutation();
    """)
    op.execute("""
        CREATE TRIGGER trg_label_values_no_delete
        BEFORE DELETE ON label_values
        FOR EACH ROW EXECUTE FUNCTION prevent_label_value_mutation();
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS prevent_feature_value_mutation() CASCADE")
    op.execute("DROP FUNCTION IF EXISTS prevent_label_value_mutation() CASCADE")
