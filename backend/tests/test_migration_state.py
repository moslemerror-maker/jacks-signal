from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _latest_head_revision() -> str:
    config = Config(str(_REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPO_ROOT / "backend" / "migrations"))
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    assert len(heads) == 1, (
        f"expected exactly one migration head, found {len(heads)}: {heads} - "
        "a branched migration history needs its own resolution, this test "
        "only handles the normal linear case"
    )
    return heads[0]


def test_database_is_stamped_at_alembic_head(db_session):
    stamped = db_session.execute(text("SELECT version_num FROM alembic_version")).scalar()
    expected = _latest_head_revision()
    assert stamped == expected, (
        f"trading_system_test is stamped at {stamped!r} but the latest migration "
        f"head is {expected!r} - the test database is behind (or ahead of/on a "
        "different branch than) the committed migrations. Run `alembic upgrade "
        "head` against it (see docs/32-migration-safety.md) rather than assuming "
        "it's current."
    )
