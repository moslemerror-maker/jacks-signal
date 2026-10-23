from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EQUITY_LIVE_DIR = ROOT / "backend" / "app" / "equity_live"


def _imported_module_names(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestOptionsIsolation:
    def test_no_source_file_in_equity_live_imports_from_options(self):
        py_files = sorted(EQUITY_LIVE_DIR.glob("*.py"))
        assert len(py_files) >= 2, "expected live_feature_service.py and candidate_scanner.py to exist"
        offenders = []
        for f in py_files:
            imports = _imported_module_names(f)
            bad = {m for m in imports if m.startswith("backend.app.options") or m == "backend.app.options"}
            if bad:
                offenders.append((f.name, bad))
        assert offenders == []

    def test_modules_import_cleanly_without_touching_options(self):
        import sys
        before = {m for m in sys.modules if m.startswith("backend.app.options")}
        import backend.app.equity_live.candidate_scanner  # noqa: F401
        import backend.app.equity_live.live_feature_service  # noqa: F401
        after = {m for m in sys.modules if m.startswith("backend.app.options")}
        assert after == before
