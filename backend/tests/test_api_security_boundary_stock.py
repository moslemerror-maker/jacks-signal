from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STOCK_LIVE_ROOT = REPO_ROOT / "backend" / "app" / "data" / "live"
OPTIONS_ROOT = REPO_ROOT / "backend" / "app" / "options"
STOCK_COLLECTOR_SCRIPT = REPO_ROOT / "scripts" / "run_stock_live_collector.py"


def _imported_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _python_files(root: Path) -> list[Path]:
    assert root.is_dir(), f"expected {root} to exist"
    return sorted(root.rglob("*.py"))


class TestStockLiveCollectorNeverImportsOptions:
    def test_no_backend_app_options_import_anywhere_in_the_live_package(self):
        offenders = []
        for path in _python_files(STOCK_LIVE_ROOT):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = _imported_module_names(tree)
            if any("backend.app.options" in name for name in imported):
                offenders.append(path)
        assert offenders == [], f"backend.app.data.live must never import backend.app.options: {offenders}"

    def test_the_entry_point_script_never_imports_options(self):
        tree = ast.parse(STOCK_COLLECTOR_SCRIPT.read_text(encoding="utf-8"), filename=str(STOCK_COLLECTOR_SCRIPT))
        imported = _imported_module_names(tree)
        offenders = [name for name in imported if "backend.app.options" in name]
        assert offenders == [], f"run_stock_live_collector.py must never import backend.app.options: {offenders}"

    def test_at_least_one_file_was_actually_scanned(self):
        assert len(_python_files(STOCK_LIVE_ROOT)) >= 4


class TestOptionsNeverImportsTheStockLiveCollector:
    def test_no_backend_app_data_live_import_anywhere_in_the_options_package(self):
        offenders = []
        for path in _python_files(OPTIONS_ROOT):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = _imported_module_names(tree)
            if any("backend.app.data.live" in name for name in imported):
                offenders.append(path)
        assert offenders == [], f"backend.app.options must never import backend.app.data.live: {offenders}"

    def test_at_least_one_file_was_actually_scanned(self):
        assert len(_python_files(OPTIONS_ROOT)) >= 6


class TestNoSharedDatabaseModelCoupling:
    def test_stock_collection_event_model_has_no_foreign_key_to_any_option_table(self):
        models_path = REPO_ROOT / "backend" / "app" / "db" / "models.py"
        source = models_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(models_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "StockCollectionEvent":
                class_source = ast.get_source_segment(source, node) or ""
                assert "ForeignKey" not in class_source, "StockCollectionEvent must never carry a foreign key"
                assert "option_" not in class_source.lower(), "StockCollectionEvent must never reference an options table"
                return
        raise AssertionError("StockCollectionEvent class not found in models.py")
