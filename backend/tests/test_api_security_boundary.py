from __future__ import annotations

import ast
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1] / "app" / "api"

FORBIDDEN_MODULE_SUBSTRINGS = (
    "dhan_client", "dhan_optionchain_client", "dhan_live_chain_source", "dhan_live_quote",
    "provider_credentials", "credential_preflight",
)
FORBIDDEN_LIVE_SESSION_NAMES = ("run_one_cycle", "run_session", "DhanLiveChainSource")


def _imported_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _api_python_files() -> list[Path]:
    assert API_ROOT.is_dir(), f"expected {API_ROOT} to exist"
    return sorted(API_ROOT.rglob("*.py"))


class TestNoDhanSpecificImportAnywhereInTheApiPackage:
    def test_no_forbidden_module_is_imported(self):
        offenders = []
        for path in _api_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = _imported_module_names(tree)
            for forbidden in FORBIDDEN_MODULE_SUBSTRINGS:
                if any(forbidden in name for name in imported):
                    offenders.append((path, forbidden))
        assert offenders == [], f"forbidden Dhan-specific import(s) found: {offenders}"

    def test_no_live_session_execution_function_is_imported(self):
        offenders = []
        for path in _api_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = _imported_module_names(tree)
            for forbidden in FORBIDDEN_LIVE_SESSION_NAMES:
                if any(forbidden in name for name in imported):
                    offenders.append((path, forbidden))
        assert offenders == [], f"forbidden live-session import(s) found: {offenders}"

    def test_at_least_one_file_was_actually_scanned(self):
        assert len(_api_python_files()) >= 6

    def test_no_requests_or_dhanhq_import_anywhere_in_the_api_package(self):
        offenders = []
        for path in _api_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = _imported_module_names(tree)
            for forbidden in ("requests", "dhanhq"):
                if any(name == forbidden or name.startswith(f"{forbidden}.") for name in imported):
                    offenders.append((path, forbidden))
        assert offenders == [], f"forbidden network-provider import(s) found: {offenders}"
