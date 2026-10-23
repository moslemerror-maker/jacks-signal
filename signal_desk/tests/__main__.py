import inspect
import sys
import tempfile
import traceback
from pathlib import Path

from . import test_engine_sim, test_research, test_stock_research, test_units

fails = 0
n = 0
for mod in (test_units, test_engine_sim, test_research, test_stock_research):
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        n += 1
        kwargs = {"tmp_path": Path(tempfile.mkdtemp())} if "tmp_path" in inspect.signature(fn).parameters else {}
        try:
            fn(**kwargs)
            print(f"ok    {mod.__name__.split('.')[-1]}::{name}")
        except Exception:
            fails += 1
            print(f"FAIL  {mod.__name__.split('.')[-1]}::{name}")
            traceback.print_exc()
print(f"\n{n - fails}/{n} passed")
sys.exit(1 if fails else 0)
