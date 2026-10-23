import inspect, sys, traceback
from . import test_lab
n = fails = 0
for name, fn in inspect.getmembers(test_lab, inspect.isfunction):
    if name.startswith("test_"):
        n += 1
        try:
            fn(); print("ok   ", name)
        except Exception:
            fails += 1; print("FAIL ", name); traceback.print_exc()
print(f"\n{n - fails}/{n} passed"); sys.exit(1 if fails else 0)
