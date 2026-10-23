from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
AF = "backend/app/research/adjustment_factors.py"
AP = "backend/app/data/instruments/adjusted_prices.py"
AE = "backend/app/research/adjustment_eligibility.py"
TESTS = ["backend/tests/test_adjustment_factors.py", "backend/tests/test_adjusted_prices_db.py", "backend/tests/test_adjusted_prices_real_data.py"]
PROOF_BONUS = ("backend/tests/test_adjustment_factors.py::TestBonusFormulaEquivalence",)
PROOF_FIRST = ("backend/tests/test_adjustment_factors.py::TestFirstSessionBoundEquivalence",)
BACKUP = pathlib.Path(tempfile.gettempdir()) / "c3_mutation_check_backup"

KILLED, SURVIVED, EQUIVALENT_PROVEN, NOOP_IDENTICAL, ERROR, FLAGGED = (
    "KILLED", "SURVIVED", "EQUIVALENT_PROVEN", "NOOP_IDENTICAL", "ERROR", "FLAGGED")


@dataclasses.dataclass(frozen=True)
class Mutant:
    file: str
    old: str
    new: str
    label: str
    proof_tests: tuple = ()
    control: str | None = None


MUTANTS = [
    Mutant(AF, "        return b / a  # new face value / old face value", "        return a / b", "CONTROL kill: split factor inverted", control="kill"),
    Mutant(AF, 'ADJUSTMENT_BASIS_VERSION = "c3-adjustment-basis/1"', 'ADJUSTMENT_BASIS_VERSION = "c3-adjustment-basis/1"',
           "CONTROL no-op: replacement identical to the original", control="noop"),
    Mutant(AF, "        if in_interval(ev.ex_date, start, as_of):", "        if start <= ev.ex_date <= as_of:", "PIT: start boundary made inclusive"),
    Mutant(AF, "        if in_interval(ev.ex_date, start, as_of):", "        if start < ev.ex_date < as_of:", "PIT: as_of boundary made exclusive"),
    Mutant(AF, "        if in_interval(ev.ex_date, start, as_of):", "        if ev.ex_date <= as_of:", "PIT: start bound dropped"),
    Mutant(AF, "        if in_interval(ev.ex_date, start, as_of):", "        if start < ev.ex_date:", "PIT: as_of bound dropped (future events leak)"),
    Mutant(AE, "    return start < ex_date <= end", "    return start <= ex_date <= end", "shared in_interval: start inclusive"),
    Mutant(AE, "    return start < ex_date <= end", "    return start < ex_date < end", "shared in_interval: end exclusive"),
    Mutant(AF, "    return b / (a + b)  # existing / (existing + bonus)", "    return a / (a + b)", "bonus factor uses bonus shares in the numerator"),
    Mutant(AF, "    return b / (a + b)  # existing / (existing + bonus)", "    return b / a", "bonus factor ignores the added shares"),
    Mutant(AF, "    return b / (a + b)  # existing / (existing + bonus)", "    return 1 / (1 + Fraction(a, b))",
           "bonus factor in the reciprocal form 1/(1+a/b)", proof_tests=PROOF_BONUS),
    Mutant(AF, "            f *= c.factor()", "            f += c.factor() - 1", "event compound: sum instead of product"),
    Mutant(AF, "            f *= c.factor()", "            f = c.factor()", "event compound: only the last component"),
    Mutant(AF, "            f *= ev.factor()", "            f += ev.factor() - 1", "cumulative: sum instead of product"),
    Mutant(AF, "            f *= ev.factor()", "            f = ev.factor()", "cumulative: only the last event"),
    Mutant(AF, "    return Fraction(d)\n\n\n@dataclasses.dataclass(frozen=True)\nclass RawBar", "    return Fraction(float(d))\n\n\n@dataclasses.dataclass(frozen=True)\nclass RawBar",
           "price read through a float (binary expansion)"),
    Mutant(AF, "        d = Decimal(repr(x))", "        d = Decimal(x)", "float price read as its binary expansion"),
    Mutant(AF, "    o, h, l, c = (price_exact(v) * factor for v in (bar.open, bar.high, bar.low, bar.close))",
           "    o, h, l, c = (price_exact(v) * factor for v in (bar.open, bar.low, bar.high, bar.close))", "OHLC: high and low swapped"),
    Mutant(AF, "    o, h, l, c = (price_exact(v) * factor for v in (bar.open, bar.high, bar.low, bar.close))",
           "    o, h, l, c = (price_exact(v) / factor for v in (bar.open, bar.high, bar.low, bar.close))", "OHLC: divided by the factor"),
    Mutant(AF, "float(o), float(h), float(l), float(c))", "float(o), float(h), float(l), float(bar.close))", "OHLC: float close left unadjusted"),
    Mutant(AF, "        if ev.event_id in seen:", "        if False:", "duplicate event id no longer refused"),
    Mutant(AF, "        if b.trade_date > as_of:", "        if False:", "a bar after as_of no longer refused (future leak)"),
    Mutant(AF, "    if len({b.trade_date for b in ordered}) != len(ordered):", "    if False:", "duplicate bar dates no longer refused"),
    Mutant(AF, "    if len({(b.series, b.timeframe) for b in ordered}) > 1:", "    if False:", "mixed series no longer refused"),
    Mutant(AF, "    if factor <= 0:", "    if False:", "non-positive factor no longer refused"),
    Mutant(AF, "    if min(o, h, l, c) <= 0:", "    if min(o, h, l, c) < 0:", "a zero price accepted"),
    Mutant(AF, "    if not (l <= o <= h and l <= c <= h and l <= h):", "    if False:", "mis-ordered raw bar accepted"),
    Mutant(AF, '    if start > as_of:\n        raise AdjustmentError(f"start {start} is after as_of {as_of}")', '    if False:\n        raise AdjustmentError("")',
           "cumulative_factor accepts start after as_of"),
    Mutant(AF, "    if not eligibility.eligible:", "    if False:", "GATE BYPASS: a blocked eligibility no longer refuses"),
    Mutant(AF, "    if eligibility.start != requested_start or eligibility.end != as_of:", "    if False:", "eligibility for another interval accepted"),
    Mutant(AF, "    if requested_start > as_of:", "    if False:", "start after as_of accepted by the builder"),
    Mutant(AF, "        if not (requested_start <= b.trade_date <= as_of) or (b.series, b.timeframe) != (series, timeframe):", "        if False:",
           "raw bars outside the request accepted"),
    Mutant(AF, "[e for e in gate_events if in_interval(e.ex_date, requested_start, as_of)]", "list(gate_events)", "out-of-interval events converted to factors"),
    Mutant(AF, "            if rec is None:", "            if False:", "missing C2 record no longer detected at factor derivation"),
    Mutant(AF, " or not terms_equal(have, attested):", ":", "component/record terms mismatch no longer detected"),
    Mutant(AF, "    if not raw_bars:\n        reasons.append(NO_BARS)", "    if False:\n        reasons.append(NO_BARS)", "no bars no longer refused"),
    Mutant(AF, "    if invalid:\n        reasons.append(INVALID_RAW_BARS)", "    if False:\n        reasons.append(INVALID_RAW_BARS)", "invalid raw bars no longer refused"),
    Mutant(AF, "    if missing and not allow_missing_bars:", "    if False:", "missing bars no longer refused"),
    Mutant(AF, "    if missing and not allow_missing_bars:", "    if missing and allow_missing_bars:", "missing-bar policy inverted"),
    Mutant(AF, '    status = ADJUSTED if quality["complete"] else ADJUSTED_WITH_GAPS', "    status = ADJUSTED", "a gappy view is reported as complete"),
    Mutant(AF, "first <= s <= as_of and s not in have", "first < s <= as_of and s not in have",
           "gap detection: lower bound made exclusive (first session)", proof_tests=PROOF_FIRST),
    Mutant(AF, "first <= s <= as_of and s not in have", "first <= s < as_of and s not in have", "gap detection: as_of session skipped"),
    Mutant(AF, "len({s for s in session_dates if min(bar_dates) <= s <= as_of})", "len({s for s in session_dates if min(bar_dates) < s <= as_of})",
           "expected-session count excludes the first session"),
    Mutant(AF, "and all(b.ex_date is not None for b in eligibility.blockers):", ":", "safe-start hint offered despite an undated blocker"),
    Mutant(AF, '"bars": [b.as_dict() for b in self.bars],', '"bars": [],', "view hash ignores the bars"),
    Mutant(AF, "for e in used],", "for e in used][:0],", "basis omits the events used"),
    Mutant(AF, '"cumulative_factor_at_requested_start": fraction_dict(bars_first_factor),', '"cumulative_factor_at_requested_start": fraction_dict(Fraction(1)),',
           "basis states a wrong cumulative factor"),
    Mutant(AF, '"as_of": as_of.isoformat(),\n        "interval_semantics"', '"as_of": "x",\n        "interval_semantics"', "basis omits the as-of date"),
    Mutant(AP, "M.trade_date >= start, M.trade_date <= as_of)", "M.trade_date > start, M.trade_date <= as_of)", "DB: bars from start exclusive"),
    Mutant(AP, "M.trade_date >= start, M.trade_date <= as_of)", "M.trade_date >= start, M.trade_date < as_of)", "DB: bars up to as_of exclusive"),
    Mutant(AP, "M.trade_date >= start, M.trade_date <= as_of)", "M.trade_date >= start)", "DB: no upper bound (future bars leak)"),
    Mutant(AP, "M.instrument_id == instrument_id, M.series == series, M.timeframe == timeframe,", "M.series == series, M.timeframe == timeframe,", "DB: all instruments' bars"),
    Mutant(AP, "M.instrument_id == instrument_id, M.series == series, M.timeframe == timeframe,", "M.instrument_id == instrument_id, M.timeframe == timeframe,", "DB: series filter dropped"),
    Mutant(AP, "M.instrument_id == instrument_id, M.series == series, M.timeframe == timeframe,", "M.instrument_id == instrument_id, M.series == series,", "DB: timeframe filter dropped"),
    Mutant(AP, "M.trade_date >= raw[0].trade_date, M.trade_date <= as_of)", "M.trade_date >= raw[0].trade_date, M.trade_date < as_of)", "DB: session calendar excludes as_of"),
    Mutant(AP, "load_adjustment_gate_inputs(session, instrument_id, start, as_of)", "load_adjustment_gate_inputs(session, instrument_id, start, start)", "DB: gate loaded for an empty interval"),
    Mutant(AP, "allow_missing_bars=allow_missing_bars)", "allow_missing_bars=True)", "DB: gaps always allowed"),
    Mutant(AP, "start=start, end=as_of, coverage_status=gate.coverage_status", 'start=start, end=as_of, coverage_status="COMPLETE"', "DB: coverage forced COMPLETE"),
]


def python_exe() -> str:
    override = os.environ.get("C3_MUTATION_PY")
    if override:
        return override
    venv = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv) if venv.exists() else sys.executable


def run(cmd: list, env_extra: dict | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


def pytest_cmd(targets) -> list:
    return [python_exe(), "-m", "pytest", *targets, "-x", "-q", "--tb=no", "-p", "no:cacheprovider"]


def classify(returncode: int, stdout: str) -> str:
    lines = (stdout or "").strip().splitlines()
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l), "")
    if returncode == 0:
        return SURVIVED
    if returncode == 1 and " failed" in summary and " error" not in summary:
        return KILLED
    return ERROR


def module_name(rel: str) -> str:
    return rel[:-3].replace("/", ".")


def sha(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def execution_proof(rel: str, mutated_text: str, env: dict) -> tuple[bool, str]:
    code = ("import importlib, inspect, hashlib, pathlib, sys\n"
            f"m = importlib.import_module({module_name(rel)!r})\n"
            "src = inspect.getsource(m).replace('\\r\\n', '\\n')\n"
            "print(hashlib.sha256(src.encode('utf-8')).hexdigest()); print(pathlib.Path(m.__file__).resolve())")
    r = run([python_exe(), "-c", code], env)
    out = (r.stdout or "").strip().splitlines()
    if r.returncode != 0 or len(out) < 2:
        return False, f"canary failed: {(r.stderr or '')[-200:]}"
    ok = out[0] == sha(mutated_text) and pathlib.Path(out[1]) == (ROOT / rel).resolve()
    return ok, f"loaded {out[1]} sha {out[0][:12]} vs mutated {sha(mutated_text)[:12]}"


def restore_pending_backup() -> None:
    manifest = BACKUP / "manifest.json"
    if manifest.exists():
        for rel, name in json.loads(manifest.read_text()).items():
            (ROOT / rel).write_bytes((BACKUP / name).read_bytes())
        shutil.rmtree(BACKUP, ignore_errors=True)
        print("WARNING: a previous run did not finish; its production files were RESTORED from the backup first.", flush=True)


def make_backup(files: dict) -> None:
    shutil.rmtree(BACKUP, ignore_errors=True)
    BACKUP.mkdir(parents=True)
    manifest = {}
    for i, (rel, data) in enumerate(files.items()):
        (BACKUP / f"f{i}.bak").write_bytes(data)
        manifest[rel] = f"f{i}.bak"
    (BACKUP / "manifest.json").write_text(json.dumps(manifest))


def evaluate(m: Mutant, originals: dict, *, verbose: bool = True) -> tuple[str, str]:
    path = ROOT / m.file
    src = originals[m.file].decode("utf-8")
    if src.count(m.old) != 1:
        return ERROR, f"pattern occurs {src.count(m.old)}x (need exactly 1)"
    mutated = src.replace(m.old, m.new)
    if mutated == src:
        return NOOP_IDENTICAL, "the mutated file is byte-identical to the original; nothing was run"
    cache = tempfile.mkdtemp(prefix="c3mut_pyc_")
    env = {"PYTHONPYCACHEPREFIX": cache, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        path.write_bytes(mutated.encode("utf-8"))
        ok, why = execution_proof(m.file, mutated, env)
        if not ok:
            return ERROR, f"the mutated code is not what was imported ({why})"
        r = run(pytest_cmd(TESTS), env)
        status = classify(r.returncode, r.stdout)
        tail = (r.stdout or "").strip().splitlines()
        first = next((l for l in tail if l.startswith("FAILED")), tail[-1] if tail else (r.stderr or "")[-120:])
        detail = first[:140]
    finally:
        path.write_bytes(originals[m.file])
        shutil.rmtree(cache, ignore_errors=True)
    if m.proof_tests:
        if status == KILLED:
            return FLAGGED, f"declared equivalent but the tests killed it: {detail}"
        if status == SURVIVED:
            proof = run(pytest_cmd(list(m.proof_tests)))
            return (EQUIVALENT_PROVEN, f"survived; proof tests pass on the original code: {', '.join(t.split('::')[-1] for t in m.proof_tests)}") \
                if proof.returncode == 0 else (SURVIVED, "declared equivalent, but its proof tests FAIL on the original code")
    return status, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[], help="run only mutants whose label contains one of these substrings (controls always run)")
    ap.add_argument("--json", help="write the results to this path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    restore_pending_backup()
    originals = {f: (ROOT / f).read_bytes() for f in {m.file for m in MUTANTS}}
    before = {f: hashlib.sha256(b).hexdigest() for f, b in originals.items()}
    t0 = time.time()
    base = run(pytest_cmd(TESTS))
    print("BASELINE (unmutated):", (base.stdout or "").strip().splitlines()[-1:] or base.stderr[-200:], "| rc", base.returncode, flush=True)
    if base.returncode != 0:
        print("baseline is not green - refusing to mutate")
        return 2
    results, controls_ok = [], True
    make_backup(originals)
    try:
        selected = [m for m in MUTANTS if m.control or not args.only or any(o.lower() in m.label.lower() for o in args.only)]
        for i, m in enumerate(selected, 1):
            status, detail = evaluate(m, originals)
            expected = {"kill": KILLED, "noop": NOOP_IDENTICAL}.get(m.control)
            if expected and status != expected:
                controls_ok = False
                print(f"CONTROL FAILED: {m.label} -> {status} (expected {expected}): {detail}", flush=True)
            results.append({"label": m.label, "status": status, "detail": detail, "file": m.file, "control": m.control})
            print(f"[{i}/{len(selected)}] {status:18} {m.label}   {detail[:120]}", flush=True)
            if not controls_ok:
                break
    finally:
        for f, b in originals.items():
            (ROOT / f).write_bytes(b)
    restored = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in originals} == before
    shutil.rmtree(BACKUP, ignore_errors=True)
    final = run(pytest_cmd(TESTS))
    print("\nfiles restored byte-for-byte:", restored)
    print("FINAL BASELINE (restored code):", (final.stdout or "").strip().splitlines()[-1:], "| rc", final.returncode)

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    real = [r for r in results if not r["control"]]
    print(f"\nMUTANTS: {len(real)} (+{len(results) - len(real)} controls) | " + " | ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" | {time.time() - t0:.0f}s")
    bad = [r for r in results if r["status"] in (SURVIVED, ERROR, FLAGGED)]
    for r in bad:
        print(f"  PROBLEM: {r['status']} {r['label']}: {r['detail']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"results": results, "counts": counts, "restored": restored}, indent=1))
    return 0 if (not bad and controls_ok and restored and final.returncode == 0) else 1


def self_test() -> int:
    assert classify(0, "191 passed in 12s") == SURVIVED
    assert classify(1, "FAILED x\n1 failed, 3 passed in 1s") == KILLED
    assert classify(1, "no summary at all") == ERROR
    assert classify(2, "ERROR collecting\n1 error in 0.1s") == ERROR
    assert classify(1, "1 failed, 1 error in 1s") == ERROR
    assert classify(127, "") == ERROR and classify(124, "") == ERROR
    os.environ["C3_MUTATION_PY"] = str(ROOT / "definitely-not-a-python")
    broken = run(pytest_cmd(TESTS))
    assert broken.returncode == 127 and classify(broken.returncode, broken.stdout) == ERROR, "a runner that cannot start must be ERROR"
    m = Mutant(AF, "x", "x", "noop")
    src = "abc x def"
    assert src.replace(m.old, m.new) == src
    print("self-test passed: broken runner -> ERROR, non-test failures -> ERROR, identical text -> no-op")
    return 0


if __name__ == "__main__":
    sys.exit(main())
