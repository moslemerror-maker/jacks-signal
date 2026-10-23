from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from decimal import Decimal

from scripts import run_o1_0b_sample_gate as g

D = dt.date
TESTS = [
    ("2015", "legacy", D(2015, 1, 29), D(2015, 1, 29)),
    ("2018", "legacy", D(2018, 6, 28), D(2018, 6, 28)),
    ("2020", "legacy", D(2020, 6, 25), D(2020, 6, 25)),
    ("2022", "legacy", D(2022, 6, 30), D(2022, 6, 30)),
    ("2024", "udiff", D(2024, 3, 28), D(2024, 3, 28)),
    ("2025", "udiff", D(2025, 9, 30), D(2025, 9, 30)),
    ("2026", "udiff", D(2026, 9, 15), D(2026, 9, 15)),
]
EXTRA_FETCH = [("legacy", D(2018, 6, 28)), ("legacy", D(2020, 6, 25)), ("legacy", D(2022, 6, 30))]
LISTED_CONTRAST = [(D(2024, 2, 5), D(2024, 3, 28), "udiff"), (D(2025, 1, 15), D(2025, 3, 27), "udiff")]


def expiry_stats(bars, expiry: dt.date) -> dict:
    rows = [b for b in bars if b.key.underlying == "NIFTY" and b.key.dtype == "IDO" and b.key.expiry == expiry]
    if not rows:
        return {"rows": 0}
    strikes = sorted({b.key.strike_paise for b in rows})
    step = g.gcd_step(strikes) or 0
    grid = set(range(strikes[0], strikes[-1] + 1, step)) if step else set(strikes)
    keys = Counter(b.key for b in rows)
    states = Counter(g.classify_trade_state(b.volume, b.open, b.high, b.low, b.close)[1] for b in rows)
    return {
        "rows": len(rows), "CE": sum(b.key.cp == "CE" for b in rows), "PE": sum(b.key.cp == "PE" for b in rows),
        "other_option_types": dict(Counter(b.key.cp for b in rows if b.key.cp not in ("CE", "PE"))),
        "unique_strikes": len(strikes), "lowest_strike": strikes[0] / 100, "highest_strike": strikes[-1] / 100, "grid_step": step / 100,
        "strike_intervals": {str(k / 100): v for k, v in sorted(Counter(y - x for x, y in zip(strikes, strikes[1:])).items())},
        "gaps_vs_finest_grid(not_a_defect_if_far_strikes_are_listed_wider)": len(grid - set(strikes)), "duplicate_contracts": sum(v - 1 for v in keys.values() if v > 1),
        "with_ohlc(traded)": states["TRADED"], "with_oi>0": sum(b.oi > 0 for b in rows), "with_volume>0": sum(b.volume > 0 for b in rows),
        "with_settle>0": sum(b.settle > 0 for b in rows), "zero_volume_rows": sum(b.volume == 0 for b in rows),
        "zero_volume_with_oi": sum(b.volume == 0 and b.oi > 0 for b in rows), "trade_states": dict(states),
        "lots": dict(Counter(b.lot for b in rows if b.lot)) or "not_in_source",
        "moneyness_note": "strike range relative to spot must be read against the day's index close",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args(argv)
    if not args.no_fetch:
        from backend.app.data.sources import nse_bhavcopy as nb
        session = nb._get_session()
        for fam, d in EXTRA_FETCH:
            side = g.fetch_one(fam, d, session)
            print("  fetched" if side.get("available", True) else "  404    ", fam, d, flush=True)
            time.sleep(1.0)
    out: dict = {"generated_at": dt.datetime.now(g.UTC).isoformat(), "source": "NSE public archive (ACTIVE_OBSERVED only)", "tests": {}}
    for label, fam, d, expiry in TESTS:
        got = g.load(fam, d)
        if got is None:
            out["tests"][label] = {"status": "FILE_NOT_PRESENT", "file": f"{fam}:{d}"}
            continue
        bars = g.legacy_bars(*got) if fam == "legacy" else g.udiff_bars(*got)
        out["tests"][label] = {"file": f"{fam}:{d}", "expiry": str(expiry), **expiry_stats(bars, expiry)}
    contrast = {}
    for mdate, expiry, fam in LISTED_CONTRAST:
        m = g.load("master", mdate)
        u = g.load(fam, mdate)
        if m is None or u is None:
            continue
        header, rows = m
        idx = {h.strip(): i for i, h in enumerate(header) if h.strip()}
        listed = sum(1 for r in rows if r and len(r) >= 21 and r[idx["DelFlg"]].strip() != "Y" and r[idx["TckrSymb"]].strip() == "NIFTY" and r[idx["FinInstrmNm"]].strip() == "OPTIDX"
                     and g.decode_nse_seconds(r[idx["XpryDt"]]).date() == expiry)
        active = sum(1 for b in g.udiff_bars(*u) if b.key.underlying == "NIFTY" and b.key.dtype == "IDO" and b.key.expiry == expiry)
        contrast[f"{mdate} -> expiry {expiry}"] = {"LISTED_OBSERVED(master)": listed, "ACTIVE_OBSERVED(bhavcopy)": active, "active_share_pct": round(100 * active / listed, 1) if listed else None}
    out["listed_vs_active"] = contrast
    g.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    p = g.OUT_ROOT / "universe_completeness.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    for k, v in out["tests"].items():
        print(k, {a: v.get(a) for a in ("file", "rows", "CE", "PE", "unique_strikes", "lowest_strike", "highest_strike", "grid_step", "strike_intervals", "duplicate_contracts", "with_ohlc(traded)", "with_oi>0", "with_volume>0", "with_settle>0", "zero_volume_rows", "lots")})
    print("listed vs active:", contrast)
    print("report:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
