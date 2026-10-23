"""Phase H, Step H1: full-universe feasibility audit + pre-registered group definitions.

Runs BEFORE any full-universe feature computation, per the governing instruction ("First determine
whether the existing Phase G feature library can safely be computed across all 299 stocks... Do not
silently discard problematic symbols. Every exclusion must be explicitly documented with a reason").

Two outputs:
  1. `audit.json` - per-universe-symbol coverage (bar count, distinct trading dates, first/last date),
     an explicit list of EXCLUDED symbols with reasons, a list of SPARSE/newly-listed symbols that are
     NOT excluded (documented, included as-is - their early sampled dates will simply yield zero rows),
     and the full bars-per-session-day distribution (session-shape diversity check).
  2. `group_definitions.json` - PRE-REGISTERED, deterministic group thresholds (price/liquidity/
     volatility median splits, computed from the full universe's own distribution) - written and
     frozen BEFORE any relative_strength_vs_universe result is looked at, per H2's explicit
     "pre-register the grouping logic before looking at results" instruction. Tier and sector groups
     need no threshold - they come directly from the existing universe.json.

REUSE, NOT A NEW FRAMEWORK: this only runs read-only SQL against the already-persisted
`intraday_equity_bars` table and reads the existing `data/derived/phase_d_research_dataset/
universe.json` - no new feature/label code, no Dhan calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.app.config import load_dotenv
from backend.app.db.session import get_session

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "data" / "derived" / "phase_d_research_dataset" / "universe.json"
OUT_DIR = ROOT / "data" / "derived" / "phase_h_full_universe_validation"

MIN_TRADING_DATES_FOR_FULL_YEAR = 200

PER_SYMBOL_COVERAGE_SQL = text(
    """
    SELECT security_id, trading_symbol, COUNT(*) AS n_bars, COUNT(DISTINCT trading_date) AS n_dates,
           MIN(trading_date) AS first_date, MAX(trading_date) AS last_date
    FROM intraday_equity_bars
    WHERE security_id = ANY(:security_ids)
    GROUP BY security_id, trading_symbol
    """
)

BARS_PER_DAY_DISTRIBUTION_SQL = text(
    """
    SELECT bars_per_day, COUNT(*) AS n_symbol_days
    FROM (
        SELECT security_id, trading_date, COUNT(*) AS bars_per_day
        FROM intraday_equity_bars
        WHERE security_id = ANY(:security_ids)
        GROUP BY security_id, trading_date
    ) t
    GROUP BY bars_per_day
    ORDER BY n_symbol_days DESC
    """
)

PER_SYMBOL_LIQUIDITY_VOL_PRICE_SQL = text(
    """
    WITH daily AS (
        SELECT security_id, trading_date, SUM(volume * close) AS day_turnover,
               (MAX(high) - MIN(low)) / NULLIF(AVG(close), 0) AS day_range_pct, AVG(close) AS day_avg_close
        FROM intraday_equity_bars
        WHERE security_id = ANY(:security_ids)
        GROUP BY security_id, trading_date
    )
    SELECT security_id, AVG(day_turnover) AS avg_daily_turnover, AVG(day_range_pct) AS avg_daily_range_pct,
           AVG(day_avg_close) AS avg_close, COUNT(*) AS n_days
    FROM daily GROUP BY security_id
    """
)


def load_universe() -> list[dict[str, Any]]:
    return json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))["universe"]


def _quantile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def classify_symbols(universe: list[dict[str, Any]], coverage_rows: dict[int, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    excluded, sparse, healthy = [], [], []
    for entry in universe:
        sid = entry["security_id"]
        cov = coverage_rows.get(sid)
        if cov is None:
            excluded.append({**entry, "reason": "zero persisted bars (not a real tradable equity in this dataset - e.g. an NSE test security)"})
        elif cov["n_dates"] < MIN_TRADING_DATES_FOR_FULL_YEAR:
            sparse.append({**entry, "n_dates": cov["n_dates"], "first_date": str(cov["first_date"]), "last_date": str(cov["last_date"]),
                           "reason": f"only {cov['n_dates']} trading dates (< {MIN_TRADING_DATES_FOR_FULL_YEAR}) - a recently listed symbol; NOT excluded, included as-is"})
        else:
            healthy.append({**entry, "n_dates": cov["n_dates"]})
    return {"excluded": excluded, "sparse": sparse, "healthy": healthy}


def run_coverage_audit(session, universe: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [e["security_id"] for e in universe]
    coverage_rows = {r["security_id"]: dict(r) for r in session.execute(PER_SYMBOL_COVERAGE_SQL, {"security_ids": ids}).mappings().all()}
    shape_rows = session.execute(BARS_PER_DAY_DISTRIBUTION_SQL, {"security_ids": ids}).fetchall()
    classified = classify_symbols(universe, coverage_rows)

    return {
        "universe_size": len(universe),
        "excluded_symbols": classified["excluded"],
        "sparse_but_included_symbols": classified["sparse"],
        "healthy_symbols_count": len(classified["healthy"]),
        "bars_per_session_day_distribution": [{"bars_per_day": r[0], "n_symbol_days": r[1]} for r in shape_rows],
        "session_shape_note": (
            "Session-shape diversity beyond the known 360-bar/375-bar dichotomy (S1.6/Phase C) exists at full-"
            "universe scale (e.g. a single 60-bar day on 2025-10-21 across 272 symbols - consistent with a genuine "
            "reduced-hours Muhurat trading session, not a data defect; a small cluster of 384-bar days around "
            "2026-08-03..08-05, the same date boundary Phase C already found unexplained at the security level). "
            "None of the intraday_features functions assume a fixed bar count per session - they derive session "
            "membership from `trading_date` and operate on however many bars a session actually has - so this "
            "diversity does not block full-universe computation. Not re-investigated further here, per the same "
            "'do not force a classification the evidence doesn't support' judgment Phase C already made."
        ),
    }


def compute_group_definitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turnovers = sorted(r["avg_daily_turnover"] for r in rows if r["avg_daily_turnover"] is not None)
    prices = sorted(r["avg_close"] for r in rows if r["avg_close"] is not None)
    vols = sorted(r["avg_daily_range_pct"] for r in rows if r["avg_daily_range_pct"] is not None and r["n_days"] >= 30)

    turnover_median = _quantile(turnovers, 0.5)
    price_median = _quantile(prices, 0.5)
    range_pct_median = _quantile(vols, 0.5)

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "Median split on each symbol's full available history (avg daily turnover / avg daily range_pct / avg close), computed BEFORE any relative_strength_vs_universe result was examined. Frozen at this value for the rest of Phase H.",
        "n_symbols": len(rows),
        "price_group": {"median_avg_close": price_median, "low_priced": "avg_close < median", "higher_priced": "avg_close >= median"},
        "liquidity_group": {"median_avg_daily_turnover": turnover_median, "higher_liquidity": "avg_daily_turnover >= median", "lower_liquidity": "avg_daily_turnover < median"},
        "volatility_group": {"median_avg_daily_range_pct": range_pct_median, "high_volatility": "avg_daily_range_pct >= median", "low_volatility": "avg_daily_range_pct < median",
                              "note": f"restricted to symbols with >=30 trading days ({len(vols)} of {len(rows)}) for a stable estimate"},
        "per_symbol_stats": {int(r["security_id"]): {"avg_daily_turnover": r["avg_daily_turnover"], "avg_daily_range_pct": r["avg_daily_range_pct"], "avg_close": r["avg_close"], "n_days": r["n_days"]} for r in rows},
    }


def run_group_definitions(session, universe: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [e["security_id"] for e in universe]
    rows = [dict(r) for r in session.execute(PER_SYMBOL_LIQUIDITY_VOL_PRICE_SQL, {"security_ids": ids}).mappings().all()]
    return compute_group_definitions(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    load_dotenv()
    universe = load_universe()
    session = get_session()
    try:
        audit = run_coverage_audit(session, universe)
        group_defs = run_group_definitions(session, universe)
    finally:
        session.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (args.out_dir / "group_definitions.json").write_text(json.dumps(group_defs, indent=2, default=str), encoding="utf-8")

    print(f"universe_size={audit['universe_size']} excluded={len(audit['excluded_symbols'])} sparse_but_included={len(audit['sparse_but_included_symbols'])} healthy={audit['healthy_symbols_count']}")
    print(f"price median={group_defs['price_group']['median_avg_close']:.2f} turnover median={group_defs['liquidity_group']['median_avg_daily_turnover']:.0f} range_pct median={group_defs['volatility_group']['median_avg_daily_range_pct']:.4f}")
    print(f"wrote {(args.out_dir / 'audit.json').relative_to(ROOT)}")
    print(f"wrote {(args.out_dir / 'group_definitions.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
