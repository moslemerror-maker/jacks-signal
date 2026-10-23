"""Phase G: deterministic representative-sample selection for feature/label research.

Selects a small, deterministic subset of the 299-symbol Phase D/F universe for feature and
label computation - explicitly NOT the full 299 symbols ("not a gigantic feature table
blindly" per the governing instruction). Zero new Dhan calls: this script only reads the
already-persisted `intraday_equity_bars` table and the existing
`data/derived/phase_d_research_dataset/universe.json`.

SELECTION METHOD (fully deterministic - no RNG, no seed, reproducible from DB state alone):

  1. For every symbol, compute from already-persisted bars:
       avg_daily_turnover  = mean over trading_date of SUM(volume * close)   (a liquidity proxy)
       avg_daily_range_pct = mean over trading_date of (day_high - day_low) / day_avg_close
                              (a simple realized-range volatility proxy)
       avg_close           = mean close across all bars                      (a price-level proxy)
  2. Within EACH of the 5 universe tiers (so every pick is cap-size-stratified by construction):
       - sort members by avg_daily_turnover ascending; pick the symbol at rank MIN, MEDIAN, MAX
         (index = round(q * (n-1)) for q in {0, 0.5, 1.0}) -> covers less-liquid / typical / highly-liquid.
       - sort members by avg_daily_range_pct ascending; pick the symbol at rank MIN and MAX
         (q in {0, 1.0}) -> covers low-volatility / high-volatility, independent of the liquidity picks.
  3. Across the WHOLE universe (not tier-scoped): add the single lowest avg_close symbol, to
     explicitly cover "low-priced" per the brief, if not already selected by step 2.
  4. Deduplicate by security_id; a symbol selected by more than one criterion keeps every
     reason it qualified under (recorded in the output, not silently collapsed to one).

This rank-based, quantile-position selection is a pure function of already-computed per-symbol
statistics - re-running this script against the same DB state always produces the same sample.
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
OUT_DIR = ROOT / "data" / "derived" / "phase_g_feature_foundation"

PER_SYMBOL_STATS_SQL = text(
    """
    WITH daily AS (
        SELECT
            security_id,
            trading_symbol,
            trading_date,
            SUM(volume) AS day_volume,
            SUM(volume * close) AS day_turnover,
            (MAX(high) - MIN(low)) / NULLIF(AVG(close), 0) AS day_range_pct,
            AVG(close) AS day_avg_close
        FROM intraday_equity_bars
        WHERE security_id = ANY(:security_ids)
        GROUP BY security_id, trading_symbol, trading_date
    )
    SELECT
        security_id,
        trading_symbol,
        AVG(day_volume) AS avg_daily_volume,
        AVG(day_turnover) AS avg_daily_turnover,
        AVG(day_range_pct) AS avg_daily_range_pct,
        AVG(day_avg_close) AS avg_close,
        COUNT(*) AS session_count
    FROM daily
    GROUP BY security_id, trading_symbol
    """
)


def load_universe() -> list[dict[str, Any]]:
    data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    return data["universe"]


def fetch_per_symbol_stats(session, security_ids: list[int]) -> dict[int, dict[str, Any]]:
    rows = session.execute(PER_SYMBOL_STATS_SQL, {"security_ids": security_ids}).mappings().all()
    return {int(r["security_id"]): dict(r) for r in rows}


def _rank_pick(members: list[dict[str, Any]], *, metric: str, quantiles: list[float]) -> list[tuple[dict[str, Any], float]]:
    ordered = sorted(members, key=lambda m: (m[metric], m["security_id"]))
    n = len(ordered)
    if n == 0:
        return []
    picks = []
    seen_indices: set[int] = set()
    for q in quantiles:
        idx = round(q * (n - 1))
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        picks.append((ordered[idx], q))
    return picks


def select_sample(universe: list[dict[str, Any]], stats_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    enriched = []
    for entry in universe:
        sid = entry["security_id"]
        s = stats_by_id.get(sid)
        if s is None:
            continue
        enriched.append({**entry, **{k: v for k, v in s.items() if k not in ("security_id", "trading_symbol")}})

    by_tier: dict[str, list[dict[str, Any]]] = {}
    for e in enriched:
        by_tier.setdefault(e["tier"], []).append(e)

    selected: dict[int, dict[str, Any]] = {}

    def add(member: dict[str, Any], reason: str) -> None:
        sid = member["security_id"]
        if sid not in selected:
            selected[sid] = {**member, "reasons": []}
        selected[sid]["reasons"].append(reason)

    for tier in sorted(by_tier):
        members = by_tier[tier]

        for member, q in _rank_pick(members, metric="avg_daily_turnover", quantiles=[0.0, 0.5, 1.0]):
            label = {0.0: "MIN", 0.5: "MEDIAN", 1.0: "MAX"}[q]
            add(member, f"tier={tier} liquidity_rank={label} (avg_daily_turnover)")

        for member, q in _rank_pick(members, metric="avg_daily_range_pct", quantiles=[0.0, 1.0]):
            label = {0.0: "MIN", 1.0: "MAX"}[q]
            add(member, f"tier={tier} volatility_rank={label} (avg_daily_range_pct)")

    lowest_price = min(enriched, key=lambda e: (e["avg_close"], e["security_id"]))
    add(lowest_price, "universe-wide lowest avg_close (low-priced coverage)")

    sample = sorted(selected.values(), key=lambda e: e["security_id"])
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "universe_file": str(UNIVERSE_FILE.relative_to(ROOT)),
        "universe_size": len(universe),
        "symbols_with_stats": len(enriched),
        "sample_size": len(sample),
        "selection_method": (
            "Deterministic rank-position selection: per universe tier, MIN/MEDIAN/MAX by avg_daily_turnover "
            "and MIN/MAX by avg_daily_range_pct, plus the single universe-wide lowest avg_close symbol. "
            "No randomness; a pure function of already-persisted intraday_equity_bars aggregates."
        ),
        "sample": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "sample_universe.json")
    args = parser.parse_args()

    load_dotenv()
    universe = load_universe()
    security_ids = [e["security_id"] for e in universe]

    session = get_session()
    try:
        stats_by_id = fetch_per_symbol_stats(session, security_ids)
    finally:
        session.close()

    result = select_sample(universe, stats_by_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"universe_size={result['universe_size']} symbols_with_stats={result['symbols_with_stats']} sample_size={result['sample_size']}")
    print(f"wrote {args.out.relative_to(ROOT)}")
    for m in result["sample"]:
        print(f"  {m['symbol']:<16} tier={m['tier']:<16} turnover={m['avg_daily_turnover']:.0f} range_pct={m['avg_daily_range_pct']:.4f} avg_close={m['avg_close']:.2f}  reasons={m['reasons']}")


if __name__ == "__main__":
    main()
