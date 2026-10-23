"""Phase J, Step 1: freeze the "mover" target definition BEFORE any new feature is tested against it.

Three candidate target families, each a genuinely different normalization philosophy (the phase's
explicit concern: "do not assume a fixed percentage threshold works equally for penny stocks and
highly liquid large caps"):

  T1 PARAMETRIC (volatility-normalized): |label_normalized_move_Xm| >= k, using the ALREADY-COMPUTED,
     ALREADY-PIT-SAFE label (move divided by the stock's own trailing realized_volatility_20).
  T2 RAW (fixed percentage, no normalization at all): mover_future_abs_return_Xm >= k - deliberately
     the naive approach, included as a contrast, not a recommendation.
  T3 EMPIRICAL (per-stock percentile rank): is this row's mover_future_abs_return_Xm in the top
     decile/5% of THAT SAME STOCK's own historical move-size distribution - a nonparametric
     normalization that needs no assumption about the shape of a stock's return distribution.

METHOD: compute the base (positive-class) rate for every (definition, threshold, horizon) combination,
overall and broken down by the FROZEN Phase H price/liquidity/volatility groups (never recomputed
here) and by universe tier. Then apply one pre-registered, mechanical SELECTION RULE (not "whichever
looks best"): among candidates whose overall base rate falls in a defensible working range, pick the
one whose base rate is most CONSISTENT (least dispersed) across the price/liquidity/volatility
groups, averaged across all 4 horizons - i.e. a target that means roughly the same thing for a
penny stock as for a large, liquid one. The selection rule and its acceptable-range bound are fixed
before this script is ever run against real data - see SELECTION_RATE_RANGE and _dispersion_score.

Reads the FROZEN Phase H dataset read-only. Zero new Dhan calls, zero new DB reads.

PHASE K CORRECTION (2026-09-28): `compute_t3` (below) was found NOT to be point-in-time safe - its
`df.groupby("symbol")[col].rank(pct=True)` ranks every row against its symbol's ENTIRE column,
including rows dated after it. Proven adversarially in `test_phase_j_mover_target_design.py`'s
`TestT3IsNotPointInTimeSafe`. `compute_t3` is KEPT, UNCHANGED, and NEVER used for any new
computation from this point on - solely so that test class continues to document the original bug.
All real target computation (the frozen selection, and Phase J's statistical re-run) now uses
`compute_t3_expanding_percentile` instead - see its own docstring for the corrected, genuinely
past-only construction and `MIN_PRIOR_OBSERVATIONS_FOR_T3` for the minimum-history rule.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_phase_h_statistical_research import HORIZONS, attach_pre_registered_groups, load_group_definitions

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_j_mover_detection"

T1_THRESHOLDS = (1.0, 1.5, 2.0)
T2_THRESHOLDS = (0.003, 0.005, 0.01, 0.02)
T3_PERCENTILES = (0.90, 0.95)

GROUP_DIMENSIONS = ("tier", "price_group", "liquidity_group", "volatility_group")
SELECTION_RATE_RANGE = (0.02, 0.30)


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(PHASE_H_DATASET)
    group_defs = load_group_definitions()
    df = attach_pre_registered_groups(df, group_defs)
    return df


def compute_t1(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    col = f"label_normalized_move_{horizon}m"
    return df[col].abs() >= threshold


def compute_t2(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    col = f"mover_future_abs_return_{horizon}m"
    return df[col] >= threshold


def compute_t3(df: pd.DataFrame, horizon: int, percentile: float) -> pd.Series:
    col = f"mover_future_abs_return_{horizon}m"
    rank = df.groupby("symbol")[col].rank(pct=True, na_option="keep")
    return rank >= percentile


MIN_PRIOR_OBSERVATIONS_FOR_T3 = 50


def _expanding_rank_fraction(values: np.ndarray, min_prior_observations: int) -> np.ndarray:
    n = len(values)
    result = np.full(n, np.nan)
    valid_mask = ~np.isnan(values)
    valid_positions = np.where(valid_mask)[0]
    if len(valid_positions) == 0:
        return result

    valid_values = values[valid_positions]
    sorted_unique = np.unique(valid_values)
    compressed = np.searchsorted(sorted_unique, valid_values) + 1
    size = len(sorted_unique)
    tree = [0] * (size + 1)

    def _update(idx: int) -> None:
        while idx <= size:
            tree[idx] += 1
            idx += idx & (-idx)

    def _query_prefix_count(idx: int) -> int:
        total = 0
        while idx > 0:
            total += tree[idx]
            idx -= idx & (-idx)
        return total

    count_prior = 0
    for pos in range(len(valid_positions)):
        orig_idx = valid_positions[pos]
        c = int(compressed[pos])
        if count_prior >= min_prior_observations:
            strictly_less = _query_prefix_count(c - 1)
            result[orig_idx] = strictly_less / count_prior
        _update(c)
        count_prior += 1
    return result


def compute_t3_expanding_percentile(df: pd.DataFrame, horizon: int, percentile: float, *, min_prior_observations: int = MIN_PRIOR_OBSERVATIONS_FOR_T3) -> pd.Series:
    col = f"mover_future_abs_return_{horizon}m"
    ordered = df.sort_values(["symbol", "bar_start_utc"])
    result = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
    for _, group in ordered.groupby("symbol", sort=False):
        values = group[col].to_numpy(dtype=float)
        rank_fraction = _expanding_rank_fraction(values, min_prior_observations)
        flags = pd.array([pd.NA if np.isnan(r) else bool(r >= percentile) for r in rank_fraction], dtype="boolean")
        result.loc[group.index] = flags
    return result


def _base_rates_by_group(flag: pd.Series, df: pd.DataFrame) -> dict[str, Any]:
    valid = flag.notna()
    overall = float(flag[valid].mean()) if valid.any() else None
    by_group: dict[str, dict[str, float]] = {}
    for dim in GROUP_DIMENSIONS:
        by_group[dim] = {}
        for val, idx in df.loc[valid].groupby(dim).groups.items():
            sub_flag = flag.loc[idx]
            by_group[dim][str(val)] = float(sub_flag.mean())
    return {"overall": overall, "n_valid": int(valid.sum()), "by_group": by_group}


def _dispersion_score(rates: dict[str, Any]) -> float | None:
    overall = rates["overall"]
    if not overall:
        return None
    all_group_rates = [v for dim in rates["by_group"].values() for v in dim.values()]
    if not all_group_rates:
        return None
    return (max(all_group_rates) - min(all_group_rates)) / overall


def evaluate_all_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = []
    for h in HORIZONS:
        for k in T1_THRESHOLDS:
            flag = compute_t1(df, h, k)
            rates = _base_rates_by_group(flag, df)
            candidates.append({"definition": "T1_parametric_vol_normalized", "horizon_minutes": h, "threshold": k, **rates, "dispersion_score": _dispersion_score(rates)})
        for k in T2_THRESHOLDS:
            flag = compute_t2(df, h, k)
            rates = _base_rates_by_group(flag, df)
            candidates.append({"definition": "T2_raw_fixed_percentage", "horizon_minutes": h, "threshold": k, **rates, "dispersion_score": _dispersion_score(rates)})
        for p in T3_PERCENTILES:
            flag = compute_t3(df, h, p)
            rates = _base_rates_by_group(flag, df)
            candidates.append({"definition": "T3_empirical_per_stock_percentile", "horizon_minutes": h, "threshold": p, **rates, "dispersion_score": _dispersion_score(rates)})
    return candidates


def select_frozen_target(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for c in candidates:
        by_pair.setdefault((c["definition"], c["threshold"]), []).append(c)

    eligible = []
    for (definition, threshold), rows in by_pair.items():
        if len(rows) != len(HORIZONS):
            continue
        in_range = all(r["overall"] is not None and SELECTION_RATE_RANGE[0] <= r["overall"] <= SELECTION_RATE_RANGE[1] for r in rows)
        if not in_range:
            continue
        dispersions = [r["dispersion_score"] for r in rows if r["dispersion_score"] is not None]
        if len(dispersions) != len(HORIZONS):
            continue
        eligible.append({"definition": definition, "threshold": threshold, "avg_dispersion_score": sum(dispersions) / len(dispersions),
                          "base_rates_by_horizon": {f"{r['horizon_minutes']}m": r["overall"] for r in rows}})

    if not eligible:
        return {"status": "NO_ELIGIBLE_CANDIDATE", "reason": f"no (definition, threshold) pair had every horizon's base rate inside {SELECTION_RATE_RANGE}"}

    eligible.sort(key=lambda e: (e["avg_dispersion_score"], e["threshold"]))
    winner = eligible[0]
    return {"status": "FROZEN", **winner, "all_eligible_candidates_ranked": eligible, "selection_rate_range": SELECTION_RATE_RANGE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = load_dataset()
    print(f"loaded {len(df)} rows from the frozen Phase H dataset")

    candidates = evaluate_all_candidates(df)
    print(f"evaluated {len(candidates)} (definition, threshold, horizon) combinations")

    frozen = select_frozen_target(candidates)
    print(f"selection result: {frozen['status']}")
    if frozen["status"] == "FROZEN":
        print(f"  frozen target: {frozen['definition']} @ threshold={frozen['threshold']}")
        print(f"  base rates by horizon: {frozen['base_rates_by_horizon']}")
        print(f"  avg dispersion score: {frozen['avg_dispersion_score']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row_count": len(df), "selection_rate_range": SELECTION_RATE_RANGE,
        "all_candidates": candidates, "frozen_target": frozen,
    }
    (args.out_dir / "mover_target_design_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.out_dir / "mover_target_definition.json").write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
    print(f"wrote {(args.out_dir / 'mover_target_design_report.json').relative_to(ROOT)}")
    print(f"wrote {(args.out_dir / 'mover_target_definition.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
