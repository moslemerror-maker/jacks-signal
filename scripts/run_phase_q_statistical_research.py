"""Phase Q: direction & movement discovery across 11 candidate information families, approved design.

FROZEN BEFORE ANY RESULT WAS EXAMINED:

TARGET: direction_label_{h}m (UP/DOWN/AMBIGUOUS), Phase K/L's exact frozen next-bar-open, 20bps-dead-zone
convention, reused verbatim, unchanged, for h in HORIZONS = (5, 15, 30, 60). MOVEMENT is a SEPARATE, already-
answered question (Phase K/J established M4 as a robust magnitude mechanism) - this phase does not re-litigate
it, only reuses pop_M4/pop_M3 as regime-breakdown slices alongside a new, broader "FULL" population (the
un-gated entire dataset) - the genuinely new question this phase asks, since Phase L/M only ever tested
direction WITHIN the narrow M3/M4-gated populations and never on the general population.

POPULATIONS = ("FULL", "M4", "M3"). FULL = every row (pop_FULL is added, all True). M4/M3 = Phase J's frozen
gates, reused unchanged.

CANDIDATES, ONE (OR A SMALL PRE-REGISTERED SET) PER FAMILY - reusing already-computed, already-frozen columns
from the Phase G/H/J/L dataset wherever a family is already represented there (per the approved design's
explicit instruction not to re-derive existing infrastructure), computing only the two genuinely NEW families:

  1. Price action        -> momentum_acceleration_10 (existing)
  2. Support/resistance   -> sr_proximity_asymmetry = distance_to_nearest_support - distance_to_nearest_resistance
                             (NEW, but a zero-risk derived combination of two already-PIT-safe existing columns)
  3. Candle structure     -> wick_asymmetry = candle_upper_wick_ratio - candle_lower_wick_ratio (same as above)
  4. Volume               -> relative_volume_20 (existing)
  5. VWAP                 -> distance_from_vwap (existing)
  6. RSI                  -> rsi_9, rsi_14 (NEW - the small pre-registered 2-window set the design calls for;
                             Cutler's RSI, not Wilder's - see momentum_oscillators.py's own docstring for why)
  7. MACD                 -> macd_histogram_12_26_9 (NEW - one small pre-registered parameterization, the
                             industry-standard 12/26/9, not a swept grid)
  8. Volatility           -> volatility_expansion_ratio (existing)
  9. Market/universe ctx  -> sector_relative_strength (existing; relative_strength_vs_universe is reused ONLY
                             as an established Family-B control, never re-tested standalone here - Phase G-I/K
                             already gave that question a firm, documented answer, and Phase M already
                             exhaustively tested breadth/dispersion/lagged-index and found nothing - retesting
                             either here would be pure duplication of already-settled questions, not new
                             research, so neither is included as a fresh Family-A candidate)
 10. Time/regime          -> intraday_regime (existing, categorical, 5 states)
 11. Combinations         -> SIX buckets from FIVE economically-motivated ideas (one idea decomposes into two
                             directionally-opposite buckets, see below) - a small, pre-registered set, not an
                             arbitrary sweep:
       momentum_x_volume        : momentum_acceleration_10 top quintile AND relative_volume_20 top quintile
       breakout_x_volume_x_rs_up: recent_breakout_up_opening_range AND relative_volume_20 top-30% AND
                                   relative_strength_vs_universe > 0 (a DIRECTIONALLY COHERENT bucket)
       breakout_x_volume_x_rs_dn: the mirror-image coherent-down bucket
       sr_x_rsi                 : |sr_proximity_asymmetry| top quintile AND RSI(14) in an overbought/oversold
                                   extreme (<=30 or >=70)
       vwap_x_volume             : |distance_from_vwap| top quintile AND relative_volume_20 top quintile
       momentum_x_breadth        : |momentum_acceleration_10| top quintile AND universe_breadth top quintile
    Why not a UP/DOWN-vs-baseline split for every combination: `_rate_test` (reused unchanged from Phase L)
    already tests UP-rate against baseline UP-rate on UP/DOWN-only rows, where UP-rate and DOWN-rate are exact
    complements - a bucket that genuinely shifts toward DOWN shows up as a significant NEGATIVE lift on the
    SAME test, with no separate DOWN-rate test needed. The one exception is the breakout+volume+RS idea, whose
    two directionally-OPPOSITE cases would cancel each other out inside one mixed bucket - split into two
    buckets, each independently tested via the same standard UP-rate test, that cancellation cannot happen.

VALIDATION METHODOLOGY - unchanged from Phase L/M, reused verbatim: `_finalize_rate_test` (centered Newey-West
on UP/DOWN-only rows), one global Bonferroni correction over a PROGRAMMATICALLY computed budget (verified by a
regression test against the actual row counts before any real result is trusted - the repeated Phase J/L/M
lesson), Family A (standalone quintile/state/combination-bucket tests) -> Family B (incremental value beyond
return_5m AND relative_strength_vs_universe, both controls, both median-split halves) -> survivors only ->
cross-stock (5 strongest symbols excluded) -> chronological (Phase H's frozen 60/20/20 split) -> economic
(Phase K's 20/40/60bps cost scenarios). A candidate is a genuine survivor only if it clears every one of these
gates - standalone significance alone proves nothing, exactly as every prior phase has insisted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.run_phase_l_statistical_research import HORIZONS, _finalize_rate_test, population_mask
from scripts.run_phase_l_statistical_research import load_datasets as load_phase_l_datasets

ROOT = Path(__file__).resolve().parents[1]
OSCILLATORS_FILE = ROOT / "data" / "derived" / "phase_q_direction_discovery" / "oscillators.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_q_direction_discovery"

POPULATIONS = ("FULL", "M4", "M3")

CONTINUOUS_CANDIDATES = [
    "momentum_acceleration_10", "sr_proximity_asymmetry", "wick_asymmetry", "relative_volume_20",
    "distance_from_vwap", "rsi_9", "rsi_14", "macd_histogram_12_26_9", "volatility_expansion_ratio",
    "sector_relative_strength",
]
CATEGORICAL_CANDIDATES = {
    "intraday_regime": ["TREND_UP", "TREND_DOWN", "HIGH_VOL_EXPANSION", "LOW_VOL_COMPRESSION", "RANGE"],
}


def load_datasets() -> pd.DataFrame:
    df = load_phase_l_datasets()
    df["pop_FULL"] = True

    osc = pd.read_parquet(OSCILLATORS_FILE)
    df = df.merge(osc, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_osc_dup"))

    df["sr_proximity_asymmetry"] = df["distance_to_nearest_support"] - df["distance_to_nearest_resistance"]
    df["wick_asymmetry"] = df["candle_upper_wick_ratio"] - df["candle_lower_wick_ratio"]
    return df


def _pct(s: pd.Series, q: float) -> float:
    return s.dropna().quantile(q)


def combo_momentum_x_volume(df: pd.DataFrame, pop_mask: pd.Series) -> pd.Series:
    m_thr = _pct(df.loc[pop_mask, "momentum_acceleration_10"], 0.80)
    v_thr = _pct(df.loc[pop_mask, "relative_volume_20"], 0.80)
    return pop_mask & (df["momentum_acceleration_10"] >= m_thr) & (df["relative_volume_20"] >= v_thr)


def combo_breakout_x_volume_x_rs_up(df: pd.DataFrame, pop_mask: pd.Series) -> pd.Series:
    v_thr = _pct(df.loc[pop_mask, "relative_volume_20"], 0.70)
    return pop_mask & (df["recent_breakout_up_opening_range"] == True) & (df["relative_volume_20"] >= v_thr) & (df["relative_strength_vs_universe"] > 0)  # noqa: E712


def combo_breakout_x_volume_x_rs_down(df: pd.DataFrame, pop_mask: pd.Series) -> pd.Series:
    v_thr = _pct(df.loc[pop_mask, "relative_volume_20"], 0.70)
    return pop_mask & (df["recent_breakout_down_opening_range"] == True) & (df["relative_volume_20"] >= v_thr) & (df["relative_strength_vs_universe"] < 0)  # noqa: E712


def combo_sr_x_rsi(df: pd.DataFrame, pop_mask: pd.Series) -> pd.Series:
    sr_thr = _pct(df.loc[pop_mask, "sr_proximity_asymmetry"].abs(), 0.80)
    return pop_mask & (df["sr_proximity_asymmetry"].abs() >= sr_thr) & ((df["rsi_14"] <= 30) | (df["rsi_14"] >= 70))


def combo_vwap_x_volume(df: pd.DataFrame, pop_mask: pd.Series) -> pd.Series:
    d_thr = _pct(df.loc[pop_mask, "distance_from_vwap"].abs(), 0.80)
    v_thr = _pct(df.loc[pop_mask, "relative_volume_20"], 0.80)
    return pop_mask & (df["distance_from_vwap"].abs() >= d_thr) & (df["relative_volume_20"] >= v_thr)


def combo_momentum_x_breadth(df: pd.DataFrame, pop_mask: pd.Series) -> pd.Series:
    m_thr = _pct(df.loc[pop_mask, "momentum_acceleration_10"].abs(), 0.80)
    b_thr = _pct(df.loc[pop_mask, "universe_breadth"], 0.80)
    return pop_mask & (df["momentum_acceleration_10"].abs() >= m_thr) & (df["universe_breadth"] >= b_thr)


COMBINATIONS: dict[str, Callable[[pd.DataFrame, pd.Series], pd.Series]] = {
    "momentum_x_volume": combo_momentum_x_volume,
    "breakout_x_volume_x_rs_up": combo_breakout_x_volume_x_rs_up,
    "breakout_x_volume_x_rs_down": combo_breakout_x_volume_x_rs_down,
    "sr_x_rsi": combo_sr_x_rsi,
    "vwap_x_volume": combo_vwap_x_volume,
    "momentum_x_breadth": combo_momentum_x_breadth,
}

N_CONTROLS = 2
N_CONTROLLED_GROUPS = 2


def test_budget() -> dict[str, int]:
    per_pop_per_horizon = len(CONTINUOUS_CANDIDATES) * 2 + sum(len(s) for s in CATEGORICAL_CANDIDATES.values()) + len(COMBINATIONS)
    n_quantities_for_b = len(CONTINUOUS_CANDIDATES) + len(CATEGORICAL_CANDIDATES) + len(COMBINATIONS)
    return {
        "A_standalone": per_pop_per_horizon * len(POPULATIONS) * len(HORIZONS),
        "B_incremental_value": len(POPULATIONS) * N_CONTROLS * n_quantities_for_b * N_CONTROLLED_GROUPS,
    }


def _rate_test(df: pd.DataFrame, subset_mask: pd.Series, direction_col: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[subset_mask, [direction_col, "symbol", "bar_start_utc"]]
    sub = sub[sub[direction_col].isin(["UP", "DOWN"])]
    if len(sub) < 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[direction_col].eq("UP").astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha)


def run_family_a(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            baseline_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()

            for quantity in CONTINUOUS_CANDIDATES:
                valid = df.loc[pop_mask, quantity].dropna()
                if len(valid) < 50:
                    results.append({"population": population, "quantity": quantity, "bucket": "n/a", "horizon_minutes": h, "skipped_reason": f"insufficient rows ({len(valid)})"})
                    continue
                low_thr, high_thr = valid.quantile(0.2), valid.quantile(0.8)
                low_mask = pop_mask & (df[quantity] <= low_thr)
                high_mask = pop_mask & (df[quantity] >= high_thr)
                results.append({"population": population, "quantity": quantity, "bucket": "lowest_quintile", "horizon_minutes": h, **_rate_test(df, low_mask, direction_col, h, alpha, baseline_rate)})
                results.append({"population": population, "quantity": quantity, "bucket": "highest_quintile", "horizon_minutes": h, **_rate_test(df, high_mask, direction_col, h, alpha, baseline_rate)})

            for quantity, states in CATEGORICAL_CANDIDATES.items():
                for state in states:
                    mask = pop_mask & (df[quantity] == state)
                    results.append({"population": population, "quantity": quantity, "bucket": f"state_{state}", "horizon_minutes": h, **_rate_test(df, mask, direction_col, h, alpha, baseline_rate)})

            for name, mask_fn in COMBINATIONS.items():
                mask = mask_fn(df, pop_mask)
                results.append({"population": population, "quantity": name, "bucket": "combination", "horizon_minutes": h, **_rate_test(df, mask, direction_col, h, alpha, baseline_rate)})
    return results


def run_family_b(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    h = 15
    direction_col = f"direction_label_{h}m"
    results = []
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        baseline_up_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
        for control_col in ("return_5m", "relative_strength_vs_universe"):
            med = df.loc[pop_mask, control_col].median()
            for group_name, group_mask in ((f"below_median_{control_col}", pop_mask & (df[control_col] < med)), (f"at_or_above_median_{control_col}", pop_mask & (df[control_col] >= med))):
                for quantity in CONTINUOUS_CANDIDATES:
                    valid = df.loc[group_mask, quantity].dropna()
                    if len(valid) < 50:
                        results.append({"population": population, "quantity": quantity, "control": control_col, "controlled_group": group_name, "skipped_reason": f"insufficient rows ({len(valid)})"})
                        continue
                    thr = valid.quantile(0.8)
                    mask = group_mask & (df[quantity] >= thr)
                    results.append({"population": population, "quantity": quantity, "control": control_col, "controlled_group": group_name, **_rate_test(df, mask, direction_col, h, alpha, baseline_up_rate)})

                for quantity, states in CATEGORICAL_CANDIDATES.items():
                    mask = group_mask & (df[quantity] == states[0])
                    results.append({"population": population, "quantity": quantity, "control": control_col, "controlled_group": group_name, **_rate_test(df, mask, direction_col, h, alpha, baseline_up_rate)})

                for name, mask_fn in COMBINATIONS.items():
                    mask = mask_fn(df, group_mask)
                    results.append({"population": population, "quantity": name, "control": control_col, "controlled_group": group_name, **_rate_test(df, mask, direction_col, h, alpha, baseline_up_rate)})
    return results


def pick_survivors(family_a: list[dict[str, Any]], family_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_significant = {(r["population"], r["quantity"], r["bucket"]) for r in family_a if r.get("significant_after_correction")}
    b_by_key: dict[tuple[str, str], list[bool]] = {}
    for r in family_b:
        if "skipped_reason" in r:
            continue
        b_by_key.setdefault((r["population"], r["quantity"]), []).append(r.get("significant_after_correction", False))

    survivors = []
    for (population, quantity, bucket) in a_significant:
        b_flags = b_by_key.get((population, quantity), [])
        if len(b_flags) == 4 and all(b_flags):
            survivors.append({"population": population, "quantity": quantity, "bucket": bucket})
    return survivors


def _survivor_mask(df: pd.DataFrame, s: dict[str, Any], base_mask: pd.Series) -> pd.Series:
    if s["quantity"] in COMBINATIONS:
        return COMBINATIONS[s["quantity"]](df, base_mask)
    if s["quantity"] in CATEGORICAL_CANDIDATES:
        state = s["bucket"].replace("state_", "")
        return base_mask & (df[s["quantity"]].astype(str) == state)
    valid = df.loc[base_mask, s["quantity"]].dropna()
    if len(valid) < 50:
        return base_mask & False
    thr = valid.quantile(0.2 if s["bucket"] == "lowest_quintile" else 0.8)
    return base_mask & ((df[s["quantity"]] <= thr) if s["bucket"] == "lowest_quintile" else (df[s["quantity"]] >= thr))


CONFIRMATORY_ALPHA = 0.05


def run_cross_stock_confirmatory(df: pd.DataFrame, survivors: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for s in survivors:
        key = f"{s['population']}_{s['quantity']}_{s['bucket']}"
        pop_mask = population_mask(df, s["population"])
        results[key] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            baseline_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
            mask = _survivor_mask(df, s, pop_mask)
            per_symbol_n = df.loc[mask & df[direction_col].isin(["UP", "DOWN"])].groupby("symbol").size()
            top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
            remainder_mask = mask & ~df["symbol"].isin(top_5)
            results[key][f"{h}m"] = {"n_symbols_excluded": len(top_5), **_rate_test(df, remainder_mask, direction_col, h, CONFIRMATORY_ALPHA, baseline_rate)}
    return results


def run_chronological_confirmatory(df: pd.DataFrame, survivors: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}

    results: dict[str, Any] = {"split_dates": {k: {"n": len(v)} for k, v in splits.items()}, "by_survivor": {}}
    for s in survivors:
        key = f"{s['population']}_{s['quantity']}_{s['bucket']}"
        pop_mask = population_mask(df, s["population"])
        results["by_survivor"][key] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            results["by_survivor"][key][f"{h}m"] = {}
            for split_name, split_dates in splits.items():
                full_mask = pop_mask & df["trading_date"].isin(split_dates)
                baseline_rate = df.loc[full_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
                mask = _survivor_mask(df, s, full_mask)
                results["by_survivor"][key][f"{h}m"][split_name] = _rate_test(df, mask, direction_col, h, CONFIRMATORY_ALPHA, baseline_rate)
    return results


def compute_baselines(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        results[population] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            sub = df.loc[pop_mask, direction_col].dropna()
            sub_binary = sub[sub.isin(["UP", "DOWN"])]
            ambiguous_rate = (sub == "AMBIGUOUS").mean() if len(sub) else None
            if len(sub_binary) == 0:
                results[population][f"{h}m"] = {"skipped_reason": "no rows"}
                continue
            majority_class = sub_binary.value_counts().idxmax()
            majority_accuracy = (sub_binary == majority_class).mean()
            results[population][f"{h}m"] = {"majority_class": majority_class, "majority_class_accuracy": float(majority_accuracy), "ambiguous_rate": float(ambiguous_rate) if ambiguous_rate is not None else None, "n": len(sub_binary)}
    return results


def run_economic_evaluation(df: pd.DataFrame, survivors: list[dict[str, Any]]) -> dict[str, Any]:
    cost_scenarios_bps = {"low_20bps": 20.0, "medium_40bps": 40.0, "high_60bps": 60.0}
    results: dict[str, Any] = {}
    for s in survivors:
        key = f"{s['population']}_{s['quantity']}_{s['bucket']}"
        pop_mask = population_mask(df, s["population"])
        results[key] = {}
        for h in HORIZONS:
            mask = _survivor_mask(df, s, pop_mask)
            ret_col = f"gross_return_{h}m"
            gross = df.loc[mask, ret_col].dropna()
            if len(gross) < 30:
                results[key][f"{h}m"] = {"skipped_reason": f"insufficient rows ({len(gross)})"}
                continue
            scenarios = {name: {"cost_bps": bps, "mean_net": float(gross.mean()) - bps / 10_000.0, "hit_rate": float(((gross - bps / 10_000.0) > 0).mean())} for name, bps in cost_scenarios_bps.items()}
            results[key][f"{h}m"] = {"n": len(gross), "gross_mean": float(gross.mean()), "cost_scenarios": scenarios}
    return results


def build_ledger(family_a, family_b, alpha: float) -> list[dict[str, Any]]:
    ledger = []
    for r in family_a:
        ledger.append({"family": "A_standalone", "hypothesis": f"{r['quantity']} ({r['bucket']}) predicts direction within {r['population']} at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_b:
        ledger.append({"family": "B_incremental_value", "hypothesis": f"{r['quantity']} adds directional value beyond {r['control']} within {r['population']} ({r.get('controlled_group')})",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(survivors, cross_stock, chrono, economic) -> dict[str, Any]:
    if not survivors:
        return {"verdict": "D", "reason": "No candidate across any of the 11 information families (or their combinations) was significant after correction AND demonstrated incremental value beyond both return_5m and relative_strength_vs_universe. No reliable directional mechanism was found.", "n_survivors": 0}

    cross_stock_holds = all(v.get("significant_after_correction") for key_results in cross_stock.values() for v in key_results.values() if "skipped_reason" not in v)
    chrono_holds = False
    for key_results in chrono.get("by_survivor", {}).values():
        for h_results in key_results.values():
            valid = [v for v in h_results.values() if "skipped_reason" not in v]
            if valid and all(v.get("significant_after_correction") for v in valid):
                chrono_holds = True

    economic_holds = False
    for key_results in economic.values():
        for h_results in key_results.values():
            if "skipped_reason" in h_results:
                continue
            if h_results["cost_scenarios"]["high_60bps"]["mean_net"] > 0:
                economic_holds = True

    if cross_stock_holds and chrono_holds and economic_holds:
        verdict, reason = "A", f"{len(survivors)} candidate(s) survived every gate including realistic 60bps costs - a genuinely new, economically plausible directional mechanism."
    elif cross_stock_holds and chrono_holds:
        verdict, reason = "B", f"{len(survivors)} candidate(s) survived every statistical and generalization gate but not the conservative 60bps cost scenario - real but not yet economically viable."
    elif cross_stock_holds or chrono_holds:
        verdict, reason = "C", f"{len(survivors)} candidate(s) passed the statistical gates but only partially replicate (cross-stock: {cross_stock_holds}, chronological: {chrono_holds})."
    else:
        verdict, reason = "D", f"{len(survivors)} candidate(s) passed the initial statistical gates but failed both cross-stock generalization and chronological replication."

    return {"verdict": verdict, "reason": reason, "n_survivors": len(survivors), "survivors": survivors, "cross_stock_holds": cross_stock_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_q_report.json")
    args = parser.parse_args()

    df = load_datasets()
    print(f"loaded {len(df)} rows, {len(df.columns)} columns")

    budget = test_budget()
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    ambiguous = compute_baselines(df)
    print("baselines computed")

    family_a = run_family_a(df, alpha)
    print(f"Family A complete: {len(family_a)} rows")
    family_b = run_family_b(df, alpha)
    print(f"Family B complete: {len(family_b)} rows")

    survivors = pick_survivors(family_a, family_b)
    print(f"survivors: {survivors}")

    cross_stock = run_cross_stock_confirmatory(df, survivors)
    chrono = run_chronological_confirmatory(df, survivors)
    economic = run_economic_evaluation(df, survivors)

    ledger = build_ledger(family_a, family_b, alpha)
    verdict = final_verdict(survivors, cross_stock, chrono, economic)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "test_budget": budget, "bonferroni_alpha": alpha, "baselines": ambiguous,
        "family_a_standalone": family_a, "family_b_incremental_value": family_b,
        "survivors": survivors, "cross_stock_confirmatory": cross_stock, "chronological_confirmatory": chrono,
        "economic_evaluation": economic, "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
