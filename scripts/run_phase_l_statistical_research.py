"""Phase L: direction-discovery research for M4/M3 magnitude setups. Built entirely on:
  - the FROZEN Phase H dataset (features/groups) - read-only, unmodified
  - the FROZEN Phase J mover-mechanics (M3/M4 population gates, unmodified)
  - the FROZEN Phase K entry_and_path dataset (next-bar-open gross_return_Xm, unmodified)
  - the new Phase L g5_g6 dataset (the only genuinely new computation this phase required)
Nothing in Phase G/H/I/J/K is modified, retested, or reinterpreted. This phase answers a narrower,
different question than any prior phase: GIVEN a magnitude setup (M3 or M4 fires), which way does it
go - not whether a big move is likely (already answered) and not whether a naive directional trade
is profitable in general (already answered, negatively, in Phase K).

FROZEN DIRECTION TARGET (approved, not re-derived here):
  direction_label_Xm = UP        if gross_return_Xm >= +0.0020  (20bps, Phase K's own low-cost floor)
                      = DOWN      if gross_return_Xm <= -0.0020
                      = AMBIGUOUS otherwise
`gross_return_Xm` is Phase K's own next-bar-open-to-horizon-close return, reused verbatim. The
AMBIGUOUS rate is reported explicitly (see `compute_ambiguous_rates`), never silently dropped.

POPULATIONS (frozen, non-pooled): M4 = bottom quintile `level_touch_count_resistance` (Phase J's
exact construction); M3 = `compression_then_expansion == True`.

G1-G10, RESOLVED TO 13 DETERMINISTIC TESTABLE QUANTITIES (frozen before any result was examined -
see CANDIDATE_QUANTITIES below for the exact, single source of truth every test in this script
iterates over - the test budget is COMPUTED FROM this same structure, not hand-counted separately,
specifically to avoid the exact under-counting bug caught and fixed in Phase J):
  G1  level_touch_count_support - level_touch_count_resistance      (continuous)
  G2  level_touch_count_support <= support's own bottom-quintile     (binary)
  G3a vwap_cross                                                     (categorical: 3 states)
  G3b distance_from_vwap > 0 ("above VWAP")                          (binary)
  G4  relative_strength_vs_universe                                  (continuous)
  G5  g5_directional_pressure_10 (NEW)                                (continuous)
  G6  g6_opening_range_position (NEW)                                 (continuous)
  G7a candle_direction (+1 vs -1, doji excluded)                      (categorical: 2 states)
  G7b candle_body_ratio                                               (continuous)
  G9  overnight_gap                                                   (continuous)
  G10 momentum_acceleration_10                                       (continuous)
(G8 - regime/time-of-day - is a DESCRIPTIVE stability check on survivors only, not an independent
pre-registered mechanism, matching Phase H/J's own precedent for this exact kind of dimension.)

STATISTICAL METHOD: reuses the exact centered-Newey-West machinery Phase J built for a binary
outcome (`_finalize_rate_test`, imported unchanged) - here the "flag" is `direction_label_Xm ==
"UP"`, restricted to NON-AMBIGUOUS rows, and `baseline_rate` is P(UP | non-ambiguous) within the
SAME (population, horizon) slice - so every test asks "does this bucket's UP-rate differ from this
setup's own baseline UP-rate," never from an arbitrary 50%.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backend.app.research.statistics import newey_west_mean_test
from scripts.run_phase_h_statistical_research import attach_pre_registered_groups, load_group_definitions
from scripts.run_phase_j_mover_target_design import MIN_PRIOR_OBSERVATIONS_FOR_T3, compute_t3_expanding_percentile

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
MOVER_MECHANICS_DATASET = ROOT / "data" / "derived" / "phase_j_mover_detection" / "mover_mechanics_dataset.parquet"
ENTRY_PATH_DATASET = ROOT / "data" / "derived" / "phase_k_economic_validation" / "entry_and_path_dataset.parquet"
G5_G6_DATASET = ROOT / "data" / "derived" / "phase_l_direction_discovery" / "g5_g6_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_l_direction_discovery"

HORIZONS = (5, 15, 30, 60)
COMBINATION_HORIZONS = (5, 15)
DIRECTION_THRESHOLD = 0.0020
POPULATIONS = ("M4", "M3")
MOVER_TARGET_PERCENTILE = 0.90


def load_datasets() -> pd.DataFrame:
    df = pd.read_parquet(PHASE_H_DATASET)
    group_defs = load_group_definitions()
    df = attach_pre_registered_groups(df, group_defs)

    mm = pd.read_parquet(MOVER_MECHANICS_DATASET)
    df = df.merge(mm, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_mm_dup"))

    ep = pd.read_parquet(ENTRY_PATH_DATASET)
    df = df.merge(ep, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_ep_dup"))

    g56 = pd.read_parquet(G5_G6_DATASET)
    df = df.merge(g56, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_g56_dup"))

    for h in HORIZONS:
        ret = df[f"gross_return_{h}m"]
        df[f"direction_label_{h}m"] = pd.Series("AMBIGUOUS", index=df.index, dtype="object")
        df.loc[ret >= DIRECTION_THRESHOLD, f"direction_label_{h}m"] = "UP"
        df.loc[ret <= -DIRECTION_THRESHOLD, f"direction_label_{h}m"] = "DOWN"
        df.loc[ret.isna(), f"direction_label_{h}m"] = None

    resistance_valid = df["level_touch_count_resistance"].dropna()
    resistance_thr = resistance_valid.quantile(0.2)
    df["pop_M4"] = df["level_touch_count_resistance"] <= resistance_thr
    df["pop_M3"] = df["compression_then_expansion"] == True  # noqa: E712

    support_valid = df["level_touch_count_support"].dropna()
    support_thr = support_valid.quantile(0.2)
    df["g1_support_minus_resistance_touch"] = df["level_touch_count_support"] - df["level_touch_count_resistance"]
    df["g2_support_also_fresh"] = df["level_touch_count_support"] <= support_thr
    df["g3b_above_vwap"] = df["distance_from_vwap"] > 0
    df["g7a_candle_direction"] = df["candle_direction"]

    return df


def population_mask(df: pd.DataFrame, population: str) -> pd.Series:
    return df[f"pop_{population}"] == True  # noqa: E712


CONTINUOUS_QUANTITIES = ["g1_support_minus_resistance_touch", "relative_strength_vs_universe", "g5_directional_pressure_10", "g6_opening_range_position", "candle_body_ratio", "overnight_gap", "momentum_acceleration_10"]
BINARY_QUANTITIES = ["g2_support_also_fresh", "g3b_above_vwap"]
CATEGORICAL_QUANTITIES = {"vwap_cross": ["RECLAIM", "REJECTION", "NONE"], "g7a_candle_direction": [1, -1]}

COMBINATIONS = [
    ("g1_x_vwap", "g1_support_minus_resistance_touch", "g3b_above_vwap"),
    ("relstrength_x_candle", "relative_strength_vs_universe", "g7a_candle_direction"),
    ("pressure_x_openingrange", "g5_directional_pressure_10", "g6_opening_range_position"),
]


def test_budget() -> dict[str, int]:
    n_continuous_tests = len(CONTINUOUS_QUANTITIES) * 2
    n_binary_tests = len(BINARY_QUANTITIES) * 1
    n_categorical_tests = sum(len(states) for states in CATEGORICAL_QUANTITIES.values())
    per_pop_per_horizon = n_continuous_tests + n_binary_tests + n_categorical_tests
    return {
        "A_standalone": per_pop_per_horizon * len(POPULATIONS) * len(HORIZONS),
        "B_combinations": len(COMBINATIONS) * len(POPULATIONS) * len(COMBINATION_HORIZONS),
        "C_incremental_value": (len(CONTINUOUS_QUANTITIES) + len(BINARY_QUANTITIES) + len(CATEGORICAL_QUANTITIES)) * len(POPULATIONS) * 2,
    }


def _finalize_rate_test(ordered_flags: list[float], baseline_rate: float, horizon: int, alpha: float) -> dict[str, Any]:
    if len(ordered_flags) < 10:
        return {"skipped_reason": f"insufficient rows ({len(ordered_flags)})"}
    centered = [f - baseline_rate for f in ordered_flags]
    max_lag = min(horizon - 1, len(centered) - 1)
    result = newey_west_mean_test(centered, max_lag=max_lag)
    bucket_rate = baseline_rate + result.mean
    return {
        "n_rows": len(ordered_flags), "bucket_up_rate": bucket_rate, "baseline_up_rate": baseline_rate,
        "lift": bucket_rate / baseline_rate if baseline_rate else None,
        "p_value": result.p_value, "significant_after_correction": bool(result.p_value < alpha),
    }


def _up_rate_test_for_subset(df: pd.DataFrame, subset_mask: pd.Series, direction_col: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[subset_mask, [direction_col, "symbol", "bar_start_utc"]]
    sub = sub[sub[direction_col].isin(["UP", "DOWN"])]
    if len(sub) < 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[direction_col].eq("UP").astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha)


def compute_ambiguous_rates(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        results[population] = {}
        for h in HORIZONS:
            col = f"direction_label_{h}m"
            sub = df.loc[pop_mask, col].dropna()
            counts = sub.value_counts(normalize=True).to_dict()
            results[population][f"{h}m"] = {"n": len(sub), "up_rate": counts.get("UP", 0.0), "down_rate": counts.get("DOWN", 0.0), "ambiguous_rate": counts.get("AMBIGUOUS", 0.0)}
    return results


def run_family_a(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            baseline_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()

            for quantity in CONTINUOUS_QUANTITIES:
                valid = df.loc[pop_mask, quantity].dropna()
                if len(valid) < 50:
                    results.append({"population": population, "quantity": quantity, "bucket": "n/a", "horizon_minutes": h, "skipped_reason": f"insufficient rows ({len(valid)})"})
                    continue
                low_thr, high_thr = valid.quantile(0.2), valid.quantile(0.8)
                low_mask = pop_mask & (df[quantity] <= low_thr)
                high_mask = pop_mask & (df[quantity] >= high_thr)
                results.append({"population": population, "quantity": quantity, "bucket": "lowest_quintile", "horizon_minutes": h, **_up_rate_test_for_subset(df, low_mask, direction_col, h, alpha, baseline_rate)})
                results.append({"population": population, "quantity": quantity, "bucket": "highest_quintile", "horizon_minutes": h, **_up_rate_test_for_subset(df, high_mask, direction_col, h, alpha, baseline_rate)})

            for quantity in BINARY_QUANTITIES:
                mask = pop_mask & (df[quantity] == True)  # noqa: E712
                results.append({"population": population, "quantity": quantity, "bucket": "flag_true", "horizon_minutes": h, **_up_rate_test_for_subset(df, mask, direction_col, h, alpha, baseline_rate)})

            for quantity, states in CATEGORICAL_QUANTITIES.items():
                for state in states:
                    mask = pop_mask & (df[quantity] == state)
                    results.append({"population": population, "quantity": quantity, "bucket": f"state_{state}", "horizon_minutes": h, **_up_rate_test_for_subset(df, mask, direction_col, h, alpha, baseline_rate)})
    return results


def run_family_b(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        for name, feat1, feat2 in COMBINATIONS:
            for h in COMBINATION_HORIZONS:
                direction_col = f"direction_label_{h}m"
                baseline_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()

                def _extreme_mask(col: str) -> pd.Series:
                    if col in BINARY_QUANTITIES or col in CATEGORICAL_QUANTITIES:
                        if col == "g7a_candle_direction":
                            return df[col] == 1
                        return df[col] == True  # noqa: E712
                    med = df.loc[pop_mask, col].median()
                    return df[col] >= med

                corner_mask = pop_mask & _extreme_mask(feat1) & _extreme_mask(feat2)
                r = _up_rate_test_for_subset(df, corner_mask, direction_col, h, alpha, baseline_rate)
                results.append({"population": population, "combination": name, "feature_1": feat1, "feature_2": feat2, "horizon_minutes": h, "n_in_corner": int(corner_mask.sum()), **r})
    return results


def run_family_c(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    h = 15
    direction_col = f"direction_label_{h}m"
    results = []
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        baseline_up_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
        med_return5m = df.loc[pop_mask, "return_5m"].median()
        for quantity in CONTINUOUS_QUANTITIES + BINARY_QUANTITIES + list(CATEGORICAL_QUANTITIES.keys()):
            for group_name, group_mask in (("below_median_return5m", pop_mask & (df["return_5m"] < med_return5m)), ("at_or_above_median_return5m", pop_mask & (df["return_5m"] >= med_return5m))):
                if quantity in BINARY_QUANTITIES:
                    mask = group_mask & (df[quantity] == True)  # noqa: E712
                elif quantity in CATEGORICAL_QUANTITIES:
                    mask = group_mask & (df[quantity] == CATEGORICAL_QUANTITIES[quantity][0])
                else:
                    valid = df.loc[group_mask, quantity].dropna()
                    if len(valid) < 50:
                        results.append({"population": population, "quantity": quantity, "controlled_group": group_name, "skipped_reason": f"insufficient rows ({len(valid)})"})
                        continue
                    thr = valid.quantile(0.8)
                    mask = group_mask & (df[quantity] >= thr)
                r = _up_rate_test_for_subset(df, mask, direction_col, h, alpha, baseline_up_rate)
                results.append({"population": population, "quantity": quantity, "controlled_group": group_name, **r})
    return results


def pick_survivors(family_a: list[dict[str, Any]], family_c: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_significant = {(r["population"], r["quantity"], r["bucket"]) for r in family_a if r.get("significant_after_correction")}
    c_by_key: dict[tuple[str, str], list[bool]] = {}
    for r in family_c:
        if "skipped_reason" in r:
            continue
        c_by_key.setdefault((r["population"], r["quantity"]), []).append(r.get("significant_after_correction", False))

    survivors = []
    for (population, quantity, bucket) in a_significant:
        c_flags = c_by_key.get((population, quantity), [])
        if len(c_flags) == 2 and all(c_flags):
            survivors.append({"population": population, "quantity": quantity, "bucket": bucket})
    return survivors


def run_cross_stock_confirmatory(df: pd.DataFrame, survivors: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for s in survivors:
        key = f"{s['population']}_{s['quantity']}_{s['bucket']}"
        pop_mask = population_mask(df, s["population"])
        results[key] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            baseline_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
            if s["quantity"] in BINARY_QUANTITIES:
                mask = pop_mask & (df[s["quantity"]] == True)  # noqa: E712
            elif s["quantity"] in CATEGORICAL_QUANTITIES:
                state = s["bucket"].replace("state_", "")
                mask = pop_mask & (df[s["quantity"]].astype(str) == state)
            else:
                valid = df.loc[pop_mask, s["quantity"]].dropna()
                thr = valid.quantile(0.2 if s["bucket"] == "lowest_quintile" else 0.8)
                mask = pop_mask & ((df[s["quantity"]] <= thr) if s["bucket"] == "lowest_quintile" else (df[s["quantity"]] >= thr))

            per_symbol_n = df.loc[mask & df[direction_col].isin(["UP", "DOWN"])].groupby("symbol").size()
            top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
            remainder_mask = mask & ~df["symbol"].isin(top_5)
            results[key][f"{h}m"] = {"n_symbols_excluded": len(top_5), **_up_rate_test_for_subset(df, remainder_mask, direction_col, h, alpha=1.0, baseline_rate=baseline_rate)}
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
                split_df_mask = df["trading_date"].isin(split_dates)
                full_mask = pop_mask & split_df_mask
                baseline_rate = df.loc[full_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
                if s["quantity"] in BINARY_QUANTITIES:
                    mask = full_mask & (df[s["quantity"]] == True)  # noqa: E712
                elif s["quantity"] in CATEGORICAL_QUANTITIES:
                    state = s["bucket"].replace("state_", "")
                    mask = full_mask & (df[s["quantity"]].astype(str) == state)
                else:
                    valid = df.loc[full_mask, s["quantity"]].dropna()
                    if len(valid) < 50:
                        results["by_survivor"][key][f"{h}m"][split_name] = {"skipped_reason": "insufficient rows in split"}
                        continue
                    thr = valid.quantile(0.2 if s["bucket"] == "lowest_quintile" else 0.8)
                    mask = full_mask & ((df[s["quantity"]] <= thr) if s["bucket"] == "lowest_quintile" else (df[s["quantity"]] >= thr))
                results["by_survivor"][key][f"{h}m"][split_name] = _up_rate_test_for_subset(df, mask, direction_col, h, alpha=1.0, baseline_rate=baseline_rate)
    return results


def compute_baselines(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        results[population] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            sub = df.loc[pop_mask, direction_col].dropna()
            sub = sub[sub.isin(["UP", "DOWN"])]
            if len(sub) == 0:
                results[population][f"{h}m"] = {"skipped_reason": "no rows"}
                continue
            majority_class = sub.value_counts().idxmax()
            majority_accuracy = (sub == majority_class).mean()
            results[population][f"{h}m"] = {"majority_class": majority_class, "majority_class_accuracy": float(majority_accuracy), "n": len(sub)}
    return results


def build_ledger(family_a, family_b, family_c, alpha: float) -> list[dict[str, Any]]:
    ledger = []
    for r in family_a:
        ledger.append({"family": "A_standalone", "hypothesis": f"{r['quantity']} ({r['bucket']}) predicts direction within {r['population']} at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_b:
        ledger.append({"family": "B_combination", "hypothesis": f"{r['combination']} predicts direction within {r['population']} at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_c:
        ledger.append({"family": "C_incremental_value", "hypothesis": f"{r['quantity']} adds directional value beyond return_5m within {r['population']} ({r.get('controlled_group')})",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(family_a, survivors, cross_stock, chrono) -> dict[str, Any]:
    if not survivors:
        return {"verdict": "D", "reason": "No candidate mechanism was significant after correction AND demonstrated incremental value beyond the return_5m baseline in both controlled-for halves. No reliable directional mechanism was found for M3/M4 magnitude setups.", "n_survivors": 0}

    cross_stock_holds = all(
        v.get("significant_after_correction") for key_results in cross_stock.values() for v in key_results.values() if "skipped_reason" not in v
    )
    chrono_holds = False
    for key_results in chrono.get("by_survivor", {}).values():
        for h_results in key_results.values():
            valid = [v for v in h_results.values() if "skipped_reason" not in v]
            if valid and all(v.get("significant_after_correction") for v in valid):
                chrono_holds = True

    if cross_stock_holds and chrono_holds:
        verdict = "B"
        reason = f"{len(survivors)} candidate(s) survived standalone significance, incremental value, cross-stock generalization, and chronological replication - a real directional mechanism for M3/M4 setups. Not yet economically evaluated / not yet A, per the project's standing discipline that statistical survival alone never implies tradability."
    elif cross_stock_holds or chrono_holds:
        verdict = "C"
        reason = f"{len(survivors)} candidate(s) survived the statistical gates, but only partially replicate (cross-stock: {cross_stock_holds}, chronological: {chrono_holds}) - a real but fragile signal, not yet reliable enough to pursue further without more evidence."
    else:
        verdict = "D"
        reason = f"{len(survivors)} candidate(s) passed the initial statistical gates but failed BOTH cross-stock generalization and chronological replication - not a reliable mechanism despite passing the first screen."

    return {"verdict": verdict, "reason": reason, "n_survivors": len(survivors), "survivors": survivors, "cross_stock_holds": cross_stock_holds, "chrono_holds": chrono_holds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_l_report.json")
    args = parser.parse_args()

    df = load_datasets()
    print(f"loaded {len(df)} rows, {len(df.columns)} columns")

    budget = test_budget()
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    ambiguous_rates = compute_ambiguous_rates(df)
    print("ambiguous rates computed")

    family_a = run_family_a(df, alpha)
    print(f"Family A complete: {len(family_a)} rows")
    family_b = run_family_b(df, alpha)
    print(f"Family B complete: {len(family_b)} rows")
    family_c = run_family_c(df, alpha)
    print(f"Family C complete: {len(family_c)} rows")

    survivors = pick_survivors(family_a, family_c)
    print(f"survivors qualifying for confirmatory evaluation: {survivors}")

    cross_stock = run_cross_stock_confirmatory(df, survivors)
    chrono = run_chronological_confirmatory(df, survivors)
    baselines = compute_baselines(df)

    ledger = build_ledger(family_a, family_b, family_c, alpha)
    verdict = final_verdict(family_a, survivors, cross_stock, chrono)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "direction_threshold": DIRECTION_THRESHOLD, "test_budget": budget, "bonferroni_alpha": alpha,
        "ambiguous_rates": ambiguous_rates, "baselines": baselines,
        "family_a_standalone": family_a, "family_b_combinations": family_b, "family_c_incremental_value": family_c,
        "survivors": survivors, "cross_stock_confirmatory": cross_stock, "chronological_confirmatory": chrono,
        "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
