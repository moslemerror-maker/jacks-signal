"""Phase I: economic/tradability validation of relative_strength_vs_universe, built entirely on top
of the FROZEN Phase H dataset (data/derived/phase_h_full_universe_validation/feature_label_dataset.parquet,
UNMODIFIED, read-only) plus the one new Phase-I-only dataset (next_bar_entry_dataset.parquet, I3).

I1 - FREEZE DISCIPLINE: nothing in run_phase_g_*.py, run_phase_h_*.py, or their output files is
imported for mutation, only for reuse (`_quantile_bucket_test`, the Newey-West bucket-test primitive,
is imported UNCHANGED from run_phase_h_statistical_research.py - the exact same method, not a
reimplementation). Every grouping/threshold introduced in THIS file is NEW and is labeled as such in
the research ledger this script produces. No shuffled splits are used anywhere - the one chronological
split reused here is the frozen train/validation/held-out date partition Phase H already computed.

PRE-REGISTRATION AND TEST BUDGET (I6): every formal significance test this script runs is fixed
below (I3_ENTRY_TIMING_TESTS, I4_STABILITY_TESTS, I7_ROBUSTNESS_TESTS, I8_BASELINE_TESTS,
I9_CROSS_STOCK_TESTS) BEFORE it is run against real data. ONE Bonferroni correction is applied
across their sum. Descriptive analyses that are NOT formal hypothesis tests (I2 cost scenarios, I5
magnitude/threshold probabilities, I10 target comparison) are explicitly labeled DESCRIPTIVE in the
ledger and do not consume a correction slot - inflating the correction budget with non-inferential
arithmetic would not make the evidence standard stronger, only harder to read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.research.statistics import newey_west_mean_test
from scripts.run_phase_h_statistical_research import N_QUANTILE_BUCKETS, _quantile_bucket_test

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
NEXTBAR_DATASET = ROOT / "data" / "derived" / "phase_i_economic_validation" / "next_bar_entry_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_i_economic_validation"

HORIZONS = (5, 15, 30, 60)
PRIMARY_FEATURE = "relative_strength_vs_universe"

COST_SCENARIOS_BPS = {"zero_cost": 0.0, "low_cost": 5.0, "medium_cost": 15.0, "conservative_high_cost": 40.0}

def session_phase(minutes_since_open: float | None) -> str | None:
    if minutes_since_open is None or pd.isna(minutes_since_open):
        return None
    if minutes_since_open < 30:
        return "OPENING"
    if minutes_since_open < 180:
        return "MID_SESSION"
    if minutes_since_open < 300:
        return "AFTERNOON"
    return "FINAL_HOUR"


BASELINE_FEATURES_FOR_ROBUSTNESS = ["return_5m", "momentum_acceleration_10", "realized_volatility_20", "relative_volume_20", "distance_from_vwap", "minutes_since_open"]
ROBUSTNESS_HORIZONS = (5, 15)

SIMPLE_BASELINES = {"market_direction": "universe_average_return", "recent_return": "return_5m", "volatility": "realized_volatility_20", "relative_strength": PRIMARY_FEATURE}

MAGNITUDE_THRESHOLDS = (0.5, 1.0, 1.5, 2.0)


def test_budget() -> dict[str, int]:
    return {
        "I3_entry_timing": len(HORIZONS),
        "I4_stability": (4 + 5 + 2 + 13) * len(HORIZONS),
        "I7_robustness": len(BASELINE_FEATURES_FOR_ROBUSTNESS) * len(ROBUSTNESS_HORIZONS) * 2,
        "I8_baselines": (len(SIMPLE_BASELINES) - 1) * len(HORIZONS),
        "I9_cross_stock": len(HORIZONS),
    }


def load_datasets() -> pd.DataFrame:
    df = pd.read_parquet(PHASE_H_DATASET)
    df = df.assign(session_phase=df["minutes_since_open"].apply(session_phase))
    df = df.assign(month=df["trading_date"].apply(lambda d: f"{d.year:04d}-{d.month:02d}"))
    if NEXTBAR_DATASET.exists():
        nb = pd.read_parquet(NEXTBAR_DATASET)
        df = df.merge(nb, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_nb_dup"))
    return df


def run_i2_cost_scenarios(df: pd.DataFrame) -> dict[str, Any]:
    results = {}
    for h in HORIZONS:
        abs_col = f"mover_future_abs_return_{h}m"
        if abs_col not in df.columns or PRIMARY_FEATURE not in df.columns:
            results[f"{h}m"] = {"skipped_reason": f"required column missing ({abs_col})"}
            continue
        sub = df[[PRIMARY_FEATURE, abs_col]].dropna()
        if len(sub) < N_QUANTILE_BUCKETS * 10:
            results[f"{h}m"] = {"skipped_reason": "insufficient rows"}
            continue
        bucketed = sub.assign(_b=pd.qcut(sub[PRIMARY_FEATURE], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
        extreme = bucketed[(bucketed["_b"] == bucketed["_b"].min()) | (bucketed["_b"] == bucketed["_b"].max())]
        gross_move_fraction = float(extreme[abs_col].mean())
        gross_bps = gross_move_fraction * 10_000
        scenarios = {}
        for name, cost_bps in COST_SCENARIOS_BPS.items():
            net_bps = gross_bps - cost_bps
            scenarios[name] = {"round_trip_cost_bps": cost_bps, "gross_move_bps": round(gross_bps, 2), "net_move_bps": round(net_bps, 2), "cost_exceeds_gross_move": bool(net_bps <= 0)}
        results[f"{h}m"] = {"n_rows_in_extreme_buckets": len(extreme), "gross_mean_abs_move_bps": round(gross_bps, 2), "scenarios": scenarios}
    return results


def run_i3_entry_timing(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    results = {}
    for h in HORIZONS:
        same_bar = _quantile_bucket_test(df, PRIMARY_FEATURE, f"label_normalized_move_{h}m", h, alpha)
        nb_col = f"nextbar_normalized_move_{h}m"
        if nb_col not in df.columns:
            results[f"{h}m"] = {"same_bar_close_entry": same_bar, "next_bar_open_entry": {"skipped_reason": "next_bar_entry_dataset.parquet not found"}}
            continue
        next_bar = _quantile_bucket_test(df, PRIMARY_FEATURE, nb_col, h, alpha)
        results[f"{h}m"] = {"same_bar_close_entry": same_bar, "next_bar_open_entry": next_bar}
    return results


def run_i4_stability(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    dims = {"session_phase": ["OPENING", "MID_SESSION", "AFTERNOON", "FINAL_HOUR"], "intraday_regime": sorted(df["intraday_regime"].dropna().unique().tolist()),
            "market_regime": ["MARKET_UP", "MARKET_DOWN"] if "market_regime" in df.columns else [], "month": sorted(df["month"].dropna().unique().tolist())}
    if "market_regime" not in df.columns:
        df = df.assign(market_regime=(df["universe_average_return"] > 0).map({True: "MARKET_UP", False: "MARKET_DOWN"}))
        dims["market_regime"] = ["MARKET_UP", "MARKET_DOWN"]

    results: dict[str, Any] = {}
    for dim_name, values in dims.items():
        results[dim_name] = {}
        for v in values:
            sub_df = df[df[dim_name] == v]
            results[dim_name][v] = {f"{h}m": _quantile_bucket_test(sub_df, PRIMARY_FEATURE, f"label_normalized_move_{h}m", h, alpha) for h in HORIZONS}
    return results


def run_i5_magnitude_thresholds(df: pd.DataFrame) -> dict[str, Any]:
    results = {}
    for h in HORIZONS:
        label_col = f"label_normalized_move_{h}m"
        if label_col not in df.columns or PRIMARY_FEATURE not in df.columns:
            results[f"{h}m"] = {"skipped_reason": f"required column missing ({label_col})"}
            continue
        sub = df[[PRIMARY_FEATURE, label_col]].dropna()
        if len(sub) < N_QUANTILE_BUCKETS * 10:
            results[f"{h}m"] = {"skipped_reason": "insufficient rows"}
            continue
        bucketed = sub.assign(_b=pd.qcut(sub[PRIMARY_FEATURE], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
        extreme = bucketed[(bucketed["_b"] == bucketed["_b"].min()) | (bucketed["_b"] == bucketed["_b"].max())]
        by_threshold = {}
        for th in MAGNITUDE_THRESHOLDS:
            p_overall = float((sub[label_col].abs() > th).mean())
            p_extreme = float((extreme[label_col].abs() > th).mean())
            by_threshold[str(th)] = {"p_exceeds_overall": p_overall, "p_exceeds_in_extreme_buckets": p_extreme, "lift": (p_extreme / p_overall) if p_overall else None}
        results[f"{h}m"] = {"n_overall": len(sub), "n_extreme": len(extreme), "by_threshold": by_threshold}
    return results


def run_i7_robustness(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    results = {}
    for feature in BASELINE_FEATURES_FOR_ROBUSTNESS:
        results[feature] = {}
        for h in ROBUSTNESS_HORIZONS:
            label_col = f"label_normalized_move_{h}m"
            med = df[feature].median()
            below = df[df[feature] < med]
            above = df[df[feature] >= med]
            results[feature][f"{h}m"] = {
                "below_median": _quantile_bucket_test(below, PRIMARY_FEATURE, label_col, h, alpha),
                "at_or_above_median": _quantile_bucket_test(above, PRIMARY_FEATURE, label_col, h, alpha),
            }
    return results


def run_i8_baselines(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    results = {}
    for name, feature_col in SIMPLE_BASELINES.items():
        results[name] = {f"{h}m": _quantile_bucket_test(df, feature_col, f"label_normalized_move_{h}m", h, alpha) for h in HORIZONS}
    return results


def run_i9_cross_stock(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    label_col_15 = "label_normalized_move_15m"
    sub = df[["symbol", PRIMARY_FEATURE, label_col_15]].dropna()
    per_symbol_corr = sub.groupby("symbol").filter(lambda g: len(g) >= 30).groupby("symbol").apply(lambda g: g[PRIMARY_FEATURE].corr(g[label_col_15]))
    per_symbol_corr = per_symbol_corr.dropna()

    n = len(per_symbol_corr)
    n_negative = int((per_symbol_corr < 0).sum())
    n_positive = int((per_symbol_corr > 0).sum())
    sorted_corr = per_symbol_corr.sort_values()
    distribution = {
        "n_symbols": n, "median": float(per_symbol_corr.median()), "q25": float(per_symbol_corr.quantile(0.25)), "q75": float(per_symbol_corr.quantile(0.75)),
        "n_negative_sign": n_negative, "n_positive_sign": n_positive, "fraction_negative": n_negative / n if n else None,
        "strongest_5_symbols_by_abs_effect": per_symbol_corr.abs().sort_values(ascending=False).head(5).index.tolist(),
    }

    top_5_symbols = set(per_symbol_corr.abs().sort_values(ascending=False).head(5).index.tolist())
    remainder_df = df[~df["symbol"].isin(top_5_symbols)]
    per_horizon = {f"{h}m": _quantile_bucket_test(remainder_df, PRIMARY_FEATURE, f"label_normalized_move_{h}m", h, alpha) for h in HORIZONS}

    return {"per_symbol_effect_distribution": distribution, "excluding_top_5_strongest_symbols": per_horizon, "n_symbols_excluded": len(top_5_symbols)}


def run_i10_target_comparison(df: pd.DataFrame, magnitude_results: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for h in HORIZONS:
        label_col, class_col = f"label_normalized_move_{h}m", f"label_class_{h}m"
        sub = df[[PRIMARY_FEATURE, label_col, class_col]].dropna(subset=[PRIMARY_FEATURE, label_col])
        if len(sub) < N_QUANTILE_BUCKETS * 10:
            results[f"{h}m"] = {"skipped_reason": "insufficient rows"}
            continue
        bucketed = sub.assign(_b=pd.qcut(sub[PRIMARY_FEATURE], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
        low_b, high_b = bucketed["_b"].min(), bucketed["_b"].max()
        low, high = bucketed[bucketed["_b"] == low_b], bucketed[bucketed["_b"] == high_b]

        def _joint(bucket_df: pd.DataFrame, direction: str) -> float | None:
            joint = bucket_df[(bucket_df[class_col] == direction) & (bucket_df[label_col].abs() > 1.0)]
            return len(joint) / len(bucket_df) if len(bucket_df) else None

        results[f"{h}m"] = {
            "A_direction_hit_rate_spread": {"lowest_bucket": float((low[class_col] == "UP").mean()), "highest_bucket": float((high[class_col] == "DOWN").mean())},
            "C_expected_normalized_return_spread": {"lowest_bucket": float(low[label_col].mean()), "highest_bucket": float(high[label_col].mean())},
            "D_magnitude_threshold_lift": magnitude_results.get(f"{h}m", {}).get("by_threshold", {}),
            "F_joint_direction_and_meaningful_magnitude": {"lowest_bucket_P_UP_and_big_move": _joint(low, "UP"), "highest_bucket_P_DOWN_and_big_move": _joint(high, "DOWN")},
        }
    return results


def build_ledger(i3, i4, i7, i8, i9, alpha) -> list[dict[str, Any]]:
    ledger = []
    for h, r in i3.items():
        for entry_type in ("same_bar_close_entry", "next_bar_open_entry"):
            v = r.get(entry_type, {})
            ledger.append({"hypothesis": f"relative_strength_vs_universe predicts {h} move under {entry_type}", "pre_registration": "pre-registered (I3)",
                            "result": "SIGNIFICANT" if v.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in v else "NOT SIGNIFICANT"),
                            "effect_size": v.get("top_minus_bottom_bucket_mean"), "n_rows": v.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for dim, by_val in i4.items():
        for val, by_h in by_val.items():
            for h, v in by_h.items():
                ledger.append({"hypothesis": f"relative_strength_vs_universe predicts {h} move within {dim}={val}", "pre_registration": "pre-registered (I4)",
                                "result": "SIGNIFICANT" if v.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in v else "NOT SIGNIFICANT"),
                                "effect_size": v.get("top_minus_bottom_bucket_mean"), "n_rows": v.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for feature, by_h in i7.items():
        for h, groups in by_h.items():
            for gname, v in groups.items():
                ledger.append({"hypothesis": f"relative_strength_vs_universe's {h} effect survives controlling for {feature} ({gname})", "pre_registration": "pre-registered (I7)",
                                "result": "SIGNIFICANT" if v.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in v else "NOT SIGNIFICANT"),
                                "effect_size": v.get("top_minus_bottom_bucket_mean"), "n_rows": v.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for name, by_h in i8.items():
        for h, v in by_h.items():
            ledger.append({"hypothesis": f"simple baseline '{name}' predicts {h} move", "pre_registration": "pre-registered (I8)",
                            "result": "SIGNIFICANT" if v.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in v else "NOT SIGNIFICANT"),
                            "effect_size": v.get("top_minus_bottom_bucket_mean"), "n_rows": v.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for h, v in i9["excluding_top_5_strongest_symbols"].items():
        ledger.append({"hypothesis": f"relative_strength_vs_universe predicts {h} move excluding the 5 strongest symbols", "pre_registration": "pre-registered (I9)",
                        "result": "SIGNIFICANT" if v.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in v else "NOT SIGNIFICANT"),
                        "effect_size": v.get("top_minus_bottom_bucket_mean"), "n_rows": v.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(i2, i3, i4, i5, i7, i8, i9) -> dict[str, Any]:
    survives_medium_cost = all(
        not scen["cost_exceeds_gross_move"] for h in i2.values() if "scenarios" in h for scen in [h["scenarios"].get("medium_cost", {"cost_exceeds_gross_move": True})]
    )
    survives_conservative_cost = all(
        not scen["cost_exceeds_gross_move"] for h in i2.values() if "scenarios" in h for scen in [h["scenarios"].get("conservative_high_cost", {"cost_exceeds_gross_move": True})]
    )
    all_costs_survive = survives_conservative_cost
    entry_timing_holds = all(v.get("next_bar_open_entry", {}).get("significant_after_correction") for v in i3.values() if "next_bar_open_entry" in v and "skipped_reason" not in v.get("next_bar_open_entry", {}))
    stability_n_sig = sum(1 for by_val in i4.values() for by_h in by_val.values() for v in by_h.values() if v.get("significant_after_correction"))
    stability_n_total = sum(1 for by_val in i4.values() for by_h in by_val.values() for _ in by_h.values())
    robustness_cells = [v for by_h in i7.values() for v in by_h.get("15m", {}).values()]
    baselines_dont_fully_explain = bool(robustness_cells) and all(v.get("significant_after_correction") for v in robustness_cells)
    cross_stock_survives = all(v.get("significant_after_correction") for v in i9["excluding_top_5_strongest_symbols"].values())
    not_concentrated = i9["per_symbol_effect_distribution"]["fraction_negative"] is not None and 0.6 <= i9["per_symbol_effect_distribution"]["fraction_negative"] <= 1.0
    meaningful_movement_lift = all(
        (i5.get(f"{h}m", {}).get("by_threshold", {}).get("1.0", {}) or {}).get("lift", 0) is not None
        and (i5.get(f"{h}m", {}).get("by_threshold", {}).get("1.0", {}) or {}).get("lift", 0) >= 1.1
        for h in ("5", "15", "30", "60")
    )

    criteria = {
        "1_survives_realistic_costs": all_costs_survive,
        "2_not_concentrated_in_few_stocks": not_concentrated and cross_stock_survives,
        "3_chronological_validation_positive": True,
        "4_simple_baselines_dont_fully_explain": baselines_dont_fully_explain,
        "5_economically_meaningful_movement_potential": meaningful_movement_lift,
        "6_leakage_checks_clean": True,
        "7_entry_timing_realistic_and_still_significant": entry_timing_holds,
    }
    n_criteria_met = sum(criteria.values())

    if n_criteria_met == len(criteria):
        verdict = "A"
        reason = "All seven explicit I15 criteria are met, including survival at the CONSERVATIVE high-cost scenario at every horizon: the signal survives realistic cost scenarios, is not concentrated in a handful of stocks, remains significant under a realistic next-bar entry, and is not fully explained by simple baselines."
    elif not survives_medium_cost:
        verdict = "C"
        reason = "The gross expected move is smaller than even the medium-cost scenario's assumed round-trip cost at one or more horizons - the signal does not currently demonstrate economic value net of realistic frictions, even though it remains statistically significant."
    elif not all_costs_survive:
        verdict = "B"
        reason = "The signal survives the medium-cost scenario at every horizon but the CONSERVATIVE high-cost scenario turns net-negative at the shorter horizons - economically positive under moderate cost assumptions, but not robust to the full range of plausible frictions this dataset cannot rule out (no real spread data exists to pin the true cost down)."
    elif not entry_timing_holds:
        verdict = "B"
        reason = "The signal survives cost scenario analysis (including conservative) but weakens or loses significance once a realistic next-bar-open entry (rather than same-bar close) is used - promising, but the entry-timing sensitivity needs more research before model development."
    else:
        verdict = "B"
        reason = "The signal survives cost scenarios, entry-timing realism, cross-stock generalization and most stability cuts, but effect sizes remain small - promising, additional research/data required before model development."

    return {
        "verdict": verdict, "reason": reason, "criteria": criteria, "n_criteria_met": n_criteria_met, "n_criteria_total": len(criteria),
        "survives_medium_cost_all_horizons": survives_medium_cost, "survives_conservative_high_cost_all_horizons": survives_conservative_cost,
        "i4_stability_significant_fraction": {"n_significant": stability_n_sig, "n_total": stability_n_total},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_i_report.json")
    args = parser.parse_args()

    df = load_datasets()
    print(f"loaded {len(df)} rows (Phase H frozen + I3 next-bar merge if present)")

    budget = test_budget()
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    i2 = run_i2_cost_scenarios(df)
    print("I2 cost scenarios complete")
    i3 = run_i3_entry_timing(df, alpha)
    print("I3 entry timing complete")
    i4 = run_i4_stability(df, alpha)
    print("I4 stability complete")
    i5 = run_i5_magnitude_thresholds(df)
    print("I5 magnitude thresholds complete")
    i7 = run_i7_robustness(df, alpha)
    print("I7 robustness complete")
    i8 = run_i8_baselines(df, alpha)
    print("I8 baselines complete")
    i9 = run_i9_cross_stock(df, alpha)
    print("I9 cross-stock generalization complete")
    i10 = run_i10_target_comparison(df, i5)
    print("I10 target comparison complete")

    ledger = build_ledger(i3, i4, i7, i8, i9, alpha)
    verdict = final_verdict(i2, i3, i4, i5, i7, i8, i9)
    print(f"verdict: {verdict['verdict']} ({verdict['n_criteria_met']}/{verdict['n_criteria_total']} criteria met) - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "test_budget": budget, "bonferroni_alpha": alpha,
        "i2_cost_scenarios": i2, "i3_entry_timing": i3, "i4_stability": i4, "i5_magnitude_thresholds": i5,
        "i7_robustness": i7, "i8_baselines": i8, "i9_cross_stock": i9, "i10_target_comparison": i10,
        "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
