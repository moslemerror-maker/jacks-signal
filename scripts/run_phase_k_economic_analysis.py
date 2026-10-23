"""Phase K, Parts 2-10: economic/tradability validation of the corrected (PIT-safe) Phase J
mover-detection evidence. Built entirely on:
  - the FROZEN Phase H dataset (features, groups) - read-only, unmodified
  - the FROZEN, CORRECTED Phase J mover-mechanics + mover_flag (compute_t3_expanding_percentile
    ONLY - the deprecated, leaky compute_t3 is never imported here)
  - the new Phase K entry_and_path dataset (realistic next-bar-open entry, gross returns, MFE/MAE)
Nothing in Phase G/H/I/J is modified. `relative_strength_vs_universe` is one of five frozen
mechanisms under test here, never assumed to be the primary predictor.

PART 2 - THE FIVE FROZEN MECHANISMS (fixed before any economic number was computed):
  M1 relative_strength_vs_universe  - DIRECTIONAL (existing baseline): lowest quintile -> long,
     highest quintile -> short (the established mean-reversion sign from Phase G/H/I).
  M2 recent_breakout_up_opening_range (flag=True) - DIRECTIONAL: long.
  M3 compression_then_expansion (flag=True) - MAGNITUDE-ONLY: Phase J never established a
     direction for this mechanism, so no directional trade is assumed - see "magnitude-only
     economics" below.
  M4 level_touch_count_resistance (lowest quantile = freshest level) - MAGNITUDE-ONLY, same caveat.
  M5 breakout_x_relative_volume (recent_breakout_up_opening_range=True AND relative_volume_20 >=
     median) - DIRECTIONAL: long - Phase J's strongest pre-registered combination.

DIRECTIONAL vs MAGNITUDE-ONLY ECONOMICS, stated explicitly (not glossed over): for a DIRECTIONAL
mechanism, "gross move" is the SIGNED next-bar-open-to-horizon-close return in the mechanism's own
implied direction (a real, executable long or short). For a MAGNITUDE-ONLY mechanism, no directional
edge was ever demonstrated in Phase J - reporting a signed trade would silently assume a direction
nobody established. Its "gross move" is instead the REALIZED, ENDPOINT |gross_return| (never the
best-case MFE, which would assume the trader could exit at the exact extremum) - explicitly reported
as an UPPER-BOUND, PERFECT-DIRECTION-KNOWLEDGE economic estimate, never as a claim of an executable
strategy.

PART 3 - ENTRY TIMING: every economic number uses `entry_and_path_dataset.parquet`'s next-bar-open
entry (see run_phase_k_entry_and_path.py) - never same-bar close.

PART 4 - COSTS: three round-trip scenarios, exactly as specified - 20 / 40 / 60 bps. Medium (40bps)
is the reference scenario for the pre-registered significance tests (Family A); all three are always
reported side by side, never only the one that looks best.

PRE-REGISTERED TEST BUDGET (fixed before any economic result was examined):
  Family A (headline economic significance, net return vs 0 at 40bps): 5 mechanisms x 4 horizons = 20
  Family B (incremental value beyond relative_strength_vs_universe, M2-M5 only): 4 x 2 halves = 8
  Family C (cross-stock generalization, excluding the 5 strongest symbols): 5 x 4 horizons = 20
  TOTAL = 48, one Bonferroni correction.

CONFIRMATORY / DESCRIPTIVE, explicitly NOT part of the corrected budget (labeled throughout):
  cost-scenario sensitivity (arithmetic on already-computed numbers), liquidity/tier/price/
  volatility group breakdown, chronological train/validation/held-out replication for the two
  strongest Family-A mechanisms (post-hoc selected, exactly Phase J's Family-E precedent), path/
  MFE-MAE descriptive statistics, and the baseline comparison table.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.research.statistics import newey_west_mean_test
from scripts.run_phase_h_statistical_research import attach_pre_registered_groups, load_group_definitions
from scripts.run_phase_j_mover_target_design import MIN_PRIOR_OBSERVATIONS_FOR_T3, compute_t3_expanding_percentile

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
MOVER_MECHANICS_DATASET = ROOT / "data" / "derived" / "phase_j_mover_detection" / "mover_mechanics_dataset.parquet"
ENTRY_PATH_DATASET = ROOT / "data" / "derived" / "phase_k_economic_validation" / "entry_and_path_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_k_economic_validation"

HORIZONS = (5, 15, 30, 60)
MOVER_TARGET_PERCENTILE = 0.90
COST_SCENARIOS_BPS = {"low_20bps": 20.0, "medium_40bps": 40.0, "high_60bps": 60.0}
REFERENCE_COST_SCENARIO = "medium_40bps"

MECHANISMS = ["M1_relative_strength_vs_universe", "M2_recent_breakout_up", "M3_compression_then_expansion", "M4_level_touch_count_resistance", "M5_breakout_x_relative_volume"]
DIRECTIONAL_MECHANISMS = {"M1_relative_strength_vs_universe", "M2_recent_breakout_up", "M5_breakout_x_relative_volume"}
MAGNITUDE_ONLY_MECHANISMS = {"M3_compression_then_expansion", "M4_level_touch_count_resistance"}


def load_datasets() -> pd.DataFrame:
    df = pd.read_parquet(PHASE_H_DATASET)
    group_defs = load_group_definitions()
    df = attach_pre_registered_groups(df, group_defs)

    mm = pd.read_parquet(MOVER_MECHANICS_DATASET)
    df = df.merge(mm, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_mm_dup"))

    ep = pd.read_parquet(ENTRY_PATH_DATASET)
    df = df.merge(ep, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_ep_dup"))

    for h in HORIZONS:
        df[f"mover_flag_{h}m"] = compute_t3_expanding_percentile(df, horizon=h, percentile=MOVER_TARGET_PERCENTILE, min_prior_observations=MIN_PRIOR_OBSERVATIONS_FOR_T3)
    return df


def _mechanism_mask_and_sign(df: pd.DataFrame, mechanism: str, horizon: int) -> tuple[pd.Series, pd.Series]:
    if mechanism == "M1_relative_strength_vs_universe":
        valid = df["relative_strength_vs_universe"].dropna()
        low_thr, high_thr = valid.quantile(0.2), valid.quantile(0.8)
        long_mask = df["relative_strength_vs_universe"] <= low_thr
        short_mask = df["relative_strength_vs_universe"] >= high_thr
        mask = long_mask | short_mask
        sign = pd.Series(0.0, index=df.index)
        sign.loc[long_mask] = 1.0
        sign.loc[short_mask] = -1.0
        return mask, sign
    if mechanism == "M2_recent_breakout_up":
        mask = df["recent_breakout_up_opening_range"] == True  # noqa: E712
        return mask, pd.Series(1.0, index=df.index)
    if mechanism == "M5_breakout_x_relative_volume":
        med = df["relative_volume_20"].median()
        mask = (df["recent_breakout_up_opening_range"] == True) & (df["relative_volume_20"] >= med)  # noqa: E712
        return mask, pd.Series(1.0, index=df.index)
    if mechanism == "M3_compression_then_expansion":
        return df["compression_then_expansion"] == True, pd.Series(float("nan"), index=df.index)  # noqa: E712
    if mechanism == "M4_level_touch_count_resistance":
        valid = df["level_touch_count_resistance"].dropna()
        thr = valid.quantile(0.2)
        return df["level_touch_count_resistance"] <= thr, pd.Series(float("nan"), index=df.index)
    raise ValueError(f"unknown mechanism {mechanism!r}")


def _oriented_returns(df: pd.DataFrame, mechanism: str, horizon: int) -> pd.Series:
    mask, sign = _mechanism_mask_and_sign(df, mechanism, horizon)
    ret_col = f"gross_return_{horizon}m"
    sub = df.loc[mask, [ret_col, "symbol", "bar_start_utc"]].dropna(subset=[ret_col])
    if mechanism in MAGNITUDE_ONLY_MECHANISMS:
        return sub.assign(_oriented=sub[ret_col].abs())
    sub = sub.assign(_oriented=sub[ret_col] * sign.loc[sub.index])
    return sub


def _cost_adjusted_stats(oriented: pd.DataFrame, alpha: float, horizon: int) -> dict[str, Any]:
    n = len(oriented)
    if n < 30:
        return {"skipped_reason": f"insufficient rows ({n})"}
    gross = oriented["_oriented"]
    ordered = oriented.sort_values(["symbol", "bar_start_utc"])["_oriented"]

    scenarios = {}
    for name, bps in COST_SCENARIOS_BPS.items():
        net = gross - (bps / 10_000.0)
        scenarios[name] = {
            "cost_bps": bps, "mean_net": float(net.mean()), "median_net": float(net.median()),
            "hit_rate": float((net > 0).mean()), "fraction_surviving_cost": float((net > 0).mean()),
        }

    ref_bps = COST_SCENARIOS_BPS[REFERENCE_COST_SCENARIO]
    net_ordered = (ordered - ref_bps / 10_000.0).tolist()
    max_lag = min(horizon - 1, len(net_ordered) - 1)
    nw = newey_west_mean_test(net_ordered, max_lag=max_lag)

    return {
        "n": n, "gross_mean": float(gross.mean()), "gross_median": float(gross.median()), "gross_stdev": float(gross.std()),
        "gross_q25": float(gross.quantile(0.25)), "gross_q75": float(gross.quantile(0.75)),
        "cost_scenarios": scenarios,
        "reference_scenario": REFERENCE_COST_SCENARIO,
        "newey_west_net_mean": nw.mean, "newey_west_p_value": nw.p_value,
        "significant_after_correction": bool(nw.p_value < alpha) and nw.mean > 0,
    }


def test_budget() -> dict[str, int]:
    return {
        "A_headline_economic": len(MECHANISMS) * len(HORIZONS),
        "B_incremental_value": (len(MECHANISMS) - 1) * 2,
        "C_cross_stock": len(MECHANISMS) * len(HORIZONS),
    }


def run_family_a(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for mechanism in MECHANISMS:
        for h in HORIZONS:
            oriented = _oriented_returns(df, mechanism, h)
            stats = _cost_adjusted_stats(oriented, alpha, h)
            results.append({"mechanism": mechanism, "horizon_minutes": h, "mechanism_type": "directional" if mechanism in DIRECTIONAL_MECHANISMS else "magnitude_only", **stats})
    return results


def run_family_b(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    h = 15
    rs_median = df["relative_strength_vs_universe"].median()
    results = []
    for mechanism in MECHANISMS:
        if mechanism == "M1_relative_strength_vs_universe":
            continue
        for group_name, group_mask in (("below_median_rs", df["relative_strength_vs_universe"] < rs_median), ("at_or_above_median_rs", df["relative_strength_vs_universe"] >= rs_median)):
            sub_df = df[group_mask]
            oriented = _oriented_returns(sub_df, mechanism, h)
            stats = _cost_adjusted_stats(oriented, alpha, h)
            results.append({"mechanism": mechanism, "controlled_group": group_name, "horizon_minutes": h, **stats})
    return results


def run_family_c(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for mechanism in MECHANISMS:
        h_for_selection = 15
        mask, _ = _mechanism_mask_and_sign(df, mechanism, h_for_selection)
        ret_col = f"gross_return_{h_for_selection}m"
        per_symbol = df.loc[mask, ["symbol", ret_col]].dropna().groupby("symbol")[ret_col].apply(lambda s: s.abs().mean())
        top_5 = per_symbol.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol) >= 5 else []
        remainder_df = df[~df["symbol"].isin(top_5)]
        for h in HORIZONS:
            oriented = _oriented_returns(remainder_df, mechanism, h)
            stats = _cost_adjusted_stats(oriented, alpha, h)
            results.append({"mechanism": mechanism, "horizon_minutes": h, "n_symbols_excluded": len(top_5), "excluded_symbols": top_5, **stats})
    return results


def run_liquidity_group_breakdown(df: pd.DataFrame) -> dict[str, Any]:
    h = 15
    dims = ["tier", "price_group", "liquidity_group", "volatility_group"]
    results: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        results[mechanism] = {}
        for dim in dims:
            results[mechanism][dim] = {}
            for val in sorted(df[dim].dropna().unique()):
                sub_df = df[df[dim] == val]
                oriented = _oriented_returns(sub_df, mechanism, h)
                if len(oriented) < 30:
                    results[mechanism][dim][str(val)] = {"skipped_reason": f"insufficient rows ({len(oriented)})"}
                    continue
                gross = oriented["_oriented"]
                net = gross - COST_SCENARIOS_BPS[REFERENCE_COST_SCENARIO] / 10_000.0
                results[mechanism][dim][str(val)] = {"n": len(oriented), "mean_net_40bps": float(net.mean()), "hit_rate_40bps": float((net > 0).mean())}
    return results


def run_path_analysis(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        results[mechanism] = {}
        for h in HORIZONS:
            mask, _ = _mechanism_mask_and_sign(df, mechanism, h)
            sub = df.loc[mask, [f"path_mfe_{h}m", f"path_mae_{h}m", f"bars_to_mfe_{h}m", f"bars_to_mae_{h}m"]].dropna()
            if len(sub) < 30:
                results[mechanism][f"{h}m"] = {"skipped_reason": f"insufficient rows ({len(sub)})"}
                continue
            mfe, mae = sub[f"path_mfe_{h}m"], sub[f"path_mae_{h}m"]
            bmfe, bmae = sub[f"bars_to_mfe_{h}m"], sub[f"bars_to_mae_{h}m"]
            results[mechanism][f"{h}m"] = {
                "n": len(sub), "mean_mfe": float(mfe.mean()), "mean_mae": float(mae.mean()),
                "median_mfe": float(mfe.median()), "median_mae": float(mae.median()),
                "fraction_favorable_before_adverse": float((bmfe < bmae).mean()),
                "fraction_mfe_at_least_2x_mae": float((mfe >= 2 * mae).mean()),
            }
    return results


def run_baselines(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    h = 15
    results: dict[str, Any] = {}

    ret_col = f"gross_return_{h}m"
    no_signal = df[[ret_col]].dropna()
    results["no_signal_baseline"] = {"n": len(no_signal), "mean_gross": float(no_signal[ret_col].mean()), "note": "unconditional next-bar-open trade, no signal at all"}

    valid = df["return_5m"].dropna()
    low_thr, high_thr = valid.quantile(0.2), valid.quantile(0.8)
    long_mask, short_mask = df["return_5m"] <= low_thr, df["return_5m"] >= high_thr
    sub = df.loc[long_mask | short_mask, [ret_col, "symbol", "bar_start_utc"]].dropna(subset=[ret_col]).copy()
    sub["_oriented"] = sub[ret_col].where(df["return_5m"] <= low_thr, -sub[ret_col])
    results["return_5m_baseline"] = _cost_adjusted_stats(sub, alpha=1.0, horizon=h)

    return results


def run_chronological_confirmatory(df: pd.DataFrame, top_mechanisms: list[str]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}

    results: dict[str, Any] = {"split_dates": {k: {"n": len(v), "first": str(min(v)) if v else None, "last": str(max(v)) if v else None} for k, v in splits.items()}, "by_mechanism": {}}
    for mechanism in top_mechanisms:
        results["by_mechanism"][mechanism] = {}
        for h in HORIZONS:
            results["by_mechanism"][mechanism][f"{h}m"] = {}
            for split_name, split_dates in splits.items():
                split_df = df[df["trading_date"].isin(split_dates)]
                oriented = _oriented_returns(split_df, mechanism, h)
                results["by_mechanism"][mechanism][f"{h}m"][split_name] = _cost_adjusted_stats(oriented, alpha=1.0, horizon=h)
    return results


def pick_top_mechanisms_by_family_a(family_a: list[dict[str, Any]], n: int = 2) -> list[str]:
    scored = [r for r in family_a if r.get("significant_after_correction") and "gross_mean" in r]
    scored.sort(key=lambda r: abs(r["gross_mean"]), reverse=True)
    seen, out = set(), []
    for r in scored:
        if r["mechanism"] not in seen:
            seen.add(r["mechanism"])
            out.append(r["mechanism"])
        if len(out) >= n:
            break
    return out


def build_ledger(family_a, family_b, family_c, alpha: float) -> list[dict[str, Any]]:
    ledger = []
    for r in family_a:
        ledger.append({"family": "A_headline_economic", "pre_registration": "pre-registered", "hypothesis": f"{r['mechanism']} shows positive net return (40bps) at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT_POSITIVE" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "gross_mean": r.get("gross_mean"), "n": r.get("n"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_b:
        ledger.append({"family": "B_incremental_value", "pre_registration": "pre-registered", "hypothesis": f"{r['mechanism']} adds economic value beyond relative_strength_vs_universe ({r['controlled_group']})",
                        "result": "SIGNIFICANT_POSITIVE" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "gross_mean": r.get("gross_mean"), "n": r.get("n"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_c:
        ledger.append({"family": "C_cross_stock", "pre_registration": "pre-registered", "hypothesis": f"{r['mechanism']} survives excluding its 5 strongest symbols at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT_POSITIVE" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "gross_mean": r.get("gross_mean"), "n": r.get("n"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(family_a, family_b, family_c, chrono) -> dict[str, Any]:
    a_sig = [r for r in family_a if r.get("significant_after_correction")]
    survives_high_cost = all(r["cost_scenarios"]["high_60bps"]["mean_net"] > 0 for r in a_sig) if a_sig else False
    b_sig = [r for r in family_b if r.get("significant_after_correction")]
    c_sig = [r for r in family_c if r.get("significant_after_correction")]
    c_total_nonskipped = [r for r in family_c if "skipped_reason" not in r]

    chrono_holds = False
    for mech_results in chrono.get("by_mechanism", {}).values():
        for h_results in mech_results.values():
            valid_splits = [v for v in h_results.values() if "skipped_reason" not in v]
            splits_positive = [v.get("newey_west_net_mean", -1) > 0 for v in valid_splits]
            if splits_positive and all(splits_positive):
                chrono_holds = True

    criteria = {
        "1_headline_economic_significance": bool(a_sig),
        "2_survives_conservative_60bps_cost": survives_high_cost,
        "3_incremental_value_beyond_relative_strength": bool(b_sig),
        "4_cross_stock_generalization": bool(c_sig) and len(c_sig) >= max(1, len(c_total_nonskipped) // 2),
        "5_chronological_replication": chrono_holds,
    }
    n_met = sum(criteria.values())

    if n_met == len(criteria):
        verdict, reason = "A", "All five criteria met: at least one mechanism is economically significant net of the conservative 60bps cost scenario, adds value beyond the existing baseline, generalizes across stocks, and replicates chronologically."
    elif not a_sig:
        verdict, reason = "D", "No mechanism shows a statistically significant positive net return at the reference (40bps) cost scenario - the mover-detection evidence does not translate into economic value under any realistic execution assumption tested."
    elif not survives_high_cost:
        verdict, reason = "C", "At least one mechanism is significant at the reference cost, but none survive the conservative 60bps scenario - the statistical signal exists but does not survive realistic economic conditions."
    elif not b_sig:
        verdict, reason = "B", "Economically significant net of costs, but none of the new mechanisms demonstrated incremental value beyond the existing relative_strength_vs_universe baseline in both controlled-for halves - promising but a material research gap remains (may simply be re-discovering the known baseline)."
    else:
        verdict, reason = "B", "Economically significant, survives conservative costs, and adds incremental value, but cross-stock generalization or chronological replication is incomplete - promising, additional research required before proceeding further."

    return {"verdict": verdict, "reason": reason, "criteria": criteria, "n_criteria_met": n_met, "n_criteria_total": len(criteria)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_k_report.json")
    args = parser.parse_args()

    df = load_datasets()
    print(f"loaded {len(df)} rows, {len(df.columns)} columns (Phase H frozen + Phase J mover-mechanics + corrected T3 + Phase K entry/path)")

    budget = test_budget()
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    family_a = run_family_a(df, alpha)
    print(f"Family A complete: {len(family_a)} rows")
    family_b = run_family_b(df, alpha)
    print(f"Family B complete: {len(family_b)} rows")
    family_c = run_family_c(df, alpha)
    print(f"Family C complete: {len(family_c)} rows")

    liquidity_breakdown = run_liquidity_group_breakdown(df)
    print("liquidity/group breakdown complete")
    path_analysis = run_path_analysis(df)
    print("path analysis complete")
    baselines = run_baselines(df, alpha)
    print("baselines complete")

    top_mechanisms = pick_top_mechanisms_by_family_a(family_a, n=2)
    chrono = run_chronological_confirmatory(df, top_mechanisms)
    print(f"chronological confirmatory complete for {top_mechanisms}")

    ledger = build_ledger(family_a, family_b, family_c, alpha)
    verdict = final_verdict(family_a, family_b, family_c, chrono)
    print(f"verdict: {verdict['verdict']} ({verdict['n_criteria_met']}/{verdict['n_criteria_total']}) - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "test_budget": budget, "bonferroni_alpha": alpha, "cost_scenarios_bps": COST_SCENARIOS_BPS,
        "family_a_headline_economic": family_a, "family_b_incremental_value": family_b, "family_c_cross_stock": family_c,
        "liquidity_group_breakdown_descriptive": liquidity_breakdown, "path_analysis_descriptive": path_analysis,
        "baselines_descriptive": baselines, "chronological_confirmatory": chrono,
        "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
