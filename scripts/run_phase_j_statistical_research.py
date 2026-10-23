"""Phase J, Step 3: mover-detection hypothesis testing against the FROZEN target from Step 1
(mover_target_definition.json - T3, per-stock 90th percentile of future |return|), using the FROZEN
Phase H dataset plus the new mover_mechanics dataset from Step 2. Nothing from Phase G/H/I is
modified; `relative_strength_vs_universe` appears here only as an existing baseline to control for,
never as the feature under test.

STATISTICAL METHOD FOR A BINARY TARGET: Phase G/H/I's `_quantile_bucket_test` compares a bucket's
MEAN of a CONTINUOUS value against zero. The mover target is binary (0/1), and the natural question
is whether a bucket's mover RATE differs from the population baseline rate - not from zero (a raw
0/1 series essentially always has a nonzero mean, so testing against zero is not a meaningful test
here). `_mover_rate_test` below reuses the exact same Newey-West HAC machinery
(`backend/app/research/statistics.py`, unchanged) on the CENTERED series (flag - baseline_rate), so
testing H0: mean(centered)=0 is exactly H0: bucket rate = baseline rate.

PRE-REGISTERED TEST BUDGET (I6/H8 discipline, continued): every family, feature list, and horizon
list below is fixed before this script runs against real data.
  FAMILY A - standalone mover association: 15 candidate features (6 existing candle-structure
    columns tested against a mover target for the first time, 5 NEW mover_mechanics columns, 4
    existing "abnormal activity" columns) x 4 horizons.
  FAMILY B - 9 pre-registered combinations (exactly the ones named in the phase instruction) x 2
    horizons (5m/15m - the horizons where early mover detection actually matters for the stated
    "early enough to act" objective).
  FAMILY C - incremental value: every Family A feature, re-tested within EACH half of a
    relative_strength_vs_universe median split, at h=15m - does the feature still show a mover-rate
    lift once the existing baseline signal is controlled for?
  FAMILY D - cross-stock generalization for the single strongest Family A feature (by |lift|,
    selected AFTER Family A runs - a disclosed, confirmatory, not pre-registered pick), x 4 horizons.
  FAMILY E - chronological 3-way OOS (reusing Phase H's FROZEN date partition, not re-derived) for
    the top 3 Family A features by |lift|, at h=15m - also confirmatory/post-hoc, disclosed as such,
    not counted as an independent pre-registered discovery.
ONE Bonferroni correction across Families A-D's fixed-in-advance test count. Family E is reported
with its own p-values but explicitly labeled confirmatory/exploratory in the research ledger, per
the phase's "distinguish exploratory from confirmatory" instruction - it does not consume a slot in
the pre-registered budget because which features it examines is itself chosen from Family A's result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.research.statistics import newey_west_mean_test
from scripts.run_phase_h_statistical_research import N_QUANTILE_BUCKETS, attach_pre_registered_groups, load_group_definitions
from scripts.run_phase_j_mover_target_design import MIN_PRIOR_OBSERVATIONS_FOR_T3, compute_t3_expanding_percentile

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
MOVER_MECHANICS_DATASET = ROOT / "data" / "derived" / "phase_j_mover_detection" / "mover_mechanics_dataset.parquet"
MOVER_TARGET_DEFINITION = ROOT / "data" / "derived" / "phase_j_mover_detection" / "mover_target_definition.json"
OUT_DIR = ROOT / "data" / "derived" / "phase_j_mover_detection"

HORIZONS = (5, 15, 30, 60)
COMBINATION_HORIZONS = (5, 15)
MOVER_TARGET_PERCENTILE = 0.90

CANDLE_FEATURES = ["candle_body_ratio", "candle_upper_wick_ratio", "candle_lower_wick_ratio", "consecutive_same_direction_count", "is_bullish_engulfing", "is_bearish_engulfing"]
MOVER_MECHANICS_FEATURES = ["compression_then_expansion", "recent_breakout_up_opening_range", "recent_breakout_down_opening_range", "level_touch_count_support", "level_touch_count_resistance"]
ABNORMAL_ACTIVITY_FEATURES = ["relative_volume_20", "volume_vs_tod_baseline", "volatility_expansion_ratio", "realized_vol_vs_tod_baseline"]
FAMILY_A_FEATURES = CANDLE_FEATURES + MOVER_MECHANICS_FEATURES + ABNORMAL_ACTIVITY_FEATURES
BINARY_FEATURES = {"is_bullish_engulfing", "is_bearish_engulfing", "compression_then_expansion", "recent_breakout_up_opening_range", "recent_breakout_down_opening_range"}

COMBINATIONS = [
    ("compression_x_volume_expansion", "compression_then_expansion", "relative_volume_20"),
    ("range_expansion_x_volume_expansion", "range_expansion_20", "relative_volume_20"),
    ("breakout_x_relative_volume", "recent_breakout_up_opening_range", "relative_volume_20"),
    ("breakout_x_vwap_position", "recent_breakout_up_opening_range", "distance_from_vwap"),
    ("sr_proximity_x_abnormal_volume", "sr_proximity", "relative_volume_20"),
    ("relative_strength_x_volatility_expansion", "relative_strength_vs_universe", "volatility_expansion_ratio"),
    ("acceleration_x_participation", "momentum_acceleration_10", "relative_volume_20"),
    ("abnormal_vol_x_abnormal_volume_tod", "realized_vol_vs_tod_baseline", "volume_vs_tod_baseline"),
    ("market_relative_x_volume_abnormality", "relative_strength_vs_universe", "relative_volume_20"),
]


def test_budget() -> dict[str, int]:
    n_binary = len(BINARY_FEATURES & set(FAMILY_A_FEATURES))
    n_continuous = len(FAMILY_A_FEATURES) - n_binary
    return {
        "A_standalone": (n_continuous * 2 + n_binary * 1) * len(HORIZONS),
        "B_combinations": len(COMBINATIONS) * len(COMBINATION_HORIZONS),
        "C_incremental_value": len(FAMILY_A_FEATURES) * 2,
        "D_cross_stock": len(HORIZONS),
    }


def load_datasets() -> pd.DataFrame:
    frozen = json.loads(MOVER_TARGET_DEFINITION.read_text(encoding="utf-8"))
    if frozen["status"] != "FROZEN" or frozen["definition"] != "T3_empirical_per_stock_percentile" or frozen["threshold"] != MOVER_TARGET_PERCENTILE:
        raise RuntimeError(f"mover_target_definition.json does not match the expected frozen target - refusing to silently proceed with a different definition: {frozen}")

    df = pd.read_parquet(PHASE_H_DATASET)
    group_defs = load_group_definitions()
    df = attach_pre_registered_groups(df, group_defs)

    mm = pd.read_parquet(MOVER_MECHANICS_DATASET)
    df = df.merge(mm, on=["security_id", "bar_start_utc"], how="left", suffixes=("", "_mm_dup"))

    for h in HORIZONS:
        df[f"mover_flag_{h}m"] = compute_t3_expanding_percentile(df, horizon=h, percentile=MOVER_TARGET_PERCENTILE, min_prior_observations=MIN_PRIOR_OBSERVATIONS_FOR_T3)

    df["sr_proximity"] = df[["distance_to_nearest_support", "distance_to_nearest_resistance"]].abs().min(axis=1, skipna=True)
    return df


def coverage_report(df: pd.DataFrame) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    periods = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}

    report: dict[str, Any] = {"overall": {}, "by_symbol_summary": {}, "by_period": {}}
    for h in HORIZONS:
        col = f"mover_flag_{h}m"
        report["overall"][f"{h}m"] = {"coverage": float(df[col].notna().mean()), "n_valid": int(df[col].notna().sum()), "n_total": len(df)}

        by_symbol = df.groupby("symbol")[col].apply(lambda s: s.notna().mean())
        report["by_symbol_summary"][f"{h}m"] = {
            "median_symbol_coverage": float(by_symbol.median()), "min_symbol_coverage": float(by_symbol.min()),
            "max_symbol_coverage": float(by_symbol.max()), "n_symbols_with_zero_coverage": int((by_symbol == 0).sum()),
            "symbols_with_zero_coverage": sorted(by_symbol[by_symbol == 0].index.tolist()),
        }

        report["by_period"][f"{h}m"] = {}
        for period_name, period_dates in periods.items():
            period_df = df[df["trading_date"].isin(period_dates)]
            report["by_period"][f"{h}m"][period_name] = {"coverage": float(period_df[col].notna().mean()) if len(period_df) else None, "n_rows": len(period_df)}
    return report


def _finalize_rate_test(ordered_flags: list[float], baseline_rate: float, horizon: int, alpha: float, n_total_valid: int) -> dict[str, Any]:
    if len(ordered_flags) < 10:
        return {"skipped_reason": f"insufficient rows ({len(ordered_flags)})"}
    centered = [f - baseline_rate for f in ordered_flags]
    max_lag = min(horizon - 1, len(centered) - 1)
    result = newey_west_mean_test(centered, max_lag=max_lag)
    bucket_rate = baseline_rate + result.mean
    return {
        "n_rows": len(ordered_flags), "bucket_rate": bucket_rate, "baseline_rate": baseline_rate,
        "lift": bucket_rate / baseline_rate if baseline_rate else None,
        "p_value": result.p_value, "confidence_interval": tuple(v + baseline_rate for v in result.confidence_interval),
        "significant_after_correction": bool(result.p_value < alpha),
    }


def _rate_test_for_subset(df: pd.DataFrame, subset_mask: pd.Series, mover_col: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[subset_mask, ["symbol", "bar_start_utc", mover_col]].dropna(subset=[mover_col])
    if len(sub) < 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[mover_col].astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha, len(df))


def run_family_a(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for feature in FAMILY_A_FEATURES:
        for h in HORIZONS:
            mover_col = f"mover_flag_{h}m"
            baseline_rate = float(df[mover_col].mean())
            valid = df[[feature, mover_col]].dropna()
            if feature in BINARY_FEATURES:
                true_mask = df[feature] == True  # noqa: E712
                r = _rate_test_for_subset(df, true_mask, mover_col, h, alpha, baseline_rate)
                results.append({"feature": feature, "horizon_minutes": h, "bucket": "flag_true", "baseline_rate": baseline_rate, **r})
            else:
                if len(valid) < N_QUANTILE_BUCKETS * 10:
                    results.append({"feature": feature, "horizon_minutes": h, "bucket": "n/a", "skipped_reason": f"insufficient rows ({len(valid)})"})
                    continue
                bucketed = valid.assign(_b=pd.qcut(valid[feature], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
                if bucketed["_b"].nunique() < 2:
                    results.append({"feature": feature, "horizon_minutes": h, "bucket": "n/a", "skipped_reason": "degenerate feature"})
                    continue
                low_b, high_b = bucketed["_b"].min(), bucketed["_b"].max()
                low_mask = df.index.isin(bucketed[bucketed["_b"] == low_b].index)
                high_mask = df.index.isin(bucketed[bucketed["_b"] == high_b].index)
                r_low = _rate_test_for_subset(df, pd.Series(low_mask, index=df.index), mover_col, h, alpha, baseline_rate)
                r_high = _rate_test_for_subset(df, pd.Series(high_mask, index=df.index), mover_col, h, alpha, baseline_rate)
                results.append({"feature": feature, "horizon_minutes": h, "bucket": "lowest_quantile", "baseline_rate": baseline_rate, **r_low})
                results.append({"feature": feature, "horizon_minutes": h, "bucket": "highest_quantile", "baseline_rate": baseline_rate, **r_high})
    return results


def run_family_b(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for name, feat1, feat2 in COMBINATIONS:
        for h in COMBINATION_HORIZONS:
            mover_col = f"mover_flag_{h}m"
            baseline_rate = float(df[mover_col].mean())

            def _extreme_mask(col: str) -> pd.Series:
                if col in BINARY_FEATURES:
                    return df[col] == True  # noqa: E712
                med = df[col].median()
                return df[col] >= med

            corner_mask = _extreme_mask(feat1) & _extreme_mask(feat2)
            r = _rate_test_for_subset(df, corner_mask, mover_col, h, alpha, baseline_rate)
            results.append({"combination": name, "feature_1": feat1, "feature_2": feat2, "horizon_minutes": h, "n_in_corner": int(corner_mask.sum()), "baseline_rate": baseline_rate, **r})
    return results


def _family_a_direction_at_15m(family_a: list[dict[str, Any]]) -> dict[str, str]:
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for r in family_a:
        if r["horizon_minutes"] == 15 and "lift" in r and r["lift"] is not None:
            by_feature.setdefault(r["feature"], []).append(r)
    direction: dict[str, str] = {}
    for feature, rows in by_feature.items():
        best = max(rows, key=lambda r: abs(r["lift"] - 1.0))
        direction[feature] = best["bucket"]
    return direction


def run_family_c(df: pd.DataFrame, alpha: float, family_a: list[dict[str, Any]], features: list[str] = FAMILY_A_FEATURES) -> list[dict[str, Any]]:
    h = 15
    mover_col = f"mover_flag_{h}m"
    baseline_rate = float(df[mover_col].mean())
    rs_median = df["relative_strength_vs_universe"].median()
    direction_by_feature = _family_a_direction_at_15m(family_a)
    results = []
    for feature in features:
        direction = direction_by_feature.get(feature, "highest_quantile")
        for group_name, group_mask in (("below_median_rs", df["relative_strength_vs_universe"] < rs_median), ("at_or_above_median_rs", df["relative_strength_vs_universe"] >= rs_median)):
            sub_df = df[group_mask]
            valid = sub_df[[feature, mover_col]].dropna()
            if feature in BINARY_FEATURES:
                mask = (sub_df[feature] == True)  # noqa: E712
                mask = mask.reindex(df.index, fill_value=False)
                r = _rate_test_for_subset(df, mask, mover_col, h, alpha, baseline_rate)
            elif len(valid) >= N_QUANTILE_BUCKETS * 10 and valid[feature].nunique() > 1:
                quantile_mask = (valid[feature] >= valid[feature].quantile(0.8)) if direction == "highest_quantile" else (valid[feature] <= valid[feature].quantile(0.2))
                mask = pd.Series(False, index=df.index)
                mask.loc[valid[quantile_mask].index] = True
                r = _rate_test_for_subset(df, mask, mover_col, h, alpha, baseline_rate)
            else:
                r = {"skipped_reason": "insufficient rows or degenerate feature in this group"}
            results.append({"feature": feature, "direction_tested": direction, "controlled_group": group_name, "horizon_minutes": h, "baseline_rate": baseline_rate, **r})
    return results


def pick_strongest_family_a(family_a: list[dict[str, Any]], n: int = 1) -> list[dict[str, Any]]:
    scored = [r for r in family_a if "lift" in r and r["lift"] is not None]
    scored.sort(key=lambda r: abs(r["lift"] - 1.0), reverse=True)
    return scored[:n]


def run_family_d(df: pd.DataFrame, strongest: dict[str, Any], alpha: float) -> dict[str, Any]:
    feature = strongest["feature"]
    results = {}
    for h in HORIZONS:
        mover_col = f"mover_flag_{h}m"
        baseline_rate = float(df[mover_col].mean())
        if feature in BINARY_FEATURES:
            mask = df[feature] == True  # noqa: E712
        else:
            valid = df[feature].dropna()
            thr = valid.quantile(0.8) if strongest["bucket"] == "highest_quantile" else valid.quantile(0.2)
            mask = (df[feature] >= thr) if strongest["bucket"] == "highest_quantile" else (df[feature] <= thr)
        per_symbol = df.loc[mask & df[mover_col].notna()].groupby("symbol")[mover_col].mean()
        top_5 = per_symbol.abs().sort_values(ascending=False).head(5).index.tolist() if len(per_symbol) >= 5 else []
        remainder_mask = mask & ~df["symbol"].isin(top_5)
        results[f"{h}m"] = {"n_symbols_excluded": len(top_5), "excluded_symbols": top_5, **_rate_test_for_subset(df, remainder_mask, mover_col, h, alpha, baseline_rate)}
    return {"feature": feature, "bucket": strongest["bucket"], "by_horizon": results}


def run_family_e(df: pd.DataFrame, top_features: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}

    h = 15
    mover_col = f"mover_flag_{h}m"
    results = {}
    for r in top_features:
        feature = r["feature"]
        by_split = {}
        for split_name, split_dates in splits.items():
            split_df = df[df["trading_date"].isin(split_dates)]
            baseline_rate = float(split_df[mover_col].mean())
            if feature in BINARY_FEATURES:
                mask = split_df[feature] == True  # noqa: E712
            else:
                valid = split_df[feature].dropna()
                if len(valid) < N_QUANTILE_BUCKETS * 10 or valid.nunique() < 2:
                    by_split[split_name] = {"skipped_reason": "insufficient rows or degenerate feature in this split"}
                    continue
                thr = valid.quantile(0.8) if r["bucket"] == "highest_quantile" else valid.quantile(0.2)
                mask = (split_df[feature] >= thr) if r["bucket"] == "highest_quantile" else (split_df[feature] <= thr)
            by_split[split_name] = _rate_test_for_subset(split_df, mask, mover_col, h, alpha=0.05, baseline_rate=baseline_rate)
        results[feature] = {"bucket": r["bucket"], "by_split": by_split}
    return {"split_dates": {k: {"n": len(v), "first": str(min(v)) if v else None, "last": str(max(v)) if v else None} for k, v in splits.items()}, "results": results}


def build_ledger(family_a, family_b, family_c, alpha) -> list[dict[str, Any]]:
    ledger = []
    for r in family_a:
        ledger.append({"family": "A_standalone", "pre_registration": "pre-registered", "hypothesis": f"{r['feature']} ({r['bucket']}) associates with the {r['horizon_minutes']}m mover target",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT SIGNIFICANT"),
                        "lift": r.get("lift"), "n_rows": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_b:
        ledger.append({"family": "B_combination", "pre_registration": "pre-registered", "hypothesis": f"{r['combination']} ({r['feature_1']} x {r['feature_2']}) associates with the {r['horizon_minutes']}m mover target",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT SIGNIFICANT"),
                        "lift": r.get("lift"), "n_rows": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_c:
        ledger.append({"family": "C_incremental_value", "pre_registration": "pre-registered", "hypothesis": f"{r['feature']} adds mover-detection value beyond relative_strength_vs_universe ({r['controlled_group']})",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT SIGNIFICANT"),
                        "lift": r.get("lift"), "n_rows": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(family_a, family_b, family_c, family_d, family_e) -> dict[str, Any]:
    a_significant = [r for r in family_a if r.get("significant_after_correction") and r.get("lift", 1.0) > 1.0]
    b_significant = [r for r in family_b if r.get("significant_after_correction") and r.get("lift", 1.0) > 1.0]
    c_significant = [r for r in family_c if r.get("significant_after_correction") and r.get("lift", 1.0) > 1.0]
    d_all_survive = all(v.get("significant_after_correction") for v in family_d["by_horizon"].values() if "skipped_reason" not in v)

    e_holds = False
    if family_e["results"]:
        for feature_result in family_e["results"].values():
            splits_sig = [v.get("significant_after_correction") for v in feature_result["by_split"].values() if "skipped_reason" not in v]
            if splits_sig and all(splits_sig):
                e_holds = True
                break

    if not a_significant:
        verdict = "C"
        reason = "No candidate feature in Family A shows a statistically significant, positive mover-rate lift after correction - the mover-detection mechanisms tested this phase are not supported by the evidence."
    elif not c_significant:
        verdict = "C"
        reason = "Some Family A features are significant standalone, but none retain a significant mover-rate lift once controlled for the existing relative_strength_vs_universe baseline in BOTH halves of the split - no candidate demonstrated genuine incremental value."
    elif not d_all_survive:
        verdict = "B"
        reason = "At least one feature adds incremental mover-detection value, but the strongest candidate's effect is not fully robust across a cross-stock generalization check - promising, needs broader validation."
    elif not e_holds:
        verdict = "B"
        reason = "Incremental value is demonstrated and cross-stock generalization holds, but the (confirmatory, post-hoc) chronological out-of-sample check does not hold consistently across all three splits for any top feature - promising, more validation needed before proceeding."
    else:
        verdict = "B"
        reason = "At least one mover-detection mechanism shows a statistically significant, incremental, cross-stock-general, chronologically-replicating lift over the existing baseline - genuinely promising evidence, though (consistent with the whole project's discipline) this alone is not yet economic/tradability validation, which would need its own Phase-I-style cost analysis before any model-readiness claim beyond B."

    return {
        "verdict": verdict, "reason": reason,
        "n_family_a_significant_positive": len(a_significant), "n_family_a_total": len(family_a),
        "n_family_b_significant_positive": len(b_significant), "n_family_b_total": len(family_b),
        "n_family_c_significant_positive": len(c_significant), "n_family_c_total": len(family_c),
        "family_d_all_survive": d_all_survive, "family_e_holds_for_at_least_one_feature": e_holds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_j_report_corrected_t3.json")
    args = parser.parse_args()

    df = load_datasets()
    print(f"loaded {len(df)} rows, {len(df.columns)} columns (Phase H frozen + Phase J mover-mechanics + CORRECTED PIT-safe mover target)")

    coverage = coverage_report(df)
    print(f"T3 coverage (corrected): overall 15m = {coverage['overall']['15m']['coverage']:.3f}")

    budget = test_budget()
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    family_a = run_family_a(df, alpha)
    print(f"Family A complete: {len(family_a)} rows")
    family_b = run_family_b(df, alpha)
    print(f"Family B complete: {len(family_b)} rows")
    family_c = run_family_c(df, alpha, family_a)
    print(f"Family C complete: {len(family_c)} rows")

    strongest_1 = pick_strongest_family_a(family_a, n=1)
    family_d = run_family_d(df, strongest_1[0], alpha) if strongest_1 else {"feature": None, "by_horizon": {}}
    print("Family D complete")

    strongest_3 = pick_strongest_family_a(family_a, n=3)
    family_e = run_family_e(df, strongest_3)
    print("Family E complete")

    ledger = build_ledger(family_a, family_b, family_c, alpha)
    verdict = final_verdict(family_a, family_b, family_c, family_d, family_e)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "t3_correction_note": "Computed against the PIT-safe expanding-percentile T3 (Phase K correction). "
                               "The original phase_j_report.json (leaky whole-dataset T3) is preserved as "
                               "phase_j_report_PRE_PHASE_K_INVALID_T3_LEAK.json and must not be cited.",
        "t3_coverage_report": coverage,
        "test_budget": budget, "bonferroni_alpha": alpha,
        "family_a_standalone": family_a, "family_b_combinations": family_b, "family_c_incremental_value": family_c,
        "family_d_cross_stock": family_d, "family_e_chronological_oos_confirmatory": family_e,
        "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger_corrected_t3.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
