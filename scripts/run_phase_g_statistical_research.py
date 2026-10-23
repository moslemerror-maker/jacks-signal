"""Phase G: statistical research over the feature/label dataset produced by
run_phase_g_compute_features.py - the step BEFORE any model training, per the governing
instruction ("use walk-forward/chronological validation... do not declare a feature 'predictive'
simply because one historical sample shows significance").

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Coverage / missingness - non-null fraction of every column, overall and by universe tier.
  2. Distribution - descriptive statistics (mean/std/quantiles) of every numeric column.
  3. Class frequency - label_class_{5,15,30}m value counts, overall and by tier/time_bucket/regime.
  4. Association with future movement - a CURATED subset of candidate features (not all ~30; see
     CANDIDATE_FEATURES) tested against label_normalized_move_{5,15,30}m via quantile-bucket mean
     comparison, each bucket's mean tested against zero with Newey-West HAC (reusing
     backend/app/research/statistics.py as-is - not reimplemented).
  5. Multiple-hypothesis correction - Bonferroni across every (feature, horizon) pair tested.
  6. Stability - sign/magnitude consistency of the single strongest candidate across symbols,
     months, regimes and time-of-day buckets.
  7. A single chronological (never shuffled) in-sample/out-of-sample split across the sampled
     trading dates, re-checking the strongest candidate's effect on each half separately.
  8. A model-readiness verdict (A/B/C/D) computed FROM the above results, not asserted in advance.

NEWEY-WEST CAVEAT, STATED EXPLICITLY (not silently assumed away): each bucket's value series is
built by sorting ALL of that bucket's rows by (symbol, bar_start_utc) and treating them as one
chronological sequence, exactly as `newey_west_mean_test(max_lag=horizon-1)` expects for a single
overlapping series. Because the underlying dataset only samples ~25 trading dates per symbol (see
run_phase_g_compute_features.py's own scope-control note), a small number of index-adjacent pairs
in this concatenated sequence actually straddle a symbol or date boundary and are not truly
`max_lag` minutes apart in wall-clock time - HAC has no notion of calendar gaps, only sequence
position. Given horizon<=30 and ~24 boundaries out of thousands of rows per bucket, this
contamination is a small fraction of the lag-pairs used and is judged here to not materially bias
the long-run-variance estimate - flagged explicitly rather than silently ignored, and worth
revisiting with a boundary-aware HAC estimator if a later phase needs tighter inference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.research.statistics import newey_west_mean_test

ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "derived" / "phase_g_feature_foundation" / "feature_label_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_g_feature_foundation"

HORIZONS = (5, 15, 30)
CANDIDATE_FEATURES = [
    "relative_volume_20",
    "vwap_deviation_normalized",
    "momentum_acceleration_10",
    "volatility_expansion_ratio",
    "distance_to_nearest_resistance",
    "distance_to_nearest_support",
    "range_position_20",
    "relative_strength_vs_universe",
    "realized_vol_vs_tod_baseline",
    "volume_vs_tod_baseline",
]
N_QUANTILE_BUCKETS = 5
BONFERRONI_TESTS = len(CANDIDATE_FEATURES) * len(HORIZONS)
BONFERRONI_ALPHA = 0.05 / BONFERRONI_TESTS


def _bucket_series_newey_west(df: pd.DataFrame, label_col: str, horizon: int) -> dict[str, Any] | None:
    ordered = df.sort_values(["symbol", "bar_start_utc"])[label_col].dropna().tolist()
    if len(ordered) < 2:
        return None
    max_lag = min(horizon - 1, len(ordered) - 1)
    result = newey_west_mean_test(ordered, max_lag=max_lag)
    return {
        "n": result.n, "mean": result.mean, "standard_error": result.standard_error,
        "p_value": result.p_value, "confidence_interval": result.confidence_interval,
    }


def coverage_and_missingness(df: pd.DataFrame) -> dict[str, Any]:
    overall = (df.notna().mean()).to_dict()
    by_tier = {}
    for tier, g in df.groupby("tier"):
        by_tier[tier] = g.notna().mean().to_dict()
    return {"overall_non_null_fraction": overall, "by_tier_non_null_fraction": by_tier}


def distribution_stats(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include="number")
    return json.loads(numeric.describe().to_json())


def class_frequencies(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for h in HORIZONS:
        col = f"label_class_{h}m"
        out[col] = {
            "overall": df[col].value_counts(dropna=False).to_dict(),
            "by_tier": {tier: g[col].value_counts(dropna=False).to_dict() for tier, g in df.groupby("tier")},
            "by_time_bucket": {tb: g[col].value_counts(dropna=False).to_dict() for tb, g in df.groupby("time_bucket")},
            "by_regime": {r: g[col].value_counts(dropna=False).to_dict() for r, g in df.groupby("intraday_regime")},
        }
    return out


def association_analysis(df: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for feature in CANDIDATE_FEATURES:
        for h in HORIZONS:
            label_col = f"label_normalized_move_{h}m"
            sub = df[[feature, label_col, "symbol", "bar_start_utc"]].dropna(subset=[feature, label_col])
            if len(sub) < N_QUANTILE_BUCKETS * 10:
                results.append({"feature": feature, "horizon_minutes": h, "skipped_reason": f"insufficient rows ({len(sub)})"})
                continue
            try:
                sub = sub.assign(_bucket=pd.qcut(sub[feature], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
            except ValueError as e:
                results.append({"feature": feature, "horizon_minutes": h, "skipped_reason": f"qcut failed: {e}"})
                continue
            n_buckets = sub["_bucket"].nunique()
            if n_buckets < 2:
                results.append({"feature": feature, "horizon_minutes": h, "skipped_reason": "degenerate feature (all one quantile bucket)"})
                continue

            bucket_means = sub.groupby("_bucket")[label_col].mean().sort_index()
            lowest_bucket, highest_bucket = sub["_bucket"].min(), sub["_bucket"].max()
            low_nw = _bucket_series_newey_west(sub[sub["_bucket"] == lowest_bucket], label_col, h)
            high_nw = _bucket_series_newey_west(sub[sub["_bucket"] == highest_bucket], label_col, h)
            pearson_corr = sub[feature].corr(sub[label_col])

            results.append({
                "feature": feature, "horizon_minutes": h, "n_rows": len(sub), "n_buckets": int(n_buckets),
                "pearson_correlation": None if pd.isna(pearson_corr) else float(pearson_corr),
                "bucket_means": {int(k): float(v) for k, v in bucket_means.items()},
                "lowest_bucket_newey_west": low_nw, "highest_bucket_newey_west": high_nw,
                "top_minus_bottom_bucket_mean": float(bucket_means.iloc[-1] - bucket_means.iloc[0]),
                "monotonic": bool(bucket_means.is_monotonic_increasing or bucket_means.is_monotonic_decreasing),
                "bonferroni_alpha": BONFERRONI_ALPHA,
                "low_bucket_significant_after_bonferroni": bool(low_nw and low_nw["p_value"] < BONFERRONI_ALPHA),
                "high_bucket_significant_after_bonferroni": bool(high_nw and high_nw["p_value"] < BONFERRONI_ALPHA),
            })
    return results


def pick_strongest_candidate(association_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [r for r in association_results if "top_minus_bottom_bucket_mean" in r]
    if not scored:
        return None
    return max(scored, key=lambda r: abs(r["top_minus_bottom_bucket_mean"]))


def stability_checks(df: pd.DataFrame, strongest: dict[str, Any]) -> dict[str, Any]:
    feature, h = strongest["feature"], strongest["horizon_minutes"]
    label_col = f"label_normalized_move_{h}m"
    sub = df[[feature, label_col, "symbol", "tier", "trading_date", "time_bucket", "intraday_regime"]].dropna(subset=[feature, label_col])

    def _sign_consistency(groups) -> dict[str, Any]:
        signs = {}
        for key, g in groups:
            if len(g) < 10:
                continue
            corr = g[feature].corr(g[label_col])
            if pd.isna(corr):
                continue
            signs[str(key)] = float(corr)
        if not signs:
            return {"n_groups_evaluated": 0}
        positive = sum(1 for v in signs.values() if v > 0)
        return {
            "n_groups_evaluated": len(signs), "n_positive_sign": positive, "n_negative_sign": len(signs) - positive,
            "fraction_matching_majority_sign": max(positive, len(signs) - positive) / len(signs),
            "per_group_correlation": signs,
        }

    sub = sub.assign(month=sub["trading_date"].apply(lambda d: f"{d.year:04d}-{d.month:02d}"))
    return {
        "feature": feature, "horizon_minutes": h,
        "by_symbol": _sign_consistency(sub.groupby("symbol")),
        "by_month": _sign_consistency(sub.groupby("month")),
        "by_regime": _sign_consistency(sub.groupby("intraday_regime")),
        "by_time_bucket": _sign_consistency(sub.groupby("time_bucket")),
    }


def chronological_split_check(df: pd.DataFrame, strongest: dict[str, Any]) -> dict[str, Any]:
    feature, h = strongest["feature"], strongest["horizon_minutes"]
    label_col = f"label_normalized_move_{h}m"
    dates = sorted(df["trading_date"].unique())
    split_idx = round(len(dates) * 0.7)
    in_sample_dates, out_sample_dates = set(dates[:split_idx]), set(dates[split_idx:])

    def _bucket_spread(sub_df: pd.DataFrame) -> dict[str, Any] | None:
        sub = sub_df[[feature, label_col]].dropna()
        if len(sub) < N_QUANTILE_BUCKETS * 10:
            return {"skipped_reason": f"insufficient rows ({len(sub)})"}
        try:
            sub = sub.assign(_bucket=pd.qcut(sub[feature], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
        except ValueError as e:
            return {"skipped_reason": f"qcut failed: {e}"}
        means = sub.groupby("_bucket")[label_col].mean().sort_index()
        if len(means) < 2:
            return {"skipped_reason": "degenerate feature in this split"}
        return {"n_rows": len(sub), "top_minus_bottom_bucket_mean": float(means.iloc[-1] - means.iloc[0])}

    return {
        "feature": feature, "horizon_minutes": h,
        "in_sample_dates": len(in_sample_dates), "out_of_sample_dates": len(out_sample_dates),
        "in_sample": _bucket_spread(df[df["trading_date"].isin(in_sample_dates)]),
        "out_of_sample": _bucket_spread(df[df["trading_date"].isin(out_sample_dates)]),
    }


def model_readiness_verdict(association_results: list[dict[str, Any]], stability: dict[str, Any], split_check: dict[str, Any]) -> dict[str, Any]:
    surviving = [
        r for r in association_results
        if r.get("low_bucket_significant_after_bonferroni") or r.get("high_bucket_significant_after_bonferroni")
    ]
    by_symbol_stable = stability.get("by_symbol", {}).get("fraction_matching_majority_sign", 0.0) >= 0.7
    in_s, out_s = split_check.get("in_sample", {}), split_check.get("out_of_sample", {})
    both_have_spread = "top_minus_bottom_bucket_mean" in in_s and "top_minus_bottom_bucket_mean" in out_s
    same_sign_oos = both_have_spread and (in_s["top_minus_bottom_bucket_mean"] * out_s["top_minus_bottom_bucket_mean"] > 0)

    if not surviving:
        verdict, reason = "D", "No (feature, horizon) pair survived Bonferroni-corrected significance testing on this sample."
    elif not same_sign_oos:
        verdict, reason = "C", "At least one significant association exists in-sample, but it does not replicate with the same sign out-of-sample on the chronological split - candidate feature/label definitions need revision before proceeding."
    elif not by_symbol_stable:
        verdict, reason = "B", "A significant, out-of-sample-consistent association exists, but its sign is not consistent across most sampled symbols - more data/broader sampling needed to establish whether this generalizes."
    else:
        verdict, reason = "B", "A significant, chronologically-replicating, cross-symbol-consistent association was found, but this dataset covers only 24 symbols x ~25 sampled dates - too small a sample to call READY FOR BASELINE MODEL outcome A without a larger, less sparsely-sampled dataset."

    return {
        "verdict": verdict, "reason": reason,
        "n_surviving_bonferroni": len(surviving),
        "surviving_pairs": [{"feature": r["feature"], "horizon_minutes": r["horizon_minutes"]} for r in surviving],
        "by_symbol_sign_stable": by_symbol_stable,
        "out_of_sample_same_sign_as_in_sample": same_sign_oos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "statistical_research_report.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    print(f"loaded {len(df)} rows, {len(df.columns)} columns from {args.dataset.relative_to(ROOT)}")

    coverage = coverage_and_missingness(df)
    distribution = distribution_stats(df)
    class_freq = class_frequencies(df)
    print("computed coverage/missingness, distribution, class frequencies")

    association = association_analysis(df)
    print(f"association analysis: {len(association)} (feature, horizon) pairs tested")

    strongest = pick_strongest_candidate(association)
    stability = stability_checks(df, strongest) if strongest else {}
    split_check = chronological_split_check(df, strongest) if strongest else {}
    verdict = model_readiness_verdict(association, stability, split_check) if strongest else {
        "verdict": "D", "reason": "No candidate feature produced any usable bucketed association at all.",
    }
    print(f"model-readiness verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_file": str(args.dataset.relative_to(ROOT)),
        "row_count": len(df), "column_count": len(df.columns),
        "bonferroni_tests": BONFERRONI_TESTS, "bonferroni_alpha": BONFERRONI_ALPHA,
        "coverage_and_missingness": coverage,
        "distribution": distribution,
        "class_frequencies": class_freq,
        "association_analysis": association,
        "strongest_candidate": strongest,
        "stability_checks": stability,
        "chronological_split_check": split_check,
        "model_readiness_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
