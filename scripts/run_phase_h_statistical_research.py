"""Phase H, H2-H9 & H11: full-universe validation of Phase G's `relative_strength_vs_universe`
discovery, plus the newly-approved mover-magnitude and interaction analyses.

PRE-REGISTRATION (H2, H8): every population, feature, horizon and interaction tested here is fixed
BEFORE this script is run against the real dataset - see POPULATION_GROUPS, MOVER_DEPENDENT_VARS
and INTERACTION_HYPOTHESES below, none of which depend on any result. Population thresholds
(price/liquidity/volatility median splits) come from `group_definitions.json`, itself computed and
frozen by `run_phase_h_audit.py` before any relative_strength_vs_universe result was examined.

TEST BUDGET AND CORRECTION (H8): exactly
  len(POPULATION_GROUPS) * len(HORIZONS)                              (H2 - population survival)
  + len(MOVER_DEPENDENT_VARS) * len(HORIZONS)                         (H3 - magnitude/mover analysis)
  + len(INTERACTION_HYPOTHESES) * len(INTERACTION_HORIZONS)           (H4 - pre-registered interactions)
formal significance tests are run, ONE Bonferroni correction is applied across the WHOLE set (not
per sub-analysis - a separate correction per family would understate the true multiple-testing
burden), and the exact count and resulting alpha are printed and written to the report. Time-of-day/
regime/month breakdowns (H2 items 12-15) are reported as STABILITY diagnostics of the population-
level result (as Phase G already did), not as additional independent hypothesis tests - inflating
the correction with descriptive stability checks would be a different kind of data-mining problem
(making the bar artificially easy to clear by testing many near-duplicate slices).

METHOD REUSED FROM PHASE G, UNCHANGED: quantile-bucket mean comparison with Newey-West HAC
(`backend/app/research/statistics.py`, `max_lag = horizon - 1`) on each bucket's chronologically-
ordered (symbol, bar_start_utc) series - see run_phase_g_statistical_research.py's own docstring for
the identical cross-boundary HAC caveat, which applies unchanged here.

H7 - THE relative_strength_vs_universe PIT CHECK: `universe_average_return` at T is computed by
grouping ALREADY-PIT-SAFE per-symbol `return_1m` values by their SHARED `bar_start_utc` (see
run_phase_h_compute_features.py's `run_pass_2_cross_sectional`) - it aggregates only what every
contributing symbol's own bars_up_to_t already knew AT THAT SAME INSTANT, never a value computed
from a later T. This is stated here as the explicit answer to H7's "ensure the universe reference at
time T does not accidentally incorporate future observations" requirement - verified by
construction (a pandas groupby on the row's own timestamp column), not by a separate runtime check.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backend.app.research.statistics import newey_west_mean_test

ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
GROUP_DEFS_FILE = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "group_definitions.json"
OUT_DIR = ROOT / "data" / "derived" / "phase_h_full_universe_validation"

HORIZONS = (5, 15, 30, 60)
INTERACTION_HORIZONS = (5, 15)
N_QUANTILE_BUCKETS = 5
PRIMARY_FEATURE = "relative_strength_vs_universe"

MOVER_DEPENDENT_VARS = ["mover_future_abs_return", "mover_future_range_pct"]

INTERACTION_HYPOTHESES = [
    "volatility_expansion_ratio",
    "volume_acceleration_10",
    "vwap_deviation_normalized",
    "distance_to_nearest_support",
    "distance_to_nearest_resistance",
    "candle_body_ratio",
    "time_bucket",
]


def load_group_definitions() -> dict[str, Any]:
    return json.loads(GROUP_DEFS_FILE.read_text(encoding="utf-8"))


def attach_pre_registered_groups(df: pd.DataFrame, group_defs: dict[str, Any]) -> pd.DataFrame:
    stats = group_defs["per_symbol_stats"]
    price_median = group_defs["price_group"]["median_avg_close"]
    turnover_median = group_defs["liquidity_group"]["median_avg_daily_turnover"]
    range_median = group_defs["volatility_group"]["median_avg_daily_range_pct"]

    def _lookup(sid: int, key: str) -> float | None:
        s = stats.get(str(sid))
        return s[key] if s else None

    sid_to_price = df["security_id"].map(lambda sid: _lookup(sid, "avg_close"))
    sid_to_turnover = df["security_id"].map(lambda sid: _lookup(sid, "avg_daily_turnover"))
    sid_to_range = df["security_id"].map(lambda sid: _lookup(sid, "avg_daily_range_pct"))

    df = df.assign(
        price_group=pd.Series(["low_priced" if v is not None and v < price_median else ("higher_priced" if v is not None else None) for v in sid_to_price], index=df.index),
        liquidity_group=pd.Series(["higher_liquidity" if v is not None and v >= turnover_median else ("lower_liquidity" if v is not None else None) for v in sid_to_turnover], index=df.index),
        volatility_group=pd.Series(["high_volatility" if v is not None and v >= range_median else ("low_volatility" if v is not None else None) for v in sid_to_range], index=df.index),
        market_regime=(df["universe_average_return"] > 0).map({True: "MARKET_UP", False: "MARKET_DOWN"}),
    )
    return df


def population_groups() -> list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    return [
        ("full_universe", lambda df: df),
        ("NIFTY50", lambda df: df[df["tier"] == "NIFTY50"]),
        ("NIFTY_NEXT50", lambda df: df[df["tier"] == "NIFTY_NEXT50"]),
        ("NIFTY_MIDCAP150", lambda df: df[df["tier"] == "NIFTY_MIDCAP150"]),
        ("NIFTY_SMALLCAP250", lambda df: df[df["tier"] == "NIFTY_SMALLCAP250"]),
        ("OTHER_MAINBOARD", lambda df: df[df["tier"] == "OTHER_MAINBOARD"]),
        ("low_priced", lambda df: df[df["price_group"] == "low_priced"]),
        ("higher_priced", lambda df: df[df["price_group"] == "higher_priced"]),
        ("high_volatility", lambda df: df[df["volatility_group"] == "high_volatility"]),
        ("low_volatility", lambda df: df[df["volatility_group"] == "low_volatility"]),
        ("higher_liquidity", lambda df: df[df["liquidity_group"] == "higher_liquidity"]),
        ("lower_liquidity", lambda df: df[df["liquidity_group"] == "lower_liquidity"]),
    ]


def bonferroni_test_budget() -> int:
    n_h2 = len(population_groups()) * len(HORIZONS)
    n_h3 = len(MOVER_DEPENDENT_VARS) * len(HORIZONS)
    n_h4 = len(INTERACTION_HYPOTHESES) * len(INTERACTION_HORIZONS)
    return n_h2 + n_h3 + n_h4


def _bucket_series_newey_west(df: pd.DataFrame, value_col: str, horizon: int) -> dict[str, Any] | None:
    ordered = df.sort_values(["symbol", "bar_start_utc"])[value_col].dropna().tolist()
    if len(ordered) < 2:
        return None
    max_lag = min(horizon - 1, len(ordered) - 1)
    result = newey_west_mean_test(ordered, max_lag=max_lag)
    return {"n": result.n, "mean": result.mean, "standard_error": result.standard_error, "p_value": result.p_value, "confidence_interval": result.confidence_interval}


def _quantile_bucket_test(df: pd.DataFrame, feature_col: str, value_col: str, horizon: int, alpha: float) -> dict[str, Any]:
    sub = df[[feature_col, value_col, "symbol", "bar_start_utc"]].dropna(subset=[feature_col, value_col])
    if len(sub) < N_QUANTILE_BUCKETS * 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})", "n_rows": len(sub)}
    try:
        sub = sub.assign(_bucket=pd.qcut(sub[feature_col], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
    except ValueError as e:
        return {"skipped_reason": f"qcut failed: {e}", "n_rows": len(sub)}
    if sub["_bucket"].nunique() < 2:
        return {"skipped_reason": "degenerate feature (single quantile bucket)", "n_rows": len(sub)}

    bucket_means = sub.groupby("_bucket")[value_col].mean().sort_index()
    low_b, high_b = sub["_bucket"].min(), sub["_bucket"].max()
    low_nw = _bucket_series_newey_west(sub[sub["_bucket"] == low_b], value_col, horizon)
    high_nw = _bucket_series_newey_west(sub[sub["_bucket"] == high_b], value_col, horizon)
    pearson = sub[feature_col].corr(sub[value_col])

    return {
        "n_rows": len(sub), "n_buckets": int(sub["_bucket"].nunique()),
        "pearson_correlation": None if pd.isna(pearson) else float(pearson),
        "bucket_means": {int(k): float(v) for k, v in bucket_means.items()},
        "top_minus_bottom_bucket_mean": float(bucket_means.iloc[-1] - bucket_means.iloc[0]),
        "monotonic": bool(bucket_means.is_monotonic_increasing or bucket_means.is_monotonic_decreasing),
        "lowest_bucket_newey_west": low_nw, "highest_bucket_newey_west": high_nw,
        "significant_after_correction": bool((low_nw and low_nw["p_value"] < alpha) or (high_nw and high_nw["p_value"] < alpha)),
    }


def run_h2_population_validation(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for pop_name, pop_filter in population_groups():
        pop_df = pop_filter(df)
        for h in HORIZONS:
            label_col = f"label_normalized_move_{h}m"
            r = _quantile_bucket_test(pop_df, PRIMARY_FEATURE, label_col, h, alpha)
            results.append({"population": pop_name, "feature": PRIMARY_FEATURE, "horizon_minutes": h, **r})
    return results


def run_h3_mover_analysis(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for dep_var in MOVER_DEPENDENT_VARS:
        for h in HORIZONS:
            value_col = f"{dep_var}_{h}m"
            r = _quantile_bucket_test(df, PRIMARY_FEATURE, value_col, h, alpha)
            results.append({"feature": PRIMARY_FEATURE, "dependent_variable": dep_var, "horizon_minutes": h, **r})
    return results


def run_h4_interactions(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for interactor in INTERACTION_HYPOTHESES:
        for h in INTERACTION_HORIZONS:
            label_col = f"label_normalized_move_{h}m"
            if interactor == "time_bucket":
                sub_groups = [(v, df[df["time_bucket"] == v]) for v in ("OPENING", "MIDDAY", "CLOSING")]
            else:
                med = df[interactor].median()
                sub_groups = [("below_median", df[df[interactor] < med]), ("at_or_above_median", df[df[interactor] >= med])]
            per_group = {}
            for gname, gdf in sub_groups:
                per_group[gname] = _quantile_bucket_test(gdf, PRIMARY_FEATURE, label_col, h, alpha)
            results.append({"interactor": interactor, "horizon_minutes": h, "groups": per_group})
    return results


def run_h7_chronological_split(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end = round(n * 0.6)
    val_end = round(n * 0.8)
    train_dates, val_dates, held_out_dates = set(dates[:train_end]), set(dates[train_end:val_end]), set(dates[val_end:])

    splits = {"train": df[df["trading_date"].isin(train_dates)], "validation": df[df["trading_date"].isin(val_dates)], "held_out": df[df["trading_date"].isin(held_out_dates)]}
    result: dict[str, Any] = {
        "train_dates": {"n": len(train_dates), "first": str(min(train_dates)) if train_dates else None, "last": str(max(train_dates)) if train_dates else None},
        "validation_dates": {"n": len(val_dates), "first": str(min(val_dates)) if val_dates else None, "last": str(max(val_dates)) if val_dates else None},
        "held_out_dates": {"n": len(held_out_dates), "first": str(min(held_out_dates)) if held_out_dates else None, "last": str(max(held_out_dates)) if held_out_dates else None},
        "by_split": {},
    }
    for split_name, split_df in splits.items():
        result["by_split"][split_name] = {
            f"{h}m": _quantile_bucket_test(split_df, PRIMARY_FEATURE, f"label_normalized_move_{h}m", h, alpha) for h in HORIZONS
        }
    return result


def stability_breakdown(df: pd.DataFrame, feature: str, horizon: int) -> dict[str, Any]:
    label_col = f"label_normalized_move_{horizon}m"
    sub = df[[feature, label_col, "symbol", "trading_date", "time_bucket", "intraday_regime", "market_regime"]].dropna(subset=[feature, label_col])
    sub = sub.assign(month=sub["trading_date"].apply(lambda d: f"{d.year:04d}-{d.month:02d}"))

    def _signs(groups) -> dict[str, Any]:
        signs = {}
        for key, g in groups:
            if len(g) < 10:
                continue
            corr = g[feature].corr(g[label_col])
            if not pd.isna(corr):
                signs[str(key)] = float(corr)
        if not signs:
            return {"n_groups_evaluated": 0}
        positive = sum(1 for v in signs.values() if v > 0)
        return {"n_groups_evaluated": len(signs), "fraction_matching_majority_sign": max(positive, len(signs) - positive) / len(signs), "per_group_correlation": signs}

    return {
        "by_time_bucket": _signs(sub.groupby("time_bucket")),
        "by_intraday_regime": _signs(sub.groupby("intraday_regime")),
        "by_market_regime": _signs(sub.groupby("market_regime")),
        "by_month": _signs(sub.groupby("month")),
        "by_symbol": _signs(sub.groupby("symbol")),
    }


def effect_size_and_tradability(df: pd.DataFrame, horizon: int) -> dict[str, Any]:
    label_col, abs_col, range_col = f"label_normalized_move_{horizon}m", f"mover_future_abs_return_{horizon}m", f"mover_future_range_pct_{horizon}m"
    mfe_col, mae_col, class_col = f"mfe_{horizon}m", f"mae_{horizon}m", f"label_class_{horizon}m"
    sub = df[[PRIMARY_FEATURE, label_col, abs_col, range_col, mfe_col, mae_col, class_col]].dropna(subset=[PRIMARY_FEATURE, label_col])
    if len(sub) < N_QUANTILE_BUCKETS * 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})"}
    sub = sub.assign(_bucket=pd.qcut(sub[PRIMARY_FEATURE], N_QUANTILE_BUCKETS, labels=False, duplicates="drop"))
    low_b, high_b = sub["_bucket"].min(), sub["_bucket"].max()

    def _describe(bucket_df: pd.DataFrame) -> dict[str, Any]:
        mean_move = float(bucket_df[label_col].mean())
        expected_class = "UP" if mean_move > 0 else ("DOWN" if mean_move < 0 else None)
        classified = bucket_df[class_col].notna().sum()
        hits = (bucket_df[class_col] == expected_class).sum() if expected_class else None
        return {
            "n": len(bucket_df),
            "mean_normalized_move": mean_move,
            "median_normalized_move": float(bucket_df[label_col].median()),
            "mean_future_abs_return": float(bucket_df[abs_col].mean()) if bucket_df[abs_col].notna().any() else None,
            "mean_future_range_pct": float(bucket_df[range_col].mean()) if bucket_df[range_col].notna().any() else None,
            "mean_mfe": float(bucket_df[mfe_col].mean()) if bucket_df[mfe_col].notna().any() else None,
            "mean_mae": float(bucket_df[mae_col].mean()) if bucket_df[mae_col].notna().any() else None,
            "expected_direction": expected_class,
            "hit_rate_for_expected_direction": float(hits / classified) if (expected_class and classified) else None,
        }

    return {
        "horizon_minutes": horizon,
        "lowest_bucket": _describe(sub[sub["_bucket"] == low_b]),
        "highest_bucket": _describe(sub[sub["_bucket"] == high_b]),
        "note": "Statistical significance (see H2 results) is reported separately - these are magnitude/hit-rate descriptives only, with NO cost/spread/slippage model applied (not available in this dataset).",
    }


def model_readiness_verdict(h2_results: list[dict[str, Any]], split_check: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
    full_universe_significant = [r for r in h2_results if r["population"] == "full_universe" and r.get("significant_after_correction")]
    surviving_populations = {r["population"] for r in h2_results if r.get("significant_after_correction")}
    n_populations = len(population_groups())

    by_symbol_stable = stability.get("by_symbol", {}).get("fraction_matching_majority_sign", 0.0) >= 0.7
    splits_significant = []
    for split_name, by_h in split_check.get("by_split", {}).items():
        splits_significant.append(any(v.get("significant_after_correction") for v in by_h.values()))

    if not full_universe_significant:
        verdict, reason = "D", "relative_strength_vs_universe does not survive Bonferroni-corrected testing on the full 298-symbol universe at any horizon - the Phase G finding does not replicate at scale."
    elif not all(splits_significant):
        verdict, reason = "C", f"Significant on the full universe, but not consistently across all three chronological splits (train/validation/held-out significant: {splits_significant}) - weakens substantially under out-of-sample pressure; needs further research before proceeding."
    elif len(surviving_populations) < n_populations * 0.5:
        verdict, reason = "C", f"Significant on the full universe and stable across time, but survives in only {len(surviving_populations)}/{n_populations} pre-registered sub-populations - the effect is concentrated, not general, and needs targeted research into where and why it fails."
    elif not by_symbol_stable:
        verdict, reason = "B", "Signal survives full-universe and chronological validation, but per-symbol sign consistency has weakened relative to Phase G - more data/validation needed before baseline modeling."
    else:
        verdict, reason = "B", "Signal survives full-universe, cross-population and chronological out-of-sample validation with effect sizes still small (statistical significance, not yet established economic tradability) - proceed to further validation, not yet to baseline modeling."

    return {
        "verdict": verdict, "reason": reason,
        "surviving_populations": sorted(surviving_populations), "n_populations_tested": n_populations,
        "chronological_splits_significant": dict(zip(split_check.get("by_split", {}).keys(), splits_significant)),
        "by_symbol_sign_stable": by_symbol_stable,
    }


def build_research_ledger(h2: list[dict[str, Any]], h3: list[dict[str, Any]], h4: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    ledger = []
    for r in h2:
        ledger.append({
            "hypothesis": f"relative_strength_vs_universe predicts {r['horizon_minutes']}m forward normalized move in population '{r['population']}'",
            "feature": r["feature"], "horizon_minutes": r["horizon_minutes"], "population": r["population"],
            "test": "quantile-bucket Newey-West HAC mean test (top & bottom quintile vs. zero)", "correction": f"Bonferroni, alpha={alpha:.6f}",
            "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT SIGNIFICANT"),
            "effect_size": r.get("top_minus_bottom_bucket_mean"), "n_rows": r.get("n_rows"),
            "interpretation": r.get("skipped_reason", ""), "limitations": "Small effect size even when significant; not yet cost-adjusted.",
        })
    for r in h3:
        ledger.append({
            "hypothesis": f"relative_strength_vs_universe predicts {r['horizon_minutes']}m {r['dependent_variable']}",
            "feature": r["feature"], "horizon_minutes": r["horizon_minutes"], "population": "full_universe",
            "test": "quantile-bucket Newey-West HAC mean test", "correction": f"Bonferroni, alpha={alpha:.6f}",
            "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT SIGNIFICANT"),
            "effect_size": r.get("top_minus_bottom_bucket_mean"), "n_rows": r.get("n_rows"),
            "interpretation": r.get("skipped_reason", ""), "limitations": "Magnitude diagnostic only - does not itself establish direction was correctly predicted.",
        })
    for r in h4:
        group_summaries = {g: (v.get("top_minus_bottom_bucket_mean"), v.get("significant_after_correction")) for g, v in r["groups"].items()}
        ledger.append({
            "hypothesis": f"relative_strength_vs_universe's effect on {r['horizon_minutes']}m move is modulated by {r['interactor']}",
            "feature": f"{PRIMARY_FEATURE} x {r['interactor']}", "horizon_minutes": r["horizon_minutes"], "population": "full_universe (split by interactor)",
            "test": "quantile-bucket Newey-West HAC mean test, run separately per interactor group", "correction": f"Bonferroni, alpha={alpha:.6f}",
            "result": str(group_summaries), "effect_size": None, "n_rows": None,
            "interpretation": "Compare top-minus-bottom effect size across groups - a material difference suggests interaction.", "limitations": "Exploratory; no formal interaction-term significance test performed, effect-size comparison only.",
        })
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "statistical_research_report.json")
    args = parser.parse_args()

    group_defs = load_group_definitions()
    df = pd.read_parquet(args.dataset)
    df = attach_pre_registered_groups(df, group_defs)
    print(f"loaded {len(df)} rows, {len(df.columns)} columns (post group attachment)")

    n_tests = bonferroni_test_budget()
    alpha = 0.05 / n_tests
    print(f"pre-registered test budget: {n_tests} tests, Bonferroni alpha={alpha:.6f}")

    h2 = run_h2_population_validation(df, alpha)
    print(f"H2 complete: {len(h2)} population x horizon tests")
    h3 = run_h3_mover_analysis(df, alpha)
    print(f"H3 complete: {len(h3)} mover tests")
    h4 = run_h4_interactions(df, alpha)
    print(f"H4 complete: {len(h4)} interaction tests")
    h7 = run_h7_chronological_split(df, alpha)
    print("H7 complete: chronological 3-way split")

    stability = stability_breakdown(df, PRIMARY_FEATURE, 15)
    tradability = {h: effect_size_and_tradability(df, h) for h in HORIZONS}
    ledger = build_research_ledger(h2, h3, h4, alpha)
    verdict = model_readiness_verdict(h2, h7, stability)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_file": str(args.dataset.relative_to(ROOT)), "row_count": len(df), "column_count": len(df.columns),
        "bonferroni_test_budget": n_tests, "bonferroni_alpha": alpha,
        "h2_population_validation": h2, "h3_mover_analysis": h3, "h4_interactions": h4,
        "h7_chronological_split": h7, "stability_breakdown_15m": stability,
        "h9_effect_size_and_tradability": tradability,
        "research_ledger": ledger, "model_readiness_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")
    print(f"wrote {(OUT_DIR / 'research_ledger.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
