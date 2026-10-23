from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.research.statistics import newey_west_mean_test

ROOT = Path(__file__).resolve().parents[1]
PHASE_T_FILE = ROOT / "data" / "derived" / "phase_t_risk_structure" / "event_paths.parquet"
PHASE_U_FILE = ROOT / "data" / "derived" / "phase_u_execution_liquidity" / "event_paths_extended.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_u_execution_liquidity"

BUCKETS = ("top_1pct_rank15m", "top_2pct_rank15m", "bottom_1pct_rank15m", "bottom_2pct_rank15m")
IS_LONG_BUCKET = {"bottom_1pct_rank15m": True, "bottom_2pct_rank15m": True, "top_1pct_rank15m": False, "top_2pct_rank15m": False}
HORIZONS = (15, 60)
MIN_N = 30
CONFIRMATORY_ALPHA = 0.05
LIQUIDITY_TIERS = ("low_liquidity", "mid_liquidity", "high_liquidity")
PARTICIPATION_LEVELS = (0.001, 0.005, 0.01, 0.02, 0.05)

EXECUTION_CONVENTIONS = {
    "signal_close": ("phase_t", "o0"), "next_open": ("phase_t", "o1"), "next_close": ("phase_u", "close1"),
    "next_vwap_proxy": ("phase_u", "vwap1"), "delay_2m": ("phase_t", "o2"), "delay_5m": ("phase_t", "o5"),
}
STANDARD_CONVENTION = "next_open"

COST_MODEL: dict[str, dict[str, float]] = {
    "A_optimistic": {"high_liquidity": 5.0, "mid_liquidity": 10.0, "low_liquidity": 20.0},
    "B_moderate": {"high_liquidity": 10.0, "mid_liquidity": 20.0, "low_liquidity": 40.0},
    "C_conservative": {"high_liquidity": 15.0, "mid_liquidity": 30.0, "low_liquidity": 60.0},
    "D_severe": {"high_liquidity": 20.0, "mid_liquidity": 50.0, "low_liquidity": 100.0},
}
ECONOMIC_BAR_SCENARIO = "C_conservative"


def test_budget() -> dict[str, int]:
    return {
        "execution_timing": len(BUCKETS) * len(EXECUTION_CONVENTIONS) * len(HORIZONS),
        "liquidity_stratified": len(BUCKETS) * len(LIQUIDITY_TIERS) * len(HORIZONS),
    }


def load_dataset() -> pd.DataFrame:
    t = pd.read_parquet(PHASE_T_FILE)
    u = pd.read_parquet(PHASE_U_FILE)
    t_cols = ["security_id", "bar_start_utc", "entry_price_o1"] + [c for c in t.columns if c.startswith(("gross_return_o0_", "gross_return_o1_", "gross_return_o2_", "gross_return_o5_"))]
    merged = u.merge(t[t_cols], on=["security_id", "bar_start_utc"], how="left", validate="one_to_one")
    return merged


def bucket_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {b: df[b].fillna(False) for b in BUCKETS}


def _return_column(convention: str, h: int) -> tuple[str, str]:
    source, tag = EXECUTION_CONVENTIONS[convention]
    return source, f"gross_return_{tag}_h{h}"


def _mean_test_cell(signed_values: pd.Series, *, max_lag_from: int, alpha: float, min_n: int = MIN_N) -> dict[str, Any]:
    vals = signed_values.dropna()
    n = len(vals)
    if n < min_n:
        return {"n": n, "skipped_reason": f"insufficient rows ({n})"}
    max_lag = min(max_lag_from, n - 1)
    result = newey_west_mean_test(vals.to_numpy(), max_lag=max_lag)
    return {
        "n": n, "mean": result.mean, "p_value": result.p_value,
        "ci_low": result.confidence_interval[0], "ci_high": result.confidence_interval[1],
        "significant_after_correction": bool(result.p_value < alpha),
    }


def run_execution_timing(df: pd.DataFrame, masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for bucket, mask in masks.items():
        sub = df.loc[mask]
        is_long = IS_LONG_BUCKET[bucket]
        for convention in EXECUTION_CONVENTIONS:
            for h in HORIZONS:
                _, col = _return_column(convention, h)
                vals = sub[col]
                signed = vals if is_long else -vals
                cell = _mean_test_cell(signed, max_lag_from=h - 1, alpha=alpha)
                results.append({"bucket": bucket, "convention": convention, "horizon_min": h, **cell})
    return results


def run_liquidity_stratified(df: pd.DataFrame, masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for bucket, mask in masks.items():
        sub = df.loc[mask]
        is_long = IS_LONG_BUCKET[bucket]
        for tier in LIQUIDITY_TIERS:
            tier_sub = sub[sub["rolling_liquidity_group"] == tier]
            for h in HORIZONS:
                _, col = _return_column(STANDARD_CONVENTION, h)
                vals = tier_sub[col]
                signed = vals if is_long else -vals
                cell = _mean_test_cell(signed, max_lag_from=h - 1, alpha=alpha)
                results.append({"bucket": bucket, "liquidity_tier": tier, "horizon_min": h, **cell})
    return results


def pick_survivors(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("significant_after_correction") and r.get("mean", 0) > 0]


def _per_event_net_return(df: pd.DataFrame, mask: pd.Series, convention: str, h: int, is_long: bool, scenario: str) -> pd.Series:
    _, col = _return_column(convention, h)
    sub = df.loc[mask, [col, "rolling_liquidity_group"]].dropna(subset=[col, "rolling_liquidity_group"])
    gross = sub[col] if is_long else -sub[col]
    round_trip_cost_bps = sub["rolling_liquidity_group"].map(COST_MODEL[scenario])
    return gross - (round_trip_cost_bps / 10_000.0)


def run_economic_evaluation(df: pd.DataFrame, masks: dict[str, pd.Series], survivors: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    out = []
    for s in survivors:
        bucket = s["bucket"]
        is_long = IS_LONG_BUCKET[bucket]
        mask = masks[bucket]
        if family == "liquidity_stratified":
            mask = mask & (df["rolling_liquidity_group"] == s["liquidity_tier"])
            convention = STANDARD_CONVENTION
        else:
            convention = s["convention"]
        h = s["horizon_min"]
        row: dict[str, Any] = {**s, "scenarios": {}}
        n_no_tier = int((masks[bucket] & df[_return_column(convention, h)[1]].notna() & df["rolling_liquidity_group"].isna()).sum())
        row["n_excluded_no_liquidity_tier"] = n_no_tier
        for scenario in COST_MODEL:
            net = _per_event_net_return(df, mask, convention, h, is_long, scenario)
            if len(net) < MIN_N:
                row["scenarios"][scenario] = {"skipped_reason": f"insufficient rows ({len(net)})"}
                continue
            row["scenarios"][scenario] = {
                "n": len(net), "mean_net": float(net.mean()), "median_net": float(net.median()),
                "hit_rate": float((net > 0).mean()),
                "worst_decile": float(net.quantile(0.10)), "best_decile": float(net.quantile(0.90)),
            }
        out.append(row)
    return out


def run_participation_capacity(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    _, col = _return_column(STANDARD_CONVENTION, HORIZONS[0])
    out: dict[str, Any] = {}
    for bucket, mask in masks.items():
        sub = df.loc[mask].dropna(subset=["avg_1m_volume_20", "entry_price_o1", "rolling_liquidity_group"])
        by_tier: dict[str, Any] = {}
        for tier in LIQUIDITY_TIERS:
            tier_sub = sub[sub["rolling_liquidity_group"] == tier]
            if len(tier_sub) < MIN_N:
                by_tier[tier] = {"n": len(tier_sub), "skipped_reason": "insufficient rows"}
                continue
            notional_per_bar = tier_sub["avg_1m_volume_20"] * tier_sub["entry_price_o1"]
            capacity = {f"{p * 100:g}pct": {"median_notional_inr": float((notional_per_bar * p).median()), "mean_notional_inr": float((notional_per_bar * p).mean())} for p in PARTICIPATION_LEVELS}
            by_tier[tier] = {"n": len(tier_sub), "capacity_by_participation": capacity}
        out[bucket] = by_tier
    return out


def run_cross_stock_confirmatory(df: pd.DataFrame, masks: dict[str, pd.Series], survivors: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    results = []
    for s in survivors:
        bucket = s["bucket"]
        mask = masks[bucket]
        if family == "liquidity_stratified":
            mask = mask & (df["rolling_liquidity_group"] == s["liquidity_tier"])
            convention = STANDARD_CONVENTION
        else:
            convention = s["convention"]
        h = s["horizon_min"]
        _, col = _return_column(convention, h)
        sub = df.loc[mask]
        per_symbol_n = sub.dropna(subset=[col]).groupby("symbol").size()
        top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
        remainder = sub[~sub["symbol"].isin(top_5)]
        signed = remainder[col] if IS_LONG_BUCKET[bucket] else -remainder[col]
        results.append({**s, "n_symbols_excluded": len(top_5), **_mean_test_cell(signed, max_lag_from=h - 1, alpha=CONFIRMATORY_ALPHA)})
    return results


def run_liquidity_extreme_exclusion(df: pd.DataFrame, masks: dict[str, pd.Series], survivors: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    per_symbol_typical = df.groupby("symbol")["rolling_median_daily_traded_value_20d"].median()
    most_extreme = per_symbol_typical.sort_values(ascending=True).head(5).index.tolist() if len(per_symbol_typical) >= 5 else []
    results = []
    for s in survivors:
        bucket = s["bucket"]
        mask = masks[bucket]
        if family == "liquidity_stratified":
            mask = mask & (df["rolling_liquidity_group"] == s["liquidity_tier"])
            convention = STANDARD_CONVENTION
        else:
            convention = s["convention"]
        h = s["horizon_min"]
        _, col = _return_column(convention, h)
        sub = df.loc[mask]
        remainder = sub[~sub["symbol"].isin(most_extreme)]
        signed = remainder[col] if IS_LONG_BUCKET[bucket] else -remainder[col]
        results.append({**s, "symbols_excluded": most_extreme, **_mean_test_cell(signed, max_lag_from=h - 1, alpha=CONFIRMATORY_ALPHA)})
    return results


def run_chronological_confirmatory(df: pd.DataFrame, masks: dict[str, pd.Series], survivors: list[dict[str, Any]], family: str) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}
    out: dict[str, Any] = {"split_sizes": {k: len(v) for k, v in splits.items()}, "by_survivor": []}
    for s in survivors:
        bucket = s["bucket"]
        base_mask = masks[bucket]
        if family == "liquidity_stratified":
            base_mask = base_mask & (df["rolling_liquidity_group"] == s["liquidity_tier"])
            convention = STANDARD_CONVENTION
        else:
            convention = s["convention"]
        h = s["horizon_min"]
        _, col = _return_column(convention, h)
        row = {**s}
        for split_name, split_dates in splits.items():
            sub = df.loc[base_mask & df["trading_date"].isin(split_dates)]
            signed = sub[col] if IS_LONG_BUCKET[bucket] else -sub[col]
            row[split_name] = _mean_test_cell(signed, max_lag_from=h - 1, alpha=CONFIRMATORY_ALPHA)
        out["by_survivor"].append(row)
    return out


def three_way_assessment(econ_rows: list[dict[str, Any]]) -> dict[str, Any]:
    statistical = len(econ_rows) > 0
    economic_rows = [r for r in econ_rows if r["scenarios"].get(ECONOMIC_BAR_SCENARIO, {}).get("mean_net", -1) > 0]
    economic = len(economic_rows) > 0
    return {
        "statistical_edge": statistical, "n_statistical_survivors": len(econ_rows),
        "economic_edge_at_conservative_scenario": economic, "n_economic_survivors_at_conservative_scenario": len(economic_rows),
        "economic_survivor_cells": [{"bucket": r["bucket"], "convention": r.get("convention"), "liquidity_tier": r.get("liquidity_tier"), "horizon_min": r["horizon_min"]} for r in economic_rows],
    }


def final_verdict(three_way: dict[str, Any], cross_stock_holds: bool, chrono_holds: bool, liquidity_extreme_holds: bool) -> dict[str, Any]:
    if not three_way["statistical_edge"]:
        return {"verdict": "E", "reason": "No execution-timing or liquidity-stratified cell was statistically significant and favorably signed. The signal does not survive this analysis.", **three_way}
    if not three_way["economic_edge_at_conservative_scenario"]:
        return {"verdict": "C", "reason": "The statistical signal survives, but no cell is net-positive under the Conservative execution-cost scenario. Realistic execution destroys the edge.", **three_way}
    executable = cross_stock_holds and chrono_holds and liquidity_extreme_holds
    if executable:
        return {"verdict": "A", "reason": "A statistically and economically significant structure survives cross-stock, chronological, and liquidity-extreme-exclusion robustness checks under the Conservative cost scenario.", **three_way, "executable_edge": True}
    return {"verdict": "B", "reason": "An economically interesting result survives the Conservative cost scenario, but execution-capacity/generalization uncertainty remains (cross-stock, chronological, or liquidity-extreme-exclusion did not fully hold).", **three_way, "executable_edge": False, "cross_stock_holds": cross_stock_holds, "chrono_holds": chrono_holds, "liquidity_extreme_holds": liquidity_extreme_holds}


def main() -> None:
    df = load_dataset()
    masks = bucket_masks(df)
    print(f"loaded {len(df)} rows, {df['security_id'].nunique()} symbols")
    for b, m in masks.items():
        print(f"  {b}: {int(m.sum())} rows")
    print(f"rolling liquidity tier defined for {df['rolling_liquidity_group'].notna().sum()}/{len(df)} rows")

    budget = test_budget()
    total = sum(budget.values())
    alpha = 0.05 / total
    print(f"test budget: {budget} -> total {total}, Bonferroni alpha={alpha:.6f}")

    execution_timing = run_execution_timing(df, masks, alpha)
    print(f"execution_timing complete: {len(execution_timing)} rows")
    liquidity_stratified = run_liquidity_stratified(df, masks, alpha)
    print(f"liquidity_stratified complete: {len(liquidity_stratified)} rows")

    et_survivors = pick_survivors(execution_timing)
    ls_survivors = pick_survivors(liquidity_stratified)
    print(f"execution_timing survivors: {len(et_survivors)}, liquidity_stratified survivors: {len(ls_survivors)}")

    et_economic = run_economic_evaluation(df, masks, et_survivors, "execution_timing")
    ls_economic = run_economic_evaluation(df, masks, ls_survivors, "liquidity_stratified")

    et_cross_stock = run_cross_stock_confirmatory(df, masks, et_survivors, "execution_timing")
    ls_cross_stock = run_cross_stock_confirmatory(df, masks, ls_survivors, "liquidity_stratified")
    et_chrono = run_chronological_confirmatory(df, masks, et_survivors, "execution_timing")
    ls_chrono = run_chronological_confirmatory(df, masks, ls_survivors, "liquidity_stratified")
    et_liq_extreme = run_liquidity_extreme_exclusion(df, masks, et_survivors, "execution_timing")
    ls_liq_extreme = run_liquidity_extreme_exclusion(df, masks, ls_survivors, "liquidity_stratified")

    participation = run_participation_capacity(df, masks)

    all_econ = et_economic + ls_economic
    three_way = three_way_assessment(all_econ)
    all_cross_stock = et_cross_stock + ls_cross_stock
    all_liq_extreme = et_liq_extreme + ls_liq_extreme
    cross_stock_holds = all(r.get("significant_after_correction") for r in all_cross_stock) if all_cross_stock else False
    liquidity_extreme_holds = all(r.get("significant_after_correction") for r in all_liq_extreme) if all_liq_extreme else False
    chrono_rows = et_chrono.get("by_survivor", []) + ls_chrono.get("by_survivor", [])
    chrono_holds = all(row.get(split, {}).get("significant_after_correction") for row in chrono_rows for split in ("train", "validation", "held_out")) if chrono_rows else False

    verdict = final_verdict(three_way, cross_stock_holds, chrono_holds, liquidity_extreme_holds)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "test_budget": budget, "total_tests": total, "alpha": alpha,
        "bucket_sizes": {b: int(m.sum()) for b, m in masks.items()},
        "liquidity_tier_coverage": {"n_defined": int(df["rolling_liquidity_group"].notna().sum()), "n_total": len(df)},
        "execution_timing": execution_timing, "liquidity_stratified": liquidity_stratified,
        "execution_timing_survivors": et_survivors, "liquidity_stratified_survivors": ls_survivors,
        "execution_timing_economic": et_economic, "liquidity_stratified_economic": ls_economic,
        "execution_timing_cross_stock": et_cross_stock, "liquidity_stratified_cross_stock": ls_cross_stock,
        "execution_timing_chronological": et_chrono, "liquidity_stratified_chronological": ls_chrono,
        "execution_timing_liquidity_extreme": et_liq_extreme, "liquidity_stratified_liquidity_extreme": ls_liq_extreme,
        "participation_capacity": participation, "three_way_assessment": three_way, "verdict": verdict,
    }
    out_path = OUT_DIR / "phase_u_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
