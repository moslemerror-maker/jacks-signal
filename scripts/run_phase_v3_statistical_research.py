from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.research.statistics import newey_west_mean_test

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "derived" / "phase_v3_liquid_mechanisms" / "dense_liquid_mechanism_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_v3_liquid_mechanisms"

HORIZONS = (5, 15, 30, 60)
INCREMENTAL_HORIZON = 15
LIQUID_TIERS = ("mid_liquidity", "high_liquidity")
MIN_N = 30
CONFIRMATORY_ALPHA = 0.05

ABNORMAL_RELATIVE_VOLUME = 2.0
M3_MIN_RETURN_5M = 0.003
M2_MAX_MINUTES_SINCE_OPEN = 30
M2_MIN_DISTANCE_FROM_OPEN = 0.005
M2_MIN_CONSECUTIVE_RUN = 3
M1_MIN_CANDLE_BODY_RATIO = 0.5

COST_MODEL: dict[str, dict[str, float]] = {
    "A_optimistic": {"high_liquidity": 5.0, "mid_liquidity": 10.0},
    "B_moderate": {"high_liquidity": 10.0, "mid_liquidity": 20.0},
    "C_conservative": {"high_liquidity": 15.0, "mid_liquidity": 30.0},
    "D_severe": {"high_liquidity": 20.0, "mid_liquidity": 50.0},
}
ECONOMIC_BAR_SCENARIO = "C_conservative"


def test_budget() -> dict[str, int]:
    return {"standalone": 4 * 2 * len(HORIZONS), "incremental": 4 * 2 * 2 * 2}


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df = df[df["rolling_liquidity_group"].isin(LIQUID_TIERS)].copy()
    returns_5m = df.groupby("bar_start_utc")["return_5m"].transform("mean")
    df["relative_strength_vs_universe"] = df["return_5m"] - returns_5m
    return df


def mechanism_masks(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    m1_up = df["breakout_up_meaningful"].fillna(False) & (df["candle_body_ratio"] >= M1_MIN_CANDLE_BODY_RATIO)
    m1_down = df["breakout_down_meaningful"].fillna(False) & (df["candle_body_ratio"] >= M1_MIN_CANDLE_BODY_RATIO)

    in_opening_window = df["minutes_since_open"].notna() & (df["minutes_since_open"] <= M2_MAX_MINUTES_SINCE_OPEN)
    m2_up = in_opening_window & (df["distance_from_session_open"] >= M2_MIN_DISTANCE_FROM_OPEN) & (df["consecutive_same_direction_count"] >= M2_MIN_CONSECUTIVE_RUN) & (df["candle_direction"] > 0)
    m2_down = in_opening_window & (df["distance_from_session_open"] <= -M2_MIN_DISTANCE_FROM_OPEN) & (df["consecutive_same_direction_count"] >= M2_MIN_CONSECUTIVE_RUN) & (df["candle_direction"] < 0)

    abnormal_vol = df["relative_volume_20"] >= ABNORMAL_RELATIVE_VOLUME
    m3_up = abnormal_vol & (df["return_5m"] >= M3_MIN_RETURN_5M)
    m3_down = abnormal_vol & (df["return_5m"] <= -M3_MIN_RETURN_5M)

    expanding = df["compression_then_expansion_flag"].fillna(False)
    m4_up = expanding & (df["candle_direction"] > 0) & (df["return_5m"] > 0)
    m4_down = expanding & (df["candle_direction"] < 0) & (df["return_5m"] < 0)

    return {
        "M1_breakout_continuation": {"up": m1_up, "down": m1_down},
        "M2_opening_drive_continuation": {"up": m2_up, "down": m2_down},
        "M3_abnormal_participation_continuation": {"up": m3_up, "down": m3_down},
        "M4_volatility_expansion_confirmation": {"up": m4_up, "down": m4_down},
    }


def _finalize_rate_test(ordered_flags: list[float], baseline_rate: float, horizon: int, alpha: float) -> dict[str, Any]:
    if len(ordered_flags) < MIN_N:
        return {"n": len(ordered_flags), "skipped_reason": f"insufficient rows ({len(ordered_flags)})"}
    centered = [f - baseline_rate for f in ordered_flags]
    max_lag = min(horizon - 1, len(centered) - 1)
    result = newey_west_mean_test(centered, max_lag=max_lag)
    realized_rate = baseline_rate + result.mean
    return {
        "n": len(ordered_flags), "realized_rate": realized_rate, "baseline_rate": baseline_rate,
        "lift": realized_rate / baseline_rate if baseline_rate else None,
        "p_value": result.p_value, "significant_after_correction": bool(result.p_value < alpha),
    }


def _rate_test(df: pd.DataFrame, mask: pd.Series, direction_col: str, implied: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[mask, [direction_col, "symbol", "bar_start_utc"]]
    sub = sub[sub[direction_col].isin(["UP", "DOWN", "NEUTRAL"])]
    if len(sub) < MIN_N:
        return {"n": len(sub), "skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[direction_col].eq(implied).astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha)


def run_standalone(df: pd.DataFrame, masks: dict[str, dict[str, pd.Series]], alpha: float) -> list[dict[str, Any]]:
    results = []
    for h in HORIZONS:
        direction_col = f"direction_label_pit_{h}"
        defined = df[direction_col].isin(["UP", "DOWN", "NEUTRAL"])
        baseline_up = (df.loc[defined, direction_col] == "UP").mean()
        baseline_down = (df.loc[defined, direction_col] == "DOWN").mean()
        for mech, sides in masks.items():
            results.append({"mechanism": mech, "side": "up", "horizon_min": h, **_rate_test(df, sides["up"], direction_col, "UP", h, alpha, baseline_up)})
            results.append({"mechanism": mech, "side": "down", "horizon_min": h, **_rate_test(df, sides["down"], direction_col, "DOWN", h, alpha, baseline_down)})
    return results


def run_incremental(df: pd.DataFrame, masks: dict[str, dict[str, pd.Series]], alpha: float) -> list[dict[str, Any]]:
    h = INCREMENTAL_HORIZON
    direction_col = f"direction_label_pit_{h}"
    defined = df[direction_col].isin(["UP", "DOWN", "NEUTRAL"])
    baseline_up = (df.loc[defined, direction_col] == "UP").mean()
    baseline_down = (df.loc[defined, direction_col] == "DOWN").mean()
    results = []
    for mech, sides in masks.items():
        for side, implied, baseline in (("up", "UP", baseline_up), ("down", "DOWN", baseline_down)):
            mask = sides[side]
            for control_col in ("return_5m", "relative_strength_vs_universe"):
                med = df.loc[mask, control_col].median()
                for group_name, group_mask in ((f"below_median_{control_col}", mask & (df[control_col] < med)), (f"at_or_above_median_{control_col}", mask & (df[control_col] >= med))):
                    results.append({"mechanism": mech, "side": side, "control": control_col, "controlled_group": group_name, **_rate_test(df, group_mask, direction_col, implied, h, alpha, baseline)})
    return results


def pick_survivors(standalone: list[dict[str, Any]], incremental: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_confirming = {(r["mechanism"], r["side"], r["horizon_min"]) for r in standalone if r.get("significant_after_correction") and r.get("lift") is not None and r["lift"] > 1.0}
    b_by_key: dict[tuple[str, str], list[bool]] = {}
    for r in incremental:
        if "skipped_reason" in r:
            continue
        b_by_key.setdefault((r["mechanism"], r["side"]), []).append(r.get("significant_after_correction", False))
    survivors = []
    for mech, side, h in a_confirming:
        flags = b_by_key.get((mech, side), [])
        if len(flags) == 4 and all(flags):
            survivors.append({"mechanism": mech, "side": side, "horizon_min": h})
    return survivors


def _survivor_mask(masks: dict[str, dict[str, pd.Series]], s: dict[str, Any]) -> pd.Series:
    return masks[s["mechanism"]][s["side"]]


def run_cross_stock_confirmatory(df: pd.DataFrame, masks: dict[str, dict[str, pd.Series]], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for s in survivors:
        h = s["horizon_min"]
        direction_col = f"direction_label_pit_{h}"
        implied = "UP" if s["side"] == "up" else "DOWN"
        mask = _survivor_mask(masks, s)
        baseline = (df.loc[df[direction_col].isin(["UP", "DOWN", "NEUTRAL"]), direction_col] == implied).mean()
        per_symbol_n = df.loc[mask & df[direction_col].notna()].groupby("symbol").size()
        top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
        remainder = mask & ~df["symbol"].isin(top_5)
        results.append({**s, "n_symbols_excluded": len(top_5), **_rate_test(df, remainder, direction_col, implied, h, CONFIRMATORY_ALPHA, baseline)})
    return results


def run_chronological_confirmatory(df: pd.DataFrame, masks: dict[str, dict[str, pd.Series]], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}
    out: dict[str, Any] = {"split_sizes": {k: len(v) for k, v in splits.items()}, "by_survivor": []}
    for s in survivors:
        h = s["horizon_min"]
        direction_col = f"direction_label_pit_{h}"
        implied = "UP" if s["side"] == "up" else "DOWN"
        mask = _survivor_mask(masks, s)
        row = {**s}
        for split_name, split_dates in splits.items():
            split_mask = mask & df["trading_date"].isin(split_dates)
            baseline = (df.loc[df["trading_date"].isin(split_dates) & df[direction_col].isin(["UP", "DOWN", "NEUTRAL"]), direction_col] == implied).mean()
            row[split_name] = _rate_test(df, split_mask, direction_col, implied, h, CONFIRMATORY_ALPHA, baseline)
        out["by_survivor"].append(row)
    return out


def run_session_half_stability(df: pd.DataFrame, masks: dict[str, dict[str, pd.Series]], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for s in survivors:
        h = s["horizon_min"]
        direction_col = f"direction_label_pit_{h}"
        implied = "UP" if s["side"] == "up" else "DOWN"
        mask = _survivor_mask(masks, s)
        for half_name, half_mask in (("morning", df["minutes_since_open"] < 180), ("afternoon", df["minutes_since_open"] >= 180)):
            sub_mask = mask & half_mask
            baseline = (df.loc[half_mask & df[direction_col].isin(["UP", "DOWN", "NEUTRAL"]), direction_col] == implied).mean()
            results.append({**s, "half": half_name, **_rate_test(df, sub_mask, direction_col, implied, h, CONFIRMATORY_ALPHA, baseline)})
    return results


def run_economic_evaluation(df: pd.DataFrame, masks: dict[str, dict[str, pd.Series]], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in survivors:
        h = s["horizon_min"]
        mask = _survivor_mask(masks, s)
        sub = df.loc[mask, [f"gross_return_{h}m", "rolling_liquidity_group"]].dropna()
        is_long = s["side"] == "up"
        gross = sub[f"gross_return_{h}m"] if is_long else -sub[f"gross_return_{h}m"]
        row: dict[str, Any] = {**s, "n": len(sub), "gross_mean": float(gross.mean()) if len(sub) else None, "scenarios": {}}
        for scenario in COST_MODEL:
            cost_bps = sub["rolling_liquidity_group"].map(COST_MODEL[scenario])
            net = gross - (cost_bps / 10_000.0)
            if len(net) < MIN_N:
                row["scenarios"][scenario] = {"skipped_reason": f"insufficient rows ({len(net)})"}
                continue
            row["scenarios"][scenario] = {"mean_net": float(net.mean()), "median_net": float(net.median()), "hit_rate": float((net > 0).mean())}
        out.append(row)
    return out


def run_capacity_evidence(df: pd.DataFrame, masks: dict[str, dict[str, pd.Series]], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in survivors:
        mask = _survivor_mask(masks, s)
        sub = df.loc[mask, "rolling_median_daily_traded_value_20d"].dropna()
        out.append({**s, "n": len(sub), "median_daily_traded_value_inr": float(sub.median()) if len(sub) else None, "p10_daily_traded_value_inr": float(sub.quantile(0.10)) if len(sub) else None})
    return out


def final_mechanism_verdicts(standalone, incremental, survivors, cross_stock, chrono, stability, economic) -> dict[str, Any]:
    verdicts: dict[str, Any] = {}
    mechanisms = ["M1_breakout_continuation", "M2_opening_drive_continuation", "M3_abnormal_participation_continuation", "M4_volatility_expansion_confirmation"]
    for mech in mechanisms:
        mech_survivors = [s for s in survivors if s["mechanism"] == mech]
        if not mech_survivors:
            verdicts[mech] = {"verdict": "FAILED", "failed_stage": "standalone_or_incremental_significance", "reason": "No side/horizon combination was both standalone-significant and incremental over return_5m/relative_strength_vs_universe."}
            continue
        cs = [r for r in cross_stock if r["mechanism"] == mech]
        cross_holds = all(r.get("significant_after_correction") for r in cs) if cs else False
        if not cross_holds:
            verdicts[mech] = {"verdict": "FAILED", "failed_stage": "cross_stock_generalization", "reason": "Did not hold after excluding the top-5 contributing symbols.", "survivors": mech_survivors}
            continue
        chrono_rows = [r for r in chrono.get("by_survivor", []) if r["mechanism"] == mech]
        chrono_holds = all(row.get(split, {}).get("significant_after_correction") for row in chrono_rows for split in ("train", "validation", "held_out")) if chrono_rows else False
        if not chrono_holds:
            verdicts[mech] = {"verdict": "FAILED", "failed_stage": "chronological_validation", "reason": "Did not hold across all three chronological splits.", "survivors": mech_survivors}
            continue
        stab_rows = [r for r in stability if r["mechanism"] == mech]
        stability_holds = all(r.get("significant_after_correction") for r in stab_rows) if stab_rows else False
        econ_rows = [r for r in economic if r["mechanism"] == mech]
        economic_holds = any(r["scenarios"].get(ECONOMIC_BAR_SCENARIO, {}).get("mean_net", -1) > 0 for r in econ_rows) if econ_rows else False
        if not economic_holds:
            verdicts[mech] = {
                "verdict": "STATISTICALLY_ROBUST_ECONOMICALLY_UNVIABLE", "failed_stage": "economic_gate",
                "reason": "Statistical signal survives chronological/cross-stock generalization but no net-positive result at the Conservative cost scenario.",
                "survivors": mech_survivors, "regime_stable": stability_holds,
            }
            continue
        verdicts[mech] = {"verdict": "VIABLE_CANDIDATE", "survivors": mech_survivors, "regime_stable": stability_holds}
    return verdicts


def main() -> None:
    df = load_dataset()
    print(f"loaded {len(df)} liquid-population rows, {df['security_id'].nunique()} symbols, {df['trading_date'].nunique()} sessions")

    budget = test_budget()
    total = sum(budget.values())
    alpha = 0.05 / total
    print(f"test budget: {budget} -> total {total}, Bonferroni alpha={alpha:.6f}")

    masks = mechanism_masks(df)
    for mech, sides in masks.items():
        print(f"  {mech}: up={int(sides['up'].sum())}, down={int(sides['down'].sum())}")

    standalone = run_standalone(df, masks, alpha)
    print(f"standalone complete: {len(standalone)} rows")
    incremental = run_incremental(df, masks, alpha)
    print(f"incremental complete: {len(incremental)} rows")

    survivors = pick_survivors(standalone, incremental)
    print(f"survivors: {survivors}")

    cross_stock = run_cross_stock_confirmatory(df, masks, survivors)
    chrono = run_chronological_confirmatory(df, masks, survivors)
    stability = run_session_half_stability(df, masks, survivors)
    economic = run_economic_evaluation(df, masks, survivors)
    capacity = run_capacity_evidence(df, masks, survivors)
    verdicts = final_mechanism_verdicts(standalone, incremental, survivors, cross_stock, chrono, stability, economic)
    print(f"verdicts: {verdicts}")

    report = {
        "test_budget": budget, "total_tests": total, "alpha": alpha,
        "population": {"n_rows": len(df), "n_symbols": int(df["security_id"].nunique()), "n_sessions": int(df["trading_date"].nunique())},
        "mechanism_counts": {m: {"up": int(s["up"].sum()), "down": int(s["down"].sum())} for m, s in masks.items()},
        "standalone": standalone, "incremental": incremental, "survivors": survivors,
        "cross_stock": cross_stock, "chronological": chrono, "session_half_stability": stability,
        "economic": economic, "capacity": capacity, "mechanism_verdicts": verdicts,
    }
    out_path = OUT_DIR / "phase_v3_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
