"""Phase S: reversal confirmation & selectivity, approved frozen design.

CORE SIGNAL: Phase R's exact, unmodified 15-minute cross-sectional rank buckets - top_1pct_rank15m,
top_2pct_rank15m, bottom_1pct_rank15m, bottom_2pct_rank15m on `cs_rank_15m`, recomputed identically (never
redefined) from the same dense dataset. Bottom buckets are LONG-implied, top buckets SHORT-implied - carried
through every confirmation candidate and the economic evaluation, never assumed long-only.

CENTRAL METHODOLOGICAL CHANGE FROM PHASE R: Family A's baseline is the CORE BUCKET's own UP-rate, not the
general population's - directly testing "does this confirmation add value beyond the Phase R rank signal
itself," the phase's actual question.

24 FROZEN CANDIDATES across 5 families (A: EMA, B: exhaustion, C: volume, D: price-action/S-R, E: multi-
timeframe structure) - every mirror pair uses an EXPLICIT directional formula, never boolean negation of the
long-side condition, since negation would asymmetrically fold the exact-zero/flat case onto the short side.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.app.research.statistics import newey_west_mean_test

ROOT = Path(__file__).resolve().parents[1]
EXTENDED_FILE = ROOT / "data" / "derived" / "phase_s_confirmation" / "dense_symbol_features_extended.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_s_confirmation"

HORIZONS = (5, 15, 30, 60)
IS_LONG_BUCKET = {"bottom_1pct_rank15m": True, "bottom_2pct_rank15m": True, "top_1pct_rank15m": False, "top_2pct_rank15m": False}


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(EXTENDED_FILE)
    df["cs_rank_15m"] = df.groupby("bar_start_utc")["return_15m"].rank(pct=True)
    universe_avg_return_1m = df.groupby("bar_start_utc")["return_1m"].transform("mean")
    df["relative_strength_vs_universe"] = df["return_1m"] - universe_avg_return_1m
    df = df.sort_values(["symbol", "bar_start_utc"]).reset_index(drop=True)
    return df


def core_bucket_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    rank = df["cs_rank_15m"]
    return {
        "top_1pct_rank15m": (rank >= 0.99).fillna(False),
        "top_2pct_rank15m": (rank >= 0.98).fillna(False),
        "bottom_1pct_rank15m": (rank <= 0.01).fillna(False),
        "bottom_2pct_rank15m": (rank <= 0.02).fillna(False),
    }


def _q(s: pd.Series, q: float) -> float:
    return s.dropna().quantile(q)


def A1_price_reclaims_ema9(df, bucket_mask, is_long):
    return df["close"] > df["ema_9"] if is_long else df["close"] < df["ema_9"]


def A2_price_reclaims_ema20(df, bucket_mask, is_long):
    return df["close"] > df["ema_20"] if is_long else df["close"] < df["ema_20"]


def A3_ema9_above_ema20(df, bucket_mask, is_long):
    return df["ema_9"] > df["ema_20"] if is_long else df["ema_9"] < df["ema_20"]


def A4_ema20_above_ema50(df, bucket_mask, is_long):
    return df["ema_20"] > df["ema_50"] if is_long else df["ema_20"] < df["ema_50"]


def A5_ema20_sloping_favorably(df, bucket_mask, is_long):
    return df["ema_20_slope_3"] > 0 if is_long else df["ema_20_slope_3"] < 0


def A6_stretched_from_ema20(df, bucket_mask, is_long):
    thr = _q(df.loc[bucket_mask, "distance_from_ema20_normalized"].abs(), 0.75)
    return bucket_mask & (df["distance_from_ema20_normalized"].abs() >= thr)


def B1_rsi_extreme_aligned(df, bucket_mask, is_long):
    return df["rsi_14"] <= 30 if is_long else df["rsi_14"] >= 70


def B2_rsi_reversing(df, bucket_mask, is_long):
    return df["rsi_14_slope_5"] > 0 if is_long else df["rsi_14_slope_5"] < 0


def B3_macd_histogram_aligned(df, bucket_mask, is_long):
    return df["macd_histogram_12_26_9"] > 0 if is_long else df["macd_histogram_12_26_9"] < 0


def B4_vwap_reclaim(df, bucket_mask, is_long):
    dfv = df["distance_from_vwap"]
    now_right_side = (dfv > 0) if is_long else (dfv < 0)
    was_wrong_recently = pd.Series(False, index=df.index)
    for lag in (1, 2, 3):
        prev_lag = dfv.shift(lag)
        was_wrong_recently = was_wrong_recently | ((prev_lag < 0) if is_long else (prev_lag > 0))
    return (now_right_side & was_wrong_recently).fillna(False)


def B5_exhaustion_sequence(df, bucket_mask, is_long):
    thr = _q(df.loc[bucket_mask, "consecutive_same_direction_count"], 0.75)
    return bucket_mask & (df["consecutive_same_direction_count"] >= thr)


def C1_elevated_volume(df, bucket_mask, is_long):
    thr = _q(df.loc[bucket_mask, "relative_volume_20"], 0.75)
    return bucket_mask & (df["relative_volume_20"] >= thr)


def C2_volume_expanding(df, bucket_mask, is_long):
    return df["relative_volume_20"] > df["relative_volume_20"].shift(3)


def C3_volume_contracting(df, bucket_mask, is_long):
    return df["relative_volume_20"] < df["relative_volume_20"].shift(3)


def C4_early_reversal_with_volume(df, bucket_mask, is_long):
    aligned = (df["return_1m"] > 0) & (df["return_3m"] > 0) if is_long else (df["return_1m"] < 0) & (df["return_3m"] < 0)
    thr = _q(df.loc[bucket_mask, "relative_volume_20"], 0.75)
    return aligned & (df["relative_volume_20"] >= thr)


def D1_near_nearest_support_resistance(df, bucket_mask, is_long):
    dist = df["distance_to_nearest_support"] if is_long else df["distance_to_nearest_resistance"]
    thr = _q(dist.where(bucket_mask).abs(), 0.25)
    return bucket_mask & (dist.abs() <= thr)


def D2_near_session_high_low_so_far(df, bucket_mask, is_long):
    level = df["session_low_so_far"] if is_long else df["session_high_so_far"]
    dist = (df["close"] - level).abs() / df["close"]
    thr = _q(dist.where(bucket_mask), 0.25)
    return bucket_mask & (dist <= thr)


def D3_near_opening_range_boundary(df, bucket_mask, is_long):
    level = df["opening_range_low"] if is_long else df["opening_range_high"]
    dist = (df["close"] - level).abs() / df["close"]
    thr = _q(dist.where(bucket_mask), 0.25)
    return bucket_mask & (dist <= thr)


def D4_near_vwap(df, bucket_mask, is_long):
    thr = _q(df.loc[bucket_mask, "distance_from_vwap"].abs(), 0.25)
    return bucket_mask & (df["distance_from_vwap"].abs() <= thr)


def D5_breakout_then_rejection(df, bucket_mask, is_long):
    breakout_col = "recent_breakout_up_opening_range" if not is_long else "recent_breakout_down_opening_range"
    level_col = "opening_range_high" if not is_long else "opening_range_low"
    had_breakout_recently = df[breakout_col].astype("boolean").fillna(False)
    for lag in range(1, 5):
        had_breakout_recently = had_breakout_recently | df[breakout_col].shift(lag).astype("boolean").fillna(False)
    rejected_back = (df["close"] < df[level_col]) if not is_long else (df["close"] > df[level_col])
    return had_breakout_recently & rejected_back


def E1_short_term_already_reversing(df, bucket_mask, is_long):
    return (df["return_1m"] > 0) & (df["return_3m"] > 0) if is_long else (df["return_1m"] < 0) & (df["return_3m"] < 0)


def E2_thirty_min_decelerating(df, bucket_mask, is_long):
    return df["return_30m"].abs() < df["return_15m"].abs() * 2.0


def E3_trend_disagreement(df, bucket_mask, is_long):
    return np.sign(df["return_5m"]) != np.sign(df["return_30m"])


def E4_against_day_trend(df, bucket_mask, is_long):
    return df["distance_from_session_open"] < 0 if is_long else df["distance_from_session_open"] > 0


CANDIDATES: dict[str, Callable] = {
    "A1_price_reclaims_ema9": A1_price_reclaims_ema9, "A2_price_reclaims_ema20": A2_price_reclaims_ema20,
    "A3_ema9_above_ema20": A3_ema9_above_ema20, "A4_ema20_above_ema50": A4_ema20_above_ema50,
    "A5_ema20_sloping_favorably": A5_ema20_sloping_favorably, "A6_stretched_from_ema20": A6_stretched_from_ema20,
    "B1_rsi_extreme_aligned": B1_rsi_extreme_aligned, "B2_rsi_reversing": B2_rsi_reversing,
    "B3_macd_histogram_aligned": B3_macd_histogram_aligned, "B4_vwap_reclaim": B4_vwap_reclaim,
    "B5_exhaustion_sequence": B5_exhaustion_sequence, "C1_elevated_volume": C1_elevated_volume,
    "C2_volume_expanding": C2_volume_expanding, "C3_volume_contracting": C3_volume_contracting,
    "C4_early_reversal_with_volume": C4_early_reversal_with_volume,
    "D1_near_nearest_support_resistance": D1_near_nearest_support_resistance,
    "D2_near_session_high_low_so_far": D2_near_session_high_low_so_far,
    "D3_near_opening_range_boundary": D3_near_opening_range_boundary, "D4_near_vwap": D4_near_vwap,
    "D5_breakout_then_rejection": D5_breakout_then_rejection,
    "E1_short_term_already_reversing": E1_short_term_already_reversing,
    "E2_thirty_min_decelerating": E2_thirty_min_decelerating, "E3_trend_disagreement": E3_trend_disagreement,
    "E4_against_day_trend": E4_against_day_trend,
}
assert len(CANDIDATES) == 24, f"expected 24 frozen candidates, got {len(CANDIDATES)}"

N_CONTROLS = 2
N_CONTROLLED_GROUPS = 2


def test_budget() -> dict[str, int]:
    n_buckets, n_candidates = len(IS_LONG_BUCKET), len(CANDIDATES)
    return {
        "A_standalone": n_buckets * n_candidates * len(HORIZONS),
        "B_incremental_value": n_buckets * n_candidates * N_CONTROLS * N_CONTROLLED_GROUPS,
    }


def _finalize_rate_test(ordered_flags: list[float], baseline_rate: float, horizon: int, alpha: float) -> dict[str, Any]:
    if len(ordered_flags) < 10:
        return {"skipped_reason": f"insufficient rows ({len(ordered_flags)})"}
    centered = [f - baseline_rate for f in ordered_flags]
    max_lag = min(horizon - 1, len(centered) - 1)
    result = newey_west_mean_test(centered, max_lag=max_lag)
    bucket_rate = baseline_rate + result.mean
    return {
        "n_rows": len(ordered_flags), "confirmed_up_rate": bucket_rate, "baseline_up_rate": baseline_rate,
        "lift": bucket_rate / baseline_rate if baseline_rate else None,
        "p_value": result.p_value, "significant_after_correction": bool(result.p_value < alpha),
    }


def _rate_test(df: pd.DataFrame, subset_mask: pd.Series, direction_col: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[subset_mask, [direction_col, "symbol", "bar_start_utc"]]
    sub = sub[sub[direction_col].isin(["UP", "DOWN"])]
    if len(sub) < 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[direction_col].eq("UP").astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha)


def run_family_a(df: pd.DataFrame, bucket_masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for bucket_name, bucket_mask in bucket_masks.items():
        is_long = IS_LONG_BUCKET[bucket_name]
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            bucket_baseline_rate = df.loc[bucket_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
            for name, fn in CANDIDATES.items():
                confirmed_mask = bucket_mask & fn(df, bucket_mask, is_long).fillna(False)
                results.append({"bucket": bucket_name, "candidate": name, "horizon_minutes": h, **_rate_test(df, confirmed_mask, direction_col, h, alpha, bucket_baseline_rate)})
    return results


def run_family_b(df: pd.DataFrame, bucket_masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    h = 15
    direction_col = f"direction_label_{h}m"
    results = []
    for bucket_name, bucket_mask in bucket_masks.items():
        is_long = IS_LONG_BUCKET[bucket_name]
        bucket_baseline_rate = df.loc[bucket_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
        for control_col in ("return_5m", "relative_strength_vs_universe"):
            med = df.loc[bucket_mask, control_col].median()
            for group_name, group_mask in ((f"below_median_{control_col}", bucket_mask & (df[control_col] < med)), (f"at_or_above_median_{control_col}", bucket_mask & (df[control_col] >= med))):
                for name, fn in CANDIDATES.items():
                    confirmed_mask = group_mask & fn(df, bucket_mask, is_long).fillna(False)
                    results.append({"bucket": bucket_name, "candidate": name, "control": control_col, "controlled_group": group_name, **_rate_test(df, confirmed_mask, direction_col, h, alpha, bucket_baseline_rate)})
    return results


def pick_survivors(family_a: list[dict[str, Any]], family_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_significant = {(r["bucket"], r["candidate"]) for r in family_a if r.get("significant_after_correction")}
    b_by_key: dict[tuple[str, str], list[bool]] = {}
    for r in family_b:
        if "skipped_reason" in r:
            continue
        b_by_key.setdefault((r["bucket"], r["candidate"]), []).append(r.get("significant_after_correction", False))
    survivors = []
    for (bucket, candidate) in a_significant:
        flags = b_by_key.get((bucket, candidate), [])
        if len(flags) == 4 and all(flags):
            survivors.append({"bucket": bucket, "candidate": candidate})
    return survivors


CONFIRMATORY_ALPHA = 0.05


def _survivor_confirmed_mask(df: pd.DataFrame, s: dict[str, Any], bucket_masks: dict[str, pd.Series]) -> pd.Series:
    bucket_mask = bucket_masks[s["bucket"]]
    is_long = IS_LONG_BUCKET[s["bucket"]]
    return bucket_mask & CANDIDATES[s["candidate"]](df, bucket_mask, is_long).fillna(False)


def run_cross_stock_confirmatory(df: pd.DataFrame, bucket_masks: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for s in survivors:
        key = f"{s['bucket']}__{s['candidate']}"
        confirmed = _survivor_confirmed_mask(df, s, bucket_masks)
        results[key] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            bucket_baseline_rate = df.loc[bucket_masks[s["bucket"]] & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
            per_symbol_n = df.loc[confirmed & df[direction_col].isin(["UP", "DOWN"])].groupby("symbol").size()
            top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
            remainder = confirmed & ~df["symbol"].isin(top_5)
            results[key][f"{h}m"] = {"n_symbols_excluded": len(top_5), **_rate_test(df, remainder, direction_col, h, CONFIRMATORY_ALPHA, bucket_baseline_rate)}
    return results


def run_chronological_confirmatory(df: pd.DataFrame, bucket_masks: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}
    results: dict[str, Any] = {"split_dates": {k: {"n": len(v)} for k, v in splits.items()}, "by_survivor": {}}
    for s in survivors:
        key = f"{s['bucket']}__{s['candidate']}"
        confirmed = _survivor_confirmed_mask(df, s, bucket_masks)
        bucket_mask = bucket_masks[s["bucket"]]
        results["by_survivor"][key] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            results["by_survivor"][key][f"{h}m"] = {}
            for split_name, split_dates in splits.items():
                split_row_mask = df["trading_date"].isin(split_dates)
                bucket_baseline_rate = df.loc[bucket_mask & split_row_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
                mask = confirmed & split_row_mask
                results["by_survivor"][key][f"{h}m"][split_name] = _rate_test(df, mask, direction_col, h, CONFIRMATORY_ALPHA, bucket_baseline_rate)
    return results


def run_economic_evaluation(df: pd.DataFrame, survivors: list[dict[str, Any]], bucket_masks: dict[str, pd.Series]) -> dict[str, Any]:
    cost_scenarios_bps = {"low_20bps": 20.0, "medium_40bps": 40.0, "high_60bps": 60.0}
    results: dict[str, Any] = {}
    for s in survivors:
        key = f"{s['bucket']}__{s['candidate']}"
        confirmed = _survivor_confirmed_mask(df, s, bucket_masks)
        is_long = IS_LONG_BUCKET[s["bucket"]]
        results[key] = {}
        for h in HORIZONS:
            ret_col = f"gross_return_{h}m"
            gross = df.loc[confirmed, ret_col].dropna()
            if len(gross) < 30:
                results[key][f"{h}m"] = {"skipped_reason": f"insufficient rows ({len(gross)})"}
                continue
            signed = gross if is_long else -gross
            scenarios = {name: {"cost_bps": bps, "mean_net": float(signed.mean()) - bps / 10_000.0, "hit_rate": float(((signed - bps / 10_000.0) > 0).mean())} for name, bps in cost_scenarios_bps.items()}
            results[key][f"{h}m"] = {"n": len(gross), "implied_side": "LONG" if is_long else "SHORT", "gross_mean": float(signed.mean()), "cost_scenarios": scenarios}
    return results


def compare_against_phase_r_alone(df: pd.DataFrame, bucket_masks: dict[str, pd.Series]) -> dict[str, Any]:
    cost_scenarios_bps = {"low_20bps": 20.0, "medium_40bps": 40.0, "high_60bps": 60.0}
    results: dict[str, Any] = {}
    for bucket_name, mask in bucket_masks.items():
        is_long = IS_LONG_BUCKET[bucket_name]
        results[bucket_name] = {}
        for h in HORIZONS:
            gross = df.loc[mask, f"gross_return_{h}m"].dropna()
            signed = gross if is_long else -gross
            scenarios = {name: {"mean_net": float(signed.mean()) - bps / 10_000.0} for name, bps in cost_scenarios_bps.items()}
            results[bucket_name][f"{h}m"] = {"n": len(gross), "gross_mean": float(signed.mean()), "cost_scenarios": scenarios}
    return results


def build_ledger(family_a, family_b, alpha: float) -> list[dict[str, Any]]:
    ledger = []
    for r in family_a:
        ledger.append({"family": "A_standalone_vs_bucket", "hypothesis": f"{r['candidate']} adds value within {r['bucket']} at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_b:
        ledger.append({"family": "B_incremental_value", "hypothesis": f"{r['candidate']} within {r['bucket']} adds value beyond {r['control']} ({r.get('controlled_group')})",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(survivors, cross_stock, chrono, economic) -> dict[str, Any]:
    if not survivors:
        return {"verdict": "D", "reason": "No confirmation candidate added significant value beyond the Phase R bucket alone AND demonstrated incremental value beyond return_5m/relative_strength_vs_universe. Confirmation does not improve the Phase R signal.", "n_survivors": 0}
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
        verdict, reason = "A", f"{len(survivors)} confirmation(s) survived every gate including realistic 60bps costs."
    elif cross_stock_holds and chrono_holds:
        verdict, reason = "B", f"{len(survivors)} confirmation(s) survived statistical and generalization gates but not the 60bps cost scenario."
    elif cross_stock_holds or chrono_holds:
        verdict, reason = "C", f"{len(survivors)} confirmation(s) passed statistical gates but only partially replicate (cross-stock: {cross_stock_holds}, chronological: {chrono_holds})."
    else:
        verdict, reason = "D", f"{len(survivors)} confirmation(s) passed initial statistical gates but failed both cross-stock and chronological replication."
    return {"verdict": verdict, "reason": reason, "n_survivors": len(survivors), "survivors": survivors, "cross_stock_holds": cross_stock_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_s_report.json")
    args = parser.parse_args()

    df = load_dataset()
    print(f"loaded {len(df)} rows, {len(df.columns)} columns")
    bucket_masks = core_bucket_masks(df)
    for name, mask in bucket_masks.items():
        print(f"  {name}: {int(mask.sum())} rows")

    budget = test_budget()
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    family_a = run_family_a(df, bucket_masks, alpha)
    print(f"Family A complete: {len(family_a)} rows")
    family_b = run_family_b(df, bucket_masks, alpha)
    print(f"Family B complete: {len(family_b)} rows")

    survivors = pick_survivors(family_a, family_b)
    print(f"survivors: {survivors}")

    cross_stock = run_cross_stock_confirmatory(df, bucket_masks, survivors)
    chrono = run_chronological_confirmatory(df, bucket_masks, survivors)
    economic = run_economic_evaluation(df, survivors, bucket_masks)
    phase_r_alone = compare_against_phase_r_alone(df, bucket_masks)

    ledger = build_ledger(family_a, family_b, alpha)
    verdict = final_verdict(survivors, cross_stock, chrono, economic)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "bucket_sizes": {k: int(v.sum()) for k, v in bucket_masks.items()},
        "test_budget": budget, "bonferroni_alpha": alpha,
        "family_a_standalone": family_a, "family_b_incremental_value": family_b,
        "survivors": survivors, "cross_stock_confirmatory": cross_stock, "chronological_confirmatory": chrono,
        "economic_evaluation": economic, "phase_r_alone_comparison": phase_r_alone,
        "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
