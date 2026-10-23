"""Phase T: entry-timing, MFE/MAE path, and risk-structure research on the frozen Phase R/S reversal signal.

Central question (unchanged from the design freeze): does entering EARLIER, with a small pre-registered
risk/exit structure, preserve enough of the raw Phase R reversal edge to become economically meaningful?

FOUR pre-registered, Bonferroni-corrected test families (budget frozen in test_budget(), see the regression
test in backend/tests/test_phase_t_statistical_research.py proving the declared count equals the actual
produced row count):

  1. entry_timing_horizon (160): 4 buckets x 5 entry offsets x 8 holding horizons - the headline experiment.
  2. mfe_vs_mae (32): 4 buckets x 8 horizons, paired test of favorable vs adverse excursion magnitude, at the
     standard (offset=1) entry.
  3. risk_exit_grid (36): 4 buckets x 9 pre-registered (stop, target) combinations (3x3, thresholds frozen
     BEFORE this script was run), offset=1, max window 60m.
  4. context_association (96): 4 buckets x 2 horizons x 6 existing Phase S context columns x 2 outcomes
     (favorable/adverse excursion), quartile-split two-sample test - "does this help size the risk/opportunity",
     never used as an entry filter.

DIRECTION: reused verbatim from Phase R/S's IS_LONG_BUCKET - never re-derived. gross_return_o{k}_h{h} and the
mfe/mae columns from the Phase T build are LONG-referenced by construction (entry_path_result's own docstring);
sign-flipped here for SHORT-implied buckets before any test, exactly Phase R's corrected convention.

STOP/TARGET RACE - DOCUMENTED APPROXIMATION: the dataset stores MFE/MAE only at 8 discrete checkpoint horizons
(1,2,3,5,10,15,30,60), not a continuous minute-by-minute path (a deliberate scope decision - see the design
freeze). simulate_stop_target() therefore determines "first touch" only to checkpoint resolution: if both a
stop and a target first cross within the SAME checkpoint window, the stop is conservatively assumed to have
triggered first. This is a coarser, explicitly non-fabricated approximation of a true continuous-time
simulation, not a claim of minute-exact execution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.research.statistics import newey_west_mean_test

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "derived" / "phase_t_risk_structure" / "event_paths.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_t_risk_structure"

ENTRY_OFFSETS = (0, 1, 2, 3, 5)
HORIZONS = (1, 2, 3, 5, 10, 15, 30, 60)
BUCKETS = ("top_1pct_rank15m", "top_2pct_rank15m", "bottom_1pct_rank15m", "bottom_2pct_rank15m")
IS_LONG_BUCKET = {"bottom_1pct_rank15m": True, "bottom_2pct_rank15m": True, "top_1pct_rank15m": False, "top_2pct_rank15m": False}
COST_SCENARIOS_BPS = {"low_20bps": 20.0, "medium_40bps": 40.0, "high_60bps": 60.0}
CONFIRMATORY_ALPHA = 0.05
MIN_N = 30

CONTEXT_COLUMNS = (
    "distance_from_ema20_normalized", "ema_20_slope_3", "realized_volatility_20",
    "relative_volume_20", "distance_from_vwap", "consecutive_same_direction_count",
)

STOP_CANDIDATES: dict[str, float | str | None] = {"none": None, "fixed_1pct": 0.01, "vol_2x": "vol_2x"}
TARGET_CANDIDATES: dict[str, float | str | None] = {"none": None, "fixed_2pct": 0.02, "vol_2x": "vol_2x"}


def test_budget() -> dict[str, int]:
    return {
        "entry_timing_horizon": len(BUCKETS) * len(ENTRY_OFFSETS) * len(HORIZONS),
        "mfe_vs_mae": len(BUCKETS) * len(HORIZONS),
        "risk_exit_grid": len(BUCKETS) * len(STOP_CANDIDATES) * len(TARGET_CANDIDATES),
        "context_association": len(BUCKETS) * 2 * len(CONTEXT_COLUMNS) * 2,
    }


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_FILE)


def bucket_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {b: df[b].fillna(False) for b in BUCKETS}


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


def _two_sample_diff_test(top_vals: pd.Series, bottom_vals: pd.Series, *, max_lag_from: int, alpha: float, min_n: int = MIN_N) -> dict[str, Any]:
    top_vals, bottom_vals = top_vals.dropna(), bottom_vals.dropna()
    if len(top_vals) < min_n or len(bottom_vals) < min_n:
        return {"n_top": len(top_vals), "n_bottom": len(bottom_vals), "skipped_reason": "insufficient rows"}
    rt = newey_west_mean_test(top_vals.to_numpy(), max_lag=min(max_lag_from, len(top_vals) - 1))
    rb = newey_west_mean_test(bottom_vals.to_numpy(), max_lag=min(max_lag_from, len(bottom_vals) - 1))
    diff = rt.mean - rb.mean
    se_diff = math.sqrt(rt.standard_error ** 2 + rb.standard_error ** 2)
    if se_diff == 0.0:
        z = math.inf if diff > 0 else (-math.inf if diff < 0 else 0.0)
    else:
        z = diff / se_diff
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {
        "n_top": len(top_vals), "n_bottom": len(bottom_vals), "mean_top": rt.mean, "mean_bottom": rb.mean,
        "diff": diff, "p_value": p_value, "significant_after_correction": bool(p_value < alpha),
    }


def run_entry_timing_horizon(df: pd.DataFrame, masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for bucket, mask in masks.items():
        sub = df.loc[mask]
        is_long = IS_LONG_BUCKET[bucket]
        for k in ENTRY_OFFSETS:
            for h in HORIZONS:
                vals = sub[f"gross_return_o{k}_h{h}"]
                signed = vals if is_long else -vals
                cell = _mean_test_cell(signed, max_lag_from=h - 1, alpha=alpha)
                results.append({"bucket": bucket, "entry_offset_min": k, "horizon_min": h, **cell})
    return results


def run_mfe_vs_mae(df: pd.DataFrame, masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for bucket, mask in masks.items():
        sub = df.loc[mask]
        is_long = IS_LONG_BUCKET[bucket]
        for h in HORIZONS:
            fav_col, adv_col = (f"mfe_o1_h{h}", f"mae_o1_h{h}") if is_long else (f"mae_o1_h{h}", f"mfe_o1_h{h}")
            diff = sub[fav_col] - sub[adv_col]
            cell = _mean_test_cell(diff, max_lag_from=h - 1, alpha=alpha)
            results.append({
                "bucket": bucket, "horizon_min": h, **cell,
                "mean_favorable_excursion": float(sub[fav_col].mean()), "median_favorable_excursion": float(sub[fav_col].median()),
                "mean_adverse_excursion": float(sub[adv_col].mean()), "median_adverse_excursion": float(sub[adv_col].median()),
                "pct_favorable_gte_adverse_at_checkpoint": float((sub[fav_col] >= sub[adv_col]).mean()),
            })
    return results


def simulate_stop_target(sub: pd.DataFrame, is_long: bool, stop_name: str, target_name: str) -> pd.Series:
    n = len(sub)
    stop_raw, target_raw = STOP_CANDIDATES[stop_name], TARGET_CANDIDATES[target_name]
    stop_level = (2.0 * sub["realized_volatility_20"]).to_numpy() if stop_raw == "vol_2x" else stop_raw
    target_level = (2.0 * sub["realized_volatility_20"]).to_numpy() if target_raw == "vol_2x" else target_raw

    stop_hit_h = np.full(n, np.inf)
    target_hit_h = np.full(n, np.inf)
    for h in HORIZONS:
        fav_col, adv_col = (f"mfe_o1_h{h}", f"mae_o1_h{h}") if is_long else (f"mae_o1_h{h}", f"mfe_o1_h{h}")
        fav, adv = sub[fav_col].to_numpy(), sub[adv_col].to_numpy()
        if stop_level is not None:
            newly = (adv >= stop_level) & np.isinf(stop_hit_h)
            stop_hit_h[newly] = h
        if target_level is not None:
            newly = (fav >= target_level) & np.isinf(target_hit_h)
            target_hit_h[newly] = h

    stop_first = stop_hit_h <= target_hit_h
    stop_wins = (~np.isinf(stop_hit_h)) & stop_first
    target_wins = (~np.isinf(target_hit_h)) & (~stop_first)
    neither = np.isinf(stop_hit_h) & np.isinf(target_hit_h)

    exit_return = np.full(n, np.nan)
    if stop_level is not None:
        exit_return = np.where(stop_wins, -stop_level, exit_return)
    if target_level is not None:
        exit_return = np.where(target_wins, target_level, exit_return)
    raw_60 = sub["gross_return_o1_h60"].to_numpy()
    raw_60_signed = raw_60 if is_long else -raw_60
    exit_return = np.where(neither, raw_60_signed, exit_return)
    return pd.Series(exit_return, index=sub.index)


def run_risk_exit_grid(df: pd.DataFrame, masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for bucket, mask in masks.items():
        sub = df.loc[mask]
        is_long = IS_LONG_BUCKET[bucket]
        for stop_name in STOP_CANDIDATES:
            for target_name in TARGET_CANDIDATES:
                exit_returns = simulate_stop_target(sub, is_long, stop_name, target_name)
                cell = _mean_test_cell(exit_returns, max_lag_from=59, alpha=alpha)
                results.append({"bucket": bucket, "stop": stop_name, "target": target_name, **cell})
    return results


def run_context_association(df: pd.DataFrame, masks: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for bucket, mask in masks.items():
        sub = df.loc[mask]
        is_long = IS_LONG_BUCKET[bucket]
        for h in (15, 30):
            fav_col, adv_col = (f"mfe_o1_h{h}", f"mae_o1_h{h}") if is_long else (f"mae_o1_h{h}", f"mfe_o1_h{h}")
            for ctx_col in CONTEXT_COLUMNS:
                ctx = sub[ctx_col]
                q1, q3 = ctx.quantile(0.25), ctx.quantile(0.75)
                top_mask, bottom_mask = ctx >= q3, ctx <= q1
                for outcome_label, col in (("favorable_excursion", fav_col), ("adverse_excursion", adv_col)):
                    test = _two_sample_diff_test(sub.loc[top_mask, col], sub.loc[bottom_mask, col], max_lag_from=h - 1, alpha=alpha)
                    results.append({"bucket": bucket, "horizon_min": h, "context_variable": ctx_col, "outcome": outcome_label, **test})
    return results


def pick_entry_timing_survivors(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("significant_after_correction") and r.get("mean", 0) > 0]


def pick_risk_exit_survivors(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("significant_after_correction") and (r.get("mean", 0) - 60.0 / 10_000.0) > 0]


def run_cross_stock_confirmatory_entry(df: pd.DataFrame, masks: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for s in survivors:
        bucket, k, h = s["bucket"], s["entry_offset_min"], s["horizon_min"]
        sub = df.loc[masks[bucket]]
        col = f"gross_return_o{k}_h{h}"
        per_symbol_n = sub.dropna(subset=[col]).groupby("symbol").size()
        top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
        remainder = sub[~sub["symbol"].isin(top_5)]
        signed = remainder[col] if IS_LONG_BUCKET[bucket] else -remainder[col]
        results.append({**s, "n_symbols_excluded": len(top_5), **_mean_test_cell(signed, max_lag_from=h - 1, alpha=CONFIRMATORY_ALPHA)})
    return results


def run_chronological_confirmatory_entry(df: pd.DataFrame, masks: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}
    out = {"split_sizes": {k: len(v) for k, v in splits.items()}, "by_survivor": []}
    for s in survivors:
        bucket, k, h = s["bucket"], s["entry_offset_min"], s["horizon_min"]
        col = f"gross_return_o{k}_h{h}"
        row = {"bucket": bucket, "entry_offset_min": k, "horizon_min": h}
        for split_name, split_dates in splits.items():
            sub = df.loc[masks[bucket] & df["trading_date"].isin(split_dates)]
            signed = sub[col] if IS_LONG_BUCKET[bucket] else -sub[col]
            row[split_name] = _mean_test_cell(signed, max_lag_from=h - 1, alpha=CONFIRMATORY_ALPHA)
        out["by_survivor"].append(row)
    return out


def run_economic_evaluation(df: pd.DataFrame, masks: dict[str, pd.Series], entry_survivors: list[dict[str, Any]], risk_survivors: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"entry_timing": [], "risk_exit": []}
    for s in entry_survivors:
        bucket, k, h = s["bucket"], s["entry_offset_min"], s["horizon_min"]
        sub = df.loc[masks[bucket]]
        vals = sub[f"gross_return_o{k}_h{h}"].dropna()
        signed = (vals if IS_LONG_BUCKET[bucket] else -vals)
        scenarios = {name: {"cost_bps": bps, "mean_net": float(signed.mean()) - bps / 10_000.0, "hit_rate": float(((signed - bps / 10_000.0) > 0).mean())} for name, bps in COST_SCENARIOS_BPS.items()}
        out["entry_timing"].append({"bucket": bucket, "entry_offset_min": k, "horizon_min": h, "n": len(signed), "gross_mean": float(signed.mean()), "cost_scenarios": scenarios})
    for s in risk_survivors:
        bucket = s["bucket"]
        sub = df.loc[masks[bucket]]
        exit_returns = simulate_stop_target(sub, IS_LONG_BUCKET[bucket], s["stop"], s["target"]).dropna()
        scenarios = {name: {"cost_bps": bps, "mean_net": float(exit_returns.mean()) - bps / 10_000.0, "hit_rate": float(((exit_returns - bps / 10_000.0) > 0).mean())} for name, bps in COST_SCENARIOS_BPS.items()}
        out["risk_exit"].append({"bucket": bucket, "stop": s["stop"], "target": s["target"], "n": len(exit_returns), "gross_mean": float(exit_returns.mean()), "cost_scenarios": scenarios})
    return out


def run_liquidity_breakdown(df: pd.DataFrame, masks: dict[str, pd.Series], entry_survivors: list[dict[str, Any]], risk_survivors: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"entry_timing": [], "risk_exit": []}
    for s in entry_survivors:
        bucket, k, h = s["bucket"], s["entry_offset_min"], s["horizon_min"]
        sub = df.loc[masks[bucket]]
        col = f"gross_return_o{k}_h{h}"
        row = {"bucket": bucket, "entry_offset_min": k, "horizon_min": h, "by_group": {}}
        for grp, g in sub.groupby("liquidity_group", observed=True):
            vals = g[col].dropna()
            signed = vals if IS_LONG_BUCKET[bucket] else -vals
            row["by_group"][str(grp)] = {"n": len(signed), "mean": float(signed.mean()) if len(signed) else None, "mean_net_60bps": float(signed.mean() - 60.0 / 10_000.0) if len(signed) else None}
        out["entry_timing"].append(row)
    for s in risk_survivors:
        bucket = s["bucket"]
        sub = df.loc[masks[bucket]]
        exit_returns = simulate_stop_target(sub, IS_LONG_BUCKET[bucket], s["stop"], s["target"])
        row = {"bucket": bucket, "stop": s["stop"], "target": s["target"], "by_group": {}}
        for grp, idx in sub.groupby("liquidity_group", observed=True).groups.items():
            vals = exit_returns.loc[idx].dropna()
            row["by_group"][str(grp)] = {"n": len(vals), "mean": float(vals.mean()) if len(vals) else None, "mean_net_60bps": float(vals.mean() - 60.0 / 10_000.0) if len(vals) else None}
        out["risk_exit"].append(row)
    return out


def final_verdict(entry_survivors, risk_survivors, cross_stock, chrono, economic) -> dict[str, Any]:
    if not entry_survivors and not risk_survivors:
        return {"verdict": "D", "reason": "No entry-timing or risk-exit structure produced a statistically significant, favorably-signed mean return after correction. No practical structure survives.", "n_survivors": 0}
    cross_holds = all(r.get("significant_after_correction") for r in cross_stock) if cross_stock else False
    chrono_holds = all(split.get("significant_after_correction") for row in chrono.get("by_survivor", []) for split in (row.get("train"), row.get("validation"), row.get("held_out")) if split) if chrono.get("by_survivor") else False
    econ_scenarios = economic.get("entry_timing", []) + economic.get("risk_exit", [])
    economic_holds = any(sc["cost_scenarios"]["high_60bps"]["mean_net"] > 0 for sc in econ_scenarios) if econ_scenarios else False
    n_survivors = len(entry_survivors) + len(risk_survivors)
    if cross_holds and chrono_holds and economic_holds:
        return {"verdict": "A", "reason": "A statistically robust, cross-stock- and chronologically-replicated structure clears the 60bps economic bar.", "n_survivors": n_survivors, "cross_stock_holds": cross_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}
    if cross_holds and chrono_holds:
        return {"verdict": "B", "reason": "Structure(s) survived statistical, cross-stock, and chronological gates but not the 60bps cost scenario.", "n_survivors": n_survivors, "cross_stock_holds": cross_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}
    if cross_holds or chrono_holds:
        return {"verdict": "C", "reason": "Structure(s) survived statistical gates and one but not both generalization checks.", "n_survivors": n_survivors, "cross_stock_holds": cross_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}
    return {"verdict": "C", "reason": "Structure(s) survived statistical significance but did not replicate across stocks or time.", "n_survivors": n_survivors, "cross_stock_holds": cross_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "phase_t_report.json")
    args = ap.parse_args()

    df = load_dataset()
    masks = bucket_masks(df)
    print(f"loaded {len(df)} rows, {df['security_id'].nunique()} symbols")
    for b, m in masks.items():
        print(f"  {b}: {int(m.sum())} rows")

    budget = test_budget()
    total = sum(budget.values())
    alpha = 0.05 / total
    print(f"test budget: {budget} -> total {total}, Bonferroni alpha={alpha:.6f}")

    entry_timing = run_entry_timing_horizon(df, masks, alpha)
    print(f"entry_timing_horizon complete: {len(entry_timing)} rows")
    mfe_mae = run_mfe_vs_mae(df, masks, alpha)
    print(f"mfe_vs_mae complete: {len(mfe_mae)} rows")
    risk_exit = run_risk_exit_grid(df, masks, alpha)
    print(f"risk_exit_grid complete: {len(risk_exit)} rows")
    context = run_context_association(df, masks, alpha)
    print(f"context_association complete: {len(context)} rows")

    entry_survivors = pick_entry_timing_survivors(entry_timing)
    risk_survivors = pick_risk_exit_survivors(risk_exit)
    print(f"entry_timing survivors: {len(entry_survivors)}, risk_exit survivors: {len(risk_survivors)}")

    cross_stock = run_cross_stock_confirmatory_entry(df, masks, entry_survivors)
    chrono = run_chronological_confirmatory_entry(df, masks, entry_survivors)
    economic = run_economic_evaluation(df, masks, entry_survivors, risk_survivors)
    liquidity = run_liquidity_breakdown(df, masks, entry_survivors, risk_survivors)
    verdict = final_verdict(entry_survivors, risk_survivors, cross_stock, chrono, economic)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "test_budget": budget, "total_tests": total, "alpha": alpha,
        "bucket_sizes": {b: int(m.sum()) for b, m in masks.items()},
        "entry_timing_horizon": entry_timing, "mfe_vs_mae": mfe_mae, "risk_exit_grid": risk_exit, "context_association": context,
        "entry_survivors": entry_survivors, "risk_survivors": risk_survivors,
        "cross_stock_confirmatory": cross_stock, "chronological_confirmatory": chrono,
        "economic_evaluation": economic, "liquidity_breakdown": liquidity, "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
