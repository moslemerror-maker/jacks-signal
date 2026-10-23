"""Phase R: cross-sectional momentum & top-gainer/loser direction discovery, approved frozen design.

DATASET: the dense, shared 5-minute cross-sectional grid built by run_phase_r_build_dense_dataset.py directly
from raw persisted 1-minute bars (NOT Phase H's sparse ~3-symbols-per-timestamp sample - see the Phase R
Design Freeze report for why). Every per-symbol ingredient column in that dataset is verified bit-identical to
the already-adversarially-PIT-tested pure functions it reproduces (backend/tests/test_phase_r_build_dense_dataset.py).

CROSS-SECTIONAL RANK - THE ONE GENUINELY NEW QUANTITY THIS PHASE INTRODUCES:
`cs_rank_Xm = groupby(bar_start_utc)[return_Xm].rank(pct=True)` - ranked WITHIN a single fixed timestamp only,
across contemporaneous stocks' already-independently-PIT-safe values. This is the same same-instant
cross-sectional aggregation principle already established safe throughout Phase G-Q for
universe_average_return/universe_breadth/relative_strength_vs_universe - structurally different from the
Phase J T3 leak, which ranked a value against its OWN column across TIME (including future rows of the SAME
stock). Never `groupby(symbol)[...].rank()` and never a whole-dataset rank - both are explicitly forbidden by
this phase's design.

TARGET: direction_label_{h}m (UP/DOWN/AMBIGUOUS), Phase K/L's exact frozen next-bar-open 20bps convention,
recomputed (not redefined) against the denser data foundation - proven bit-identical to
run_phase_k_entry_and_path.entry_and_path in the dataset-builder's own test suite.

POPULATION: FULL only for the exploratory sweep (the entire dense cross-section - cross-sectional rank is
inherently a whole-universe concept, and M3/M4 require mover_mechanics features this dense rebuild does not
compute - regime stability is instead assessed via time-of-day, matching the frozen design's own contingency).

40 FROZEN CANDIDATES (exact groups matching the approved design document):
  - 20 core threshold buckets: cs_rank_5m and cs_rank_15m, each at top/bottom {1%,2%,5%,10%,20%}
  - 4 persistence: top-10%/bottom-10% on cs_rank_5m, 2 or 3 consecutive 5-min samples
  - 4 acceleration: leader/laggard x strengthening/weakening (return_5m vs return_15m/3)
  - 2 volume confirmation: top-10%/bottom-10% x relative_volume_20 top quintile
  - 6 breakout/VWAP (A-F): leader+breakout, laggard+breakdown, leader+above-VWAP, laggard+below-VWAP,
    leader+weakening, laggard+weakening-downside
  - 4 gap+momentum: gap-up+top-rank, gap-down+bottom-rank, gap-up+weakening, gap-down+weakening

A single UP-rate-vs-baseline test (reusing `_finalize_rate_test`, unchanged) captures BOTH the continuation and
reversal hypotheses at once, exactly as Phase Q established for its combination buckets: for a TOP bucket,
lift > 1 is continuation evidence, lift < 1 is reversal evidence, and the reverse for a BOTTOM bucket - stated
as the pre-registered interpretation convention here, before any result is examined.
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
DENSE_FILE = ROOT / "data" / "derived" / "phase_r_cross_sectional" / "dense_symbol_features.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_r_cross_sectional"

HORIZONS = (5, 15, 30, 60)
THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)
RANK_WINDOWS = (5, 15)


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DENSE_FILE)
    for w in RANK_WINDOWS:
        df[f"cs_rank_{w}m"] = df.groupby("bar_start_utc")[f"return_{w}m"].rank(pct=True)
    universe_avg_return_1m = df.groupby("bar_start_utc")["return_1m"].transform("mean")
    df["relative_strength_vs_universe"] = df["return_1m"] - universe_avg_return_1m
    df["universe_coverage_n"] = df.groupby("bar_start_utc")["security_id"].transform("nunique")
    df = df.sort_values(["symbol", "bar_start_utc"]).reset_index(drop=True)
    return df


def coverage_report(df: pd.DataFrame) -> dict[str, Any]:
    per_t = df.groupby("bar_start_utc")["security_id"].nunique()
    return {
        "n_distinct_timestamps": int(len(per_t)), "min_symbols": int(per_t.min()), "median_symbols": float(per_t.median()),
        "max_symbols": int(per_t.max()), "n_timestamps_ge_250_symbols": int((per_t >= 250).sum()),
        "n_timestamps_lt_100_symbols": int((per_t < 100).sum()),
    }


def _bucket(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return mask.fillna(False)


def build_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    candidates: dict[str, pd.Series] = {}

    for w in RANK_WINDOWS:
        rank = df[f"cs_rank_{w}m"]
        for th in THRESHOLDS:
            candidates[f"top_{int(th*100)}pct_rank{w}m"] = _bucket(df, rank >= (1 - th))
            candidates[f"bottom_{int(th*100)}pct_rank{w}m"] = _bucket(df, rank <= th)

    leader = _bucket(df, df["cs_rank_5m"] >= 0.90)
    laggard = _bucket(df, df["cs_rank_5m"] <= 0.10)

    same_symbol = df["symbol"] == df["symbol"].shift(1)
    same_symbol_2 = df["symbol"] == df["symbol"].shift(2)
    leader_prev1 = leader.shift(1).where(same_symbol, other=False)
    leader_prev2 = leader.shift(2).where(same_symbol_2, other=False)
    laggard_prev1 = laggard.shift(1).where(same_symbol, other=False)
    laggard_prev2 = laggard.shift(2).where(same_symbol_2, other=False)
    candidates["persistent_leader_2consec"] = _bucket(df, leader & leader_prev1)
    candidates["persistent_leader_3consec"] = _bucket(df, leader & leader_prev1 & leader_prev2)
    candidates["persistent_laggard_2consec"] = _bucket(df, laggard & laggard_prev1)
    candidates["persistent_laggard_3consec"] = _bucket(df, laggard & laggard_prev1 & laggard_prev2)

    strengthening = df["return_5m"] > (df["return_15m"] / 3.0)
    weakening = df["return_5m"] < (df["return_15m"] / 3.0)
    candidates["leader_momentum_strengthening"] = _bucket(df, leader & strengthening)
    candidates["leader_momentum_weakening"] = _bucket(df, leader & weakening)
    candidates["laggard_downside_strengthening"] = _bucket(df, laggard & weakening)
    candidates["laggard_downside_weakening"] = _bucket(df, laggard & strengthening)

    rel_vol_thr = df["relative_volume_20"].dropna().quantile(0.80)
    high_volume = _bucket(df, df["relative_volume_20"] >= rel_vol_thr)
    candidates["leader_x_high_volume"] = _bucket(df, leader & high_volume)
    candidates["laggard_x_high_volume"] = _bucket(df, laggard & high_volume)

    above_vwap = _bucket(df, df["distance_from_vwap"] > 0)
    below_vwap = _bucket(df, df["distance_from_vwap"] < 0)
    candidates["A_leader_x_breakout"] = _bucket(df, leader & (df["recent_breakout_up_opening_range"] == True))  # noqa: E712
    candidates["B_laggard_x_breakdown"] = _bucket(df, laggard & (df["recent_breakout_down_opening_range"] == True))  # noqa: E712
    candidates["C_leader_x_above_vwap"] = _bucket(df, leader & above_vwap)
    candidates["D_laggard_x_below_vwap"] = _bucket(df, laggard & below_vwap)
    candidates["E_leader_x_weakening"] = _bucket(df, leader & weakening)
    candidates["F_laggard_x_downside_weakening"] = _bucket(df, laggard & strengthening)

    gap_up = _bucket(df, df["overnight_gap"] > 0)
    gap_down = _bucket(df, df["overnight_gap"] < 0)
    candidates["gap_up_x_top_rank"] = _bucket(df, gap_up & leader)
    candidates["gap_down_x_bottom_rank"] = _bucket(df, gap_down & laggard)
    candidates["gap_up_x_weakening"] = _bucket(df, gap_up & weakening)
    candidates["gap_down_x_weakening"] = _bucket(df, gap_down & strengthening)

    return candidates


N_CONTROLS = 2
N_CONTROLLED_GROUPS = 2


def test_budget(candidates: dict[str, pd.Series]) -> dict[str, int]:
    n = len(candidates)
    return {
        "A_standalone": n * len(HORIZONS),
        "B_incremental_value": n * N_CONTROLS * N_CONTROLLED_GROUPS,
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


def _rate_test(df: pd.DataFrame, subset_mask: pd.Series, direction_col: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[subset_mask, [direction_col, "symbol", "bar_start_utc"]]
    sub = sub[sub[direction_col].isin(["UP", "DOWN"])]
    if len(sub) < 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[direction_col].eq("UP").astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha)


def run_family_a(df: pd.DataFrame, candidates: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    results = []
    for h in HORIZONS:
        direction_col = f"direction_label_{h}m"
        baseline_rate = df.loc[df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
        for name, mask in candidates.items():
            results.append({"quantity": name, "horizon_minutes": h, **_rate_test(df, mask, direction_col, h, alpha, baseline_rate)})
    return results


def run_family_b(df: pd.DataFrame, candidates: dict[str, pd.Series], alpha: float) -> list[dict[str, Any]]:
    h = 15
    direction_col = f"direction_label_{h}m"
    results = []
    baseline_up_rate = df.loc[df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
    for control_col in ("return_5m", "relative_strength_vs_universe"):
        med = df[control_col].median()
        for group_name, group_mask in ((f"below_median_{control_col}", df[control_col] < med), (f"at_or_above_median_{control_col}", df[control_col] >= med)):
            for name, mask in candidates.items():
                full_mask = mask & group_mask
                results.append({"quantity": name, "control": control_col, "controlled_group": group_name, **_rate_test(df, full_mask, direction_col, h, alpha, baseline_up_rate)})
    return results


def pick_survivors(family_a: list[dict[str, Any]], family_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_significant = {r["quantity"] for r in family_a if r.get("significant_after_correction")}
    b_by_key: dict[str, list[bool]] = {}
    for r in family_b:
        if "skipped_reason" in r:
            continue
        b_by_key.setdefault(r["quantity"], []).append(r.get("significant_after_correction", False))
    survivors = []
    for quantity in a_significant:
        flags = b_by_key.get(quantity, [])
        if len(flags) == 4 and all(flags):
            survivors.append({"quantity": quantity})
    return survivors


CONFIRMATORY_ALPHA = 0.05


def run_cross_stock_confirmatory(df: pd.DataFrame, candidates: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for s in survivors:
        mask = candidates[s["quantity"]]
        results[s["quantity"]] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            baseline_rate = df.loc[df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
            per_symbol_n = df.loc[mask & df[direction_col].isin(["UP", "DOWN"])].groupby("symbol").size()
            top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
            remainder_mask = mask & ~df["symbol"].isin(top_5)
            results[s["quantity"]][f"{h}m"] = {"n_symbols_excluded": len(top_5), **_rate_test(df, remainder_mask, direction_col, h, CONFIRMATORY_ALPHA, baseline_rate)}
    return results


def run_chronological_confirmatory(df: pd.DataFrame, candidates: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}
    results: dict[str, Any] = {"split_dates": {k: {"n": len(v)} for k, v in splits.items()}, "by_survivor": {}}
    for s in survivors:
        mask = candidates[s["quantity"]]
        results["by_survivor"][s["quantity"]] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            results["by_survivor"][s["quantity"]][f"{h}m"] = {}
            for split_name, split_dates in splits.items():
                split_row_mask = df["trading_date"].isin(split_dates)
                baseline_rate = df.loc[split_row_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
                full_mask = mask & split_row_mask
                results["by_survivor"][s["quantity"]][f"{h}m"][split_name] = _rate_test(df, full_mask, direction_col, h, CONFIRMATORY_ALPHA, baseline_rate)
    return results


def run_regime_stability(df: pd.DataFrame, candidates: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    ist_time = df["bar_start_utc"].dt.tz_convert(IST).dt.time
    opening = (ist_time >= dt.time(9, 15)) & (ist_time < dt.time(10, 30))
    closing = (ist_time >= dt.time(14, 30)) & (ist_time <= dt.time(15, 30))
    midday = ~opening & ~closing
    regimes = {"opening": opening, "midday": midday, "closing": closing}

    results: dict[str, Any] = {}
    for s in survivors:
        mask = candidates[s["quantity"]]
        results[s["quantity"]] = {}
        for regime_name, regime_mask in regimes.items():
            results[s["quantity"]][regime_name] = {}
            for h in HORIZONS:
                direction_col = f"direction_label_{h}m"
                full_mask = mask & regime_mask
                baseline_rate = df.loc[regime_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
                results[s["quantity"]][regime_name][f"{h}m"] = _rate_test(df, full_mask, direction_col, h, CONFIRMATORY_ALPHA, baseline_rate)
    return results


def run_economic_evaluation(df: pd.DataFrame, candidates: dict[str, pd.Series], survivors: list[dict[str, Any]]) -> dict[str, Any]:
    cost_scenarios_bps = {"low_20bps": 20.0, "medium_40bps": 40.0, "high_60bps": 60.0}
    results: dict[str, Any] = {}
    for s in survivors:
        mask = candidates[s["quantity"]]
        results[s["quantity"]] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            ret_col = f"gross_return_{h}m"
            sub_dir = df.loc[mask, direction_col].dropna()
            sub_dir_binary = sub_dir[sub_dir.isin(["UP", "DOWN"])]
            baseline_binary = df.loc[df[direction_col].isin(["UP", "DOWN"]), direction_col]
            bucket_up_rate = (sub_dir_binary == "UP").mean() if len(sub_dir_binary) else None
            baseline_up_rate = (baseline_binary == "UP").mean() if len(baseline_binary) else None
            implied_long = (bucket_up_rate is not None and baseline_up_rate is not None and bucket_up_rate >= baseline_up_rate)

            gross = df.loc[mask, ret_col].dropna()
            if len(gross) < 30:
                results[s["quantity"]][f"{h}m"] = {"skipped_reason": f"insufficient rows ({len(gross)})"}
                continue
            signed_gross = gross if implied_long else -gross
            scenarios = {name: {"cost_bps": bps, "mean_net": float(signed_gross.mean()) - bps / 10_000.0, "hit_rate": float(((signed_gross - bps / 10_000.0) > 0).mean())} for name, bps in cost_scenarios_bps.items()}
            results[s["quantity"]][f"{h}m"] = {"n": len(gross), "implied_side": "LONG" if implied_long else "SHORT", "gross_mean": float(signed_gross.mean()), "cost_scenarios": scenarios}
    return results


def top_gainer_loser_report(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rank = df["cs_rank_5m"]
    buckets = {"top_5pct": rank >= 0.95, "top_10pct": rank >= 0.90, "bottom_10pct": rank <= 0.10, "bottom_5pct": rank <= 0.05}
    for name, mask in buckets.items():
        out[name] = {}
        is_top = name.startswith("top")
        for h in HORIZONS:
            direction_col, ret_col = f"direction_label_{h}m", f"gross_return_{h}m"
            sub = df.loc[mask, direction_col].dropna()
            n = len(sub)
            up_rate = (sub == "UP").mean() if n else None
            down_rate = (sub == "DOWN").mean() if n else None
            amb_rate = (sub == "AMBIGUOUS").mean() if n else None
            continuation_rate = up_rate if is_top else down_rate
            reversal_rate = down_rate if is_top else up_rate
            gross = df.loc[mask, ret_col].dropna()
            out[name][f"{h}m"] = {
                "n": n, "up_probability": up_rate, "down_probability": down_rate, "ambiguous_probability": amb_rate,
                "continuation_rate": continuation_rate, "reversal_rate": reversal_rate,
                "gross_mean_return": float(gross.mean()) if len(gross) else None,
                "net_return_20bps": float(gross.mean() - 0.0020) if len(gross) else None,
                "net_return_40bps": float(gross.mean() - 0.0040) if len(gross) else None,
                "net_return_60bps": float(gross.mean() - 0.0060) if len(gross) else None,
            }
    return out


def build_ledger(family_a, family_b, alpha: float) -> list[dict[str, Any]]:
    ledger = []
    for r in family_a:
        ledger.append({"family": "A_standalone", "hypothesis": f"{r['quantity']} predicts direction at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_b:
        ledger.append({"family": "B_incremental_value", "hypothesis": f"{r['quantity']} adds value beyond {r['control']} ({r.get('controlled_group')})",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(survivors, cross_stock, chrono, economic) -> dict[str, Any]:
    if not survivors:
        return {"verdict": "D", "reason": "No cross-sectional momentum candidate was significant after correction AND demonstrated incremental value beyond both return_5m and relative_strength_vs_universe. No reliable cross-sectional directional mechanism was found.", "n_survivors": 0}

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
        verdict, reason = "A", f"{len(survivors)} candidate(s) survived every gate including realistic 60bps costs."
    elif cross_stock_holds and chrono_holds:
        verdict, reason = "B", f"{len(survivors)} candidate(s) survived statistical and generalization gates but not the 60bps cost scenario."
    elif cross_stock_holds or chrono_holds:
        verdict, reason = "C", f"{len(survivors)} candidate(s) passed statistical gates but only partially replicate (cross-stock: {cross_stock_holds}, chronological: {chrono_holds})."
    else:
        verdict, reason = "D", f"{len(survivors)} candidate(s) passed initial statistical gates but failed both cross-stock and chronological replication."
    return {"verdict": verdict, "reason": reason, "n_survivors": len(survivors), "survivors": survivors, "cross_stock_holds": cross_stock_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_r_report.json")
    args = parser.parse_args()

    df = load_dataset()
    print(f"loaded {len(df)} rows, {len(df.columns)} columns")
    coverage = coverage_report(df)
    print(f"coverage: {coverage}")

    candidates = build_candidates(df)
    print(f"{len(candidates)} candidates built")
    assert len(candidates) == 40, f"expected 40 frozen candidates, got {len(candidates)}"

    budget = test_budget(candidates)
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    family_a = run_family_a(df, candidates, alpha)
    print(f"Family A complete: {len(family_a)} rows")
    family_b = run_family_b(df, candidates, alpha)
    print(f"Family B complete: {len(family_b)} rows")

    survivors = pick_survivors(family_a, family_b)
    print(f"survivors: {survivors}")

    cross_stock = run_cross_stock_confirmatory(df, candidates, survivors)
    chrono = run_chronological_confirmatory(df, candidates, survivors)
    regime = run_regime_stability(df, candidates, survivors)
    economic = run_economic_evaluation(df, candidates, survivors)
    top_gainer_loser = top_gainer_loser_report(df)

    ledger = build_ledger(family_a, family_b, alpha)
    verdict = final_verdict(survivors, cross_stock, chrono, economic)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "coverage": coverage, "test_budget": budget, "bonferroni_alpha": alpha,
        "family_a_standalone": family_a, "family_b_incremental_value": family_b,
        "survivors": survivors, "cross_stock_confirmatory": cross_stock, "chronological_confirmatory": chrono,
        "regime_stability": regime, "economic_evaluation": economic, "top_gainer_loser_report": top_gainer_loser,
        "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
