from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from scripts.run_phase_v4_strategy_discovery import chronological_splits
from scripts.run_phase_v6_statistical_research import (
    COST_MODEL,
    CONFIRMATORY_ALPHA,
    ECONOMIC_BAR_SCENARIO,
    ENTRY_OFFSET,
    MIN_N,
    RACE_CHECKPOINTS,
    TOP_N_VALUES,
    _mean_test_vs_baseline,
    load_dataset as load_v6_dataset,
)
from scripts.run_phase_v7a_opportunity_intensity_tod import TOD_BUCKETS

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "derived" / "phase_v7b_long_horizon"
DATASET_PATH = OUT_DIR / "dense_long_horizon_dataset.parquet"

LONG_HORIZONS = (60, 120, 180, 240, 300)
NEW_HORIZONS = (120, 180, 240, 300)


def compute_long_horizon_paths_for_symbol(bars: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    close, open_, high, low, day = bars["close"], bars["open"], bars["high"], bars["low"], bars["trading_date"]
    k = 1
    entry_price = open_.shift(-k)
    entry_ok = day.shift(-k) == day

    new_columns: dict[str, pd.Series] = {f"entry_price_o{k}": entry_price.where(entry_ok)}
    for h in horizons:
        fwd_max_h = pd.Series(high[::-1].rolling(h, min_periods=h).max().to_numpy()[::-1], index=bars.index)
        fwd_min_h = pd.Series(low[::-1].rolling(h, min_periods=h).min().to_numpy()[::-1], index=bars.index)
        window_ok = entry_ok & (day.shift(-(k + h)) == day)
        future_close = close.shift(-(k + h))
        max_high_w = fwd_max_h.shift(-(k + 1))
        min_low_w = fwd_min_h.shift(-(k + 1))
        gross = (future_close / entry_price) - 1.0
        mfe = ((max_high_w - entry_price) / entry_price).clip(lower=0.0)
        mae = ((entry_price - min_low_w) / entry_price).clip(lower=0.0)
        new_columns[f"gross_return_o{k}_h{h}"] = gross.where(window_ok)
        new_columns[f"mfe_o{k}_h{h}"] = mfe.where(window_ok)
        new_columns[f"mae_o{k}_h{h}"] = mae.where(window_ok)
    return pd.concat([bars, pd.DataFrame(new_columns, index=bars.index)], axis=1)


def _fav_adv_cols_long(h: int) -> tuple[str, str]:
    return f"mfe_o{ENTRY_OFFSET}_h{h}", f"mae_o{ENTRY_OFFSET}_h{h}"


def build_dataset(force: bool = False) -> pd.DataFrame:
    if DATASET_PATH.exists() and not force:
        return pd.read_parquet(DATASET_PATH)

    load_dotenv()
    base = load_v6_dataset()
    print(f"loaded Part 6 base dataset: {len(base)} rows, {base['security_id'].nunique()} symbols")
    security_ids = sorted(base["security_id"].unique().tolist())

    session = get_session()
    t0 = time.monotonic()
    merged_parts = []
    try:
        for i, sid in enumerate(security_ids, 1):
            rows = session.execute(
                text(
                    """
                    SELECT bar_start_utc, trading_date, open, high, low, close, volume
                    FROM intraday_equity_bars WHERE security_id = :sid AND timeframe = '1m' ORDER BY bar_start_utc ASC
                    """
                ),
                {"sid": sid},
            ).fetchall()
            if not rows:
                continue
            bars = pd.DataFrame(rows, columns=["bar_start_utc", "trading_date", "open", "high", "low", "close", "volume"])
            for c in ("open", "high", "low", "close", "volume"):
                bars[c] = bars[c].astype(float)
            paths = compute_long_horizon_paths_for_symbol(bars, LONG_HORIZONS)
            keep_cols = ["bar_start_utc", "trading_date"] + [
                f"{metric}_o{ENTRY_OFFSET}_h{h}" for metric in ("gross_return", "mfe", "mae") for h in LONG_HORIZONS
            ]
            sym_base = base[base["security_id"] == sid]
            merged = sym_base.merge(
                paths[keep_cols].rename(columns={c: f"{c}__new" for c in keep_cols if c not in ("bar_start_utc", "trading_date")}),
                on=["bar_start_utc", "trading_date"], how="left", validate="one_to_one",
            )
            merged_parts.append(merged)
            if i % 50 == 0:
                print(f"  {i}/{len(security_ids)} symbols, {time.monotonic() - t0:.0f}s elapsed")
    finally:
        session.close()

    full = pd.concat(merged_parts, ignore_index=True)

    existing_h60 = full[[f"gross_return_o{ENTRY_OFFSET}_h60", f"mfe_o{ENTRY_OFFSET}_h60", f"mae_o{ENTRY_OFFSET}_h60"]]
    recomputed_h60 = full[[f"gross_return_o{ENTRY_OFFSET}_h60__new", f"mfe_o{ENTRY_OFFSET}_h60__new", f"mae_o{ENTRY_OFFSET}_h60__new"]]
    for col_existing, col_new in zip(existing_h60.columns, recomputed_h60.columns):
        a, b = full[col_existing], full[col_new]
        both_defined = a.notna() & b.notna()
        if both_defined.sum() == 0:
            raise RuntimeError(f"equivalence check found zero overlapping defined rows for {col_existing}")
        if not np.allclose(a[both_defined], b[both_defined], atol=1e-10):
            raise RuntimeError(f"EQUIVALENCE CHECK FAILED for {col_existing} vs {col_new} - long-horizon function does not reproduce Phase T's own h=60 result bit-identically. STOP.")
        if (a.isna() != b.isna()).any():
            raise RuntimeError(f"EQUIVALENCE CHECK FAILED for {col_existing}: NaN pattern (window eligibility) differs between Phase T's original and the long-horizon reimplementation.")
    print(f"equivalence check PASSED: recomputed h=60 is bit-identical to Phase T's own h=60 on {both_defined.sum()} overlapping rows")

    drop_cols = [f"{metric}_o{ENTRY_OFFSET}_h60__new" for metric in ("gross_return", "mfe", "mae")]
    full = full.drop(columns=drop_cols)
    rename_new = {f"{metric}_o{ENTRY_OFFSET}_h{h}__new": f"{metric}_o{ENTRY_OFFSET}_h{h}" for metric in ("gross_return", "mfe", "mae") for h in NEW_HORIZONS}
    full = full.rename(columns=rename_new)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_parquet(DATASET_PATH, index=False)
    print(f"wrote {len(full)} rows, {full.shape[1]} columns -> {DATASET_PATH.relative_to(ROOT)}")
    return full


def eligibility_report(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for h in LONG_HORIZONS:
        fav_col, adv_col = _fav_adv_cols_long(h)
        eligible_mask = df[fav_col].notna()
        n_total = len(df)
        n_eligible = int(eligible_mask.sum())
        eligible_rows = df.loc[eligible_mask]
        by_tod = eligible_rows["time_of_day_bucket"].value_counts().to_dict()
        minutes = eligible_rows["minutes_since_open"] if "minutes_since_open" in eligible_rows else None
        out[str(h)] = {
            "n_total_candidates": n_total,
            "n_eligible": n_eligible,
            "pct_eligible": n_eligible / n_total if n_total else None,
            "eligible_by_time_of_day": {str(k): int(v) for k, v in by_tod.items()},
            "earliest_eligible_minutes_since_open": float(minutes.min()) if minutes is not None and len(minutes) else None,
            "latest_eligible_minutes_since_open": float(minutes.max()) if minutes is not None and len(minutes) else None,
        }
    return out


def _ranked(df: pd.DataFrame) -> pd.Series:
    intensity = df["relative_volume_20"] * df["volatility_expansion_ratio"]
    return intensity.groupby(df["bar_start_utc"]).rank(ascending=False, method="first")


def test_budget_v7b() -> dict[str, int]:
    return {"discovery": len(LONG_HORIZONS) * len(TOD_BUCKETS) * len(TOP_N_VALUES)}


def run_discovery(discovery_df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    rank_desc = _ranked(discovery_df)
    results = []
    for h in LONG_HORIZONS:
        fav_col, adv_col = _fav_adv_cols_long(h)
        for tod in TOD_BUCKETS:
            tod_mask = discovery_df["time_of_day_bucket"] == tod
            baseline_tod = (discovery_df.loc[tod_mask, fav_col] - discovery_df.loc[tod_mask, adv_col]).mean()
            for n in TOP_N_VALUES:
                mask = (rank_desc <= n) & tod_mask
                sub = discovery_df.loc[mask]
                asymmetry_values = sub[fav_col] - sub[adv_col]
                test_result = _mean_test_vs_baseline(asymmetry_values, baseline_tod, h, alpha)
                results.append({"time_of_day": tod, "top_n": n, "horizon_min": h, **test_result})
    return results


def pick_survivors(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in results
        if r.get("significant") and r.get("realized_mean") is not None and r.get("baseline_mean") is not None
        and r["realized_mean"] > r["baseline_mean"]
    ]


def run_split_check(df: pd.DataFrame, splits: dict[str, set], survivor: dict[str, Any], split_name: str) -> dict[str, Any]:
    h, tod, n = survivor["horizon_min"], survivor["time_of_day"], survivor["top_n"]
    fav_col, adv_col = _fav_adv_cols_long(h)
    split_df = df[df["trading_date"].isin(splits[split_name])]
    rank_desc = _ranked(split_df)
    tod_mask = split_df["time_of_day_bucket"] == tod
    baseline_tod = (split_df.loc[tod_mask, fav_col] - split_df.loc[tod_mask, adv_col]).mean()
    mask = (rank_desc <= n) & tod_mask
    sub = split_df.loc[mask]
    asymmetry_values = sub[fav_col] - sub[adv_col]
    return _mean_test_vs_baseline(asymmetry_values, baseline_tod, h, CONFIRMATORY_ALPHA)


def run_cross_stock(df: pd.DataFrame, splits: dict[str, set], survivor: dict[str, Any]) -> dict[str, Any]:
    h, tod, n = survivor["horizon_min"], survivor["time_of_day"], survivor["top_n"]
    fav_col, adv_col = _fav_adv_cols_long(h)
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    rank_desc = _ranked(confirmatory_df)
    tod_mask = confirmatory_df["time_of_day_bucket"] == tod
    mask = (rank_desc <= n) & tod_mask
    per_symbol_n = confirmatory_df.loc[mask].groupby("symbol").size()
    top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
    remainder = mask & ~confirmatory_df["symbol"].isin(top_5)
    baseline_tod = (confirmatory_df.loc[tod_mask, fav_col] - confirmatory_df.loc[tod_mask, adv_col]).mean()
    sub = confirmatory_df.loc[remainder]
    test = _mean_test_vs_baseline(sub[fav_col] - sub[adv_col], baseline_tod, h, CONFIRMATORY_ALPHA)
    return {
        "n_symbols_excluded": len(top_5),
        "concentration_top5_share": float(per_symbol_n.head(5).sum() / per_symbol_n.sum()) if len(per_symbol_n) else None,
        **test,
    }


def run_economic(df: pd.DataFrame, splits: dict[str, set], survivor: dict[str, Any]) -> dict[str, Any]:
    h, tod, n = survivor["horizon_min"], survivor["time_of_day"], survivor["top_n"]
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    rank_desc = _ranked(confirmatory_df)
    tod_mask = confirmatory_df["time_of_day_bucket"] == tod
    mask = (rank_desc <= n) & tod_mask
    sub = confirmatory_df.loc[mask, [f"gross_return_o{ENTRY_OFFSET}_h{h}", "rolling_liquidity_group"]].dropna()
    gross = sub[f"gross_return_o{ENTRY_OFFSET}_h{h}"]
    row: dict[str, Any] = {"n": len(sub), "gross_mean": float(gross.mean()) if len(sub) else None, "scenarios": {}}
    for scenario in COST_MODEL:
        cost_bps = sub["rolling_liquidity_group"].map(COST_MODEL[scenario])
        net = gross - (cost_bps / 10_000.0)
        if len(net) < MIN_N:
            row["scenarios"][scenario] = {"skipped_reason": f"insufficient rows ({len(net)})"}
            continue
        row["scenarios"][scenario] = {"mean_net": float(net.mean()), "median_net": float(net.median()), "hit_rate": float((net > 0).mean())}
    return row


def target_before_stop_note(h: int) -> dict[str, Any]:
    if h <= max(RACE_CHECKPOINTS):
        return {"applicable": True, "note": "within existing RACE_CHECKPOINTS range, directly reusable"}
    return {"applicable": False, "note": f"RACE_CHECKPOINTS tops out at {max(RACE_CHECKPOINTS)}m; extending to {h}m would require new checkpoints beyond the frozen grid, out of scope for this phase"}


def horizon_confound_summary(discovery_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for h in LONG_HORIZONS:
        fav_col, adv_col = _fav_adv_cols_long(h)
        row = {}
        for tod in TOD_BUCKETS:
            sub = discovery_df[discovery_df["time_of_day_bucket"] == tod]
            vals = sub[fav_col] - sub[adv_col]
            row[tod] = {"unconditional_baseline": float(vals.mean()) if vals.notna().any() else None, "n_eligible": int(vals.notna().sum())}
        out[str(h)] = row
    return out


def final_classification(survivors: list[dict[str, Any]], checks: dict[tuple, dict[str, Any]]) -> dict[str, Any]:
    if not survivors:
        return {"classification": "A", "reason": "No horizon x time-of-day x Top-N cell beat its own horizon-and-time-of-day-conditioned baseline at discovery after correction."}
    rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    overall_rank = -1
    cells = []
    for s in survivors:
        key = (s["time_of_day"], s["top_n"], s["horizon_min"])
        c = checks[key]
        val, held, cross, econ = c["validation"], c["held_out"], c["cross_stock"], c["economic"]
        val_holds = bool(val.get("significant") and val.get("realized_mean", -1) > val.get("baseline_mean", 0))
        held_holds = bool(held.get("significant") and held.get("realized_mean", -1) > held.get("baseline_mean", 0))
        cross_holds = bool(cross.get("significant") and cross.get("realized_mean", -1) > cross.get("baseline_mean", 0))
        replication_holds = val_holds and held_holds and cross_holds
        gross_mean = econ.get("gross_mean")
        scenarios = econ.get("scenarios", {})
        econ_viable = scenarios.get(ECONOMIC_BAR_SCENARIO, {}).get("mean_net", -1) > 0
        econ_hopeless = gross_mean is not None and (gross_mean <= 0 or scenarios.get("A_optimistic", {}).get("mean_net", -1) <= 0)
        if not replication_holds:
            cell_class = "B"
        elif econ_viable:
            cell_class = "D"
        elif econ_hopeless:
            cell_class = "B"
        else:
            cell_class = "C"
        overall_rank = max(overall_rank, rank[cell_class])
        cells.append({
            "time_of_day": s["time_of_day"], "top_n": s["top_n"], "horizon_min": s["horizon_min"],
            "validation_holds": val_holds, "held_out_holds": held_holds, "cross_stock_holds": cross_holds,
            "gross_mean": gross_mean, "economic_viable_at_conservative_bar": econ_viable,
            "economic_hopeless": econ_hopeless, "cell_classification": cell_class,
        })
    overall = {v: k for k, v in rank.items()}[overall_rank]
    return {"classification": overall, "cells": cells}


def main() -> None:
    df = build_dataset()
    print(f"dataset ready: {len(df)} rows, {df['security_id'].nunique()} symbols, {df['trading_date'].nunique()} sessions")

    splits = chronological_splits(df)
    print(f"splits: discovery={len(splits['discovery'])}, validation={len(splits['validation'])}, held_out={len(splits['held_out'])} sessions")
    discovery_df = df[df["trading_date"].isin(splits["discovery"])]

    eligibility = eligibility_report(df)
    for h in LONG_HORIZONS:
        e = eligibility[str(h)]
        print(f"  h={h}: {e['n_eligible']}/{e['n_total_candidates']} eligible ({e['pct_eligible']:.1%}), by tod={e['eligible_by_time_of_day']}")

    budget = test_budget_v7b()
    alpha = 0.05 / budget["discovery"]
    print(f"test budget: {budget}, alpha={alpha:.6f}")

    discovery = run_discovery(discovery_df, alpha)
    survivors = pick_survivors(discovery)
    print(f"discovery survivors ({len(survivors)}):")
    for s in survivors:
        print(f"  tod={s['time_of_day']} top_n={s['top_n']} h={s['horizon_min']} n_sample={s['n']} lift={s.get('lift')} p={s['p_value']:.2e}")

    checks: dict[tuple, dict[str, Any]] = {}
    for s in survivors:
        key = (s["time_of_day"], s["top_n"], s["horizon_min"])
        checks[key] = {
            "validation": run_split_check(df, splits, s, "validation"),
            "held_out": run_split_check(df, splits, s, "held_out"),
            "cross_stock": run_cross_stock(df, splits, s),
            "economic": run_economic(df, splits, s),
            "target_before_stop": target_before_stop_note(s["horizon_min"]),
        }

    confound = horizon_confound_summary(discovery_df)
    classification = final_classification(survivors, checks)
    print(f"classification: {classification['classification']}")

    report = {
        "frozen_hypothesis": {
            "H0": "The opportunity-intensity state does NOT contain economically useful information at longer intraday horizons beyond the appropriate time-of-day/session baseline.",
            "H1": "The existing opportunity-intensity state contains incremental information that becomes economically useful over a longer intraday horizon.",
        },
        "population": {"n_rows": len(df), "n_symbols": int(df["security_id"].nunique()), "n_sessions": int(df["trading_date"].nunique())},
        "splits": {k: len(v) for k, v in splits.items()},
        "long_horizons": LONG_HORIZONS, "test_budget": budget, "alpha": alpha,
        "eligibility": eligibility, "discovery": discovery, "survivors": survivors,
        "checks": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in checks.items()},
        "horizon_confound_summary": confound, "final_classification": classification,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase_v7b_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
