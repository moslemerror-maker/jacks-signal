from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_phase_v4_strategy_discovery import chronological_splits
from scripts.run_phase_v6_statistical_research import (
    ALL_HORIZONS,
    COST_MODEL,
    CONFIRMATORY_ALPHA,
    DISCOVERY_HORIZONS,
    ECONOMIC_BAR_SCENARIO,
    ENTRY_OFFSET,
    MIN_N,
    TOP_N_VALUES,
    _fav_adv_cols,
    _mean_test_vs_baseline,
    baseline_asymmetry,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "derived" / "phase_v7a_opportunity_intensity_tod"

TOD_BUCKETS = ("opening", "mid", "late")


def _ranked(df: pd.DataFrame) -> pd.Series:
    intensity = df["relative_volume_20"] * df["volatility_expansion_ratio"]
    return intensity.groupby(df["bar_start_utc"]).rank(ascending=False, method="first")


def baseline_asymmetry_by_tod(df: pd.DataFrame, h: int, tod: str) -> float:
    return baseline_asymmetry(df[df["time_of_day_bucket"] == tod], h, True)


def test_budget_v7a() -> dict[str, int]:
    return {
        "discovery_by_tod": len(TOD_BUCKETS) * len(TOP_N_VALUES) * len(DISCOVERY_HORIZONS),
        "confirmatory_by_tod": len(TOD_BUCKETS) * len(TOP_N_VALUES) * len(ALL_HORIZONS) * 2,
    }


def run_discovery_by_tod(discovery_df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    rank_desc = _ranked(discovery_df)
    results = []
    for h in DISCOVERY_HORIZONS:
        fav_col, adv_col = _fav_adv_cols(h, True)
        for tod in TOD_BUCKETS:
            tod_mask = discovery_df["time_of_day_bucket"] == tod
            baseline_tod = baseline_asymmetry_by_tod(discovery_df, h, tod)
            for n in TOP_N_VALUES:
                mask = (rank_desc <= n) & tod_mask
                sub = discovery_df.loc[mask]
                asymmetry_values = sub[fav_col] - sub[adv_col]
                test_result = _mean_test_vs_baseline(asymmetry_values, baseline_tod, h, alpha)
                results.append({
                    "family": "top_n_ranking_by_tod", "time_of_day": tod, "top_n": n, "horizon_min": h,
                    **test_result,
                })
    return results


def pick_survivors_by_tod(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in results
        if r.get("significant") and r.get("realized_mean") is not None and r.get("baseline_mean") is not None
        and r["realized_mean"] > r["baseline_mean"]
    ]


def run_confirmatory_by_tod(df: pd.DataFrame, splits: dict[str, set]) -> list[dict[str, Any]]:
    results = []
    for split_name in ("validation", "held_out"):
        split_df = df[df["trading_date"].isin(splits[split_name])]
        rank_desc = _ranked(split_df)
        for h in ALL_HORIZONS:
            fav_col, adv_col = _fav_adv_cols(h, True)
            for tod in TOD_BUCKETS:
                tod_mask = split_df["time_of_day_bucket"] == tod
                baseline_tod = baseline_asymmetry_by_tod(split_df, h, tod)
                for n in TOP_N_VALUES:
                    mask = (rank_desc <= n) & tod_mask
                    sub = split_df.loc[mask]
                    asymmetry_values = sub[fav_col] - sub[adv_col]
                    test_result = _mean_test_vs_baseline(asymmetry_values, baseline_tod, h, CONFIRMATORY_ALPHA)
                    results.append({
                        "split": split_name, "time_of_day": tod, "top_n": n, "horizon_min": h,
                        **test_result,
                    })
    return results


def _confirmatory_cells_for(survivor: dict[str, Any], confirmatory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for split_name in ("validation", "held_out"):
        for row in confirmatory:
            if (row["split"] == split_name and row["time_of_day"] == survivor["time_of_day"]
                    and row["top_n"] == survivor["top_n"] and row["horizon_min"] == survivor["horizon_min"]):
                out[split_name] = row
                break
    return out


def run_cross_stock_by_tod(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    rank_desc = _ranked(confirmatory_df)
    results = []
    for s in survivors:
        h, tod, n = s["horizon_min"], s["time_of_day"], s["top_n"]
        tod_mask = confirmatory_df["time_of_day_bucket"] == tod
        mask = (rank_desc <= n) & tod_mask
        per_symbol_n = confirmatory_df.loc[mask].groupby("symbol").size()
        top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
        remainder = mask & ~confirmatory_df["symbol"].isin(top_5)
        baseline_tod = baseline_asymmetry_by_tod(confirmatory_df, h, tod)
        fav_col, adv_col = _fav_adv_cols(h, True)
        sub = confirmatory_df.loc[remainder]
        test = _mean_test_vs_baseline(sub[fav_col] - sub[adv_col], baseline_tod, h, CONFIRMATORY_ALPHA)
        results.append({
            **s, "n_symbols_excluded": len(top_5),
            "concentration_top5_share": float(per_symbol_n.head(5).sum() / per_symbol_n.sum()) if len(per_symbol_n) else None,
            **test,
        })
    return results


def run_economic_by_tod(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    rank_desc = _ranked(confirmatory_df)
    out = []
    for s in survivors:
        h, tod, n = s["horizon_min"], s["time_of_day"], s["top_n"]
        tod_mask = confirmatory_df["time_of_day_bucket"] == tod
        mask = (rank_desc <= n) & tod_mask
        sub = confirmatory_df.loc[mask, [f"gross_return_o{ENTRY_OFFSET}_h{h}", "rolling_liquidity_group"]].dropna()
        gross = sub[f"gross_return_o{ENTRY_OFFSET}_h{h}"]
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


def opening_confound_summary(discovery_df: pd.DataFrame) -> dict[str, Any]:
    n_total = len(discovery_df)
    out: dict[str, Any] = {"bucket_row_share": {}, "bucket_unconditional_baseline": {}}
    for tod in TOD_BUCKETS:
        sub = discovery_df[discovery_df["time_of_day_bucket"] == tod]
        out["bucket_row_share"][tod] = len(sub) / n_total if n_total else None
        out["bucket_unconditional_baseline"][tod] = {
            h: baseline_asymmetry(sub, h, True) for h in DISCOVERY_HORIZONS
        }
    return out


def final_classification(survivors: list[dict[str, Any]], confirmatory: list[dict[str, Any]],
                          cross_stock: list[dict[str, Any]], economic: list[dict[str, Any]]) -> dict[str, Any]:
    if not survivors:
        return {"classification": "A", "reason": "No bucket x Top-N x horizon cell beat its own time-of-day-conditioned baseline at discovery after correction. Time-of-day confounded / explains the pooled Part 6 result."}
    cells = []
    worst_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    overall_rank = -1
    for s, cs, econ in zip(survivors, cross_stock, economic):
        conf = _confirmatory_cells_for(s, confirmatory)
        val_holds = bool(conf.get("validation", {}).get("significant", False) and conf["validation"].get("realized_mean", -1) > conf["validation"].get("baseline_mean", 0))
        held_holds = bool(conf.get("held_out", {}).get("significant", False) and conf["held_out"].get("realized_mean", -1) > conf["held_out"].get("baseline_mean", 0))
        cross_holds = bool(cs.get("significant", False) and cs.get("realized_mean", -1) > cs.get("baseline_mean", 0))
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
        overall_rank = max(overall_rank, worst_rank[cell_class])
        cells.append({
            "time_of_day": s["time_of_day"], "top_n": s["top_n"], "horizon_min": s["horizon_min"],
            "validation_holds": val_holds, "held_out_holds": held_holds, "cross_stock_holds": cross_holds,
            "gross_mean": gross_mean, "economic_viable_at_conservative_bar": econ_viable,
            "economic_hopeless": econ_hopeless, "cell_classification": cell_class,
        })
    overall = {v: k for k, v in worst_rank.items()}[overall_rank]
    return {"classification": overall, "cells": cells}


def main() -> None:
    df = load_dataset()
    print(f"loaded {len(df)} rows, {df['security_id'].nunique()} symbols, {df['trading_date'].nunique()} sessions")
    splits = chronological_splits(df)
    print(f"splits: discovery={len(splits['discovery'])}, validation={len(splits['validation'])}, held_out={len(splits['held_out'])} sessions")
    discovery_df = df[df["trading_date"].isin(splits["discovery"])]

    budget = test_budget_v7a()
    disc_alpha = 0.05 / budget["discovery_by_tod"]
    print(f"test budget: {budget}, discovery alpha={disc_alpha:.6f}")

    discovery = run_discovery_by_tod(discovery_df, disc_alpha)
    survivors = pick_survivors_by_tod(discovery)
    print(f"discovery survivors ({len(survivors)}):")
    for s in survivors:
        print(f"  tod={s['time_of_day']} top_n={s['top_n']} h={s['horizon_min']} n_sample={s['n']} lift={s['lift']:.3f} p={s['p_value']:.2e}")

    confirmatory = run_confirmatory_by_tod(df, splits)
    cross_stock = run_cross_stock_by_tod(df, splits, survivors)
    economic = run_economic_by_tod(df, splits, survivors)
    confound = opening_confound_summary(discovery_df)
    classification = final_classification(survivors, confirmatory, cross_stock, economic)
    print(f"classification: {classification['classification']}")

    report = {
        "frozen_hypothesis": {
            "H0": "The apparent Phase V6 opportunity-intensity advantage is fully explained by time-of-day.",
            "H1": "Opportunity intensity retains incremental predictive/opportunity value within time-of-day conditions.",
        },
        "time_of_day_bucket_definition": {"opening": "minutes_since_open <= 30", "mid": "30 < minutes_since_open < 180", "late": "minutes_since_open >= 180", "source": "scripts/run_phase_v4_strategy_discovery.py:120-123 (frozen, unmodified)"},
        "population": {"n_rows": len(df), "n_symbols": int(df["security_id"].nunique()), "n_sessions": int(df["trading_date"].nunique())},
        "splits": {k: len(v) for k, v in splits.items()},
        "test_budget": budget, "discovery_alpha": disc_alpha, "confirmatory_alpha": CONFIRMATORY_ALPHA,
        "discovery_by_tod": discovery, "survivors": survivors,
        "confirmatory_by_tod": confirmatory, "cross_stock": cross_stock, "economic": economic,
        "opening_confound_summary": confound, "final_classification": classification,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase_v7a_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
