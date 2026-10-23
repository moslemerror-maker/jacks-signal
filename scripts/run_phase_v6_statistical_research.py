from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.research.statistics import newey_west_mean_test
from scripts.run_phase_v4_strategy_discovery import chronological_splits

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "derived" / "phase_v6_opportunity" / "dense_opportunity_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_v6_opportunity"

ENTRY_OFFSET = 1
DISCOVERY_HORIZONS = (15, 30)
ALL_HORIZONS = (5, 15, 30, 60)
RACE_CHECKPOINTS = (1, 2, 3, 5, 10, 15, 30, 60)
MIN_N = 30
CONFIRMATORY_ALPHA = 0.05

TARGET_VOL_MULT = 1.5
STOP_VOL_MULT = 1.0

COST_MODEL: dict[str, dict[str, float]] = {
    "A_optimistic": {"high_liquidity": 5.0, "mid_liquidity": 10.0},
    "B_moderate": {"high_liquidity": 10.0, "mid_liquidity": 20.0},
    "C_conservative": {"high_liquidity": 15.0, "mid_liquidity": 30.0},
    "D_severe": {"high_liquidity": 20.0, "mid_liquidity": 50.0},
}
ECONOMIC_BAR_SCENARIO = "C_conservative"

TOP_N_VALUES = (10, 20, 50)


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_FILE)


def _fav_adv_cols(h: int, is_long: bool) -> tuple[str, str]:
    mfe_col, mae_col = f"mfe_o{ENTRY_OFFSET}_h{h}", f"mae_o{ENTRY_OFFSET}_h{h}"
    return (mfe_col, mae_col) if is_long else (mae_col, mfe_col)


def state_masks(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    compressed = df["volatility_expansion_ratio"] <= 0.7
    abnormal_vol = df["relative_volume_20"] >= 2.0
    near_level = df["nearest_level_proximity"] <= df["nearest_level_proximity"].quantile(0.10)
    top_wick, bottom_wick = df["wick_asymmetry"] >= df["wick_asymmetry"].quantile(0.75), df["wick_asymmetry"] <= df["wick_asymmetry"].quantile(0.25)

    states: dict[str, dict[str, Any]] = {
        "S1_compression_volume_up": {"mask": compressed & abnormal_vol & (df["return_5m"] > 0), "is_long": True},
        "S1_compression_volume_down": {"mask": compressed & abnormal_vol & (df["return_5m"] < 0), "is_long": False},
        "S2_level_volume_up": {"mask": near_level & abnormal_vol & (df["return_5m"] > 0), "is_long": True},
        "S2_level_volume_down": {"mask": near_level & abnormal_vol & (df["return_5m"] < 0), "is_long": False},
        "S3_vwap_reclaim": {"mask": df["vwap_reclaim"].fillna(False), "is_long": True},
        "S4_vwap_rejection": {"mask": df["vwap_rejection"].fillna(False), "is_long": False},
        "S5_failed_breakout_up_fade": {"mask": df["failed_breakout_up"].fillna(False), "is_long": False},
        "S6_failed_breakout_down_fade": {"mask": df["failed_breakout_down"].fillna(False), "is_long": True},
        "S7_wick_asymmetry_up": {"mask": top_wick, "is_long": True},
        "S7_wick_asymmetry_down": {"mask": bottom_wick, "is_long": False},
        "S8_abnormal_volume_up": {"mask": abnormal_vol & (df["return_5m"] > 0), "is_long": True},
        "S8_abnormal_volume_down": {"mask": abnormal_vol & (df["return_5m"] < 0), "is_long": False},
    }
    from scripts.run_phase_v3_statistical_research import mechanism_masks as part3_mechanism_masks
    m3m4 = part3_mechanism_masks(df)
    states["S9_fade_M3_up_trigger"] = {"mask": m3m4["M3_abnormal_participation_continuation"]["up"], "is_long": False}
    states["S9_fade_M3_down_trigger"] = {"mask": m3m4["M3_abnormal_participation_continuation"]["down"], "is_long": True}
    states["S10_fade_M4_up_trigger"] = {"mask": m3m4["M4_volatility_expansion_confirmation"]["up"], "is_long": False}
    states["S10_fade_M4_down_trigger"] = {"mask": m3m4["M4_volatility_expansion_confirmation"]["down"], "is_long": True}
    return states


def _mean_test_vs_baseline(values: pd.Series, baseline_mean: float, horizon: int, alpha: float) -> dict[str, Any]:
    vals = values.dropna()
    if len(vals) < MIN_N:
        return {"n": len(vals), "skipped_reason": f"insufficient rows ({len(vals)})"}
    centered = (vals - baseline_mean).tolist()
    max_lag = min(horizon - 1, len(centered) - 1)
    result = newey_west_mean_test(centered, max_lag=max_lag)
    realized_mean = baseline_mean + result.mean
    return {
        "n": len(vals), "realized_mean": realized_mean, "baseline_mean": baseline_mean,
        "lift": realized_mean / baseline_mean if baseline_mean else None,
        "p_value": result.p_value, "significant": bool(result.p_value < alpha),
    }


def baseline_asymmetry(df: pd.DataFrame, h: int, is_long: bool) -> float:
    fav_col, adv_col = _fav_adv_cols(h, is_long)
    return (df[fav_col] - df[adv_col]).mean()


def baseline_target_before_stop_rate(df: pd.DataFrame, h: int, is_long: bool) -> float:
    rate = race_target_before_stop(df, is_long)
    return rate.mean()


def race_target_before_stop(df: pd.DataFrame, is_long: bool) -> pd.Series:
    n = len(df)
    target_level = (TARGET_VOL_MULT * df["realized_volatility_20"]).to_numpy()
    stop_level = (STOP_VOL_MULT * df["realized_volatility_20"]).to_numpy()
    target_hit = np.full(n, np.inf)
    stop_hit = np.full(n, np.inf)
    for h in RACE_CHECKPOINTS:
        fav_col, adv_col = _fav_adv_cols(h, is_long)
        fav, adv = df[fav_col].to_numpy(), df[adv_col].to_numpy()
        newly_t = (fav >= target_level) & np.isinf(target_hit)
        target_hit[newly_t] = h
        newly_s = (adv >= stop_level) & np.isinf(stop_hit)
        stop_hit[newly_s] = h
    stop_first = stop_hit <= target_hit
    win = (~np.isinf(target_hit)) & (~stop_first)
    return pd.Series(win.astype(float), index=df.index)


def run_state_discovery(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    states = state_masks(df)
    results = []
    for h in DISCOVERY_HORIZONS:
        fav_adv_baselines = {is_long: baseline_asymmetry(df, h, is_long) for is_long in (True, False)}
        tbs_baselines = {is_long: baseline_target_before_stop_rate(df, h, is_long) for is_long in (True, False)}
        for name, spec in states.items():
            mask, is_long = spec["mask"].fillna(False), spec["is_long"]
            fav_col, adv_col = _fav_adv_cols(h, is_long)
            sub = df.loc[mask]
            asymmetry_values = sub[fav_col] - sub[adv_col]
            results.append({"family": "state_asymmetry", "state": name, "horizon_min": h, "is_long": is_long,
                             **_mean_test_vs_baseline(asymmetry_values, fav_adv_baselines[is_long], h, alpha)})
            tbs = race_target_before_stop(sub, is_long) if len(sub) >= MIN_N else pd.Series(dtype=float)
            results.append({"family": "target_before_stop", "state": name, "horizon_min": h, "is_long": is_long,
                             **_mean_test_vs_baseline(tbs, tbs_baselines[is_long], h, alpha)})
    return results


def run_top_n_discovery(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    intensity = df["relative_volume_20"] * df["volatility_expansion_ratio"]
    for h in DISCOVERY_HORIZONS:
        baseline_long = baseline_asymmetry(df, h, True)
        for n in TOP_N_VALUES:
            rank_desc = intensity.groupby(df["bar_start_utc"]).rank(ascending=False, method="first")
            mask = rank_desc <= n
            fav_col, adv_col = _fav_adv_cols(h, True)
            sub = df.loc[mask]
            asymmetry_values = sub[fav_col] - sub[adv_col]
            results.append({"family": "top_n_ranking", "n": n, "horizon_min": h, **_mean_test_vs_baseline(asymmetry_values, baseline_long, h, alpha)})
    return results


def run_top_n_confirmatory(df: pd.DataFrame, splits: dict[str, set]) -> list[dict[str, Any]]:
    results = []
    for split_name in ("validation", "held_out"):
        split_df = df[df["trading_date"].isin(splits[split_name])]
        intensity = split_df["relative_volume_20"] * split_df["volatility_expansion_ratio"]
        for h in ALL_HORIZONS:
            baseline_long = baseline_asymmetry(split_df, h, True)
            for n in TOP_N_VALUES:
                rank_desc = intensity.groupby(split_df["bar_start_utc"]).rank(ascending=False, method="first")
                mask = rank_desc <= n
                fav_col, adv_col = _fav_adv_cols(h, True)
                sub = split_df.loc[mask]
                asymmetry_values = sub[fav_col] - sub[adv_col]
                results.append({"split": split_name, "n": n, "horizon_min": h, **_mean_test_vs_baseline(asymmetry_values, baseline_long, h, CONFIRMATORY_ALPHA)})
    return results


def test_budget_discovery() -> dict[str, int]:
    return {"state_asymmetry_and_tbs": 16 * len(DISCOVERY_HORIZONS) * 2, "top_n_ranking": len(TOP_N_VALUES) * len(DISCOVERY_HORIZONS)}


def pick_survivors(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    survivors = []
    for r in results:
        if r.get("family") == "top_n_ranking":
            continue
        if r.get("significant") and r.get("realized_mean") is not None and r.get("baseline_mean") is not None and r["realized_mean"] > r["baseline_mean"]:
            survivors.append(r)
    return survivors


def _mask_for_state(df: pd.DataFrame, state_name: str) -> tuple[pd.Series, bool]:
    spec = state_masks(df)[state_name]
    return spec["mask"].fillna(False), spec["is_long"]


def run_confirmatory(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for s in survivors:
        h = s["horizon_min"]
        row: dict[str, Any] = {**s}
        for split_name in ("validation", "held_out"):
            split_df = df[df["trading_date"].isin(splits[split_name])]
            mask, is_long = _mask_for_state(split_df, s["state"])
            if s["family"] == "state_asymmetry":
                baseline = baseline_asymmetry(split_df, h, is_long)
                fav_col, adv_col = _fav_adv_cols(h, is_long)
                sub = split_df.loc[mask]
                row[split_name] = _mean_test_vs_baseline(sub[fav_col] - sub[adv_col], baseline, h, CONFIRMATORY_ALPHA)
            else:
                baseline = baseline_target_before_stop_rate(split_df, h, is_long)
                sub = split_df.loc[mask]
                tbs = race_target_before_stop(sub, is_long) if len(sub) >= MIN_N else pd.Series(dtype=float)
                row[split_name] = _mean_test_vs_baseline(tbs, baseline, h, CONFIRMATORY_ALPHA)
        results.append(row)
    return results


def run_cross_stock_confirmatory(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    results = []
    for s in survivors:
        h = s["horizon_min"]
        mask, is_long = _mask_for_state(confirmatory_df, s["state"])
        per_symbol_n = confirmatory_df.loc[mask].groupby("symbol").size()
        top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
        remainder = mask & ~confirmatory_df["symbol"].isin(top_5)
        if s["family"] == "state_asymmetry":
            baseline = baseline_asymmetry(confirmatory_df, h, is_long)
            fav_col, adv_col = _fav_adv_cols(h, is_long)
            sub = confirmatory_df.loc[remainder]
            test = _mean_test_vs_baseline(sub[fav_col] - sub[adv_col], baseline, h, CONFIRMATORY_ALPHA)
        else:
            baseline = baseline_target_before_stop_rate(confirmatory_df, h, is_long)
            sub = confirmatory_df.loc[remainder]
            tbs = race_target_before_stop(sub, is_long) if len(sub) >= MIN_N else pd.Series(dtype=float)
            test = _mean_test_vs_baseline(tbs, baseline, h, CONFIRMATORY_ALPHA)
        results.append({**s, "n_symbols_excluded": len(top_5), "concentration_top5_share": float(per_symbol_n.head(5).sum() / per_symbol_n.sum()) if len(per_symbol_n) else None, **test})
    return results


def run_regime_stability(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    results = []
    for s in survivors:
        h = s["horizon_min"]
        mask, is_long = _mask_for_state(confirmatory_df, s["state"])
        for tod in ("opening", "mid", "late"):
            tod_mask = confirmatory_df["time_of_day_bucket"] == tod
            sub_mask = mask & tod_mask
            if s["family"] == "state_asymmetry":
                baseline = baseline_asymmetry(confirmatory_df.loc[tod_mask], h, is_long)
                fav_col, adv_col = _fav_adv_cols(h, is_long)
                sub = confirmatory_df.loc[sub_mask]
                test = _mean_test_vs_baseline(sub[fav_col] - sub[adv_col], baseline, h, CONFIRMATORY_ALPHA)
            else:
                baseline = baseline_target_before_stop_rate(confirmatory_df.loc[tod_mask], h, is_long)
                sub = confirmatory_df.loc[sub_mask]
                tbs = race_target_before_stop(sub, is_long) if len(sub) >= MIN_N else pd.Series(dtype=float)
                test = _mean_test_vs_baseline(tbs, baseline, h, CONFIRMATORY_ALPHA)
            results.append({**s, "time_of_day": tod, **test})
    return results


def run_economic_evaluation(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    out = []
    for s in survivors:
        h = s["horizon_min"]
        mask, is_long = _mask_for_state(confirmatory_df, s["state"])
        sub = confirmatory_df.loc[mask, [f"gross_return_o{ENTRY_OFFSET}_h{h}", "rolling_liquidity_group"]].dropna()
        gross = sub[f"gross_return_o{ENTRY_OFFSET}_h{h}"] if is_long else -sub[f"gross_return_o{ENTRY_OFFSET}_h{h}"]
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


def run_mfe_mae_detail(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    out = []
    for s in survivors:
        h = s["horizon_min"]
        mask, is_long = _mask_for_state(confirmatory_df, s["state"])
        fav_col, adv_col = _fav_adv_cols(h, is_long)
        sub = confirmatory_df.loc[mask]
        fav, adv = sub[fav_col].dropna(), sub[adv_col].dropna()
        out.append({
            **s, "n": len(fav), "mean_mfe": float(fav.mean()) if len(fav) else None, "median_mfe": float(fav.median()) if len(fav) else None,
            "mean_mae": float(adv.mean()) if len(adv) else None, "median_mae": float(adv.median()) if len(adv) else None,
            "mfe_mae_ratio_mean": float(fav.mean() / adv.mean()) if len(adv) and adv.mean() else None,
        })
    return out


def final_verdicts(survivors: list[dict[str, Any]], confirmatory: list[dict[str, Any]], cross_stock: list[dict[str, Any]], economic: list[dict[str, Any]]) -> dict[str, Any]:
    if not survivors:
        return {"verdict": "NONE_SURVIVED", "reason": "No state x side x horizon cell showed significantly better asymmetry or target-before-stop rate than the unconditional population on the discovery period with the hypothesis-confirming direction."}
    out = []
    for s, c, cs, econ in zip(survivors, confirmatory, cross_stock, economic):
        val_holds = c["validation"].get("significant", False)
        held_out_holds = c["held_out"].get("significant", False)
        cross_holds = cs.get("significant", False)
        econ_holds = "scenarios" in econ and econ["scenarios"].get(ECONOMIC_BAR_SCENARIO, {}).get("mean_net", -1) > 0
        if val_holds and held_out_holds and cross_holds and econ_holds:
            verdict = "TRADE_CANDIDATE"
        elif val_holds and held_out_holds and cross_holds:
            verdict = "WAIT_STATISTICALLY_ROBUST_NOT_ECONOMIC"
        else:
            verdict = "NO_TRADE_FAILED_OOS_OR_CROSS_STOCK"
        out.append({**s, "verdict": verdict, "validation_significant": val_holds, "held_out_significant": held_out_holds, "cross_stock_significant": cross_holds, "economic_holds": econ_holds, "concentration_top5_share": cs.get("concentration_top5_share")})
    return {"candidates": out}


def main() -> None:
    df = load_dataset()
    print(f"loaded {len(df)} rows, {df['security_id'].nunique()} symbols, {df['trading_date'].nunique()} sessions")
    splits = chronological_splits(df)
    print(f"splits: discovery={len(splits['discovery'])}, validation={len(splits['validation'])}, held_out={len(splits['held_out'])} sessions")
    discovery_df = df[df["trading_date"].isin(splits["discovery"])]

    budget = test_budget_discovery()
    total = sum(budget.values())
    alpha = 0.05 / total
    print(f"discovery test budget: {budget} -> total {total}, Bonferroni alpha={alpha:.6f}")

    state_results = run_state_discovery(discovery_df, alpha)
    top_n_results = run_top_n_discovery(discovery_df, alpha)
    print(f"state_asymmetry_and_tbs={len(state_results)}, top_n={len(top_n_results)}")

    all_discovery = state_results + top_n_results
    survivors = pick_survivors(all_discovery)
    print(f"survivors ({len(survivors)}): {survivors}")

    confirmatory = run_confirmatory(df, splits, survivors)
    cross_stock = run_cross_stock_confirmatory(df, splits, survivors)
    regime = run_regime_stability(df, splits, survivors)
    economic = run_economic_evaluation(df, splits, survivors)
    mfe_mae_detail = run_mfe_mae_detail(df, splits, survivors)
    top_n_confirmatory = run_top_n_confirmatory(df, splits)
    verdicts = final_verdicts(survivors, confirmatory, cross_stock, economic)
    print(f"verdicts: {verdicts}")

    report = {
        "population": {"n_rows": len(df), "n_symbols": int(df["security_id"].nunique()), "n_sessions": int(df["trading_date"].nunique())},
        "splits": {k: len(v) for k, v in splits.items()}, "test_budget_discovery": budget, "total_discovery_tests": total, "alpha": alpha,
        "state_discovery": state_results, "top_n_discovery": top_n_results, "survivors": survivors,
        "confirmatory": confirmatory, "cross_stock": cross_stock, "regime_stability": regime,
        "economic": economic, "mfe_mae_detail": mfe_mae_detail, "top_n_confirmatory": top_n_confirmatory, "verdicts": verdicts,
    }
    out_path = OUT_DIR / "phase_v6_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
