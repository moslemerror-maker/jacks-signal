from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.research.statistics import newey_west_mean_test
from scripts.run_phase_v3_statistical_research import mechanism_masks as part3_mechanism_masks

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "derived" / "phase_v3_liquid_mechanisms" / "dense_liquid_mechanism_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_v4_strategy_discovery"

LIQUID_TIERS = ("mid_liquidity", "high_liquidity")
DISCOVERY_HORIZONS = (15, 30)
ALL_HORIZONS = (5, 15, 30, 60)
MIN_N = 30
CONFIRMATORY_ALPHA = 0.05

OPENING_WINDOW_MAX_MIN = 30
LATE_SESSION_MIN_MIN = 180
VWAP_RECLAIM_LOOKBACK = 1
BREAKOUT_FAILURE_LOOKBACK = (1, 2, 3, 4, 5)
EARLY_SNAPSHOT_WINDOW = (28, 32)

COST_MODEL: dict[str, dict[str, float]] = {
    "A_optimistic": {"high_liquidity": 5.0, "mid_liquidity": 10.0},
    "B_moderate": {"high_liquidity": 10.0, "mid_liquidity": 20.0},
    "C_conservative": {"high_liquidity": 15.0, "mid_liquidity": 30.0},
    "D_severe": {"high_liquidity": 20.0, "mid_liquidity": 50.0},
}
ECONOMIC_BAR_SCENARIO = "C_conservative"


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df = df[df["rolling_liquidity_group"].isin(LIQUID_TIERS)].copy()
    df = df.sort_values(["security_id", "bar_start_utc"]).reset_index(drop=True)
    returns_5m_mean = df.groupby("bar_start_utc")["return_5m"].transform("mean")
    df["relative_strength_vs_universe"] = df["return_5m"] - returns_5m_mean
    df["universe_breadth"] = df.groupby("bar_start_utc")["return_5m"].transform(lambda s: (s > 0).mean())
    df["cs_rank_15m"] = df.groupby("bar_start_utc")["return_15m"].rank(pct=True)
    return df


def chronological_splits(df: pd.DataFrame) -> dict[str, set]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    disc_end, val_end = round(n * 0.5), round(n * 0.75)
    return {"discovery": set(dates[:disc_end]), "validation": set(dates[disc_end:val_end]), "held_out": set(dates[val_end:])}


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    high, low, open_, close = df["high"], df["low"], df["open"], df["close"]
    rng = (high - low).replace(0.0, np.nan)
    df["candle_upper_wick_ratio"] = ((high - np.maximum(open_, close)) / rng).fillna(0.0)
    df["candle_lower_wick_ratio"] = ((np.minimum(open_, close) - low) / rng).fillna(0.0)
    df["close_location_in_range"] = ((close - low) / rng).fillna(0.5)
    df["wick_asymmetry"] = df["candle_upper_wick_ratio"] - df["candle_lower_wick_ratio"]

    or_rng = (df["opening_range_high"] - df["opening_range_low"]).replace(0.0, np.nan)
    df["opening_range_position"] = ((close - df["opening_range_low"]) / or_rng)

    df["nearest_level_proximity"] = pd.concat([df["distance_to_nearest_support"].abs(), df["distance_to_nearest_resistance"].abs()], axis=1).min(axis=1)

    df["time_of_day_bucket"] = np.where(
        df["minutes_since_open"] <= OPENING_WINDOW_MAX_MIN, "opening",
        np.where(df["minutes_since_open"] >= LATE_SESSION_MIN_MIN, "late", "mid"),
    )

    grp = df.groupby("security_id")
    dvwap_prev = grp["distance_from_vwap"].shift(VWAP_RECLAIM_LOOKBACK)
    df["vwap_reclaim"] = (dvwap_prev < 0) & (df["distance_from_vwap"] >= 0)
    df["vwap_rejection"] = (dvwap_prev > 0) & (df["distance_from_vwap"] <= 0)
    df["vwap_reclaim"] = df["vwap_reclaim"].where(dvwap_prev.notna())
    df["vwap_rejection"] = df["vwap_rejection"].where(dvwap_prev.notna())

    in_snapshot_window = df["minutes_since_open"].between(*EARLY_SNAPSHOT_WINDOW)
    snapshot_value = df["distance_from_session_open"].where(in_snapshot_window)
    df["early_session_direction"] = snapshot_value.groupby([df["security_id"], df["trading_date"]]).transform(lambda s: s.ffill().bfill())
    opposite_sign = np.sign(df["early_session_direction"]) * np.sign(df["distance_from_session_open"]) < 0
    both_material = (df["early_session_direction"].abs() >= 0.005) & (df["distance_from_session_open"].abs() >= 0.003)
    df["opening_reversal_signal"] = opposite_sign & both_material & (df["minutes_since_open"] > EARLY_SNAPSHOT_WINDOW[1])

    meaningful_level_up = pd.concat([df["previous_day_high"], df["opening_range_high"]], axis=1).max(axis=1)
    meaningful_level_down = pd.concat([df["previous_day_low"], df["opening_range_low"]], axis=1).min(axis=1)
    recent_up_any = pd.concat([grp["breakout_up_meaningful"].shift(k).fillna(False) for k in BREAKOUT_FAILURE_LOOKBACK], axis=1).any(axis=1)
    recent_down_any = pd.concat([grp["breakout_down_meaningful"].shift(k).fillna(False) for k in BREAKOUT_FAILURE_LOOKBACK], axis=1).any(axis=1)
    df["failed_breakout_up"] = recent_up_any & (close < meaningful_level_up)
    df["failed_breakout_down"] = recent_down_any & (close > meaningful_level_down)

    prev_open, prev_close = grp["open"].shift(1), grp["close"].shift(1)
    prev_dir = np.sign(prev_close - prev_open)
    cur_dir = np.sign(close - open_)
    engulf_bull = (prev_dir < 0) & (cur_dir > 0) & (open_ <= prev_close) & (close >= prev_open)
    engulf_bear = (prev_dir > 0) & (cur_dir < 0) & (open_ >= prev_close) & (close <= prev_open)
    has_prev = prev_open.notna()
    df["two_bar_engulfing_bullish"] = engulf_bull.where(has_prev)
    df["two_bar_engulfing_bearish"] = engulf_bear.where(has_prev)
    return df


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
        "p_value": result.p_value, "significant": bool(result.p_value < alpha),
    }


def _rate_test(df: pd.DataFrame, mask: pd.Series, direction_col: str, implied: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[mask.fillna(False), [direction_col, "symbol", "bar_start_utc"]]
    sub = sub[sub[direction_col].isin(["UP", "DOWN", "NEUTRAL"])]
    if len(sub) < MIN_N:
        return {"n": len(sub), "skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[direction_col].eq(implied).astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha)


def baseline_rates(df: pd.DataFrame, horizon: int) -> dict[str, float]:
    direction_col = f"direction_label_pit_{horizon}"
    defined = df[direction_col].isin(["UP", "DOWN", "NEUTRAL"])
    sub = df.loc[defined, direction_col]
    return {"UP": (sub == "UP").mean(), "DOWN": (sub == "DOWN").mean(), "MOVE": (sub != "NEUTRAL").mean()}


def movement_mask_test(df: pd.DataFrame, mask: pd.Series, horizon: int, alpha: float, baseline_move: float) -> dict[str, Any]:
    direction_col = f"direction_label_pit_{horizon}"
    sub = df.loc[mask.fillna(False), direction_col]
    sub = sub[sub.isin(["UP", "DOWN", "NEUTRAL"])]
    if len(sub) < MIN_N:
        return {"n": len(sub), "skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = (sub != "NEUTRAL").astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_move, horizon, alpha)


SINGLE_FEATURES = (
    "candle_body_ratio", "wick_asymmetry", "close_location_in_range", "opening_range_position",
    "relative_volume_20", "directional_pressure_10", "volatility_expansion_ratio", "nearest_level_proximity",
    "relative_strength_vs_universe", "cs_rank_15m", "universe_breadth", "overnight_gap",
)

TOP_N_VALUES = (10, 20, 50)


def quartile_masks(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    return df[col] >= q3, df[col] <= q1


def run_single_feature_discovery(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for h in DISCOVERY_HORIZONS:
        direction_col = f"direction_label_pit_{h}"
        base = baseline_rates(df, h)
        for feat in SINGLE_FEATURES:
            top, bottom = quartile_masks(df, feat)
            results.append({"family": "single_feature", "feature": feat, "cell": "top_implies_up", "horizon_min": h, **_rate_test(df, top, direction_col, "UP", h, alpha, base["UP"])})
            results.append({"family": "single_feature", "feature": feat, "cell": "bottom_implies_down", "horizon_min": h, **_rate_test(df, bottom, direction_col, "DOWN", h, alpha, base["DOWN"])})
    return results


def named_hypothesis_masks(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    compressed = df["volatility_expansion_ratio"] <= 0.7
    abnormal_vol = df["relative_volume_20"] >= 2.0
    near_level = df["nearest_level_proximity"] <= df["nearest_level_proximity"].quantile(0.10)
    rs_extreme_high = df["relative_strength_vs_universe"] >= df["relative_strength_vs_universe"].quantile(0.90)
    rs_extreme_low = df["relative_strength_vs_universe"] <= df["relative_strength_vs_universe"].quantile(0.10)
    gap_up, gap_down = df["overnight_gap"] >= 0.005, df["overnight_gap"] <= -0.005

    return {
        "H1_opening_reversal": {"mask": df["opening_reversal_signal"].fillna(False), "target": "direction", "implied": "sign_of_current", "horizons": (30, 60)},
        "H2_vwap_reclaim": {"mask": df["vwap_reclaim"].fillna(False), "target": "direction", "implied": "UP", "horizons": (15, 30)},
        "H2_vwap_rejection": {"mask": df["vwap_rejection"].fillna(False), "target": "direction", "implied": "DOWN", "horizons": (15, 30)},
        "H3_failed_breakout_up": {"mask": df["failed_breakout_up"].fillna(False), "target": "direction", "implied": "DOWN", "horizons": (15, 30)},
        "H3_failed_breakout_down": {"mask": df["failed_breakout_down"].fillna(False), "target": "direction", "implied": "UP", "horizons": (15, 30)},
        "H4_gap_up_continuation": {"mask": gap_up, "target": "direction", "implied": "UP", "horizons": (15, 30)},
        "H4_gap_down_continuation": {"mask": gap_down, "target": "direction", "implied": "DOWN", "horizons": (15, 30)},
        "H5_compression_plus_volume": {"mask": compressed & abnormal_vol, "target": "movement", "implied": "MOVE", "horizons": (15, 30)},
        "H6_level_plus_volume": {"mask": near_level & abnormal_vol, "target": "movement", "implied": "MOVE", "horizons": (15, 30)},
        "H7_rs_extreme_opening": {"mask": rs_extreme_high & (df["time_of_day_bucket"] == "opening"), "target": "direction", "implied": "UP", "horizons": (30,)},
        "H7_rs_weak_opening": {"mask": rs_extreme_low & (df["time_of_day_bucket"] == "opening"), "target": "direction", "implied": "DOWN", "horizons": (30,)},
    }


def run_named_hypothesis_discovery(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    masks = named_hypothesis_masks(df)
    results = []
    for name, spec in masks.items():
        for h in spec["horizons"]:
            direction_col = f"direction_label_pit_{h}"
            if spec["target"] == "movement":
                base_move = baseline_rates(df, h)["MOVE"]
                results.append({"family": "named_hypothesis", "hypothesis": name, "horizon_min": h, **movement_mask_test(df, spec["mask"], h, alpha, base_move)})
            elif spec["implied"] == "sign_of_current":
                sub_mask = spec["mask"]
                up_part = sub_mask & (df["distance_from_session_open"] > 0)
                down_part = sub_mask & (df["distance_from_session_open"] < 0)
                base = baseline_rates(df, h)
                results.append({"family": "named_hypothesis", "hypothesis": name, "side": "up", "horizon_min": h, **_rate_test(df, up_part, direction_col, "UP", h, alpha, base["UP"])})
                results.append({"family": "named_hypothesis", "hypothesis": name, "side": "down", "horizon_min": h, **_rate_test(df, down_part, direction_col, "DOWN", h, alpha, base["DOWN"])})
            else:
                base = baseline_rates(df, h)
                results.append({"family": "named_hypothesis", "hypothesis": name, "horizon_min": h, **_rate_test(df, spec["mask"], direction_col, spec["implied"], h, alpha, base[spec["implied"]])})
    return results


def run_fade_discovery(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    masks = part3_mechanism_masks(df)
    results = []
    for h in DISCOVERY_HORIZONS:
        direction_col = f"direction_label_pit_{h}"
        base = baseline_rates(df, h)
        for mech, sides in masks.items():
            results.append({"family": "H_FADE", "mechanism": mech, "trigger_side": "up", "horizon_min": h, **_rate_test(df, sides["up"], direction_col, "DOWN", h, alpha, base["DOWN"])})
            results.append({"family": "H_FADE", "mechanism": mech, "trigger_side": "down", "horizon_min": h, **_rate_test(df, sides["down"], direction_col, "UP", h, alpha, base["UP"])})
    return results


def run_top_n_discovery(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    extremity = (df["cs_rank_15m"] - 0.5).abs()
    for h in DISCOVERY_HORIZONS:
        direction_col = f"direction_label_pit_{h}"
        base_move = baseline_rates(df, h)["MOVE"]
        for n in TOP_N_VALUES:
            rank_desc = extremity.groupby(df["bar_start_utc"]).rank(ascending=False, method="first")
            mask = rank_desc <= n
            results.append({"family": "top_n_ranking", "n": n, "horizon_min": h, **movement_mask_test(df, mask, h, alpha, base_move)})
    return results


def pick_survivors(all_discovery_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    survivors = []
    for r in all_discovery_results:
        if r.get("family") == "top_n_ranking":
            continue
        if r.get("significant") and r.get("lift") is not None and r["lift"] > 1.0:
            survivors.append(r)
    return survivors


def _mask_for_survivor(df: pd.DataFrame, s: dict[str, Any]) -> tuple[pd.Series, str]:
    family = s["family"]
    if family == "single_feature":
        top, bottom = quartile_masks(df, s["feature"])
        return (top, "UP") if s["cell"] == "top_implies_up" else (bottom, "DOWN")
    if family == "named_hypothesis":
        masks = named_hypothesis_masks(df)
        spec = masks[s["hypothesis"]]
        if spec["target"] == "movement":
            return spec["mask"], "MOVE"
        if spec["implied"] == "sign_of_current":
            if s["side"] == "up":
                return spec["mask"] & (df["distance_from_session_open"] > 0), "UP"
            return spec["mask"] & (df["distance_from_session_open"] < 0), "DOWN"
        return spec["mask"], spec["implied"]
    if family == "H_FADE":
        masks = part3_mechanism_masks(df)
        implied = "DOWN" if s["trigger_side"] == "up" else "UP"
        return masks[s["mechanism"]][s["trigger_side"]], implied
    raise ValueError(f"unknown family {family}")


def run_confirmatory(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for s in survivors:
        h = s["horizon_min"]
        row: dict[str, Any] = {**s}
        for split_name in ("validation", "held_out"):
            split_df = df[df["trading_date"].isin(splits[split_name])]
            mask, implied = _mask_for_survivor(split_df, s)
            if implied == "MOVE":
                base_move = baseline_rates(split_df, h)["MOVE"]
                row[split_name] = movement_mask_test(split_df, mask, h, CONFIRMATORY_ALPHA, base_move)
            else:
                direction_col = f"direction_label_pit_{h}"
                base = baseline_rates(split_df, h)
                row[split_name] = _rate_test(split_df, mask, direction_col, implied, h, CONFIRMATORY_ALPHA, base.get(implied, base["UP"]))
        results.append(row)
    return results


def run_cross_stock_confirmatory(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    for s in survivors:
        h = s["horizon_min"]
        mask, implied = _mask_for_survivor(confirmatory_df, s)
        per_symbol_n = confirmatory_df.loc[mask.fillna(False)].groupby("symbol").size()
        top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
        remainder_mask = mask & ~confirmatory_df["symbol"].isin(top_5)
        if implied == "MOVE":
            base_move = baseline_rates(confirmatory_df, h)["MOVE"]
            test = movement_mask_test(confirmatory_df, remainder_mask, h, CONFIRMATORY_ALPHA, base_move)
        else:
            direction_col = f"direction_label_pit_{h}"
            base = baseline_rates(confirmatory_df, h)
            test = _rate_test(confirmatory_df, remainder_mask, direction_col, implied, h, CONFIRMATORY_ALPHA, base.get(implied, base["UP"]))
        results.append({**s, "n_symbols_excluded": len(top_5), **test})
    return results


def run_regime_stability(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    for s in survivors:
        h = s["horizon_min"]
        mask, implied = _mask_for_survivor(confirmatory_df, s)
        for half_name, half_mask in (("morning", confirmatory_df["minutes_since_open"] < 180), ("afternoon", confirmatory_df["minutes_since_open"] >= 180)):
            sub_mask = mask & half_mask
            if implied == "MOVE":
                base_move = baseline_rates(confirmatory_df.loc[half_mask], h)["MOVE"]
                test = movement_mask_test(confirmatory_df, sub_mask, h, CONFIRMATORY_ALPHA, base_move)
            else:
                direction_col = f"direction_label_pit_{h}"
                base = baseline_rates(confirmatory_df.loc[half_mask], h)
                test = _rate_test(confirmatory_df, sub_mask, direction_col, implied, h, CONFIRMATORY_ALPHA, base.get(implied, base["UP"]))
            results.append({**s, "half": half_name, **test})
    return results


def run_economic_evaluation(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    out = []
    for s in survivors:
        h = s["horizon_min"]
        mask, implied = _mask_for_survivor(confirmatory_df, s)
        if implied == "MOVE":
            out.append({**s, "skipped_reason": "movement-only hypothesis has no directional economic side"})
            continue
        sub = confirmatory_df.loc[mask.fillna(False), [f"gross_return_{h}m", "rolling_liquidity_group"]].dropna()
        is_long = implied == "UP"
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


def run_capacity_evidence(df: pd.DataFrame, splits: dict[str, set], survivors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirmatory_df = df[df["trading_date"].isin(splits["validation"] | splits["held_out"])]
    out = []
    for s in survivors:
        mask, _ = _mask_for_survivor(confirmatory_df, s)
        sub = confirmatory_df.loc[mask.fillna(False), "rolling_median_daily_traded_value_20d"].dropna()
        out.append({**s, "n": len(sub), "median_daily_traded_value_inr": float(sub.median()) if len(sub) else None})
    return out


def run_top_n_confirmatory(df: pd.DataFrame, splits: dict[str, set]) -> list[dict[str, Any]]:
    results = []
    for split_name in ("validation", "held_out"):
        split_df = df[df["trading_date"].isin(splits[split_name])]
        extremity = (split_df["cs_rank_15m"] - 0.5).abs()
        for h in ALL_HORIZONS:
            base_move = baseline_rates(split_df, h)["MOVE"]
            for n in TOP_N_VALUES:
                rank_desc = extremity.groupby(split_df["bar_start_utc"]).rank(ascending=False, method="first")
                mask = rank_desc <= n
                results.append({"split": split_name, "n": n, "horizon_min": h, **movement_mask_test(split_df, mask, h, CONFIRMATORY_ALPHA, base_move)})
    return results


def final_verdicts(survivors: list[dict[str, Any]], confirmatory: list[dict[str, Any]], cross_stock: list[dict[str, Any]], regime: list[dict[str, Any]], economic: list[dict[str, Any]]) -> dict[str, Any]:
    if not survivors:
        return {"verdict": "E", "reason": "No single-feature, named-hypothesis, or H_FADE cell was both statistically significant on the discovery period AND confirmed its own hypothesized direction (lift > 1). No validated predictive information was found."}
    out = []
    for s, c, cs, econ in zip(survivors, confirmatory, cross_stock, economic):
        val_holds = c["validation"].get("significant", False)
        held_out_holds = c["held_out"].get("significant", False)
        cross_holds = cs.get("significant", False)
        econ_holds = "scenarios" in econ and econ["scenarios"].get(ECONOMIC_BAR_SCENARIO, {}).get("mean_net", -1) > 0
        if val_holds and held_out_holds and cross_holds and econ_holds:
            verdict = "A_VALIDATED_AND_ECONOMICALLY_VIABLE"
        elif val_holds and held_out_holds and cross_holds:
            verdict = "B_STATISTICALLY_ROBUST_ECONOMICALLY_UNVIABLE_OR_NA"
        else:
            verdict = "C_FAILED_OOS_OR_CROSS_STOCK_CONFIRMATION"
        out.append({**s, "verdict": verdict, "validation_significant": val_holds, "held_out_significant": held_out_holds, "cross_stock_significant": cross_holds, "economic_holds": econ_holds})
    return {"candidates": out}


def test_budget_discovery() -> dict[str, int]:
    return {
        "single_feature": len(SINGLE_FEATURES) * 2 * len(DISCOVERY_HORIZONS),
        "named_hypothesis": 22,
        "H_FADE": 4 * 2 * len(DISCOVERY_HORIZONS),
        "top_n_ranking": len(TOP_N_VALUES) * len(DISCOVERY_HORIZONS),
    }


def main() -> None:
    df = load_dataset()
    print(f"loaded {len(df)} liquid-population rows, {df['security_id'].nunique()} symbols, {df['trading_date'].nunique()} sessions")
    df = add_feature_columns(df)
    splits = chronological_splits(df)
    print(f"splits: discovery={len(splits['discovery'])}, validation={len(splits['validation'])}, held_out={len(splits['held_out'])} sessions")

    discovery_df = df[df["trading_date"].isin(splits["discovery"])]
    print(f"discovery population: {len(discovery_df)} rows")

    budget = test_budget_discovery()
    total = sum(budget.values())
    alpha = 0.05 / total
    print(f"discovery test budget: {budget} -> total {total}, Bonferroni alpha={alpha:.6f}")

    single_feature = run_single_feature_discovery(discovery_df, alpha)
    named = run_named_hypothesis_discovery(discovery_df, alpha)
    fade = run_fade_discovery(discovery_df, alpha)
    top_n = run_top_n_discovery(discovery_df, alpha)
    print(f"single_feature={len(single_feature)}, named_hypothesis={len(named)}, H_FADE={len(fade)}, top_n={len(top_n)}")

    all_discovery = single_feature + named + fade + top_n
    survivors = pick_survivors(all_discovery)
    print(f"survivors ({len(survivors)}): {survivors}")

    confirmatory = run_confirmatory(df, splits, survivors)
    cross_stock = run_cross_stock_confirmatory(df, splits, survivors)
    regime = run_regime_stability(df, splits, survivors)
    economic = run_economic_evaluation(df, splits, survivors)
    capacity = run_capacity_evidence(df, splits, survivors)
    top_n_confirmatory = run_top_n_confirmatory(df, splits)
    verdicts = final_verdicts(survivors, confirmatory, cross_stock, regime, economic)
    print(f"verdicts: {verdicts}")

    report = {
        "population": {"n_rows": len(df), "n_symbols": int(df["security_id"].nunique()), "n_sessions": int(df["trading_date"].nunique())},
        "splits": {k: len(v) for k, v in splits.items()}, "test_budget_discovery": budget, "total_discovery_tests": total, "alpha": alpha,
        "single_feature": single_feature, "named_hypothesis": named, "H_FADE": fade, "top_n_discovery": top_n,
        "survivors": survivors, "confirmatory": confirmatory, "cross_stock": cross_stock, "regime_stability": regime,
        "economic": economic, "capacity": capacity, "top_n_confirmatory": top_n_confirmatory, "verdicts": verdicts,
    }
    out_path = OUT_DIR / "phase_v4_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
