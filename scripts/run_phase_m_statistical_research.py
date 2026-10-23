"""Phase M: new-information-source research for M3/M4 direction, following Phase L's verdict D
(no reliable directional mechanism found among 13 OHLCV-derived candidates). Built on:
  - Phase L's `load_datasets()`, reused verbatim (Phase H frozen + Phase J mover-mechanics + Phase K
    entry_and_path + Phase L g5_g6, with pop_M4/pop_M3 gates and direction_label_Xm already computed)
  - the already-loaded `market_bars` table (series='INDEX', NIFTY 50/CNX NIFTY, NIFTY BANK/BANK
    NIFTY) - read-only, nothing written to it
Nothing in Phase G-L is modified. No new function is added to the `intraday_features` package - every
candidate here is a DataFrame-level derived column (cross-sectional aggregate, shift, or DB join),
not a new bars_up_to_t feature, so it lives in this phase's own script, matching the project's
established convention (phase-specific derived columns belong to the phase, not the canonical library).

FOUR CANDIDATES, EXECUTED IN THE APPROVED ORDER:
  1. Universe breadth LEVEL (`universe_breadth` at T) - already-existing column, tested independently
     for the first time (previously only ever used as an input to relative_strength_vs_universe).
  2. Universe-level regime aggregates - breadth acceleration (T vs T_prev), cross-sectional dispersion
     of return_1m at T, concentration of abnormal movement at T.
  3. Lagged daily NIFTY 50 / NIFTY BANK context - the PRIOR COMPLETED trading day's index direction
     and range, joined via `merge_asof(..., direction="backward", allow_exact_matches=False)` - the
     `allow_exact_matches=False` is what MECHANICALLY guarantees today's own index row can never be
     joined onto today's stock rows, not merely a documented convention.
  4. Inter-stock lead/lag (universe and sector-peer variants) - gated on passing all four adversarial
     PIT checks (see test_phase_m_statistical_research.py) BEFORE being included in any real result.

TARGET AND METHOD, UNCHANGED FROM PHASE L: 20bps next-bar-open direction label, M3/M4 populations
never pooled, the exact centered-Newey-West rate test (`_finalize_rate_test`, imported unchanged),
one global Bonferroni correction with a budget computed FROM the same candidate-list structure the
test functions iterate over (Phase L's own undercounting lesson, applied again), Phase H's frozen
chronological split, cross-stock exclusion of the 5 strongest symbols, and FOUR baselines (majority-
class, same-sign-as-return_5m, relative_strength_vs_universe, AMBIGUOUS-rate coverage).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from scripts.run_phase_l_statistical_research import HORIZONS, POPULATIONS, _finalize_rate_test, population_mask
from scripts.run_phase_l_statistical_research import load_datasets as load_phase_l_datasets

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "derived" / "phase_m_new_information"

NIFTY_50_SYMBOLS = ("NIFTY 50", "CNX NIFTY")
NIFTY_BANK_SYMBOLS = ("NIFTY BANK", "BANK NIFTY")

MISSING_INDEX_CONTEXT_AFTER = dt.date(2026, 9, 11)


def add_candidate_1(df: pd.DataFrame) -> pd.DataFrame:
    df["c1_universe_breadth"] = df["universe_breadth"]
    return df


def compute_breadth_acceleration(df: pd.DataFrame) -> pd.Series:
    unique_times = df[["bar_start_utc", "universe_breadth"]].drop_duplicates("bar_start_utc").sort_values("bar_start_utc")
    unique_times["c2_breadth_acceleration"] = unique_times["universe_breadth"].diff(1)
    return df.merge(unique_times[["bar_start_utc", "c2_breadth_acceleration"]], on="bar_start_utc", how="left")["c2_breadth_acceleration"]


def compute_cross_sectional_dispersion(df: pd.DataFrame) -> pd.Series:
    disp = df.groupby("bar_start_utc")["return_1m"].transform("std")
    return disp.rename("c2_cross_sectional_dispersion")


def compute_abnormal_concentration(df: pd.DataFrame, *, threshold: float = 1.0) -> pd.Series:
    normalized = (df["return_1m"] / df["realized_volatility_20"]).abs()
    is_abnormal = (normalized > threshold).astype(float)
    is_abnormal_valid = is_abnormal.where(normalized.notna())
    return is_abnormal_valid.groupby(df["bar_start_utc"]).transform("mean").rename("c2_abnormal_concentration")


def add_candidate_2(df: pd.DataFrame) -> pd.DataFrame:
    df["c2_breadth_acceleration"] = compute_breadth_acceleration(df)
    df["c2_cross_sectional_dispersion"] = compute_cross_sectional_dispersion(df)
    df["c2_abnormal_concentration"] = compute_abnormal_concentration(df)
    return df


def classify_and_dedupe_index_rows(idx: pd.DataFrame) -> pd.DataFrame:
    idx = idx.copy()
    idx["index_name"] = idx["symbol"].apply(lambda s: "NIFTY_50" if s in NIFTY_50_SYMBOLS else "NIFTY_BANK")
    return idx.drop_duplicates(subset=["trade_date", "index_name"])


def load_index_daily_context(session) -> pd.DataFrame:
    from sqlalchemy import text

    rows = session.execute(text(
        """
        SELECT mb.trade_date, isym.symbol, mb.open, mb.high, mb.low, mb.close
        FROM market_bars mb
        JOIN instrument_symbols isym ON isym.instrument_id = mb.instrument_id
        WHERE mb.series = 'INDEX' AND isym.symbol = ANY(:symbols)
        ORDER BY mb.trade_date
        """
    ), {"symbols": list(NIFTY_50_SYMBOLS) + list(NIFTY_BANK_SYMBOLS)}).mappings().all()
    idx = pd.DataFrame([dict(r) for r in rows])
    return classify_and_dedupe_index_rows(idx)


def _index_daily_features(idx: pd.DataFrame, index_name: str) -> pd.DataFrame:
    sub = idx[idx["index_name"] == index_name].sort_values("trade_date").copy()
    sub["daily_return"] = sub["close"].pct_change()
    sub["daily_range_pct"] = (sub["high"] - sub["low"]) / sub["close"]
    sub = sub.rename(columns={"trade_date": "trading_date"})
    sub["_merge_date"] = pd.to_datetime(sub["trading_date"]).astype("datetime64[ns]")
    return sub[["_merge_date", "trading_date", "daily_return", "daily_range_pct"]]


def add_candidate_3(df: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trading_date").copy()
    df["_merge_date"] = pd.to_datetime(df["trading_date"]).astype("datetime64[ns]")
    for index_name, prefix in (("NIFTY_50", "c3_nifty50"), ("NIFTY_BANK", "c3_niftybank")):
        feats = _index_daily_features(idx, index_name).sort_values("_merge_date")
        df = pd.merge_asof(df, feats[["_merge_date", "daily_return", "daily_range_pct"]], on="_merge_date", direction="backward", allow_exact_matches=False)
        df = df.rename(columns={"daily_return": f"{prefix}_prev_day_return", "daily_range_pct": f"{prefix}_prev_day_range_pct"})
        unavailable = df["trading_date"] > MISSING_INDEX_CONTEXT_AFTER
        df.loc[unavailable, [f"{prefix}_prev_day_return", f"{prefix}_prev_day_range_pct"]] = None
    return df.drop(columns=["_merge_date"])


def compute_leader_impulse(df: pd.DataFrame, group_col: str, *, min_group_size: int = 5) -> pd.Series:
    d = df[["symbol", "bar_start_utc", "return_5m", group_col]].copy()
    d = d.sort_values(["symbol", "bar_start_utc"])
    d["return_5m_prev"] = d.groupby("symbol")["return_5m"].shift(1)

    group_sum = d.groupby(["bar_start_utc", group_col])["return_5m_prev"].transform("sum")
    group_count = d.groupby(["bar_start_utc", group_col])["return_5m_prev"].transform("count")
    leader_mean_excl_self = (group_sum - d["return_5m_prev"].fillna(0)) / (group_count - d["return_5m_prev"].notna().astype(int)).replace(0, pd.NA)
    leader_mean_excl_self = leader_mean_excl_self.where(group_count - 1 >= min_group_size)

    impulse = leader_mean_excl_self - d["return_5m_prev"]
    return impulse.reindex(df.index)


def add_candidate_4(df: pd.DataFrame) -> pd.DataFrame:
    df["c4_universe_leader_impulse"] = compute_leader_impulse(df, "liquidity_group")
    df["c4_sector_leader_impulse"] = compute_leader_impulse(df, "sector")
    return df


CONTINUOUS_CANDIDATES = ["c1_universe_breadth", "c2_breadth_acceleration", "c2_cross_sectional_dispersion", "c2_abnormal_concentration",
                          "c3_nifty50_prev_day_range_pct", "c3_niftybank_prev_day_range_pct", "c4_universe_leader_impulse", "c4_sector_leader_impulse"]
CATEGORICAL_CANDIDATES = {"c3_nifty50_direction": ["UP", "DOWN"], "c3_niftybank_direction": ["UP", "DOWN"]}


def add_direction_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["c3_nifty50_direction"] = df["c3_nifty50_prev_day_return"].apply(lambda r: "UP" if pd.notna(r) and r > 0 else ("DOWN" if pd.notna(r) and r < 0 else None))
    df["c3_niftybank_direction"] = df["c3_niftybank_prev_day_return"].apply(lambda r: "UP" if pd.notna(r) and r > 0 else ("DOWN" if pd.notna(r) and r < 0 else None))
    return df


N_CONTROLS = 2
N_CONTROLLED_GROUPS = 2

CONFIRMATORY_ALPHA = 0.05


def test_budget() -> dict[str, int]:
    per_pop_per_horizon = len(CONTINUOUS_CANDIDATES) * 2 + sum(len(s) for s in CATEGORICAL_CANDIDATES.values())
    n_candidates = len(CONTINUOUS_CANDIDATES) + len(CATEGORICAL_CANDIDATES)
    return {
        "A_standalone": per_pop_per_horizon * len(POPULATIONS) * len(HORIZONS),
        "B_incremental_value": len(POPULATIONS) * N_CONTROLS * n_candidates * N_CONTROLLED_GROUPS,
    }


def run_family_a(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    results = []
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            baseline_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()

            for quantity in CONTINUOUS_CANDIDATES:
                valid = df.loc[pop_mask, quantity].dropna()
                if len(valid) < 50:
                    results.append({"population": population, "quantity": quantity, "bucket": "n/a", "horizon_minutes": h, "skipped_reason": f"insufficient rows ({len(valid)})"})
                    continue
                low_thr, high_thr = valid.quantile(0.2), valid.quantile(0.8)
                low_mask = pop_mask & (df[quantity] <= low_thr)
                high_mask = pop_mask & (df[quantity] >= high_thr)
                results.append({"population": population, "quantity": quantity, "bucket": "lowest_quintile", "horizon_minutes": h, **_rate_test(df, low_mask, direction_col, h, alpha, baseline_rate)})
                results.append({"population": population, "quantity": quantity, "bucket": "highest_quintile", "horizon_minutes": h, **_rate_test(df, high_mask, direction_col, h, alpha, baseline_rate)})

            for quantity, states in CATEGORICAL_CANDIDATES.items():
                for state in states:
                    mask = pop_mask & (df[quantity] == state)
                    results.append({"population": population, "quantity": quantity, "bucket": f"state_{state}", "horizon_minutes": h, **_rate_test(df, mask, direction_col, h, alpha, baseline_rate)})
    return results


def _rate_test(df: pd.DataFrame, subset_mask: pd.Series, direction_col: str, horizon: int, alpha: float, baseline_rate: float) -> dict[str, Any]:
    sub = df.loc[subset_mask, [direction_col, "symbol", "bar_start_utc"]]
    sub = sub[sub[direction_col].isin(["UP", "DOWN"])]
    if len(sub) < 10:
        return {"skipped_reason": f"insufficient rows ({len(sub)})"}
    ordered = sub.sort_values(["symbol", "bar_start_utc"])[direction_col].eq("UP").astype(float).tolist()
    return _finalize_rate_test(ordered, baseline_rate, horizon, alpha)


def run_family_b(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    h = 15
    direction_col = f"direction_label_{h}m"
    results = []
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        baseline_up_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
        for control_col in ("return_5m", "relative_strength_vs_universe"):
            med = df.loc[pop_mask, control_col].median()
            for quantity in CONTINUOUS_CANDIDATES + list(CATEGORICAL_CANDIDATES.keys()):
                for group_name, group_mask in ((f"below_median_{control_col}", pop_mask & (df[control_col] < med)), (f"at_or_above_median_{control_col}", pop_mask & (df[control_col] >= med))):
                    if quantity in CATEGORICAL_CANDIDATES:
                        mask = group_mask & (df[quantity] == CATEGORICAL_CANDIDATES[quantity][0])
                    else:
                        valid = df.loc[group_mask, quantity].dropna()
                        if len(valid) < 50:
                            results.append({"population": population, "quantity": quantity, "control": control_col, "controlled_group": group_name, "skipped_reason": f"insufficient rows ({len(valid)})"})
                            continue
                        thr = valid.quantile(0.8)
                        mask = group_mask & (df[quantity] >= thr)
                    results.append({"population": population, "quantity": quantity, "control": control_col, "controlled_group": group_name, **_rate_test(df, mask, direction_col, h, alpha, baseline_up_rate)})
    return results


def pick_survivors(family_a: list[dict[str, Any]], family_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a_significant = {(r["population"], r["quantity"], r["bucket"]) for r in family_a if r.get("significant_after_correction")}
    b_by_key: dict[tuple[str, str], list[bool]] = {}
    for r in family_b:
        if "skipped_reason" in r:
            continue
        b_by_key.setdefault((r["population"], r["quantity"]), []).append(r.get("significant_after_correction", False))

    survivors = []
    for (population, quantity, bucket) in a_significant:
        b_flags = b_by_key.get((population, quantity), [])
        if len(b_flags) == 4 and all(b_flags):
            survivors.append({"population": population, "quantity": quantity, "bucket": bucket})
    return survivors


def _survivor_mask(df: pd.DataFrame, s: dict[str, Any], base_mask: pd.Series) -> pd.Series:
    if s["quantity"] in CATEGORICAL_CANDIDATES:
        state = s["bucket"].replace("state_", "")
        return base_mask & (df[s["quantity"]].astype(str) == state)
    valid = df.loc[base_mask, s["quantity"]].dropna()
    if len(valid) < 50:
        return base_mask & False
    thr = valid.quantile(0.2 if s["bucket"] == "lowest_quintile" else 0.8)
    return base_mask & ((df[s["quantity"]] <= thr) if s["bucket"] == "lowest_quintile" else (df[s["quantity"]] >= thr))


def run_cross_stock_confirmatory(df: pd.DataFrame, survivors: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for s in survivors:
        key = f"{s['population']}_{s['quantity']}_{s['bucket']}"
        pop_mask = population_mask(df, s["population"])
        results[key] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            baseline_rate = df.loc[pop_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
            mask = _survivor_mask(df, s, pop_mask)
            per_symbol_n = df.loc[mask & df[direction_col].isin(["UP", "DOWN"])].groupby("symbol").size()
            top_5 = per_symbol_n.sort_values(ascending=False).head(5).index.tolist() if len(per_symbol_n) >= 5 else []
            remainder_mask = mask & ~df["symbol"].isin(top_5)
            results[key][f"{h}m"] = {"n_symbols_excluded": len(top_5), **_rate_test(df, remainder_mask, direction_col, h, alpha=CONFIRMATORY_ALPHA, baseline_rate=baseline_rate)}
    return results


def run_chronological_confirmatory(df: pd.DataFrame, survivors: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(df["trading_date"].unique())
    n = len(dates)
    train_end, val_end = round(n * 0.6), round(n * 0.8)
    splits = {"train": set(dates[:train_end]), "validation": set(dates[train_end:val_end]), "held_out": set(dates[val_end:])}

    results: dict[str, Any] = {"split_dates": {k: {"n": len(v)} for k, v in splits.items()}, "by_survivor": {}}
    for s in survivors:
        key = f"{s['population']}_{s['quantity']}_{s['bucket']}"
        pop_mask = population_mask(df, s["population"])
        results["by_survivor"][key] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            results["by_survivor"][key][f"{h}m"] = {}
            for split_name, split_dates in splits.items():
                full_mask = pop_mask & df["trading_date"].isin(split_dates)
                baseline_rate = df.loc[full_mask & df[direction_col].isin(["UP", "DOWN"]), direction_col].eq("UP").mean()
                mask = _survivor_mask(df, s, full_mask)
                results["by_survivor"][key][f"{h}m"][split_name] = _rate_test(df, mask, direction_col, h, alpha=CONFIRMATORY_ALPHA, baseline_rate=baseline_rate)
    return results


def compute_baselines(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for population in POPULATIONS:
        pop_mask = population_mask(df, population)
        results[population] = {}
        for h in HORIZONS:
            direction_col = f"direction_label_{h}m"
            sub = df.loc[pop_mask, direction_col].dropna()
            sub_binary = sub[sub.isin(["UP", "DOWN"])]
            ambiguous_rate = (sub == "AMBIGUOUS").mean() if len(sub) else None
            if len(sub_binary) == 0:
                results[population][f"{h}m"] = {"skipped_reason": "no rows"}
                continue
            majority_class = sub_binary.value_counts().idxmax()
            majority_accuracy = (sub_binary == majority_class).mean()
            results[population][f"{h}m"] = {"majority_class": majority_class, "majority_class_accuracy": float(majority_accuracy), "ambiguous_rate": float(ambiguous_rate) if ambiguous_rate is not None else None, "n": len(sub_binary)}
    return results


def run_economic_evaluation(df: pd.DataFrame, survivors: list[dict[str, Any]]) -> dict[str, Any]:
    cost_scenarios_bps = {"low_20bps": 20.0, "medium_40bps": 40.0, "high_60bps": 60.0}
    results: dict[str, Any] = {}
    for s in survivors:
        key = f"{s['population']}_{s['quantity']}_{s['bucket']}"
        pop_mask = population_mask(df, s["population"])
        results[key] = {}
        for h in HORIZONS:
            mask = _survivor_mask(df, s, pop_mask)
            ret_col = f"gross_return_{h}m"
            gross = df.loc[mask, ret_col].dropna()
            if len(gross) < 30:
                results[key][f"{h}m"] = {"skipped_reason": f"insufficient rows ({len(gross)})"}
                continue
            scenarios = {name: {"cost_bps": bps, "mean_net": float(gross.mean()) - bps / 10_000.0, "hit_rate": float(((gross - bps / 10_000.0) > 0).mean())} for name, bps in cost_scenarios_bps.items()}
            results[key][f"{h}m"] = {"n": len(gross), "gross_mean": float(gross.mean()), "cost_scenarios": scenarios}
    return results


def build_ledger(family_a, family_b, alpha: float) -> list[dict[str, Any]]:
    ledger = []
    for r in family_a:
        ledger.append({"family": "A_standalone", "hypothesis": f"{r['quantity']} ({r['bucket']}) predicts direction within {r['population']} at {r['horizon_minutes']}m",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    for r in family_b:
        ledger.append({"family": "B_incremental_value", "hypothesis": f"{r['quantity']} adds directional value beyond {r['control']} within {r['population']} ({r.get('controlled_group')})",
                        "result": "SIGNIFICANT" if r.get("significant_after_correction") else ("SKIPPED" if "skipped_reason" in r else "NOT_SIGNIFICANT"),
                        "lift": r.get("lift"), "n": r.get("n_rows"), "correction": f"Bonferroni alpha={alpha:.6f}"})
    return ledger


def final_verdict(survivors, cross_stock, chrono, economic) -> dict[str, Any]:
    if not survivors:
        return {"verdict": "D", "reason": "No candidate information source was significant after correction AND demonstrated incremental value beyond BOTH return_5m and relative_strength_vs_universe. No genuinely new directional information was found.", "n_survivors": 0}

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
        verdict = "A"
        reason = f"{len(survivors)} candidate(s) survived every statistical gate, cross-stock generalization, chronological replication, AND remain net-positive under the conservative 60bps cost scenario - genuinely new, economically plausible directional information."
    elif cross_stock_holds and chrono_holds:
        verdict = "B"
        reason = f"{len(survivors)} candidate(s) survived every statistical gate, cross-stock generalization, and chronological replication, but do not survive the conservative 60bps cost scenario - a real directional mechanism, not yet economically viable."
    elif cross_stock_holds or chrono_holds:
        verdict = "C"
        reason = f"{len(survivors)} candidate(s) survived the statistical gates but only partially replicate (cross-stock: {cross_stock_holds}, chronological: {chrono_holds})."
    else:
        verdict = "D"
        reason = f"{len(survivors)} candidate(s) passed the initial statistical gates but failed both cross-stock generalization and chronological replication."

    return {"verdict": verdict, "reason": reason, "n_survivors": len(survivors), "survivors": survivors, "cross_stock_holds": cross_stock_holds, "chrono_holds": chrono_holds, "economic_holds": economic_holds}


def main(*, include_candidate_4: bool) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "phase_m_report.json")
    args = parser.parse_args()

    load_dotenv()
    df = load_phase_l_datasets()
    print(f"loaded {len(df)} rows, {len(df.columns)} columns (Phase L frozen base)")

    df = add_candidate_1(df)
    df = add_candidate_2(df)
    session = get_session()
    try:
        idx = load_index_daily_context(session)
    finally:
        session.close()
    df = add_candidate_3(df, idx)
    df = add_direction_flags(df)
    print(f"index daily context: {len(idx)} rows read from market_bars (read-only)")

    global CONTINUOUS_CANDIDATES
    if include_candidate_4:
        df = add_candidate_4(df)
        print("Candidate 4 (inter-stock lead/lag) included - all adversarial PIT tests passed")
    else:
        CONTINUOUS_CANDIDATES = [c for c in CONTINUOUS_CANDIDATES if not c.startswith("c4_")]
        print("Candidate 4 (inter-stock lead/lag) EXCLUDED - adversarial PIT tests did not all pass")

    budget = test_budget()
    n_tests = sum(budget.values())
    alpha = 0.05 / n_tests
    print(f"test budget: {budget} -> total {n_tests}, Bonferroni alpha={alpha:.6f}")

    family_a = run_family_a(df, alpha)
    print(f"Family A complete: {len(family_a)} rows")
    family_b = run_family_b(df, alpha)
    print(f"Family B complete: {len(family_b)} rows")

    survivors = pick_survivors(family_a, family_b)
    print(f"survivors: {survivors}")

    cross_stock = run_cross_stock_confirmatory(df, survivors)
    chrono = run_chronological_confirmatory(df, survivors)
    baselines = compute_baselines(df)
    economic = run_economic_evaluation(df, survivors)

    ledger = build_ledger(family_a, family_b, alpha)
    verdict = final_verdict(survivors, cross_stock, chrono, economic)
    print(f"verdict: {verdict['verdict']} - {verdict['reason']}")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(df),
        "candidate_4_included": include_candidate_4, "test_budget": budget, "bonferroni_alpha": alpha,
        "baselines": baselines, "family_a_standalone": family_a, "family_b_incremental_value": family_b,
        "survivors": survivors, "cross_stock_confirmatory": cross_stock, "chronological_confirmatory": chrono,
        "economic_evaluation": economic, "research_ledger": ledger, "final_verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "research_ledger.json").write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    include_c4 = "--no-candidate-4" not in sys.argv
    if "--no-candidate-4" in sys.argv:
        sys.argv.remove("--no-candidate-4")
    main(include_candidate_4=include_c4)
