from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backend.app.research.intraday_features.bar_math import Bar
from backend.app.research.intraday_features.labels import next_bar_entry_path_result
from backend.app.research.intraday_features.liquidity import rolling_median_daily_traded_value
from scripts.run_phase_u_build_execution_liquidity import (
    HORIZONS,
    compute_execution_variants_for_symbol,
    compute_rolling_liquidity_classification,
)

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _synthetic_symbol_frame(n_sessions: int = 3, bars_per_session: int = 80, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        trading_date = dt.date(2026, 1, 5) + dt.timedelta(days=s)
        open_ist = dt.datetime(trading_date.year, trading_date.month, trading_date.day, 9, 15, tzinfo=IST)
        price = 100.0 + s * 5
        for i in range(bars_per_session):
            o = price
            c = price * (1 + rng.normal(scale=0.003))
            h = max(o, c) * (1 + abs(rng.normal(scale=0.001)))
            l = min(o, c) * (1 - abs(rng.normal(scale=0.001)))
            vol = 1000 + rng.integers(0, 5000)
            ts = (open_ist + dt.timedelta(minutes=i)).astimezone(UTC)
            rows.append({"bar_start_utc": ts, "trading_date": trading_date, "open": o, "high": h, "low": l, "close": c, "volume": float(vol)})
            price = c
    return pd.DataFrame(rows)


def _to_bars(df: pd.DataFrame) -> list[Bar]:
    return [Bar(r.bar_start_utc, r.trading_date, r.open, r.high, r.low, r.close, r.volume) for r in df.itertuples()]


class TestEquivalenceToPureFunction:
    def test_close1_and_vwap1_match_next_bar_entry_path_result_across_many_rows(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_execution_variants_for_symbol(df)
        bars = _to_bars(df)
        mismatches = []
        for i in range(0, len(df), 5):
            bars_up_to_t = bars[: i + 1]
            future_bars = bars[i + 1 :]
            for basis in ("close", "vwap_proxy"):
                col_basis = "close1" if basis == "close" else "vwap1"
                for h in HORIZONS:
                    ref = next_bar_entry_path_result(bars_up_to_t, future_bars, horizon_bars=h, price_basis=basis)
                    row = vectorized.iloc[i]
                    vec_gross = row[f"gross_return_{col_basis}_h{h}"]
                    if ref is None:
                        if not pd.isna(vec_gross):
                            mismatches.append((i, basis, h, "expected None, got a value", vec_gross))
                        continue
                    if pd.isna(vec_gross):
                        mismatches.append((i, basis, h, "expected a value, got NaN"))
                        continue
                    if not np.isclose(row[f"entry_price_{col_basis}"], ref.entry_price, rtol=1e-9):
                        mismatches.append((i, basis, h, "entry_price", row[f"entry_price_{col_basis}"], ref.entry_price))
                    if not np.isclose(vec_gross, ref.gross_return, rtol=1e-9):
                        mismatches.append((i, basis, h, "gross_return", vec_gross, ref.gross_return))
                    if not np.isclose(row[f"mfe_{col_basis}_h{h}"], ref.max_upside_excursion, rtol=1e-9, atol=1e-12):
                        mismatches.append((i, basis, h, "mfe"))
                    if not np.isclose(row[f"mae_{col_basis}_h{h}"], ref.max_downside_excursion, rtol=1e-9, atol=1e-12):
                        mismatches.append((i, basis, h, "mae"))
        assert mismatches == [], f"{len(mismatches)} mismatches, first few: {mismatches[:10]}"


class TestRollingLiquidityClassificationEquivalence:
    def test_matches_pure_function_per_symbol(self):
        rng = np.random.default_rng(3)
        n_days = 45
        rows = []
        for sid in (1, 2, 3):
            base = {1: 1_000.0, 2: 10_000.0, 3: 100_000.0}[sid]
            for d in range(n_days):
                rows.append({"security_id": sid, "trading_date": dt.date(2026, 1, 1) + dt.timedelta(days=d), "traded_value": base * (1 + rng.normal(scale=0.05))})
        daily = pd.DataFrame(rows)
        out = compute_rolling_liquidity_classification(daily)
        for sid in (1, 2, 3):
            sub = out[out["security_id"] == sid].sort_values("trading_date").reset_index(drop=True)
            ref = rolling_median_daily_traded_value(sub["traded_value"].tolist(), window=20)
            for i in range(n_days):
                vec_val = sub.loc[i, "rolling_median_daily_traded_value_20d"]
                if ref[i] is None:
                    assert pd.isna(vec_val)
                else:
                    assert vec_val == pytest.approx(ref[i])

    def test_first_20_days_have_no_liquidity_group(self):
        rng = np.random.default_rng(1)
        rows = [{"security_id": 1, "trading_date": dt.date(2026, 1, 1) + dt.timedelta(days=d), "traded_value": 1000.0 + rng.normal()} for d in range(25)]
        out = compute_rolling_liquidity_classification(pd.DataFrame(rows))
        assert out.iloc[:20]["rolling_liquidity_group"].isna().all()
        assert out.iloc[20:]["rolling_liquidity_group"].notna().all()

    def test_cross_sectional_tertile_is_relative_to_the_same_date_only(self):
        rows = []
        for d in range(25):
            date = dt.date(2026, 1, 1) + dt.timedelta(days=d)
            rows.append({"security_id": 1, "trading_date": date, "traded_value": 1_000.0})
            rows.append({"security_id": 2, "trading_date": date, "traded_value": 10_000.0})
            rows.append({"security_id": 3, "trading_date": date, "traded_value": 100_000.0})
        out = compute_rolling_liquidity_classification(pd.DataFrame(rows))
        last_date = dt.date(2026, 1, 1) + dt.timedelta(days=24)
        last = out[out["trading_date"] == last_date]
        assert last.set_index("security_id").loc[1, "rolling_liquidity_group"] == "low_liquidity"
        assert last.set_index("security_id").loc[3, "rolling_liquidity_group"] == "high_liquidity"
