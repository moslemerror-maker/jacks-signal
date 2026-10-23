from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backend.app.research.intraday_features.bar_math import Bar
from backend.app.research.intraday_features.labels import entry_path_result
from scripts.run_phase_t_build_event_paths import (
    ENTRY_OFFSETS,
    HORIZONS,
    compute_liquidity_tertiles,
    compute_paths_for_symbol,
)

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _synthetic_symbol_frame(n_sessions: int = 3, bars_per_session: int = 80, seed: int = 7) -> pd.DataFrame:
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
    def test_every_offset_horizon_matches_entry_path_result_across_many_rows(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_paths_for_symbol(df)
        bars = _to_bars(df)
        mismatches = []
        sample_indices = list(range(0, len(df), 7))
        for i in sample_indices:
            bars_up_to_t = bars[: i + 1]
            future_bars = bars[i + 1 :]
            for k in ENTRY_OFFSETS:
                for h in HORIZONS:
                    ref = entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=k, horizon_bars=h)
                    row = vectorized.iloc[i]
                    vec_entry = row[f"entry_price_o{k}"]
                    vec_gross = row[f"gross_return_o{k}_h{h}"]
                    vec_mfe = row[f"mfe_o{k}_h{h}"]
                    vec_mae = row[f"mae_o{k}_h{h}"]
                    if ref is None:
                        if not pd.isna(vec_gross):
                            mismatches.append((i, k, h, "expected None, vectorized had a value", vec_gross))
                        continue
                    if pd.isna(vec_gross):
                        mismatches.append((i, k, h, "expected a value, vectorized was NaN", ref))
                        continue
                    if not np.isclose(vec_entry, ref.entry_price, rtol=1e-9):
                        mismatches.append((i, k, h, "entry_price", vec_entry, ref.entry_price))
                    if not np.isclose(vec_gross, ref.gross_return, rtol=1e-9):
                        mismatches.append((i, k, h, "gross_return", vec_gross, ref.gross_return))
                    if not np.isclose(vec_mfe, ref.max_upside_excursion, rtol=1e-9, atol=1e-12):
                        mismatches.append((i, k, h, "mfe", vec_mfe, ref.max_upside_excursion))
                    if not np.isclose(vec_mae, ref.max_downside_excursion, rtol=1e-9, atol=1e-12):
                        mismatches.append((i, k, h, "mae", vec_mae, ref.max_downside_excursion))
        assert mismatches == [], f"{len(mismatches)} mismatches, first few: {mismatches[:10]}"


class TestLiquidityTertiles:
    def test_three_groups_with_correct_relative_ordering(self):
        class _FakeSession:
            def execute(self, *_args, **_kwargs):
                class _Result:
                    def fetchall(self_inner):
                        rows = []
                        for sid, base_value in [(1, 1_000.0), (2, 10_000.0), (3, 100_000.0), (4, 1_100.0), (5, 11_000.0), (6, 110_000.0)]:
                            for d in range(5):
                                rows.append((sid, dt.date(2026, 1, 1) + dt.timedelta(days=d), base_value * (1 + 0.01 * d)))
                        return rows
                return _Result()
        out = compute_liquidity_tertiles(_FakeSession(), [1, 2, 3, 4, 5, 6])
        assert set(out["liquidity_group"].unique()) == {"low_liquidity", "mid_liquidity", "high_liquidity"}
        low_syms = set(out.loc[out["liquidity_group"] == "low_liquidity", "security_id"])
        high_syms = set(out.loc[out["liquidity_group"] == "high_liquidity", "security_id"])
        assert low_syms == {1, 4}
        assert high_syms == {3, 6}
