from __future__ import annotations

import random
import uuid
from datetime import date, timedelta

import pytest

from backend.app.research.cross_sectional import (
    DailyCrossSectionalSeries,
    _CrossSectionalRow,
    compute_daily_cross_sectional_series,
)
from backend.app.research.statistics import naive_iid_mean_test, newey_west_mean_test

_FEATURE_ID = uuid.uuid4()
_FEATURE_KEY = str(_FEATURE_ID)


def _row(instrument_id, day, resolved, realized_return, signal_positive):
    return _CrossSectionalRow(
        instrument_id=instrument_id,
        entry_date=day,
        label_resolved=resolved,
        realized_return=realized_return,
        features={_FEATURE_KEY: 0.01 if signal_positive else -0.01},
    )


_INSTRUMENTS = [uuid.UUID(int=i) for i in range(5)]
_DAYS = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]


class TestZeroSignalDays:
    def test_a_day_with_no_signaling_instruments_is_excluded_from_signal_conditional(self):
        rows_by_date = {
            _DAYS[0]: [
                _row(_INSTRUMENTS[0], _DAYS[0], True, 0.02, signal_positive=False),
                _row(_INSTRUMENTS[1], _DAYS[0], True, -0.01, signal_positive=False),
            ]
        }
        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        assert _DAYS[0] not in series.signal_conditional
        assert series.portfolio[_DAYS[0]] == 0.0
        assert _DAYS[0] in series.zero_signal_dates

    def test_a_day_where_the_only_signaling_instrument_is_unresolved_counts_as_zero_signal(self):
        rows_by_date = {
            _DAYS[0]: [
                _row(_INSTRUMENTS[0], _DAYS[0], resolved=False, realized_return=None, signal_positive=True),
            ]
        }
        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        assert _DAYS[0] not in series.signal_conditional
        assert series.portfolio[_DAYS[0]] == 0.0
        assert _DAYS[0] in series.zero_signal_dates

    def test_every_date_remains_in_the_portfolio_series_regardless_of_signal(self):
        rows_by_date = {
            _DAYS[0]: [_row(_INSTRUMENTS[0], _DAYS[0], True, 0.05, signal_positive=True)],
            _DAYS[1]: [_row(_INSTRUMENTS[0], _DAYS[1], True, 0.02, signal_positive=False)],
        }
        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        assert set(series.portfolio.keys()) == {_DAYS[0], _DAYS[1]}
        assert _DAYS[1] in series.zero_signal_dates
        assert _DAYS[0] not in series.zero_signal_dates

    def test_a_signaling_day_appears_in_both_series_with_identical_value(self):
        rows_by_date = {
            _DAYS[0]: [
                _row(_INSTRUMENTS[0], _DAYS[0], True, 0.05, signal_positive=True),
                _row(_INSTRUMENTS[1], _DAYS[0], True, 0.03, signal_positive=True),
            ]
        }
        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        assert series.signal_conditional[_DAYS[0]] == pytest.approx(0.04)
        assert series.portfolio[_DAYS[0]] == pytest.approx(0.04)
        assert series.signal_conditional[_DAYS[0]] == series.portfolio[_DAYS[0]]


class TestSingleExtremeInstrumentDoesNotDominate:
    def test_equal_weight_mean_is_exact_arithmetic_mean_not_skewed(self):
        rows_by_date = {
            _DAYS[0]: [
                _row(_INSTRUMENTS[0], _DAYS[0], True, 10.0, signal_positive=True),
                _row(_INSTRUMENTS[1], _DAYS[0], True, 0.01, signal_positive=True),
                _row(_INSTRUMENTS[2], _DAYS[0], True, 0.02, signal_positive=True),
                _row(_INSTRUMENTS[3], _DAYS[0], True, -0.01, signal_positive=True),
            ]
        }
        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        expected = (10.0 + 0.01 + 0.02 - 0.01) / 4
        assert series.portfolio[_DAYS[0]] == pytest.approx(expected)

    def test_extreme_instruments_influence_is_bounded_to_its_equal_share(self):
        base_others = [0.01, 0.02, -0.01]

        def _aggregate_with_extreme(extreme_value):
            rows_by_date = {
                _DAYS[0]: [
                    _row(_INSTRUMENTS[0], _DAYS[0], True, extreme_value, signal_positive=True),
                    *[
                        _row(_INSTRUMENTS[i + 1], _DAYS[0], True, v, signal_positive=True)
                        for i, v in enumerate(base_others)
                    ],
                ]
            }
            return compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID).portfolio[
                _DAYS[0]
            ]

        low = _aggregate_with_extreme(1.0)
        high = _aggregate_with_extreme(2.0)
        k = 1 + len(base_others)
        assert (high - low) == pytest.approx(1.0 / k)


class TestMaxSimultaneousPositions:
    def test_signal_conditional_series_is_always_uncapped(self):
        rows_by_date = {
            _DAYS[0]: [
                _row(_INSTRUMENTS[i], _DAYS[0], True, 0.01 * (i + 1), signal_positive=True)
                for i in range(5)
            ]
        }
        uncapped = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        capped = compute_daily_cross_sectional_series(
            rows_by_date, _FEATURE_ID, max_simultaneous_positions=2
        )
        assert uncapped.signal_conditional[_DAYS[0]] == capped.signal_conditional[_DAYS[0]]
        assert uncapped.portfolio[_DAYS[0]] != capped.portfolio[_DAYS[0]]

    def test_cap_selection_is_deterministic_by_instrument_id_not_by_return(self):
        rows_by_date = {
            _DAYS[0]: [
                _row(_INSTRUMENTS[0], _DAYS[0], True, -100.0, signal_positive=True),
                _row(_INSTRUMENTS[1], _DAYS[0], True, 100.0, signal_positive=True),
            ]
        }
        capped = compute_daily_cross_sectional_series(
            rows_by_date, _FEATURE_ID, max_simultaneous_positions=1
        )
        assert capped.portfolio[_DAYS[0]] == pytest.approx(-100.0)

    def test_cap_larger_than_active_count_has_no_effect(self):
        rows_by_date = {
            _DAYS[0]: [_row(_INSTRUMENTS[0], _DAYS[0], True, 0.05, signal_positive=True)]
        }
        uncapped = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        capped = compute_daily_cross_sectional_series(
            rows_by_date, _FEATURE_ID, max_simultaneous_positions=10
        )
        assert uncapped.portfolio[_DAYS[0]] == capped.portfolio[_DAYS[0]]


class TestCrossSectionalCorrelationAdversarial:

    N_INSTRUMENTS = 8
    T_DAYS = 300

    @classmethod
    def _build_rows_by_date(cls, seed: int, idiosyncratic_sigma: float):
        rng = random.Random(seed)
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(cls.T_DAYS)]
        instruments = [uuid.UUID(int=i) for i in range(cls.N_INSTRUMENTS)]
        rows_by_date = {}
        for day in days:
            common_shock = rng.uniform(-1.0, 1.0)
            day_rows = []
            for instrument in instruments:
                idiosyncratic = rng.gauss(0.0, idiosyncratic_sigma)
                day_rows.append(
                    _CrossSectionalRow(
                        instrument_id=instrument,
                        entry_date=day,
                        label_resolved=True,
                        realized_return=common_shock + idiosyncratic,
                        features={_FEATURE_KEY: 0.01},
                    )
                )
            rows_by_date[day] = day_rows
        return rows_by_date

    def test_naive_pooled_panel_understates_uncertainty_by_roughly_sqrt_n(self):
        rows_by_date = self._build_rows_by_date(seed=20260918, idiosyncratic_sigma=0.01)

        pooled_panel_values = [
            row.realized_return for day_rows in rows_by_date.values() for row in day_rows
        ]
        naive = naive_iid_mean_test(pooled_panel_values)

        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        aggregated_values = [series.portfolio[d] for d in sorted(series.portfolio)]
        aggregated = newey_west_mean_test(aggregated_values, max_lag=0)

        ratio = aggregated.standard_error / naive.standard_error
        expected_ratio = self.N_INSTRUMENTS**0.5

        assert ratio == pytest.approx(expected_ratio, rel=0.3), (
            f"aggregated/naive SE ratio was {ratio:.3f}, expected close to "
            f"sqrt({self.N_INSTRUMENTS})={expected_ratio:.3f}"
        )
        assert naive.standard_error < aggregated.standard_error

    def test_naive_pooled_panel_is_systematically_more_optimistic_than_the_aggregate(self):
        rows_by_date = self._build_rows_by_date(seed=20260918, idiosyncratic_sigma=0.01)

        pooled_panel_values = [
            row.realized_return for day_rows in rows_by_date.values() for row in day_rows
        ]
        naive = naive_iid_mean_test(pooled_panel_values)

        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        aggregated_values = [series.portfolio[d] for d in sorted(series.portfolio)]
        aggregated = newey_west_mean_test(aggregated_values, max_lag=0)

        assert naive.p_value < aggregated.p_value

    def test_aggregation_reduces_sample_size_from_n_times_t_to_t(self):
        rows_by_date = self._build_rows_by_date(seed=1, idiosyncratic_sigma=0.01)
        pooled_panel_count = sum(len(v) for v in rows_by_date.values())
        series = compute_daily_cross_sectional_series(rows_by_date, _FEATURE_ID)
        assert pooled_panel_count == self.N_INSTRUMENTS * self.T_DAYS
        assert len(series.portfolio) == self.T_DAYS
