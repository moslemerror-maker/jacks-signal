from __future__ import annotations

import pytest

from backend.app.research.intraday_features.liquidity import rolling_median_daily_traded_value


class TestRollingMedianDailyTradedValue:
    def test_first_window_days_are_none(self):
        out = rolling_median_daily_traded_value([1.0] * 25, window=20)
        assert out[:20] == [None] * 20
        assert all(v is not None for v in out[20:])

    def test_value_at_window_excludes_the_day_itself(self):
        values = [10.0] * 20 + [999999.0]
        out = rolling_median_daily_traded_value(values, window=20)
        assert out[20] == pytest.approx(10.0)

    def test_exact_median_of_trailing_window(self):
        values = list(range(1, 21)) + [0.0]
        out = rolling_median_daily_traded_value(values, window=20)
        assert out[20] == pytest.approx(10.5)

    def test_window_slides_forward_one_day_at_a_time(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = rolling_median_daily_traded_value(values, window=3)
        assert out[:3] == [None, None, None]
        assert out[3] == pytest.approx(2.0)
        assert out[4] == pytest.approx(3.0)

    def test_invalid_window_rejected(self):
        with pytest.raises(ValueError):
            rolling_median_daily_traded_value([1.0, 2.0], window=0)
