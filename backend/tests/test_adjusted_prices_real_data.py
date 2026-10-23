from __future__ import annotations

from datetime import date as D
from fractions import Fraction as F

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from backend.app.config import database_url
from backend.app.data.instruments.adjusted_prices import read_adjusted_bars
from backend.app.data.instruments.corporate_actions import assess_adjustment_eligibility
from backend.app.db.models import InstrumentSymbol
from backend.app.research import adjustment_factors as af

FULL = (D(2015, 1, 1), D(2026, 9, 11))
BARS_FP = "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY t.id::text), '')) FROM market_bars t"


@pytest.fixture(scope="module")
def real():
    try:
        engine = create_engine(database_url())
        conn = engine.connect()
    except OperationalError:
        pytest.skip("dev database not reachable")
    tx = conn.begin()
    try:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        active = conn.execute(text(
            "SELECT count(*) FROM corporate_action_term_verifications v WHERE v.action='VERIFIED' AND NOT EXISTS "
            "(SELECT 1 FROM corporate_action_term_verifications r WHERE r.action='REVOKED' AND r.revokes_id=v.id)")).scalar()
        n_bars = conn.execute(text("SELECT count(*) FROM market_bars")).scalar()
    except (OperationalError, ProgrammingError):
        tx.rollback(); conn.close()
        pytest.skip("dev database lacks the Stage 3.9 schema")
    if active != 10 or n_bars < 70000:
        tx.rollback(); conn.close()
        pytest.skip(f"dev database is not at the end-of-C2 state ({active} active C2 records, {n_bars} bars)")
    session = Session(bind=conn)
    ids = dict(session.execute(select(InstrumentSymbol.symbol, InstrumentSymbol.instrument_id).where(InstrumentSymbol.effective_to.is_(None))).all())
    fp = tuple(conn.execute(text(BARS_FP)).one())
    yield session, ids, fp, conn
    assert tuple(conn.execute(text(BARS_FP)).one()) == fp, "market_bars changed during the real-data tests"
    session.close(); tx.rollback(); conn.close()


def view(real, name, start, as_of, **kw):
    session, ids, _, _ = real
    return read_adjusted_bars(session, ids[name], start, as_of, **kw)


def groups(v):
    out = {}
    for b in v.bars:
        out.setdefault(b.factor, []).append(b.raw.trade_date)
    return {f: (min(d), max(d), len(d)) for f, d in out.items()}


def codes(v):
    return {b.code for b in v.eligibility.blockers}


class TestJswsteelSplit10To1:
    def test_the_full_window_is_adjusted_with_an_exact_one_tenth_before_the_ex_date(self, real):
        v = view(real, "JSWSTEEL", *FULL)
        assert v.status == af.ADJUSTED and v.eligibility.eligible and not v.eligibility.blockers and v.data_quality["complete"]
        assert v.data_quality["raw_bar_count"] == v.data_quality["expected_session_count"] == 2887 and v.data_quality["missing_sessions"] == []
        assert groups(v) == {F(1, 10): (D(2015, 1, 1), D(2017, 1, 3), 495), F(1): (D(2017, 1, 4), D(2026, 9, 11), 2392)}
        (ev,) = v.basis["events"]
        assert ev["ex_date"] == "2017-01-04" and ev["event_factor"] == {"numerator": "1", "denominator": "10"}
        assert [(c["kind"], c["term_a"], c["term_b"], c["tier"]) for c in ev["components"]] == [("SPLIT", "10", "1", 1)]

    def test_the_bars_around_the_split_have_the_exact_expected_values(self, real):
        b = {x.raw.trade_date: x for x in view(real, "JSWSTEEL", *FULL).bars}
        assert (b[D(2017, 1, 3)].open, b[D(2017, 1, 3)].high, b[D(2017, 1, 3)].low, b[D(2017, 1, 3)].close) == (164.1, 165.36, 163.1, 164.455)
        assert af.fraction_text(b[D(2017, 1, 3)].close_exact) == "32891/200" and b[D(2017, 1, 3)].factor == F(1, 10)
        assert (b[D(2016, 12, 30)].close, b[D(2017, 1, 2)].close) == (162.75, 163.43)
        for d, raw in ((D(2017, 1, 4), (166.75, 166.75, 160.6, 163.05)), (D(2017, 1, 5), (164.05, 176.65, 163.5, 175.65))):
            assert b[d].factor == 1 and (b[d].open, b[d].high, b[d].low, b[d].close) == raw == (b[d].raw.open, b[d].raw.high, b[d].raw.low, b[d].raw.close)

    def test_the_split_no_longer_looks_like_a_ninety_percent_crash(self, real):
        b = {x.raw.trade_date: x for x in view(real, "JSWSTEEL", *FULL).bars}
        pre, post = b[D(2017, 1, 3)], b[D(2017, 1, 4)]
        assert post.raw.close / pre.raw.close - 1 < -0.9
        assert abs(post.close / pre.close - 1) < 0.01

    @pytest.mark.parametrize("as_of,before_factor,n_events", [
        (D(2017, 1, 3), F(1), 0),
        (D(2017, 1, 4), F(1, 10), 1),
        (D(2017, 1, 5), F(1, 10), 1),
    ])
    def test_as_of_before_on_and_after_the_event(self, real, as_of, before_factor, n_events):
        v = view(real, "JSWSTEEL", FULL[0], as_of)
        assert v.status == af.ADJUSTED and len(v.basis["events"]) == n_events
        assert v.bars[0].factor == before_factor and v.bars[-1].raw.trade_date == as_of
        if as_of == D(2017, 1, 3):
            assert all(x.factor == 1 and (x.open, x.high, x.low, x.close) == (x.raw.open, x.raw.high, x.raw.low, x.raw.close) for x in v.bars)

    def test_a_request_starting_on_the_ex_date_is_untouched(self, real):
        v = view(real, "JSWSTEEL", D(2017, 1, 4), FULL[1])
        assert v.status == af.ADJUSTED and set(x.factor for x in v.bars) == {F(1)} and v.basis["events"] == []

    def test_the_view_is_reproducible_and_never_touches_the_raw_bars(self, real):
        a, b = view(real, "JSWSTEEL", *FULL), view(real, "JSWSTEEL", *FULL)
        assert a == b and a.view_sha256 == b.view_sha256 and a.basis_sha256 == b.basis_sha256 and len(a.view_sha256) == 64
        assert tuple(real[3].execute(text(BARS_FP)).one()) == real[2]

    def test_every_adjusted_bar_keeps_ohlc_ordering(self, real):
        for x in view(real, "JSWSTEEL", *FULL).bars:
            assert x.low <= min(x.open, x.close) and max(x.open, x.close) <= x.high and x.low <= x.high

    def test_the_views_eligibility_is_exactly_the_gates(self, real):
        session, ids, _, _ = real
        assert view(real, "JSWSTEEL", *FULL).eligibility == assess_adjustment_eligibility(session, ids["JSWSTEEL"], *FULL)


class TestBajfinanceCompound:
    def test_the_2016_event_is_a_bonus_and_a_split_that_compound_to_exactly_one_tenth(self, real):
        v = view(real, "BAJFINANCE", D(2016, 9, 7), D(2016, 9, 8))
        assert v.status == af.ADJUSTED and [(x.raw.trade_date, x.factor) for x in v.bars] == [(D(2016, 9, 7), F(1, 10)), (D(2016, 9, 8), F(1))]
        (ev,) = v.basis["events"]
        assert {c["kind"]: (c["term_a"], c["term_b"], c["factor"]) for c in ev["components"]} == {
            "BONUS": ("1", "1", {"numerator": "1", "denominator": "2"}), "SPLIT": ("10", "2", {"numerator": "1", "denominator": "5"})}
        assert ev["event_factor"] == {"numerator": "1", "denominator": "10"}
        assert v.bars[0].raw.close == 11393.3 and v.bars[0].close == 1139.33 and af.fraction_text(v.bars[0].close_exact) == "113933/100"

    def test_before_the_unresolved_2025_events_the_whole_history_is_adjusted(self, real):
        v = view(real, "BAJFINANCE", FULL[0], D(2025, 6, 13))
        assert v.status == af.ADJUSTED and groups(v) == {F(1, 10): (D(2015, 1, 1), D(2016, 9, 7), 416), F(1): (D(2016, 9, 8), D(2025, 6, 13), 2162)}

    def test_a_request_reaching_the_2025_events_is_refused_because_the_bonus_4_to_1_is_unresolved(self, real):
        for end in (D(2025, 6, 16), FULL[1]):
            v = view(real, "BAJFINANCE", FULL[0], end)
            assert v.status == af.REFUSED and v.bars == () and v.safe_start_hint == D(2025, 6, 16)
            assert codes(v) == {"COMPONENT_NOT_MANUALLY_VERIFIED", "EVENT_NOT_MANUALLY_VERIFIED", "NO_ACTIVE_C2_VERIFICATION", "UNRESOLVED_REGISTER_ITEM"}
            assert v.eligibility.barrier_dates == (D(2025, 6, 16),)

    def test_the_verified_2025_split_alone_does_not_unlock_the_date(self, real):
        assert view(real, "BAJFINANCE", D(2025, 6, 13), D(2025, 6, 16)).status == af.REFUSED

    def test_a_request_starting_on_the_2025_ex_date_is_adjusted_with_no_event(self, real):
        v = view(real, "BAJFINANCE", D(2025, 6, 16), FULL[1])
        assert v.status == af.ADJUSTED and v.basis["events"] == [] and set(x.factor for x in v.bars) == {F(1)}


class TestBlockedInstrumentsStayBlocked:
    @pytest.mark.parametrize("name,expected", [
        ("HDFCBANK", {"UNRESOLVED_REGISTER_ITEM"}),
        ("KOTAKBANK", {"UNRESOLVED_REGISTER_ITEM"}),
        ("BAJFINANCE", {"COMPONENT_NOT_MANUALLY_VERIFIED", "EVENT_NOT_MANUALLY_VERIFIED", "NO_ACTIVE_C2_VERIFICATION", "UNRESOLVED_REGISTER_ITEM"}),
        ("RELIANCE", {"UNSUPPORTED_ACTION_TYPE", "UNRESOLVED_REGISTER_ITEM"}),
    ])
    def test_the_full_window_is_refused_with_no_adjusted_value(self, real, name, expected):
        v = view(real, name, *FULL)
        assert v.status == af.REFUSED and af.ELIGIBILITY_BLOCKED in v.refusal_reasons and v.bars == () and v.basis is None
        assert codes(v) == expected and v.eligibility.coverage_status == "COMPLETE"

    def test_reliance_is_blocked_by_the_unsupported_rights_and_demerger_and_by_the_register_even_for_verified_bonuses(self, real):
        v = view(real, "RELIANCE", *FULL)
        unsupported = {(str(b.ex_date), b.action_class) for b in v.eligibility.blockers if b.code == "UNSUPPORTED_ACTION_TYPE"}
        assert unsupported == {("2020-05-13", "RIGHTS"), ("2023-07-20", "DEMERGER")}
        reg = {(str(b.ex_date), b.action_class) for b in v.eligibility.blockers if b.code == "UNRESOLVED_REGISTER_ITEM"}
        assert {("2017-09-07", "BONUS"), ("2024-10-28", "BONUS")} <= reg
        assert view(real, "RELIANCE", D(2017, 1, 1), D(2017, 12, 31)).status == af.REFUSED

    def test_hdfcbank_kotakbank_and_reliance_adjust_only_the_intervals_that_do_not_straddle_a_barrier(self, real):
        h = view(real, "HDFCBANK", FULL[0], D(2025, 8, 25))
        assert h.status == af.ADJUSTED and groups(h) == {F(1, 2): (D(2015, 1, 1), D(2019, 9, 18), 1163), F(1): (D(2019, 9, 19), D(2025, 8, 25), 1465)}
        assert view(real, "HDFCBANK", D(2025, 8, 26), FULL[1]).status == af.ADJUSTED
        k = view(real, "KOTAKBANK", D(2015, 7, 8), FULL[1])
        assert k.status == af.ADJUSTED and groups(k) == {F(1, 5): (D(2015, 7, 8), D(2026, 1, 13), 2597), F(1): (D(2026, 1, 14), D(2026, 9, 11), 163)}
        r = view(real, "RELIANCE", D(2024, 10, 28), FULL[1])
        assert r.status == af.ADJUSTED and set(x.factor for x in r.bars) == {F(1)}

    def test_the_safe_start_hint_points_at_the_last_barrier(self, real):
        assert view(real, "HDFCBANK", *FULL).safe_start_hint == D(2025, 8, 26)
        assert view(real, "KOTAKBANK", *FULL).safe_start_hint == D(2015, 7, 8)
        assert view(real, "RELIANCE", *FULL).safe_start_hint == D(2024, 10, 28)

    def test_no_instrument_data_quality_problem_exists_in_the_pilot(self, real):
        for name in ("JSWSTEEL", "HDFCBANK", "KOTAKBANK", "BAJFINANCE", "RELIANCE"):
            dq = view(real, name, *FULL).data_quality
            assert dq["complete"] and dq["raw_bar_count"] == dq["expected_session_count"] == 2887 and dq["invalid_raw_bars"] == []
