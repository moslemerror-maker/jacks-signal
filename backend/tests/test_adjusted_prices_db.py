from __future__ import annotations

import dataclasses
import uuid
from datetime import date as D
from fractions import Fraction as F

import pytest
from sqlalchemy import func, select, text

from backend.app.data.instruments.adjusted_prices import read_adjusted_bars
from backend.app.data.instruments.corporate_actions import assess_adjustment_eligibility
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionCoverageAssessment as Assessment,
    CorporateActionEvent,
    CorporateActionTermVerification,
    CorporateActionUnresolvedEvent as Unresolved,
    CorporateActionVerificationLog,
    CorporateActionVerificationStatus as VS,
    Instrument,
    MarketBar,
    SourceFormat,
)
from backend.app.research import adjustment_factors as af
from backend.tests.c2_helpers import insert_c2_record
from backend.tests.test_adjustment_eligibility_db import AGM, BONUS, DIV, FULL, SPLIT, add_event, assess, conflicted, reg

JSW_BARS = [
    (D(2016, 12, 30), 1599.0, 1632.45, 1595.65, 1627.5),
    (D(2017, 1, 2), 1628.0, 1641.85, 1610.1, 1634.3),
    (D(2017, 1, 3), 1641.0, 1653.6, 1631.0, 1644.55),
    (D(2017, 1, 4), 166.75, 166.75, 160.6, 163.05),
    (D(2017, 1, 5), 164.05, 176.65, 163.5, 175.65),
]
EX = D(2017, 1, 4)


@pytest.fixture(autouse=True)
def _no_foreign_bars(db_session):
    n = db_session.scalar(select(func.count()).select_from(MarketBar))
    assert n == 0, f"the test database holds {n} committed market_bars; C3 calendar tests need a clean one"


def new_instrument(db) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db.add(inst)
    db.flush()
    return inst


def add_bars(db, inst, rows, series="EQ", tf="1D"):
    for d, o, h, l, c in rows:
        db.add(MarketBar(instrument_id=inst.id, trade_date=d, series=series, timeframe=tf, open=o, high=h, low=l, close=c,
                         source_format=SourceFormat.NSE_UDIFF, source_file="test.csv"))
    db.flush()


def jsw(db, *, verified=True, register=(), window=FULL, assessment=True):
    inst = new_instrument(db)
    add_bars(db, inst, JSW_BARS)
    add_event(db, inst, EX, "Fv Splt Frm Rs 10 To Re 1", SPLIT(fv=(10.0, 1.0)), legacy=not verified)
    if assessment:
        assess(db, inst, register, window=window)
    return inst


def bars_fingerprint(db):
    return tuple(db.execute(text("SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY t.id::text), '')) FROM market_bars t")).one())


def table_counts(db):
    return tuple(db.scalar(select(func.count()).select_from(m)) for m in
                 (MarketBar, CorporateActionEvent, CorporateActionComponent, Assessment, Unresolved, CorporateActionVerificationLog,
                  CorporateActionTermVerification))


def factors(v):
    return {b.raw.trade_date: b.factor for b in v.bars}


def blocker_codes(v):
    return sorted({b.code for b in v.eligibility.blockers})


class TestPilotPositive:
    def test_dates_before_on_and_after_the_split(self, db_session):
        inst = jsw(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        assert v.status == af.ADJUSTED and v.eligibility.eligible and v.refusal_reasons == ()
        assert factors(v) == {D(2016, 12, 30): F(1, 10), D(2017, 1, 2): F(1, 10), D(2017, 1, 3): F(1, 10), D(2017, 1, 4): F(1), D(2017, 1, 5): F(1)}
        b = {x.raw.trade_date: x for x in v.bars}
        assert (b[D(2017, 1, 3)].open, b[D(2017, 1, 3)].high, b[D(2017, 1, 3)].low, b[D(2017, 1, 3)].close) == (164.1, 165.36, 163.1, 164.455)
        assert b[D(2017, 1, 3)].close_exact == F(32891, 200)
        assert (b[D(2017, 1, 4)].open, b[D(2017, 1, 4)].close) == (166.75, 163.05)
        assert (b[D(2017, 1, 5)].high, b[D(2017, 1, 5)].close) == (176.65, 175.65)

    def test_the_view_describes_itself(self, db_session):
        inst = jsw(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        assert v.instrument_id == str(inst.id) and v.series == "EQ" and v.timeframe == "1D" and v.as_of == D(2017, 1, 5)
        assert v.basis["version"] == af.ADJUSTMENT_BASIS_VERSION and v.basis["price_basis_adjusted"] == af.PRICE_BASIS_ADJUSTED
        assert v.basis["cumulative_factor_at_requested_start"] == {"numerator": "1", "denominator": "10"}
        ev = v.basis["events"]
        assert len(ev) == 1 and ev[0]["ex_date"] == "2017-01-04" and ev[0]["event_factor"] == {"numerator": "1", "denominator": "10"}
        comp = ev[0]["components"][0]
        assert (comp["kind"], comp["term_a"], comp["term_b"], comp["tier"]) == ("SPLIT", "10", "1", 1)
        rec = db_session.scalar(select(CorporateActionTermVerification.id).where(CorporateActionTermVerification.action == "VERIFIED"))
        assert comp["c2_record_id"] == str(rec)
        assert v.basis["eligibility"]["coverage_status"] == "COMPLETE" and v.basis["eligibility"]["governing_assessment_id"]
        first = db_session.scalar(select(MarketBar.id).where(MarketBar.instrument_id == inst.id, MarketBar.trade_date == D(2016, 12, 30)))
        assert v.bars[0].raw.bar_id == str(first)
        assert "NOT_ESTABLISHED" in v.basis["point_in_time"] and v.basis["dividends_adjusted"] is False

    @pytest.mark.parametrize("as_of,expected", [
        (D(2017, 1, 3), {D(2016, 12, 30): F(1), D(2017, 1, 2): F(1), D(2017, 1, 3): F(1)}),
        (D(2017, 1, 4), {D(2016, 12, 30): F(1, 10), D(2017, 1, 2): F(1, 10), D(2017, 1, 3): F(1, 10), D(2017, 1, 4): F(1)}),
        (D(2017, 1, 5), {D(2016, 12, 30): F(1, 10), D(2017, 1, 2): F(1, 10), D(2017, 1, 3): F(1, 10), D(2017, 1, 4): F(1), D(2017, 1, 5): F(1)}),
    ])
    def test_as_of_before_on_and_after_the_event(self, db_session, as_of, expected):
        inst = jsw(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), as_of)
        assert v.status == af.ADJUSTED and factors(v) == expected
        assert (not v.basis["events"]) == (as_of < EX)

    def test_a_request_starting_on_the_ex_date_excludes_the_event(self, db_session):
        inst = jsw(db_session)
        v = read_adjusted_bars(db_session, inst.id, EX, D(2017, 1, 5))
        assert v.status == af.ADJUSTED and factors(v) == {EX: F(1), D(2017, 1, 5): F(1)} and v.basis["events"] == []

    def test_the_same_request_twice_is_identical(self, db_session):
        inst = jsw(db_session)
        a = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        b = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        assert a == b and a.view_sha256 == b.view_sha256 and a.basis_sha256 == b.basis_sha256

    def test_every_adjusted_bar_keeps_ohlc_ordering(self, db_session):
        inst = jsw(db_session)
        for b in read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5)).bars:
            assert b.low <= min(b.open, b.close) and max(b.open, b.close) <= b.high
            assert b.low_exact <= min(b.open_exact, b.close_exact) and max(b.open_exact, b.close_exact) <= b.high_exact


class TestRawBarsAndStateAreNeverModified:
    def test_reading_writes_nothing_at_all(self, db_session):
        inst = jsw(db_session)
        before_bars, before_counts = bars_fingerprint(db_session), table_counts(db_session)
        for _ in range(3):
            read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
            read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 3))
        assert bars_fingerprint(db_session) == before_bars and table_counts(db_session) == before_counts
        assert not db_session.new and not db_session.dirty and not db_session.deleted

    def test_a_refused_read_writes_nothing_either(self, db_session):
        inst = jsw(db_session, verified=False)
        before_bars, before_counts = bars_fingerprint(db_session), table_counts(db_session)
        assert read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5)).status == af.REFUSED
        assert bars_fingerprint(db_session) == before_bars and table_counts(db_session) == before_counts
        assert not db_session.new and not db_session.dirty and not db_session.deleted

    def test_the_raw_values_in_the_view_are_the_stored_values_and_the_stored_values_stay(self, db_session):
        inst = jsw(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        stored = db_session.execute(select(MarketBar.trade_date, MarketBar.open, MarketBar.high, MarketBar.low, MarketBar.close)
                                    .where(MarketBar.instrument_id == inst.id).order_by(MarketBar.trade_date)).all()
        assert [(b.raw.trade_date, b.raw.open, b.raw.high, b.raw.low, b.raw.close) for b in v.bars] == [tuple(r) for r in stored] == JSW_BARS

    def test_the_view_object_cannot_be_mutated(self, db_session):
        inst = jsw(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.status = "ADJUSTED"
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.bars[0].raw.close = 1.0


class TestTheViewUsesTheEstablishedGate:
    def _refused(self, db, inst, start=D(2016, 12, 30), end=D(2017, 1, 5), **kw):
        v = read_adjusted_bars(db, inst.id, start, end, **kw)
        assert v.status == af.REFUSED and af.ELIGIBILITY_BLOCKED in v.refusal_reasons
        assert v.bars == () and v.basis is None and v.basis_sha256 is None
        assert v.eligibility == assess_adjustment_eligibility(db, inst.id, start, end)
        return v

    def test_unverified_event_blocks(self, db_session):
        inst = new_instrument(db_session); add_bars(db_session, inst, JSW_BARS)
        add_event(db_session, inst, EX, "Split", SPLIT(VS.MACHINE_PARSED), status=VS.UNVERIFIED); assess(db_session, inst)
        assert {"EVENT_NOT_MANUALLY_VERIFIED", "COMPONENT_NOT_MANUALLY_VERIFIED"} <= set(blocker_codes(self._refused(db_session, inst)))

    def test_a_legacy_verified_component_without_c2_evidence_blocks(self, db_session):
        v = self._refused(db_session, jsw(db_session, verified=False))
        assert "NO_ACTIVE_C2_VERIFICATION" in blocker_codes(v) and v.eligibility.blockers[0].detail["legacy_manually_verified_without_c2_evidence"]

    @pytest.mark.parametrize("ctype,code", [("RIGHTS", "UNSUPPORTED_ACTION_TYPE"), ("DEMERGER", "UNSUPPORTED_ACTION_TYPE"), ("MERGER", "UNSUPPORTED_ACTION_TYPE")])
    def test_unsupported_price_affecting_actions_block(self, db_session, ctype, code):
        inst = new_instrument(db_session); add_bars(db_session, inst, JSW_BARS)
        add_event(db_session, inst, EX, f"{ctype} 1:1", (ctype, VS.MACHINE_PARSED, {}, f"{ctype} 1:1"), status=VS.UNVERIFIED); assess(db_session, inst)
        assert code in blocker_codes(self._refused(db_session, inst))

    def test_an_unrecognised_other_text_blocks_but_a_benign_agm_and_a_dividend_do_not(self, db_session):
        inst = new_instrument(db_session); add_bars(db_session, inst, JSW_BARS)
        add_event(db_session, inst, EX, "Something Unknown", ("OTHER", VS.UNVERIFIED, {}, "Special Cum Something"), status=VS.UNVERIFIED); assess(db_session, inst)
        assert "UNCLASSIFIED_COMPONENT" in blocker_codes(self._refused(db_session, inst))
        ok = new_instrument(db_session); add_bars(db_session, ok, JSW_BARS)
        add_event(db_session, ok, EX, "AGM/Dividend", AGM(), DIV(), status=VS.UNVERIFIED); assess(db_session, ok)
        v = read_adjusted_bars(db_session, ok.id, D(2016, 12, 30), D(2017, 1, 5))
        assert v.status == af.ADJUSTED and set(factors(v).values()) == {F(1)} and v.basis["events"] == []

    def test_a_blocking_register_item_blocks_even_a_verified_event(self, db_session):
        v = self._refused(db_session, jsw(db_session, register=[reg(EX, "SPLIT")]))
        assert blocker_codes(v) == ["UNRESOLVED_REGISTER_ITEM"] and v.eligibility.barrier_dates == (EX,)

    def test_an_info_register_item_does_not_block(self, db_session):
        inst = jsw(db_session, register=[reg(EX, "DIAGNOSTIC", "WATCH_JUMP", "INFO")])
        assert read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5)).status == af.ADJUSTED

    def test_a_dateless_blocking_register_item_blocks_every_request(self, db_session):
        inst = jsw(db_session, register=[reg(None, "OTHER")])
        assert self._refused(db_session, inst, D(2017, 1, 4), D(2017, 1, 5)).safe_start_hint is None

    def test_no_assessment_means_unknown_coverage_and_a_refusal(self, db_session):
        v = self._refused(db_session, jsw(db_session, assessment=False))
        assert blocker_codes(v) == ["COVERAGE_NOT_COMPLETE"] and v.eligibility.coverage_status == "UNKNOWN"

    def test_a_conflicted_assessment_refuses(self, db_session):
        inst = jsw(db_session, assessment=False); assess(db_session, inst, evidence=conflicted())
        assert blocker_codes(self._refused(db_session, inst)) == ["COVERAGE_NOT_COMPLETE"]

    def test_an_as_of_beyond_the_assessed_window_refuses(self, db_session):
        inst = jsw(db_session, window=(D(2015, 1, 1), D(2017, 1, 5)))
        assert read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5)).status == af.ADJUSTED
        assert blocker_codes(self._refused(db_session, inst, D(2016, 12, 30), D(2017, 3, 1))) == ["COVERAGE_NOT_COMPLETE"]

    def test_a_disputed_component_blocks(self, db_session):
        inst = new_instrument(db_session); add_bars(db_session, inst, JSW_BARS)
        add_event(db_session, inst, EX, "Split", SPLIT(VS.DISPUTED), status=VS.MANUALLY_VERIFIED, legacy=True); assess(db_session, inst)
        assert "COMPONENT_NOT_MANUALLY_VERIFIED" in blocker_codes(self._refused(db_session, inst))

    def test_a_revoked_c2_verification_blocks_again(self, db_session):
        inst = jsw(db_session)
        assert read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5)).status == af.ADJUSTED
        rec = db_session.scalar(select(CorporateActionTermVerification).where(CorporateActionTermVerification.action == "VERIFIED"))
        insert_c2_record(db_session, rec.component_id, "SPLIT", 10, 1, action="REVOKED", revokes_id=rec.id)
        db_session.flush()
        v = self._refused(db_session, inst)
        assert "NO_ACTIVE_C2_VERIFICATION" in blocker_codes(v)

    def test_the_views_eligibility_equals_the_gates_for_every_state_and_window(self, db_session):
        inst = jsw(db_session, register=[reg(EX, "SPLIT")])
        for start, end in [(D(2016, 12, 30), D(2017, 1, 5)), (D(2016, 12, 30), D(2017, 1, 3)), (EX, D(2017, 1, 5)), (D(2017, 1, 3), EX), (EX, EX)]:
            v = read_adjusted_bars(db_session, inst.id, start, end)
            assert v.eligibility == assess_adjustment_eligibility(db_session, inst.id, start, end)
            assert (v.status == af.REFUSED) == (not v.eligibility.eligible)

    def test_the_barrier_only_blocks_intervals_that_straddle_it(self, db_session):
        inst = jsw(db_session, register=[reg(EX, "SPLIT")])
        assert read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 3)).status == af.ADJUSTED
        assert read_adjusted_bars(db_session, inst.id, EX, D(2017, 1, 5)).status == af.ADJUSTED
        blocked = read_adjusted_bars(db_session, inst.id, D(2017, 1, 3), D(2017, 1, 5))
        assert blocked.status == af.REFUSED and blocked.safe_start_hint == EX

    def test_start_after_as_of_is_rejected(self, db_session):
        with pytest.raises(ValueError):
            read_adjusted_bars(db_session, jsw(db_session).id, D(2017, 1, 5), D(2017, 1, 4))


class TestBarriersAndCompounds:
    def _kotak(self, db):
        inst = new_instrument(db)
        add_bars(db, inst, [(D(2015, 7, 7), 1000.0, 1010.0, 990.0, 1005.0), (D(2015, 7, 8), 500.0, 505.0, 495.0, 502.0),
                            (D(2026, 1, 13), 2000.0, 2010.0, 1990.0, 2005.0), (D(2026, 1, 14), 400.0, 402.0, 398.0, 401.0),
                            (D(2026, 9, 11), 420.0, 425.0, 418.0, 424.0)])
        add_event(db, inst, D(2015, 7, 8), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        add_event(db, inst, D(2026, 1, 14), "Split 5 to 1", SPLIT(fv=(5.0, 1.0)))
        assess(db, inst, [reg(D(2015, 7, 8), "BONUS")])
        return inst

    def test_the_full_window_stays_blocked_by_the_unresolved_bonus_whatever_else_is_verified(self, db_session):
        inst = self._kotak(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2015, 7, 7), D(2026, 9, 11))
        assert v.status == af.REFUSED and v.bars == () and D(2015, 7, 8) in v.eligibility.barrier_dates
        assert {"UNRESOLVED_REGISTER_ITEM", "EVENT_NOT_MANUALLY_VERIFIED"} <= set(blocker_codes(v))

    def test_a_request_that_starts_after_the_barrier_is_adjusted_by_the_verified_split_only(self, db_session):
        inst = self._kotak(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2015, 7, 8), D(2026, 9, 11))
        assert v.status == af.ADJUSTED
        assert factors(v) == {D(2015, 7, 8): F(1, 5), D(2026, 1, 13): F(1, 5), D(2026, 1, 14): F(1), D(2026, 9, 11): F(1)}

    def _bajfinance(self, db):
        inst = new_instrument(db)
        add_bars(db, inst, [(D(2016, 9, 7), 11295.0, 11740.0, 11294.0, 11393.3), (D(2016, 9, 8), 1153.9, 1173.6, 1080.0, 1162.8),
                            (D(2025, 6, 13), 9150.0, 9365.5, 9147.0, 9331.0), (D(2025, 6, 16), 956.0, 956.0, 925.0, 938.0),
                            (D(2026, 9, 11), 900.0, 910.0, 895.0, 905.0)])
        add_event(db, inst, D(2016, 9, 8), "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- To Rs 2/-", BONUS(), SPLIT(fv=(10.0, 2.0)))
        add_event(db, inst, D(2025, 6, 16), "Bonus 4:1", BONUS(ratio=(4, 1)))
        add_event(db, inst, D(2025, 6, 16), "Face Value Split (Sub-Division) - From Rs 2/- To Re 1/-", SPLIT(fv=(2.0, 1.0)))
        assess(db, inst)
        return inst

    def test_the_2016_compound_event_is_one_tenth(self, db_session):
        inst = self._bajfinance(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 9, 7), D(2016, 9, 8))
        assert v.status == af.ADJUSTED and factors(v) == {D(2016, 9, 7): F(1, 10), D(2016, 9, 8): F(1)}
        (ev,) = v.basis["events"]
        assert sorted((c["kind"], c["factor"]["denominator"]) for c in ev["components"]) == [("BONUS", "2"), ("SPLIT", "5")]
        assert ev["event_factor"] == {"numerator": "1", "denominator": "10"}

    def test_the_2025_pair_of_same_date_events_is_one_tenth(self, db_session):
        inst = self._bajfinance(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2025, 6, 13), D(2025, 6, 16))
        assert v.status == af.ADJUSTED and factors(v) == {D(2025, 6, 13): F(1, 10), D(2025, 6, 16): F(1)}
        assert sorted(e["event_factor"]["denominator"] for e in v.basis["events"]) == ["2", "5"]

    def test_both_pairs_compound_to_exactly_one_hundredth(self, db_session):
        inst = self._bajfinance(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 9, 7), D(2026, 9, 11))
        assert v.status == af.ADJUSTED
        assert factors(v) == {D(2016, 9, 7): F(1, 100), D(2016, 9, 8): F(1, 10), D(2025, 6, 13): F(1, 10), D(2025, 6, 16): F(1), D(2026, 9, 11): F(1)}
        assert v.basis["cumulative_factor_at_requested_start"] == {"numerator": "1", "denominator": "100"}
        assert af.fraction_text(v.bars[0].close_exact) == "113933/1000"

    def test_an_unverified_2025_bonus_blocks_the_window_containing_it_but_not_the_2016_window(self, db_session):
        inst = new_instrument(db_session)
        add_bars(db_session, inst, [(D(2016, 9, 7), 11295.0, 11740.0, 11294.0, 11393.3), (D(2016, 9, 8), 1153.9, 1173.6, 1080.0, 1162.8),
                                    (D(2025, 6, 13), 9150.0, 9365.5, 9147.0, 9331.0), (D(2025, 6, 16), 956.0, 956.0, 925.0, 938.0)])
        add_event(db_session, inst, D(2016, 9, 8), "compound", BONUS(), SPLIT(fv=(10.0, 2.0)))
        add_event(db_session, inst, D(2025, 6, 16), "Bonus 4:1", BONUS(VS.MACHINE_PARSED, (4, 1)), status=VS.UNVERIFIED)
        add_event(db_session, inst, D(2025, 6, 16), "split", SPLIT(fv=(2.0, 1.0)))
        assess(db_session, inst, [reg(D(2025, 6, 16), "BONUS"), reg(D(2025, 6, 16), "SPLIT")])
        assert read_adjusted_bars(db_session, inst.id, D(2016, 9, 7), D(2025, 6, 16)).status == af.REFUSED
        ok = read_adjusted_bars(db_session, inst.id, D(2016, 9, 7), D(2025, 6, 13))
        assert ok.status == af.ADJUSTED and factors(ok) == {D(2016, 9, 7): F(1, 10), D(2016, 9, 8): F(1), D(2025, 6, 13): F(1)}

    def test_the_insertion_order_of_verified_events_never_changes_a_factor(self, db_session):
        rows = [(D(2016, 9, 8), "Bonus 1:1", BONUS()), (D(2016, 9, 8), "split", SPLIT(fv=(10.0, 2.0))),
                (D(2025, 6, 16), "Bonus 4:1", BONUS(ratio=(4, 1))), (D(2025, 6, 16), "split", SPLIT(fv=(2.0, 1.0)))]
        seen = []
        for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
            inst = new_instrument(db_session)
            add_bars(db_session, inst, [(D(2016, 9, 7), 100.0, 110.0, 90.0, 105.0), (D(2016, 9, 8), 10.0, 11.0, 9.0, 10.5), (D(2025, 6, 13), 9.0, 11.0, 8.0, 10.0), (D(2025, 6, 16), 1.0, 1.1, 0.9, 1.0)])
            for i in order:
                add_event(db_session, inst, rows[i][0], rows[i][1], rows[i][2])
            assess(db_session, inst)
            seen.append(factors(read_adjusted_bars(db_session, inst.id, D(2016, 9, 7), D(2026, 9, 11))))
        assert seen[0] == seen[1] == seen[2] and seen[0][D(2016, 9, 7)] == F(1, 100)


class TestPointInTimeAgainstTheDatabase:
    def test_a_later_event_never_changes_an_earlier_as_of_view(self, db_session):
        inst = jsw(db_session)
        before = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        add_bars(db_session, inst, [(D(2018, 6, 1), 200.0, 210.0, 190.0, 205.0), (D(2018, 6, 4), 100.0, 105.0, 95.0, 101.0)])
        add_event(db_session, inst, D(2018, 6, 4), "Bonus 1:1", BONUS())
        add_event(db_session, inst, D(2019, 1, 1), "Rights", ("RIGHTS", VS.UNVERIFIED, {}, "Rights 1:4"), status=VS.UNVERIFIED)
        after = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        assert after == before and after.view_sha256 == before.view_sha256
        later = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2018, 6, 4))
        assert later.status == af.ADJUSTED and later.bars[0].factor == F(1, 20)
        assert read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2019, 1, 1)).status == af.REFUSED

    def test_an_event_of_another_instrument_is_irrelevant(self, db_session):
        a, b = jsw(db_session), new_instrument(db_session)
        add_event(db_session, b, D(2017, 1, 3), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        assert read_adjusted_bars(db_session, a.id, D(2016, 12, 30), D(2017, 1, 5)).status == af.ADJUSTED

    def test_no_event_in_the_interval_means_factor_one_for_every_bar(self, db_session):
        inst = new_instrument(db_session); add_bars(db_session, inst, JSW_BARS); assess(db_session, inst)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        assert v.status == af.ADJUSTED and set(factors(v).values()) == {F(1)} and v.basis["events"] == []
        assert all((b.open, b.high, b.low, b.close) == (b.raw.open, b.raw.high, b.raw.low, b.raw.close) for b in v.bars)


class TestDataQualityAgainstTheDatabase:
    def test_an_instrument_with_no_bars_is_refused(self, db_session):
        inst = new_instrument(db_session); assess(db_session, inst)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        assert v.status == af.REFUSED and v.refusal_reasons == (af.NO_BARS,)

    def test_an_unknown_instrument_is_refused_not_an_error(self, db_session):
        v = read_adjusted_bars(db_session, uuid.uuid4(), D(2016, 12, 30), D(2017, 1, 5))
        assert v.status == af.REFUSED and af.NO_BARS in v.refusal_reasons

    def test_a_missing_session_seen_on_another_instrument_is_a_gap(self, db_session):
        inst = jsw(db_session)
        other = new_instrument(db_session)
        add_bars(db_session, other, JSW_BARS + [(D(2017, 1, 6), 10.0, 11.0, 9.0, 10.0)])
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 6))
        assert v.status == af.REFUSED and af.MISSING_BARS in v.refusal_reasons and v.data_quality["missing_sessions"] == ["2017-01-06"]
        assert v.bars == () and v.data_quality["complete"] is False

    def test_allowing_gaps_yields_a_view_that_says_it_has_gaps(self, db_session):
        inst = jsw(db_session)
        other = new_instrument(db_session)
        add_bars(db_session, other, JSW_BARS + [(D(2017, 1, 6), 10.0, 11.0, 9.0, 10.0)])
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 6), allow_missing_bars=True)
        assert v.status == af.ADJUSTED_WITH_GAPS and len(v.bars) == 5 and v.data_quality["complete"] is False
        assert v.data_quality["missing_sessions"] == ["2017-01-06"] and v.data_quality["expected_session_count"] == 6

    def test_a_gap_in_the_middle_never_changes_the_factors_of_the_bars_that_exist(self, db_session):
        inst = new_instrument(db_session)
        add_bars(db_session, inst, [b for b in JSW_BARS if b[0] != D(2017, 1, 3)])
        add_event(db_session, inst, EX, "split", SPLIT(fv=(10.0, 1.0))); assess(db_session, inst)
        other = new_instrument(db_session); add_bars(db_session, other, JSW_BARS)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5), allow_missing_bars=True)
        assert v.data_quality["missing_sessions"] == ["2017-01-03"] and factors(v)[D(2017, 1, 2)] == F(1, 10) and factors(v)[EX] == F(1)

    def test_a_zero_price_bar_is_refused_as_invalid(self, db_session):
        inst = new_instrument(db_session)
        add_bars(db_session, inst, [(D(2017, 1, 2), 0.0, 0.0, 0.0, 0.0), (D(2017, 1, 3), 1.0, 2.0, 1.0, 2.0)]); assess(db_session, inst)
        v = read_adjusted_bars(db_session, inst.id, D(2017, 1, 2), D(2017, 1, 3))
        assert v.status == af.REFUSED and af.INVALID_RAW_BARS in v.refusal_reasons and v.data_quality["invalid_raw_bars"][0][0] == "2017-01-02"

    def test_other_series_and_timeframes_are_never_mixed_in(self, db_session):
        inst = jsw(db_session)
        add_bars(db_session, inst, [(D(2017, 1, 3), 1.0, 2.0, 1.0, 2.0)], series="BL")
        add_bars(db_session, inst, [(D(2017, 1, 3), 3.0, 4.0, 3.0, 4.0)], tf="5m")
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 5))
        assert v.status == af.ADJUSTED and len(v.bars) == 5 and all(b.raw.series == "EQ" and b.raw.timeframe == "1D" for b in v.bars)
        assert {(x.raw.series, x.raw.timeframe) for x in read_adjusted_bars(db_session, inst.id, D(2017, 1, 3), D(2017, 1, 3), series="BL").bars} == {("BL", "1D")}

    def test_bars_after_the_as_of_are_never_returned(self, db_session):
        inst = jsw(db_session)
        v = read_adjusted_bars(db_session, inst.id, D(2016, 12, 30), D(2017, 1, 3))
        assert max(b.raw.trade_date for b in v.bars) == D(2017, 1, 3) and len(v.bars) == 3
