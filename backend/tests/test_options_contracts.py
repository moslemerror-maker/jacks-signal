from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from backend.app.options.aliases import ContractAlias, resolve
from backend.app.options.contracts import ContractIdentity, ContractSpec, spec_as_of
from backend.app.options.provenance import PROVENANCE_CLASSES, check_provenance_class
from scripts.run_o1_0b_sample_gate import ContractKey, legacy_bars, load, udiff_bars

D = dt.date
FIXTURE = "DEVELOPMENT_FIXTURE"
PRODUCTION = "PRODUCTION_RESEARCH_DATASET"


def nifty_call(strike=2200000, expiry=D(2024, 7, 25)) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", expiry, strike, "CE")


def nifty_fut(expiry=D(2024, 7, 25)) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDF", expiry, None, None)


class TestIdentityConstructionAndValidation:
    def test_a_valid_option_and_a_valid_future_construct(self):
        assert nifty_call().option_type == "CE"
        assert nifty_fut().strike_paise is None

    def test_strike_is_an_exact_int_never_a_float(self):
        with pytest.raises(ValueError, match="never a float"):
            ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), 22000.0, "CE")  # type: ignore[arg-type]

    def test_strike_must_be_positive_for_options(self):
        for bad in (0, -100):
            with pytest.raises(ValueError, match="positive integer strike_paise"):
                ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), bad, "CE")

    def test_option_requires_a_strike(self):
        with pytest.raises(ValueError, match="positive integer strike_paise"):
            ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), None, "CE")

    def test_option_requires_a_valid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), 2200000, "XX")  # type: ignore[arg-type]

    def test_american_and_european_types_are_both_accepted_and_distinct(self):
        e = ContractIdentity("SBIN", "STO", D(2010, 7, 29), 300000, "CE")
        a = ContractIdentity("SBIN", "STO", D(2010, 7, 29), 300000, "CA")
        assert e != a

    def test_future_must_not_carry_a_strike(self):
        with pytest.raises(ValueError, match="must have strike_paise=None"):
            ContractIdentity("NIFTY", "IDF", D(2024, 7, 25), 2200000, None)

    def test_future_must_not_carry_an_option_type(self):
        with pytest.raises(ValueError, match="must have option_type=None"):
            ContractIdentity("NIFTY", "IDF", D(2024, 7, 25), None, "CE")

    def test_unknown_derivative_class_rejected(self):
        with pytest.raises(ValueError, match="derivative_class"):
            ContractIdentity("NIFTY", "OPTIDX", D(2024, 7, 25), 2200000, "CE")  # type: ignore[arg-type]

    def test_empty_underlying_rejected(self):
        with pytest.raises(ValueError, match="underlying"):
            ContractIdentity("", "IDO", D(2024, 7, 25), 2200000, "CE")

    def test_strike_rupees_display(self):
        assert nifty_call(2235050).strike_rupees == "22350.50"
        with pytest.raises(ValueError):
            nifty_fut().strike_rupees

    def test_frozen_immutable(self):
        c = nifty_call()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.strike_paise = 1  # type: ignore[misc]


class TestIdentityEqualityAndHashing:
    def test_equal_fields_are_equal_and_hash_equal(self):
        a = nifty_call()
        b = ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), 2200000, "CE")
        assert a == b and hash(a) == hash(b)

    @pytest.mark.parametrize("field, value", [("underlying", "BANKNIFTY"), ("derivative_class", "STO"), ("expiry", D(2024, 8, 1)),
                                              ("strike_paise", 2300000), ("option_type", "PE")])
    def test_any_differing_field_makes_it_unequal(self, field, value):
        base = dataclasses.asdict(nifty_call())
        base[field] = value
        assert ContractIdentity(**base) != nifty_call()

    def test_usable_as_a_dict_key_and_set_member(self):
        a, b = nifty_call(), ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), 2200000, "CE")
        d = {a: "first"}
        d[b] = "second"
        assert len(d) == 1 and d[a] == "second"
        assert len({a, b, nifty_fut()}) == 2

    def test_futures_and_options_on_the_same_underlying_expiry_are_distinct(self):
        assert nifty_fut() != nifty_call() and hash(nifty_fut()) != hash(nifty_call())


class TestContractSpec:
    def test_valid_spec(self):
        s = ContractSpec(nifty_call(), lot_size=50, effective_from=D(2024, 1, 1), effective_to=D(2024, 6, 30), provenance_class=FIXTURE)
        assert s.covers(D(2024, 3, 1)) and not s.covers(D(2024, 7, 1))

    def test_open_ended_spec_covers_everything_from_its_start(self):
        s = ContractSpec(nifty_call(), lot_size=65, effective_from=D(2026, 1, 1), effective_to=None, provenance_class=FIXTURE)
        assert s.covers(D(2030, 1, 1)) and not s.covers(D(2025, 12, 31))

    def test_lot_size_must_be_positive(self):
        with pytest.raises(ValueError, match="lot_size"):
            ContractSpec(nifty_call(), lot_size=0, effective_from=D(2024, 1, 1), effective_to=None, provenance_class=FIXTURE)

    def test_effective_to_before_effective_from_rejected(self):
        with pytest.raises(ValueError, match="effective_to"):
            ContractSpec(nifty_call(), lot_size=50, effective_from=D(2024, 6, 1), effective_to=D(2024, 1, 1), provenance_class=FIXTURE)

    def test_provenance_class_is_mandatory(self):
        with pytest.raises(TypeError):
            ContractSpec(nifty_call(), lot_size=50, effective_from=D(2024, 1, 1), effective_to=None)  # type: ignore[call-arg]

    def test_provenance_class_rejects_an_invalid_value(self):
        with pytest.raises(ValueError, match="provenance_class"):
            ContractSpec(nifty_call(), lot_size=50, effective_from=D(2024, 1, 1), effective_to=None, provenance_class="MADE_UP")  # type: ignore[arg-type]

    def test_provenance_class_rejects_none_no_silent_default(self):
        with pytest.raises(ValueError, match="provenance_class"):
            ContractSpec(nifty_call(), lot_size=50, effective_from=D(2024, 1, 1), effective_to=None, provenance_class=None)  # type: ignore[arg-type]


class TestSpecAsOf:
    def test_finds_the_covering_spec_among_several(self):
        specs = [
            ContractSpec(nifty_call(), lot_size=50, effective_from=D(2024, 1, 1), effective_to=D(2024, 6, 30), provenance_class=FIXTURE),
            ContractSpec(nifty_call(), lot_size=25, effective_from=D(2024, 7, 1), effective_to=D(2024, 11, 20), provenance_class=FIXTURE),
            ContractSpec(nifty_call(), lot_size=75, effective_from=D(2024, 11, 21), effective_to=D(2026, 1, 5), provenance_class=FIXTURE),
            ContractSpec(nifty_call(), lot_size=65, effective_from=D(2026, 1, 6), effective_to=None, provenance_class=FIXTURE),
        ]
        assert spec_as_of(specs, D(2024, 3, 1)).lot_size == 50
        assert spec_as_of(specs, D(2024, 7, 4)).lot_size == 25
        assert spec_as_of(specs, D(2025, 1, 1)).lot_size == 75
        assert spec_as_of(specs, D(2026, 9, 18)).lot_size == 65
        assert spec_as_of(specs, D(2020, 1, 1)) is None

    def test_never_applies_a_later_spec_to_an_earlier_date(self):
        specs = [ContractSpec(nifty_call(), lot_size=65, effective_from=D(2026, 1, 6), effective_to=None, provenance_class=FIXTURE)]
        assert spec_as_of(specs, D(2024, 7, 5)) is None

    def test_overlapping_specs_raise_rather_than_silently_pick_one(self):
        specs = [
            ContractSpec(nifty_call(), lot_size=50, effective_from=D(2024, 1, 1), effective_to=D(2024, 8, 1), provenance_class=FIXTURE),
            ContractSpec(nifty_call(), lot_size=25, effective_from=D(2024, 7, 1), effective_to=D(2024, 12, 1), provenance_class=FIXTURE),
        ]
        with pytest.raises(ValueError, match="overlapping effective windows"):
            spec_as_of(specs, D(2024, 7, 15))

    def test_empty_specs_returns_none(self):
        assert spec_as_of([], D(2024, 1, 1)) is None


class TestProvenance:
    def test_both_supported_values(self):
        assert set(PROVENANCE_CLASSES) == {"DEVELOPMENT_FIXTURE", "PRODUCTION_RESEARCH_DATASET"}

    @pytest.mark.parametrize("bad", [None, "", "development_fixture", "PRODUCTION", "development fixture", 0])
    def test_rejects_everything_else(self, bad):
        with pytest.raises(ValueError, match="provenance_class"):
            check_provenance_class(bad)

    def test_accepts_both_valid_values(self):
        assert check_provenance_class(FIXTURE) == FIXTURE
        assert check_provenance_class(PRODUCTION) == PRODUCTION


class TestContractAlias:
    def test_source_id_does_not_affect_canonical_identity(self):
        c1, c2 = nifty_call(), nifty_call()
        a1 = ContractAlias(c1, source="upstox", token="NSE_FO|999", first_seen=D(2024, 1, 1), last_seen=D(2024, 7, 25), provenance_class=FIXTURE)
        a2 = ContractAlias(c2, source="nse_archives", token="177686", first_seen=D(2024, 1, 1), last_seen=D(2024, 7, 25), provenance_class=FIXTURE)
        assert a1.contract == a2.contract == c1
        assert a1 != a2

    def test_alias_is_a_distinct_type_from_identity_never_equal(self):
        c = nifty_call()
        a = ContractAlias(c, source="nse_archives", token="177686", first_seen=D(2024, 1, 1), last_seen=D(2024, 7, 25), provenance_class=FIXTURE)
        assert a != c and c != a

    def test_one_identity_can_have_several_live_aliases_at_once(self):
        c = nifty_call()
        aliases = [
            ContractAlias(c, source="nse_archives", token="177686", first_seen=D(2024, 1, 1), last_seen=D(2024, 7, 25), provenance_class=FIXTURE),
            ContractAlias(c, source="upstox", token="NSE_FO|999", first_seen=D(2024, 1, 1), last_seen=D(2024, 7, 25), provenance_class=FIXTURE),
        ]
        assert len({a.token for a in aliases}) == 2 and all(a.contract == c for a in aliases)

    def test_a_token_can_be_recycled_across_time_for_different_contracts(self):
        c1 = ContractIdentity("ESCORTS", "STO", D(2024, 1, 25), 315000, "CE")
        c2 = ContractIdentity("HINDPETRO", "STO", D(2025, 3, 27), 26000, "PE")
        old = ContractAlias(c1, source="nse_archives", token="114864", first_seen=D(2024, 1, 1), last_seen=D(2024, 1, 25), provenance_class=FIXTURE)
        new = ContractAlias(c2, source="nse_archives", token="114864", first_seen=D(2025, 1, 1), last_seen=D(2025, 3, 27), provenance_class=FIXTURE)
        assert resolve([old, new], source="nse_archives", token="114864", as_of=D(2024, 1, 10)) == c1
        assert resolve([old, new], source="nse_archives", token="114864", as_of=D(2025, 2, 1)) == c2

    def test_resolve_returns_none_when_nothing_matches(self):
        assert resolve([], source="nse_archives", token="1", as_of=D(2024, 1, 1)) is None

    def test_resolve_raises_on_an_ambiguous_token(self):
        c1, c2 = nifty_call(2200000), nifty_call(2250000)
        overlapping = [
            ContractAlias(c1, source="nse_archives", token="X", first_seen=D(2024, 1, 1), last_seen=D(2024, 12, 31), provenance_class=FIXTURE),
            ContractAlias(c2, source="nse_archives", token="X", first_seen=D(2024, 6, 1), last_seen=D(2024, 12, 31), provenance_class=FIXTURE),
        ]
        with pytest.raises(ValueError, match="ambiguous token"):
            resolve(overlapping, source="nse_archives", token="X", as_of=D(2024, 7, 1))

    def test_empty_source_or_token_rejected(self):
        with pytest.raises(ValueError, match="source"):
            ContractAlias(nifty_call(), source="", token="1", first_seen=D(2024, 1, 1), last_seen=D(2024, 1, 1), provenance_class=FIXTURE)
        with pytest.raises(ValueError, match="token"):
            ContractAlias(nifty_call(), source="nse_archives", token="", first_seen=D(2024, 1, 1), last_seen=D(2024, 1, 1), provenance_class=FIXTURE)

    def test_last_seen_before_first_seen_rejected(self):
        with pytest.raises(ValueError, match="last_seen"):
            ContractAlias(nifty_call(), source="nse_archives", token="1", first_seen=D(2024, 6, 1), last_seen=D(2024, 1, 1), provenance_class=FIXTURE)

    def test_provenance_class_is_mandatory_on_alias_too(self):
        with pytest.raises(TypeError):
            ContractAlias(nifty_call(), source="nse_archives", token="1", first_seen=D(2024, 1, 1), last_seen=D(2024, 1, 1))  # type: ignore[call-arg]


def key_to_identity(key: ContractKey) -> ContractIdentity | None:
    if key.dtype not in ("IDF", "STF", "IDO", "STO"):
        return None
    strike = None if key.cp is None else key.strike_paise
    if key.dtype in ("IDF", "STF"):
        strike = None
    return ContractIdentity(key.underlying, key.dtype, key.expiry, strike, key.cp)


def load_fixture_identities(family: str, day: dt.date) -> list[ContractIdentity]:
    got = load(family, day)
    if got is None:
        pytest.skip(f"O1.0B NSE sample not present on disk for {family}:{day}")
    bars = legacy_bars(*got) if family == "legacy" else udiff_bars(*got)
    out = []
    for b in bars:
        ident = key_to_identity(b.key)
        if ident is not None:
            out.append(ident)
    return out


class TestFixtureReplay:

    def test_every_parsed_identity_is_tagged_a_fixture_when_wrapped_as_an_alias(self):
        idents = load_fixture_identities("udiff", D(2026, 9, 18))
        aliases = [ContractAlias(c, source="nse_archives", token=str(i), first_seen=D(2026, 9, 18), last_seen=D(2026, 9, 18), provenance_class=FIXTURE)
                  for i, c in enumerate(idents)]
        assert aliases and all(a.provenance_class == FIXTURE for a in aliases)

    def test_no_duplicate_identities_within_one_fixture_file(self):
        idents = load_fixture_identities("udiff", D(2026, 9, 18))
        assert len(idents) == len(set(idents))

    def test_strike_paise_round_trips_exactly_for_a_real_contract(self):
        idents = load_fixture_identities("udiff", D(2025, 9, 30))
        nifty_calls = [c for c in idents if c.underlying == "NIFTY" and c.derivative_class == "IDO" and c.option_type == "CE"]
        assert nifty_calls and all(isinstance(c.strike_paise, int) and c.strike_paise % 50 == 0 for c in nifty_calls[:20])

    @pytest.mark.parametrize("family, day, underlying, dclass, expected_count", [
        ("udiff", D(2024, 7, 5), "NIFTY", "IDO", 1634),
        ("udiff", D(2024, 7, 5), "BANKNIFTY", "IDO", 1636),
        ("udiff", D(2026, 9, 18), "NIFTY", "IDO", 1700),
        ("udiff", D(2026, 9, 18), "BANKNIFTY", "IDO", 927),
    ])
    def test_established_contract_counts_are_unchanged_by_the_new_identity_type(self, family, day, underlying, dclass, expected_count):
        idents = load_fixture_identities(family, day)
        matching = [c for c in idents if c.underlying == underlying and c.derivative_class == dclass]
        assert len(matching) == expected_count

    def test_documented_regression_case_nifty_and_banknifty_2025_09_30(self):
        idents = load_fixture_identities("udiff", D(2025, 9, 30))
        nifty = [c for c in idents if c.underlying == "NIFTY" and c.derivative_class == "IDO"]
        bn = [c for c in idents if c.underlying == "BANKNIFTY" and c.derivative_class == "IDO"]
        assert len(nifty) == 1572
        assert len(bn) == 921
        assert len(bn) == len({c for c in bn})
        bn_expiring = sorted({c.strike_paise for c in bn if c.expiry == D(2025, 9, 30)})
        import math
        finest_step = 0
        for a, b in zip(bn_expiring, bn_expiring[1:]):
            finest_step = math.gcd(finest_step, b - a)
        assert finest_step == 10000
        assert not hasattr(nifty[0], "provenance_class")

    def test_futures_rows_in_the_fixture_carry_no_strike_or_option_type(self):
        idents = load_fixture_identities("udiff", D(2026, 9, 18))
        futures = [c for c in idents if c.derivative_class in ("IDF", "STF")]
        assert futures and all(c.strike_paise is None and c.option_type is None for c in futures)
