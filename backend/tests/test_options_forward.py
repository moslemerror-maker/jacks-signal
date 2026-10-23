from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

import pytest

from backend.app.options.contracts import ContractIdentity
from backend.app.options.forward import ForwardResolution, resolve_forward_from_live_future, resolve_forward_spot_proxy
from backend.app.options.live_quote import LiveQuote

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
FUT_CONTRACT = ContractIdentity("NIFTY", "IDF", D(2026, 9, 29), None, None)
OPT_CONTRACT = ContractIdentity("NIFTY", "IDO", D(2026, 9, 29), 2450000, "CE")


def futq(**overrides) -> LiveQuote:
    fields = dict(
        contract=FUT_CONTRACT, observed_at=dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC), last_price=Decimal("23350"),
        bid=Decimal("23349"), ask=Decimal("23351"), volume=1000, open_interest=50000, underlying_last_price=Decimal("23329"),
        provider_reported_iv=None, provider_reported_greeks=None, source="dhan", source_row_hash=None, provenance_class=FIXTURE,
    )
    fields.update(overrides)
    return LiveQuote.from_raw(**fields)


class TestForwardResolutionValidation:
    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match="source"):
            ForwardResolution(forward=Decimal(100), source="GUESSED", underlying_last_price=Decimal(100), detail="x")  # type: ignore[arg-type]

    def test_nonpositive_forward_rejected(self):
        with pytest.raises(ValueError, match="forward"):
            ForwardResolution(forward=Decimal(0), source="SPOT_PROXY_UNVERIFIED", underlying_last_price=Decimal(100), detail="x")

    def test_nonpositive_underlying_rejected(self):
        with pytest.raises(ValueError, match="underlying_last_price"):
            ForwardResolution(forward=Decimal(100), source="SPOT_PROXY_UNVERIFIED", underlying_last_price=Decimal(-1), detail="x")

    def test_frozen(self):
        r = resolve_forward_spot_proxy(Decimal(100))
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.forward = Decimal(200)  # type: ignore[misc]


class TestSpotProxyFallback:
    def test_uses_spot_as_forward_tagged_unverified(self):
        r = resolve_forward_spot_proxy(Decimal("23329"))
        assert r.forward == Decimal("23329") and r.source == "SPOT_PROXY_UNVERIFIED"
        assert r.is_verified is False

    def test_nonpositive_input_rejected(self):
        with pytest.raises(ValueError, match="underlying_last_price"):
            resolve_forward_spot_proxy(Decimal(0))


class TestVerifiedLiveFuturesPath:
    def test_valid_future_quote_resolves_verified(self):
        q = futq()
        r = resolve_forward_from_live_future(q)
        assert r.forward == Decimal("23350") and r.source == "VERIFIED_LIVE_FUTURES" and r.is_verified is True
        assert r.underlying_last_price == Decimal("23329")

    def test_underlying_last_price_falls_back_to_futures_price_when_unset(self):
        q = futq(underlying_last_price=None)
        r = resolve_forward_from_live_future(q)
        assert r.underlying_last_price == Decimal("23350")

    def test_non_futures_contract_rejected(self):
        q = futq(contract=OPT_CONTRACT)
        with pytest.raises(ValueError, match="futures contract"):
            resolve_forward_from_live_future(q)

    def test_no_quote_rejected(self):
        q = futq(last_price=None, bid=None, ask=None)
        with pytest.raises(ValueError, match="not usable"):
            resolve_forward_from_live_future(q)

    def test_anomalous_quote_rejected(self):
        q = futq(last_price=Decimal(0))
        with pytest.raises(ValueError, match="not usable"):
            resolve_forward_from_live_future(q)
