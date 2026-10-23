from __future__ import annotations

import pytest

from backend.app.data.sources import credential_preflight as cp
from backend.app.data.sources import provider_credentials as pc


@pytest.fixture(autouse=True)
def no_real_env(monkeypatch):
    monkeypatch.setattr(cp, "load_dotenv", lambda: None)
    monkeypatch.setattr(pc, "load_dotenv", lambda: None)
    for k in ("DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN"):
        monkeypatch.delenv(k, raising=False)


FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.dGVzdC1zaWduYXR1cmU"


class TestDhanCredentialPreflight:
    def test_missing_both(self):
        r = cp.dhan_credential_preflight()
        assert r.provider == "dhan" and r.status == "MISSING"

    def test_missing_one(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "123456")
        r = cp.dhan_credential_preflight()
        assert r.status == "MISSING" and "DHAN_ACCESS_TOKEN" in r.detail

    def test_non_numeric_client_id_is_malformed(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "not-a-number")
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", FAKE_JWT)
        r = cp.dhan_credential_preflight()
        assert r.status == "MALFORMED" and "DHAN_CLIENT_ID" in r.detail

    def test_non_jwt_shaped_token_is_malformed(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "123456")
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", "not-a-jwt-at-all")
        r = cp.dhan_credential_preflight()
        assert r.status == "MALFORMED" and "DHAN_ACCESS_TOKEN" in r.detail

    def test_present_and_well_formed_is_ok(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "123456")
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", FAKE_JWT)
        r = cp.dhan_credential_preflight()
        assert r.status == "OK"
        assert "NOT verified" in r.detail

    def test_whitespace_only_client_id_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "   ")
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", FAKE_JWT)
        r = cp.dhan_credential_preflight()
        assert r.status == "MISSING"

    def test_never_echoes_the_credential_value(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "123456")
        secret_token = "eyJhbGciOiJIUzI1NiJ9.SECRETPAYLOAD12345.SECRETSIGNATURE6789"
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", secret_token)
        r = cp.dhan_credential_preflight()
        assert secret_token not in r.detail
        assert "SECRETPAYLOAD12345" not in str(r)

    def test_result_status_is_validated(self):
        with pytest.raises(ValueError, match="status"):
            cp.CredentialPreflightResult("dhan", "SUSPICIOUSLY_GOOD", "x")
