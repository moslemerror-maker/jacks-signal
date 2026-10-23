from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import pathlib
import zipfile

import pytest

from backend.app.data.raw.adapter import FetchedContent, IngestStatus, ProviderAdapter, ingest
from backend.app.data.raw.schema_registry import (NSE_FO_BHAVCOPY_LEGACY, NSE_FO_BHAVCOPY_UDIFF, NSE_FO_CONTRACT_MASTER, SchemaMismatch, SchemaRegistry, SchemaSpec,
                                                  default_registry, format_version_of)
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import nse_fo_archive as nfa
from backend.app.data.sources.errors import InvalidContentError

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
D = dt.date
LEGACY_HDR = list(NSE_FO_BHAVCOPY_LEGACY.named_columns) + [""]
UDIFF_HDR_A = list(NSE_FO_BHAVCOPY_UDIFF.named_columns) + ["Rsvd01", "Rsvd02", "Rsvd03", "Rsvd04"]
UDIFF_HDR_B = list(NSE_FO_BHAVCOPY_UDIFF.named_columns) + ["Rsvd1", "Rsvd2", "Rsvd3", "Rsvd4"]


def zipped(header, rows=1) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for i in range(rows):
        w.writerow([f"v{i}"] * len(header))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("x.csv", buf.getvalue())
    return out.getvalue()


def gzipped(header, rows=1) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for i in range(rows):
        w.writerow([f"v{i}"] * len(header))
    return gzip.compress(buf.getvalue().encode())


class TestSchemaRegistry:
    reg = default_registry()

    def test_legacy_with_trailing_empty_column(self):
        spec, fv = self.reg.classify("nse_archives", "fo_bhavcopy_legacy", LEGACY_HDR)
        assert spec.schema_id == "nse.fo.bhavcopy.legacy/1" and fv == format_version_of(LEGACY_HDR)

    def test_udiff_reserved_column_rename_is_the_same_schema_but_a_different_format_version(self):
        a, fa = self.reg.classify("nse_archives", "fo_bhavcopy_udiff", UDIFF_HDR_A)
        b, fb = self.reg.classify("nse_archives", "fo_bhavcopy_udiff", UDIFF_HDR_B)
        assert a.schema_id == b.schema_id == "nse.fo.bhavcopy.udiff/1" and fa != fb

    def test_header_whitespace_is_stripped(self):
        spec, _ = self.reg.classify("nse_archives", "fo_bhavcopy_udiff", [" " + c + " " for c in UDIFF_HDR_A])
        assert spec.schema_id.startswith("nse.fo.bhavcopy.udiff")

    @pytest.mark.parametrize("mutate, kind", [
        (lambda h: h + ["NewBusinessColumn"], "unknown NAMED columns"),
        (lambda h: [c for c in h if c != "OpnIntrst"], "missing columns"),
        (lambda h: [h[1], h[0]] + h[2:], "reordered"),
        (lambda h: [c.replace("SttlmPric", "SettlementPrice") for c in h], "unknown NAMED columns"),
    ])
    def test_a_change_to_the_named_columns_is_a_hard_stop(self, mutate, kind):
        with pytest.raises(SchemaMismatch, match=kind):
            self.reg.classify("nse_archives", "fo_bhavcopy_udiff", mutate(UDIFF_HDR_A))

    def test_a_schema_mismatch_is_an_invalid_content_error(self):
        assert issubclass(SchemaMismatch, InvalidContentError)

    def test_unregistered_dataset(self):
        with pytest.raises(SchemaMismatch, match="no schema registered"):
            self.reg.classify("nse_archives", "nope", ["a"])
        with pytest.raises(SchemaMismatch):
            self.reg.inspect("other_provider", "fo_bhavcopy_udiff", b"x")

    def test_a_schema_cannot_be_edited_in_place(self):
        with pytest.raises(ValueError, match="already registered"):
            SchemaRegistry([NSE_FO_BHAVCOPY_LEGACY, NSE_FO_BHAVCOPY_LEGACY])

    def test_a_new_schema_version_can_be_added_alongside_and_the_matching_one_wins(self):
        v2 = SchemaSpec("nse.fo.bhavcopy.udiff/2", "nse_archives", "fo_bhavcopy_udiff", "zip", NSE_FO_BHAVCOPY_UDIFF.named_columns + ("Added",))
        reg = SchemaRegistry([NSE_FO_BHAVCOPY_UDIFF, v2])
        assert reg.classify("nse_archives", "fo_bhavcopy_udiff", UDIFF_HDR_A)[0].schema_id.endswith("/1")
        assert reg.classify("nse_archives", "fo_bhavcopy_udiff", UDIFF_HDR_A + ["Added"])[0].schema_id.endswith("/2")

    def test_inspect_reads_zip_and_gzip(self):
        info = self.reg.inspect("nse_archives", "fo_bhavcopy_legacy", zipped(LEGACY_HDR, rows=3))
        assert (info.schema_id, info.data_rows) == ("nse.fo.bhavcopy.legacy/1", 3)
        info = self.reg.inspect("nse_archives", "fo_contract_master", gzipped(list(NSE_FO_CONTRACT_MASTER.named_columns) + ["Rsvd01"], rows=2))
        assert info.schema_id == "nse.fo.contract_master/1" and len(NSE_FO_CONTRACT_MASTER.named_columns) == 135

    @pytest.mark.parametrize("payload, why", [
        (b"<html>Access Denied</html>", "not a valid zip"),
        (b"", "not a valid zip"),
        (zipped(LEGACY_HDR, rows=0), "no data rows"),
    ])
    def test_unusable_content_is_rejected(self, payload, why):
        with pytest.raises(InvalidContentError, match=why):
            self.reg.inspect("nse_archives", "fo_bhavcopy_legacy", payload)

    def test_a_zip_with_two_members_is_rejected(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("a.csv", "x\n1\n")
            z.writestr("b.csv", "x\n1\n")
        with pytest.raises(InvalidContentError, match="exactly one member"):
            self.reg.inspect("nse_archives", "fo_bhavcopy_legacy", out.getvalue())

    def test_a_gzip_that_is_not_gzip(self):
        with pytest.raises(InvalidContentError, match="not a valid gzip"):
            self.reg.inspect("nse_archives", "fo_contract_master", b"plain text, not gzip")


class FakeAdapter:
    provider_id = "fake"
    fetcher_version = "fake/1"
    license_status = "REQUIRES_PROVIDER_CONFIRMATION"

    def __init__(self, responses):
        self.responses, self.calls = dict(responses), 0
        self.registry = default_registry()

    def datasets(self):
        return ("fo_bhavcopy_legacy",)

    def source_url(self, dataset, source_key):
        return f"https://fake.invalid/{source_key}"

    def fetch(self, dataset, source_key):
        self.calls += 1
        r = self.responses.get(source_key)
        return None if r is None else FetchedContent(r, 200, self.source_url(dataset, source_key), {"Last-Modified": "x"}, "application/zip")

    def validate(self, dataset, content):
        return self.registry.inspect("nse_archives", "fo_bhavcopy_legacy", content)


GOOD = zipped(LEGACY_HDR, rows=2)


class TestIngest:
    @pytest.fixture
    def store(self, tmp_path):
        return RawStore(tmp_path / "raw")

    def test_the_fake_satisfies_the_protocol(self):
        assert isinstance(FakeAdapter({}), ProviderAdapter)
        assert isinstance(nfa.NseFoArchiveAdapter(), ProviderAdapter)

    def test_created_then_skipped_with_no_second_request(self, store):
        a = FakeAdapter({"k.zip": GOOD})
        r1 = ingest(store, a, "fo_bhavcopy_legacy", "k.zip", now=T0)
        assert r1.status is IngestStatus.CREATED and r1.version == 1 and r1.sha256 == hashlib.sha256(GOOD).hexdigest()
        r2 = ingest(store, a, "fo_bhavcopy_legacy", "k.zip", now=T0)
        assert r2.status is IngestStatus.SKIPPED_EXISTING and a.calls == 1 and r2.sha256 == r1.sha256

    def test_provenance_is_recorded_on_the_artifact(self, store):
        a = FakeAdapter({"k.zip": GOOD})
        ingest(store, a, "fo_bhavcopy_legacy", "k.zip", now=T0)
        m = store.latest("fake", "fo_bhavcopy_legacy", "k.zip")
        assert (m.schema_id, m.fetcher_version, m.license_status, m.fetched_at) == ("nse.fo.bhavcopy.legacy/1", "fake/1", "REQUIRES_PROVIDER_CONFIRMATION", T0.isoformat())
        assert m.format_version == format_version_of(LEGACY_HDR) and m.response_headers == {"Last-Modified": "x"} and m.source_url == "https://fake.invalid/k.zip"

    def test_refresh_with_identical_bytes_is_unchanged(self, store):
        a = FakeAdapter({"k.zip": GOOD})
        ingest(store, a, "fo_bhavcopy_legacy", "k.zip")
        r = ingest(store, a, "fo_bhavcopy_legacy", "k.zip", refresh=True)
        assert r.status is IngestStatus.UNCHANGED and a.calls == 2 and len(store.versions("fake", "fo_bhavcopy_legacy", "k.zip")) == 1

    def test_refresh_with_different_valid_bytes_is_a_new_version(self, store):
        a = FakeAdapter({"k.zip": GOOD})
        first = ingest(store, a, "fo_bhavcopy_legacy", "k.zip")
        a.responses["k.zip"] = zipped(LEGACY_HDR, rows=5)
        r = ingest(store, a, "fo_bhavcopy_legacy", "k.zip", refresh=True)
        assert r.status is IngestStatus.NEW_VERSION and r.version == 2
        assert store.latest("fake", "fo_bhavcopy_legacy", "k.zip").supersedes_sha256 == first.sha256 and store.read_bytes("fake", "fo_bhavcopy_legacy", "k.zip", 1) == GOOD

    def test_not_available_stores_and_quarantines_nothing(self, store):
        r = ingest(store, FakeAdapter({}), "fo_bhavcopy_legacy", "holiday.zip")
        assert r.status is IngestStatus.NOT_AVAILABLE and store.manifest_entries() == [] and not (store.root / "_quarantine").exists()

    @pytest.mark.parametrize("bad, why", [(b"<html>blocked</html>", "not a valid zip"), (zipped(LEGACY_HDR + ["Surprise"]), "unknown NAMED columns"), (zipped(LEGACY_HDR, rows=0), "no data rows")])
    def test_bad_content_is_quarantined_and_the_key_is_retried_next_time(self, store, bad, why):
        a = FakeAdapter({"k.zip": bad})
        r = ingest(store, a, "fo_bhavcopy_legacy", "k.zip", now=T0)
        assert r.status is IngestStatus.QUARANTINED and why in r.reason
        assert store.exists("fake", "fo_bhavcopy_legacy", "k.zip") is False and list((store.root / "_quarantine").rglob("*"))
        a.responses["k.zip"] = GOOD
        assert ingest(store, a, "fo_bhavcopy_legacy", "k.zip").status is IngestStatus.CREATED and a.calls == 2

    def test_a_tampered_stored_file_makes_a_refresh_refuse(self, store):
        from backend.app.data.raw.artifact import RawIntegrityError
        a = FakeAdapter({"k.zip": GOOD})
        ingest(store, a, "fo_bhavcopy_legacy", "k.zip")
        store.artifact_path("fake", "fo_bhavcopy_legacy", "k.zip").write_bytes(b"tampered")
        with pytest.raises(RawIntegrityError):
            ingest(store, a, "fo_bhavcopy_legacy", "k.zip", refresh=True)

    def test_unknown_dataset(self, store):
        with pytest.raises(ValueError, match="no dataset"):
            ingest(store, FakeAdapter({}), "nope", "k")

    def test_transport_errors_propagate_and_store_nothing(self, store):
        class Boom(FakeAdapter):
            def fetch(self, dataset, source_key):
                raise ConnectionError("network down")
        with pytest.raises(ConnectionError):
            ingest(store, Boom({}), "fo_bhavcopy_legacy", "k.zip")
        assert store.manifest_entries() == []


class FakeResponse:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code, self.content = status, content
        self.headers = headers or {"Last-Modified": "Fri, 05 Jul 2024 13:38:00 GMT", "Content-Type": "application/zip", "Set-Cookie": "secret=1"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, response):
        self.response, self.requests = response, []

    def get(self, url, headers=None, timeout=None):
        self.requests.append((url, headers, timeout))
        return self.response


class TestNseAdapter:
    @pytest.mark.parametrize("dataset, day, key, url_tail", [
        ("fo_bhavcopy_legacy", D(2015, 1, 5), "fo05JAN2015bhav.csv.zip", "/historical/DERIVATIVES/2015/JAN/fo05JAN2015bhav.csv.zip"),
        ("fo_bhavcopy_udiff", D(2024, 7, 8), "BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip", "/fo/BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip"),
        ("fo_contract_master", D(2024, 2, 5), "NSE_FO_contract_05022024.csv.gz", "/fo/NSE_FO_contract_05022024.csv.gz"),
    ])
    def test_keys_and_urls(self, dataset, day, key, url_tail):
        assert nfa.source_key_for(dataset, day) == key
        assert nfa.NseFoArchiveAdapter().source_url(dataset, key) == nfa.BASE + url_tail

    @pytest.mark.parametrize("dataset, key", [("fo_bhavcopy_legacy", "../fo05JAN2015bhav.csv.zip"), ("fo_bhavcopy_legacy", "fo05JAN2015.csv"), ("fo_bhavcopy_udiff", "x.zip"),
                                              ("fo_contract_master", "NSE_FO_contract_5022024.csv.gz"), ("nope", "k")])
    def test_malformed_keys_never_reach_a_url(self, dataset, key):
        with pytest.raises(ValueError):
            nfa.NseFoArchiveAdapter().source_url(dataset, key)

    def test_fetch_uses_the_browser_headers_and_keeps_only_whitelisted_response_headers(self):
        s = FakeSession(FakeResponse(200, GOOD))
        got = nfa.NseFoArchiveAdapter(session=s).fetch("fo_bhavcopy_legacy", "fo05JAN2015bhav.csv.zip")
        url, headers, timeout = s.requests[0]
        assert got.content == GOOD and got.http_status == 200 and "Mozilla" in headers["User-Agent"] and headers["Referer"].startswith("https://www.nseindia.com") and timeout == 40
        assert got.headers["Last-Modified"] and "Set-Cookie" not in got.headers and got.media_type == "application/zip"

    def test_404_is_none_and_other_errors_raise(self):
        assert nfa.NseFoArchiveAdapter(session=FakeSession(FakeResponse(404))).fetch("fo_bhavcopy_legacy", "fo05JAN2015bhav.csv.zip") is None
        with pytest.raises(RuntimeError):
            nfa.NseFoArchiveAdapter(session=FakeSession(FakeResponse(503))).fetch("fo_bhavcopy_legacy", "fo05JAN2015bhav.csv.zip")

    def test_end_to_end_with_a_fake_session(self, tmp_path):
        store = RawStore(tmp_path / "raw")
        a = nfa.NseFoArchiveAdapter(session=FakeSession(FakeResponse(200, zipped(UDIFF_HDR_B, rows=2))))
        r = ingest(store, a, "fo_bhavcopy_udiff", nfa.source_key_for("fo_bhavcopy_udiff", D(2024, 7, 8)), now=T0)
        m = store.latest("nse_archives", "fo_bhavcopy_udiff", "BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip")
        assert r.status is IngestStatus.CREATED and m.license_status == "UNCLEAR" and m.schema_id == "nse.fo.bhavcopy.udiff/1"

    def test_html_error_page_served_with_200_is_quarantined(self, tmp_path):
        store = RawStore(tmp_path / "raw")
        a = nfa.NseFoArchiveAdapter(session=FakeSession(FakeResponse(200, b"<html>Access Denied</html>", {"Content-Type": "text/html"})))
        assert ingest(store, a, "fo_bhavcopy_legacy", "fo05JAN2015bhav.csv.zip").status is IngestStatus.QUARANTINED


ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "raw" / "nse_fo_sample"
_FAMILY = {"legacy": "fo_bhavcopy_legacy", "udiff": "fo_bhavcopy_udiff", "master": "fo_contract_master"}
sample_files = sorted(p for p in SAMPLE.glob("*/*") if p.is_file() and not p.name.endswith(".meta.json") and p.parent.name in _FAMILY) if SAMPLE.exists() else []
needs_sample = pytest.mark.skipif(len(sample_files) < 32, reason="O1.0B raw sample not present (data/raw/nse_fo_sample)")


class ReplayAdapter:
    provider_id = "nse_archives"
    fetcher_version = "replay/1"
    license_status = "UNCLEAR"

    def __init__(self):
        self.real = nfa.NseFoArchiveAdapter()

    def datasets(self):
        return self.real.datasets()

    def source_url(self, dataset, key):
        return self.real.source_url(dataset, key)

    def fetch(self, dataset, key):
        family = next(f for f, d in _FAMILY.items() if d == dataset)
        p = SAMPLE / family / key
        side = json.loads(p.with_name(p.name + ".meta.json").read_text())
        return FetchedContent(p.read_bytes(), side["http_status"], side["source_url"], side["response_headers"], "application/octet-stream")

    def validate(self, dataset, content):
        return self.real.validate(dataset, content)


@needs_sample
class TestRealSample:
    def test_every_real_file_matches_a_registered_schema_and_the_format_versions_are_the_ones_o1_0b_found(self):
        reg = default_registry()
        seen = {}
        for p in sample_files:
            dataset = _FAMILY[p.parent.name]
            info = reg.inspect("nse_archives", dataset, p.read_bytes())
            seen.setdefault(dataset, {}).setdefault((info.schema_id, info.format_version), []).append(p.name)
        assert {k: len(v) for k, v in seen.items()} == {"fo_bhavcopy_legacy": 1, "fo_bhavcopy_udiff": 2, "fo_contract_master": 1}
        assert {sid for v in seen.values() for sid, _ in v} == {"nse.fo.bhavcopy.legacy/1", "nse.fo.bhavcopy.udiff/1", "nse.fo.contract_master/1"}
        by_fv = {fv: names for (_, fv), names in seen["fo_bhavcopy_udiff"].items()}
        fv_of = {n: fv for fv, names in by_fv.items() for n in names}
        early = {fv_of[f"BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"] for d in ("20240102", "20240205", "20240328")}
        late = {fv_of[f"BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"] for d in ("20240704", "20240705", "20250115", "20260918")}
        assert len(early) == 1 and len(late) == 1 and early != late

    def test_replaying_all_32_files_into_a_fresh_store_is_lossless_idempotent_and_verifiable(self, tmp_path):
        store = RawStore(tmp_path / "raw")
        a = ReplayAdapter()
        for p in sample_files:
            r = ingest(store, a, _FAMILY[p.parent.name], p.name, now=T0)
            assert r.status is IngestStatus.CREATED, (p.name, r)
            side = json.loads(p.with_name(p.name + ".meta.json").read_text())
            assert r.sha256 == side["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
            m = store.latest("nse_archives", _FAMILY[p.parent.name], p.name)
            assert m.response_headers.get("Last-Modified") == side["response_headers"].get("Last-Modified") and m.size == side["content_length"]
        assert len(store.manifest_entries()) == len(sample_files) >= 32 and store.verify_all() == []
        d1 = store.write_manifest()[1]
        for p in sample_files:
            assert ingest(store, a, _FAMILY[p.parent.name], p.name).status is IngestStatus.SKIPPED_EXISTING
        assert store.write_manifest()[1] == d1 and store.verify_all() == []

    def test_real_master_has_no_named_columns_outside_the_registered_schema(self):
        h = next(p for p in sample_files if p.parent.name == "master")
        text = gzip.decompress(h.read_bytes()).decode()
        header = next(csv.reader(io.StringIO(text)))
        assert NSE_FO_CONTRACT_MASTER.named_of(tuple(c.strip() for c in header)) == NSE_FO_CONTRACT_MASTER.named_columns
        assert [c for c in header if c.startswith("Rsvd")] == [f"Rsvd{i:02d}" for i in range(1, 16)]
