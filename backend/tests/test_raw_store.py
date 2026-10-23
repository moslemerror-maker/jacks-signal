from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest

from backend.app.data.raw.artifact import RawArtifactMeta, RawIntegrityError, check_component
from backend.app.data.raw.store import PutStatus, RawStore, sha256_bytes

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
HEADERS = {"Last-Modified": "Fri, 05 Jul 2024 13:38:00 GMT", "ETag": '"abc"'}


def put(store, content=b"alpha", key="fo05JUL2024bhav.csv.zip", **kw):
    args = dict(provider="prov", dataset="ds", source_key=key, content=content, source_url="https://example.invalid/" + key, http_status=200,
                response_headers=HEADERS, fetcher_version="t/1", fetched_at=T0)
    args.update(kw)
    return store.put(**args)


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "raw")


class TestKeySafety:
    @pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "", ".hidden", "a..b", "C:x", "x y", "a" * 129, "-lead", "x\x00y"])
    def test_unsafe_components_are_rejected(self, bad):
        with pytest.raises(ValueError):
            check_component("k", bad)

    @pytest.mark.parametrize("ok", ["fo05JUL2024bhav.csv.zip", "NSE_FO_contract_18092026.csv.gz", "a", "k=v", "A-b_c.1"])
    def test_safe_components(self, ok):
        assert check_component("k", ok) == ok

    def test_store_never_builds_a_path_outside_root(self, store):
        with pytest.raises(ValueError):
            store.artifact_path("prov", "ds", "../../etc/passwd")
        with pytest.raises(ValueError):
            store.artifact_path("..", "ds", "k")


class TestPut:
    def test_first_put_creates_bytes_and_sidecar(self, store):
        r = put(store)
        assert r.status is PutStatus.CREATED and r.meta.version == 1 and r.meta.supersedes_sha256 is None
        assert r.path.read_bytes() == b"alpha" and r.meta.sha256 == sha256_bytes(b"alpha") and r.meta.size == 5
        side = json.loads(store.sidecar_path(r.path).read_text())
        assert side["fetched_at"] == T0.isoformat() and side["response_headers"]["Last-Modified"] == HEADERS["Last-Modified"] and side["license_status"] == "UNCLEAR"

    def test_identical_bytes_are_a_no_op(self, store):
        first = put(store)
        mtime = first.path.stat().st_mtime_ns
        again = put(store, fetched_at=T0 + dt.timedelta(days=1))
        assert again.status is PutStatus.UNCHANGED and again.meta == first.meta and first.path.stat().st_mtime_ns == mtime
        assert len(store.versions("prov", "ds", "fo05JUL2024bhav.csv.zip")) == 1

    def test_different_bytes_become_a_new_version_and_the_old_one_is_untouched(self, store):
        v1 = put(store, b"alpha")
        v2 = put(store, b"BETA-corrected", fetched_at=T0 + dt.timedelta(days=3))
        assert v2.status is PutStatus.NEW_VERSION and v2.meta.version == 2 and v2.meta.supersedes_sha256 == v1.meta.sha256
        assert v2.path.name.endswith(".v2") and v1.path.read_bytes() == b"alpha" and v2.path.read_bytes() == b"BETA-corrected"
        assert store.read_bytes("prov", "ds", "fo05JUL2024bhav.csv.zip") == b"BETA-corrected"
        assert store.read_bytes("prov", "ds", "fo05JUL2024bhav.csv.zip", version=1) == b"alpha"

    def test_reverting_to_older_bytes_is_still_a_new_version(self, store):
        put(store, b"a")
        put(store, b"b")
        r = put(store, b"a")
        assert r.status is PutStatus.NEW_VERSION and r.meta.version == 3

    def test_a_lost_race_on_the_data_file_fails_instead_of_overwriting(self, store, monkeypatch):
        r = put(store, b"alpha")
        monkeypatch.setattr(store, "versions", lambda *a, **k: [])
        with pytest.raises(FileExistsError):
            put(store, b"a different response")
        assert r.path.read_bytes() == b"alpha"

    def test_a_lost_race_on_the_sidecar_never_rewrites_it(self, store, monkeypatch):
        r = put(store, b"alpha")
        original = store.sidecar_path(r.path).read_text()
        r.path.unlink()
        monkeypatch.setattr(store, "versions", lambda *a, **k: [])
        with pytest.raises(FileExistsError):
            put(store, b"a different response")
        assert store.sidecar_path(r.path).read_text() == original

    def test_a_tampered_file_blocks_every_later_put_and_is_not_overwritten(self, store):
        r = put(store)
        r.path.write_bytes(b"tampered")
        with pytest.raises(RawIntegrityError, match="immutable"):
            put(store, b"alpha")
        with pytest.raises(RawIntegrityError):
            put(store, b"something else")
        assert r.path.read_bytes() == b"tampered"

    def test_a_file_without_a_sidecar_is_refused_not_adopted(self, store):
        r = put(store)
        store.sidecar_path(r.path).unlink()
        with pytest.raises(RawIntegrityError, match="without a provenance sidecar"):
            put(store)
        assert r.path.read_bytes() == b"alpha"

    def test_a_sidecar_without_its_file_is_refused(self, store):
        r = put(store)
        r.path.unlink()
        with pytest.raises(RawIntegrityError, match="without its data file"):
            put(store)

    def test_naive_timestamps_are_normalised_to_utc(self, store):
        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        r = put(store, fetched_at=dt.datetime(2026, 9, 21, 17, 30, tzinfo=ist))
        assert r.meta.fetched_at == dt.datetime(2026, 9, 21, 12, 0, tzinfo=UTC).isoformat()

    def test_schema_and_license_are_recorded(self, store):
        r = put(store, schema_id="s/1", format_version="abcd", license_status="REQUIRES_PROVIDER_CONFIRMATION", media_type="application/zip")
        assert (r.meta.schema_id, r.meta.format_version, r.meta.license_status, r.meta.media_type) == ("s/1", "abcd", "REQUIRES_PROVIDER_CONFIRMATION", "application/zip")


class TestReads:
    def test_read_bytes_rehashes_every_time(self, store):
        r = put(store)
        r.path.write_bytes(b"corrupt")
        with pytest.raises(RawIntegrityError):
            store.read_bytes("prov", "ds", "fo05JUL2024bhav.csv.zip")

    def test_missing_key(self, store):
        assert store.latest("prov", "ds", "nope") is None and store.exists("prov", "ds", "nope") is False
        with pytest.raises(FileNotFoundError):
            store.read_bytes("prov", "ds", "nope")

    def test_a_sidecar_that_describes_a_different_key_is_rejected(self, store):
        r = put(store)
        side = store.sidecar_path(r.path)
        d = json.loads(side.read_text())
        d["source_key"] = "other"
        side.write_text(json.dumps(d))
        with pytest.raises(RawIntegrityError, match="not the path it sits at"):
            store.versions("prov", "ds", "fo05JUL2024bhav.csv.zip")


class TestVerifyAll:
    def test_clean_store(self, store):
        put(store, b"a", key="k1")
        put(store, b"b", key="k1")
        put(store, b"c", key="k2")
        assert store.verify_all() == []

    def test_empty_or_missing_root(self, tmp_path):
        assert RawStore(tmp_path / "nowhere").verify_all() == []

    def test_each_problem_kind_is_reported(self, store):
        a = put(store, b"aaaa", key="hash")
        b = put(store, b"bbbb", key="size")
        c = put(store, b"cccc", key="orphan_side")
        d = put(store, b"dddd", key="orphan_file")
        a.path.write_bytes(b"AAAA")
        b.path.write_bytes(b"bb")
        c.path.unlink()
        store.sidecar_path(d.path).unlink()
        (store.root / "prov" / "ds" / "junk.meta.json").write_text("{not json")
        kinds = {(p.path.split("/")[-1], p.kind) for p in store.verify_all()}
        assert ("hash", "HASH_MISMATCH") in kinds and ("size", "SIZE_MISMATCH") in kinds and ("size", "HASH_MISMATCH") in kinds
        assert ("orphan_side.meta.json", "ORPHAN_SIDECAR") in kinds and ("orphan_file", "ORPHAN_FILE") in kinds and ("junk.meta.json", "ORPHAN_SIDECAR") in kinds

    def test_bad_sidecar_content(self, store):
        r = put(store)
        store.sidecar_path(r.path).write_text(json.dumps({"provider": "prov"}))
        assert any(p.kind == "BAD_SIDECAR" for p in store.verify_all())

    def test_version_gap_is_reported(self, store):
        put(store, b"a")
        v2 = put(store, b"b")
        v3 = put(store, b"c")
        v2.path.unlink()
        store.sidecar_path(v2.path).unlink()
        assert {p.kind for p in store.verify_all()} == {"VERSION_GAP"}
        assert v3.path.exists()

    def test_a_sidecar_moved_to_the_wrong_place(self, store):
        r = put(store, key="k1")
        wrong = r.path.parent / "k2"
        wrong.write_bytes(r.path.read_bytes())
        store.sidecar_path(wrong).write_text(store.sidecar_path(r.path).read_text())
        assert any(p.kind == "PATH_MISMATCH" for p in store.verify_all())

    def test_quarantine_is_outside_the_verified_tree(self, store):
        put(store)
        store.quarantine(provider="prov", dataset="ds", source_key="k", content=b"<html>", source_url="u", http_status=200, reason="html")
        assert store.verify_all() == []


class TestQuarantine:
    def test_bad_bytes_are_kept_outside_the_normal_tree_and_the_key_stays_unfetched(self, store):
        p = store.quarantine(provider="prov", dataset="ds", source_key="k.zip", content=b"<html>oops", source_url="u", http_status=200, reason="not a zip",
                             response_headers=HEADERS, quarantined_at=T0)
        assert "_quarantine" in p.parts and p.read_bytes() == b"<html>oops"
        side = json.loads(store.sidecar_path(p).read_text())
        assert side["reason"] == "not a zip" and side["sha256"] == sha256_bytes(b"<html>oops") and side["quarantined_at"] == T0.isoformat()
        assert store.exists("prov", "ds", "k.zip") is False

    def test_content_addressed_same_bytes_once_different_bytes_both_kept(self, store):
        args = dict(provider="prov", dataset="ds", source_key="k.zip", source_url="u", http_status=200, reason="r")
        a1 = store.quarantine(content=b"one", **args)
        a2 = store.quarantine(content=b"one", **args)
        b = store.quarantine(content=b"two", **args)
        assert a1 == a2 and a1 != b and a1.read_bytes() == b"one" and b.read_bytes() == b"two"
        assert len(list(a1.parent.glob("*.meta.json"))) == 2

    def test_quarantined_bytes_can_later_be_stored_normally(self, store):
        store.quarantine(provider="prov", dataset="ds", source_key="k", content=b"bad", source_url="u", http_status=200, reason="r")
        assert put(store, b"good", key="k").status is PutStatus.CREATED


class TestManifest:
    def test_manifest_is_derived_deterministic_and_ordered(self, store):
        put(store, b"z", key="zz")
        put(store, b"a", key="aa")
        put(store, b"a2", key="aa")
        path, digest = store.write_manifest()
        lines = [json.loads(l) for l in path.read_text().splitlines()]
        assert [(l["source_key"], l["version"]) for l in lines] == [("aa", 1), ("aa", 2), ("zz", 1)]
        assert store.write_manifest()[1] == digest
        assert sha256_bytes(path.read_bytes()) == digest

    def test_manifest_is_rebuilt_from_sidecars_not_trusted(self, store):
        put(store, b"a", key="aa")
        path, d1 = store.write_manifest()
        path.write_text("garbage")
        assert store.write_manifest()[1] == d1
        assert store.verify_all() == []

    def test_empty_store_manifest(self, store):
        path, digest = store.write_manifest()
        assert path.read_bytes() == b"" and digest == sha256_bytes(b"")


class TestMeta:
    def base(self, **over):
        d = dict(provider="p", dataset="d", source_key="k", version=1, source_url="u", fetched_at=T0.isoformat(), http_status=200, sha256="0" * 64, size=0,
                 media_type=None, response_headers={}, fetcher_version="f")
        d.update(over)
        return d

    def test_round_trip(self):
        m = RawArtifactMeta(**self.base())
        assert RawArtifactMeta.from_json(m.to_json()) == m

    @pytest.mark.parametrize("over", [dict(version=0), dict(sha256="xyz"), dict(sha256="A" * 64), dict(size=-1), dict(license_status="PERMITTED"),
                                      dict(version=2), dict(version=1, supersedes_sha256="1" * 64), dict(provider="../x")])
    def test_invalid_records_are_rejected(self, over):
        with pytest.raises(ValueError):
            RawArtifactMeta(**self.base(**over))

    def test_unknown_and_missing_fields_are_rejected_on_read(self):
        d = dataclasses.asdict(RawArtifactMeta(**self.base()))
        with pytest.raises(RawIntegrityError, match="unknown fields"):
            RawArtifactMeta.from_json(json.dumps({**d, "extra": 1}))
        d.pop("sha256")
        with pytest.raises(RawIntegrityError, match="missing"):
            RawArtifactMeta.from_json(json.dumps(d))
