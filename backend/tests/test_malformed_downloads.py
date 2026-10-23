from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from backend.app.data.normalizers._io import read_csv_from_zip
from backend.app.data.normalizers.index_history import normalize_index_history


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


class TestCorruptedZip:
    def test_completely_non_zip_content_raises_badzipfile(self, tmp_path):
        bad_file = tmp_path / "corrupted.zip"
        bad_file.write_bytes(b"this is not a zip file at all, just text")

        with pytest.raises(zipfile.BadZipFile):
            read_csv_from_zip(bad_file)

    def test_empty_file_raises_badzipfile(self, tmp_path):
        empty_file = tmp_path / "empty.zip"
        empty_file.write_bytes(b"")

        with pytest.raises(zipfile.BadZipFile):
            read_csv_from_zip(empty_file)

    def test_truncated_zip_raises(self, tmp_path):
        real_zip = tmp_path / "real.zip"
        _write_zip(real_zip, {"data.csv": b"SYMBOL,SERIES\nRELIANCE,EQ\n" * 100})
        full_bytes = real_zip.read_bytes()

        truncated = tmp_path / "truncated.zip"
        truncated.write_bytes(full_bytes[: len(full_bytes) // 2])

        with pytest.raises(zipfile.BadZipFile):
            read_csv_from_zip(truncated)

    def test_zip_with_no_csv_raises_value_error(self, tmp_path):
        no_csv = tmp_path / "no_csv.zip"
        _write_zip(no_csv, {"readme.txt": b"not a csv"})

        with pytest.raises(ValueError, match="expected exactly 1 CSV"):
            read_csv_from_zip(no_csv)

    def test_zip_with_two_csvs_raises_value_error(self, tmp_path):
        two_csv = tmp_path / "two_csv.zip"
        _write_zip(two_csv, {"a.csv": b"x", "b.csv": b"y"})

        with pytest.raises(ValueError, match="expected exactly 1 CSV"):
            read_csv_from_zip(two_csv)

    def test_zip_with_empty_csv_raises(self, tmp_path):
        empty_csv_zip = tmp_path / "empty_csv.zip"
        _write_zip(empty_csv_zip, {"data.csv": b""})

        with pytest.raises(Exception):  # noqa: B017 - documenting actual pandas behavior
            read_csv_from_zip(empty_csv_zip)


class TestMalformedIndexJson:
    def test_invalid_json_syntax_raises(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json,,,")

        with pytest.raises(json.JSONDecodeError):
            normalize_index_history(bad_json)

    def test_empty_file_raises_json_error(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("")

        with pytest.raises(json.JSONDecodeError):
            normalize_index_history(empty)

    def test_valid_json_missing_data_key_returns_no_bars(self, tmp_path):
        no_data_key = tmp_path / "no_data.json"
        no_data_key.write_text(json.dumps({"unexpected": "shape"}))

        bars, rejected = normalize_index_history(no_data_key)
        assert bars == []
        assert rejected == []

    def test_valid_json_empty_data_array_returns_no_bars(self, tmp_path):
        empty_data = tmp_path / "empty_data.json"
        empty_data.write_text(json.dumps({"data": []}))

        bars, rejected = normalize_index_history(empty_data)
        assert bars == []
        assert rejected == []

    def test_data_key_with_wrong_type_raises(self, tmp_path):
        wrong_type = tmp_path / "wrong_type.json"
        wrong_type.write_text(json.dumps({"data": "Internal Server Error"}))

        with pytest.raises(Exception):  # noqa: B017 - documenting actual behavior
            normalize_index_history(wrong_type)
