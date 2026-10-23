from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


def read_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(
                f"{path}: expected exactly 1 CSV in archive, found {csv_names}"
            )
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(f)


def clean_float(v: object) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return float(v)


def clean_int(v: object) -> int | None:
    f = clean_float(v)
    return int(f) if f is not None else None


def clean_isin(v: object) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s and s.upper() != "NAN" else None


def jsonable(v: object) -> object:
    if isinstance(v, float) and pd.isna(v):
        return None
    return v
