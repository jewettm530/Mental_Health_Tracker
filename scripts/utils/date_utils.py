"""Date and timestamp helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def to_datetime(series: pd.Series) -> pd.Series:
    """Convert a Series to timezone-naive pandas datetimes when possible."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return dt.dt.tz_convert(None)
    except TypeError:
        return dt


def parse_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    if column in df.columns:
        df[column] = to_datetime(df[column])
    return df


def add_date_column(df: pd.DataFrame, datetime_col: str, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if datetime_col in df.columns:
        df[datetime_col] = to_datetime(df[datetime_col])
        df[date_col] = df[datetime_col].dt.date.astype("string")
    return df


def standardize_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    df = df.copy()
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce").dt.date.astype("string")
    return df


def find_first_datetime_column(df: pd.DataFrame, candidates: Iterable[str] | None = None) -> str | None:
    if candidates is None:
        candidates = ["date", "startDate", "endDate", "createdAt", "updatedAt", "timestamp", "time"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def make_daily_index(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame({"date": dates.date.astype("string")})
