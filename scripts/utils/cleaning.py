"""General dataframe cleaning helpers."""

from __future__ import annotations

import re

import pandas as pd


def clean_column_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


def convert_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    exclude = set(exclude or [])
    for col in df.columns:
        if col in exclude:
            continue
        converted = pd.to_numeric(df[col], errors="ignore")
        df[col] = converted
    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    df = drop_empty_columns(df)
    df = normalize_text_columns(df)
    return df


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str = "date", how: str = "outer") -> pd.DataFrame:
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    return left.merge(right, on=on, how=how)
