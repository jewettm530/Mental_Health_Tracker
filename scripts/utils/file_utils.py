"""Reusable file helpers."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_folder(path: Path | str) -> None:
    """Remove all files/subfolders inside a folder, but keep the folder itself."""
    path = ensure_dir(path)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def save_csv(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def read_csv_if_exists(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def save_json(data: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def list_zip_files(zip_path: Path | str) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as z:
        return z.namelist()


def require_file(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def first_existing(paths: Iterable[Path | str]) -> Path | None:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None
