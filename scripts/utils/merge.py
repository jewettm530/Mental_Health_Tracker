"""Merge helpers for daily mental health tracking datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.date_utils import standardize_date
from utils.file_utils import ensure_dir, save_csv, save_json
from utils.paths import APPLE_CLEAN_DIR, MERGED_DIR, STOIC_CLEAN_DIR

DESCRIPTIVE_SUFFIXES = ("_text", "_labels", "_label")
DESCRIPTIVE_COLUMNS = {"symptoms_text", "triggers_text", "automatic_thoughts_text", "recovery_methods_text", "lowest_mood_duration_label", "symptom_labels", "trigger_labels", "recovery_labels"}


def _read_daily_file(path: Path, prefix: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df = standardize_date(df, "date")
    if prefix:
        rename = {c: f"{prefix}_{c}" for c in df.columns if c != "date" and not c.startswith(prefix)}
        df = df.rename(columns=rename)
    return df


def load_clean_daily_datasets() -> dict[str, pd.DataFrame]:
    """Load separated clean daily tables."""
    files = {
        # Do not also merge stoic_daily_wide; it duplicates these same columns.
        "stoic_mental_health": (STOIC_CLEAN_DIR / "stoic_mental_health_daily.csv", None),
        # Do not prefix sleep. The Apple sleep processor already creates a clear
        # canonical variable: sleep_hours_asleep.
        "apple_sleep": (APPLE_CLEAN_DIR / "apple_daily_sleep.csv", None),
        "apple_activity": (APPLE_CLEAN_DIR / "apple_daily_activity.csv", "activity"),
        "apple_heart": (APPLE_CLEAN_DIR / "apple_daily_heart.csv", "heart"),
        "apple_body": (APPLE_CLEAN_DIR / "apple_daily_body.csv", "body"),
        "apple_respiratory": (APPLE_CLEAN_DIR / "apple_daily_respiratory.csv", "resp"),
        "apple_workouts": (APPLE_CLEAN_DIR / "apple_daily_workouts.csv", "workout"),
        "apple_activity_summary": (APPLE_CLEAN_DIR / "apple_activity_summary.csv", "ring"),
    }
    return {name: _read_daily_file(path, prefix) for name, (path, prefix) in files.items()}


def merge_daily_sources(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge all daily datasets on date."""
    master = pd.DataFrame(columns=["date"])
    for _, df in datasets.items():
        if df.empty or "date" not in df.columns:
            continue
        if master.empty or (len(master.columns) == 1 and master["date"].empty):
            master = df.copy()
        else:
            overlap = [c for c in df.columns if c != "date" and c in master.columns]
            if overlap:
                df = df.drop(columns=overlap)
            master = master.merge(df, on="date", how="outer")
    if not master.empty:
        master = standardize_date(master, "date").sort_values("date").reset_index(drop=True)
    return master


def build_correlation_ready_dataset(master: pd.DataFrame) -> pd.DataFrame:
    """Keep date plus genuinely numeric columns only."""
    if master.empty:
        return pd.DataFrame(columns=["date"])

    out = pd.DataFrame({"date": master["date"]})
    for col in master.columns:
        if col == "date" or col in DESCRIPTIVE_COLUMNS or col.endswith(DESCRIPTIVE_SUFFIXES):
            continue
        # Keep total hours asleep as the main sleep variable; skip sleep-stage/time-in-bed
        # columns in correlation_ready to avoid noisy/repetitive analysis.
        lower = col.lower()
        if lower in {"sleep_hours_awake", "sleep_hours_in_bed", "sleep_hours_total_tracked"}:
            continue
        converted = pd.to_numeric(master[col], errors="coerce")
        # Require at least two values and at least two unique values for correlation usefulness.
        if converted.notna().sum() >= 2 and converted.nunique(dropna=True) >= 2:
            out[col] = converted
    return out


def build_master_daily_dataset(output_dir: Path = MERGED_DIR) -> None:
    ensure_dir(output_dir)
    datasets = load_clean_daily_datasets()
    master = merge_daily_sources(datasets)
    corr_ready = build_correlation_ready_dataset(master)

    save_csv(master, output_dir / "master_daily.csv")
    save_csv(corr_ready, output_dir / "correlation_ready_daily.csv")

    inventory = {name: {"rows": len(df), "columns": list(df.columns)} for name, df in datasets.items()}
    inventory["master_daily"] = {"rows": len(master), "columns": list(master.columns)}
    inventory["correlation_ready_daily"] = {"rows": len(corr_ready), "columns": list(corr_ready.columns)}
    save_json(inventory, output_dir / "merge_inventory.json")
