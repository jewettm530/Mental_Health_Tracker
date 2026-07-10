"""Apple Health import and processing helpers."""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from utils.date_utils import add_date_column, to_datetime
from utils.file_utils import clear_folder, ensure_dir, save_csv, save_json
from utils.paths import APPLE_CLEAN_DIR, APPLE_HEALTH_RAW, APPLE_IMPORTED_DIR

# Keep the useful metrics for mental health correlation work.
KEEP_TYPES = {
    "HKQuantityTypeIdentifierHeartRate",
    "HKQuantityTypeIdentifierRestingHeartRate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "HKQuantityTypeIdentifierStepCount",
    "HKQuantityTypeIdentifierAppleExerciseTime",
    "HKQuantityTypeIdentifierAppleStandTime",
    "HKQuantityTypeIdentifierDistanceWalkingRunning",
    "HKQuantityTypeIdentifierFlightsClimbed",
    "HKQuantityTypeIdentifierActiveEnergyBurned",
    "HKQuantityTypeIdentifierBasalEnergyBurned",
    "HKCategoryTypeIdentifierSleepAnalysis",
    "HKQuantityTypeIdentifierBodyMass",
    "HKQuantityTypeIdentifierRespiratoryRate",
    "HKQuantityTypeIdentifierOxygenSaturation",
}

TYPE_TO_SHORT_NAME = {
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_heart_rate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_sdnn",
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierAppleExerciseTime": "exercise_minutes",
    "HKQuantityTypeIdentifierAppleStandTime": "stand_minutes",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "walking_running_distance",
    "HKQuantityTypeIdentifierFlightsClimbed": "flights_climbed",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
    "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_energy",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep_analysis",
    "HKQuantityTypeIdentifierBodyMass": "body_mass",
    "HKQuantityTypeIdentifierRespiratoryRate": "respiratory_rate",
    "HKQuantityTypeIdentifierOxygenSaturation": "oxygen_saturation",
}

SUM_TYPES = {"steps", "exercise_minutes", "stand_minutes", "walking_running_distance", "flights_climbed", "active_energy", "basal_energy"}
MEAN_TYPES = {"heart_rate", "resting_heart_rate", "hrv_sdnn", "body_mass", "respiratory_rate", "oxygen_saturation"}


def find_export_xml(zip_path: Path = APPLE_HEALTH_RAW) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        matches = [name for name in z.namelist() if name.endswith("export.xml")]
    if not matches:
        raise FileNotFoundError("Could not find export.xml inside Apple Health zip.")
    return matches[0]


def _iter_xml_elements(zip_path: Path, tags: Iterable[str]):
    tags = set(tags)
    xml_name = find_export_xml(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(xml_name) as xml_file:
            for _, elem in ET.iterparse(xml_file, events=("end",)):
                if elem.tag in tags:
                    yield elem.tag, dict(elem.attrib)
                elem.clear()


def import_apple_health_zip(zip_path: Path = APPLE_HEALTH_RAW, output_dir: Path = APPLE_IMPORTED_DIR) -> None:
    """Stream selected Apple Health data from health.zip into imported CSVs."""
    ensure_dir(output_dir)
    clear_folder(output_dir)

    records: list[dict[str, Any]] = []
    workouts: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for tag, attrs in _iter_xml_elements(zip_path, tags=["Record", "Workout", "ActivitySummary"]):
        if tag == "Record":
            record_type = attrs.get("type")
            if record_type in KEEP_TYPES:
                attrs["type_short"] = TYPE_TO_SHORT_NAME.get(record_type, record_type)
                records.append(attrs)
        elif tag == "Workout":
            workouts.append(attrs)
        elif tag == "ActivitySummary":
            summaries.append(attrs)

    records_df = pd.DataFrame(records)
    workouts_df = pd.DataFrame(workouts)
    summaries_df = pd.DataFrame(summaries)

    save_csv(records_df, output_dir / "apple_health_records_selected.csv")
    save_csv(workouts_df, output_dir / "apple_health_workouts.csv")
    save_csv(summaries_df, output_dir / "apple_health_activity_summary.csv")

    save_json(
        {
            "records_selected_rows": len(records_df),
            "workout_rows": len(workouts_df),
            "activity_summary_rows": len(summaries_df),
            "selected_types": sorted(KEEP_TYPES),
        },
        output_dir / "apple_health_import_inventory.json",
    )


def _load_records(input_dir: Path = APPLE_IMPORTED_DIR) -> pd.DataFrame:
    path = input_dir / "apple_health_records_selected.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    # Preserve the original Apple Health value string for categorical records like sleep.
    df["value_raw"] = df.get("value")
    df["value_numeric"] = pd.to_numeric(df.get("value"), errors="coerce")
    for col in ["startDate", "endDate", "creationDate"]:
        if col in df.columns:
            df[col] = to_datetime(df[col])
    df = add_date_column(df, "startDate")
    return df


def build_daily_activity(records_df: pd.DataFrame, output_dir: Path = APPLE_CLEAN_DIR) -> pd.DataFrame:
    if records_df.empty:
        out = pd.DataFrame(columns=["date"])
    else:
        df = records_df[records_df["type_short"].isin(SUM_TYPES)].copy()
        out = df.pivot_table(index="date", columns="type_short", values="value_numeric", aggfunc="sum").reset_index()
    save_csv(out, output_dir / "apple_daily_activity.csv")
    return out


def build_daily_heart(records_df: pd.DataFrame, output_dir: Path = APPLE_CLEAN_DIR) -> pd.DataFrame:
    heart_types = {"heart_rate", "resting_heart_rate", "hrv_sdnn"}
    if records_df.empty:
        out = pd.DataFrame(columns=["date"])
    else:
        df = records_df[records_df["type_short"].isin(heart_types)].copy()
        out = df.pivot_table(index="date", columns="type_short", values="value_numeric", aggfunc="mean").reset_index()
    save_csv(out, output_dir / "apple_daily_heart.csv")
    return out


def build_daily_body(records_df: pd.DataFrame, output_dir: Path = APPLE_CLEAN_DIR) -> pd.DataFrame:
    body_types = {"body_mass"}
    if records_df.empty:
        out = pd.DataFrame(columns=["date"])
    else:
        df = records_df[records_df["type_short"].isin(body_types)].copy()
        out = df.pivot_table(index="date", columns="type_short", values="value_numeric", aggfunc="mean").reset_index()
    save_csv(out, output_dir / "apple_daily_body.csv")
    return out


def build_daily_respiratory(records_df: pd.DataFrame, output_dir: Path = APPLE_CLEAN_DIR) -> pd.DataFrame:
    resp_types = {"respiratory_rate", "oxygen_saturation"}
    if records_df.empty:
        out = pd.DataFrame(columns=["date"])
    else:
        df = records_df[records_df["type_short"].isin(resp_types)].copy()
        out = df.pivot_table(index="date", columns="type_short", values="value_numeric", aggfunc="mean").reset_index()
    save_csv(out, output_dir / "apple_daily_respiratory.csv")
    return out


def build_daily_sleep(records_df: pd.DataFrame, output_dir: Path = APPLE_CLEAN_DIR) -> pd.DataFrame:
    """Build daily sleep totals.

    The main downstream variable is sleep_hours_asleep. Apple Health may export
    multiple sleep stages (core, deep, REM, awake, in bed). For mental-health
    analysis, the default should be total hours asleep rather than separate stages.

    Awake/in-bed time is preserved when available, but analysis/plots use
    sleep_hours_asleep.
    """
    empty_cols = ["date", "sleep_hours_asleep", "sleep_hours_in_bed", "sleep_hours_awake"]
    if records_df.empty:
        out = pd.DataFrame(columns=empty_cols)
        save_csv(out, output_dir / "apple_daily_sleep.csv")
        return out

    df = records_df[records_df["type_short"] == "sleep_analysis"].copy()
    if df.empty or "startDate" not in df.columns or "endDate" not in df.columns:
        out = pd.DataFrame(columns=empty_cols)
        save_csv(out, output_dir / "apple_daily_sleep.csv")
        return out

    df["duration_hours"] = (df["endDate"] - df["startDate"]).dt.total_seconds() / 3600
    df = df[df["duration_hours"].notna() & (df["duration_hours"] > 0)].copy()
    if df.empty:
        out = pd.DataFrame(columns=empty_cols)
        save_csv(out, output_dir / "apple_daily_sleep.csv")
        return out

    df["sleep_value"] = df.get("value_raw", "unknown").astype("string").str.lower()

    def classify_sleep(value: str) -> str:
        value = str(value).lower()
        if "awake" in value:
            return "awake"
        if "inbed" in value or "in_bed" in value or "in bed" in value:
            return "in_bed"
        if any(term in value for term in ["asleep", "core", "deep", "rem"]):
            return "asleep"
        return "other"

    df["sleep_bucket"] = df["sleep_value"].map(classify_sleep)
    pivot = df.pivot_table(index="date", columns="sleep_bucket", values="duration_hours", aggfunc="sum").reset_index()

    out = pd.DataFrame({"date": pivot["date"]})
    out["sleep_hours_asleep"] = pivot.get("asleep", pd.Series([pd.NA] * len(pivot)))
    out["sleep_hours_in_bed"] = pivot.get("in_bed", pd.Series([pd.NA] * len(pivot)))
    out["sleep_hours_awake"] = pivot.get("awake", pd.Series([pd.NA] * len(pivot)))

    # Keep total tracked only as a sanity check, not as the main sleep variable.
    numeric_cols = [c for c in pivot.columns if c != "date"]
    out["sleep_hours_total_tracked"] = pivot[numeric_cols].sum(axis=1, min_count=1) if numeric_cols else pd.NA

    save_csv(out, output_dir / "apple_daily_sleep.csv")
    return out


def process_workouts(input_dir: Path = APPLE_IMPORTED_DIR, output_dir: Path = APPLE_CLEAN_DIR) -> pd.DataFrame:
    path = input_dir / "apple_health_workouts.csv"
    if not path.exists():
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "apple_daily_workouts.csv")
        return out

    df = pd.read_csv(path, low_memory=False)
    if df.empty or "startDate" not in df.columns:
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "apple_daily_workouts.csv")
        return out

    df["startDate"] = to_datetime(df["startDate"])
    df["date"] = df["startDate"].dt.date.astype("string")
    for col in ["duration", "totalDistance", "totalEnergyBurned"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_map = {"workout_count": ("workoutActivityType", "count")}
    if "duration" in df.columns:
        agg_map["workout_duration_total"] = ("duration", "sum")
    if "totalDistance" in df.columns:
        agg_map["workout_distance_total"] = ("totalDistance", "sum")
    if "totalEnergyBurned" in df.columns:
        agg_map["workout_energy_total"] = ("totalEnergyBurned", "sum")

    out = df.groupby("date").agg(**agg_map).reset_index()
    save_csv(out, output_dir / "apple_daily_workouts.csv")
    return out


def process_activity_summary(input_dir: Path = APPLE_IMPORTED_DIR, output_dir: Path = APPLE_CLEAN_DIR) -> pd.DataFrame:
    path = input_dir / "apple_health_activity_summary.csv"
    if not path.exists():
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "apple_activity_summary.csv")
        return out

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        out = pd.DataFrame(columns=["date"])
    else:
        if "dateComponents" in df.columns:
            df = df.rename(columns={"dateComponents": "date"})
        for col in df.columns:
            if col != "date":
                
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        out = df
    save_csv(out, output_dir / "apple_activity_summary.csv")
    return out


def process_apple_health(input_dir: Path = APPLE_IMPORTED_DIR, output_dir: Path = APPLE_CLEAN_DIR) -> None:
    """Build separated clean Apple Health daily datasets."""
    ensure_dir(output_dir)
    records = _load_records(input_dir)
    build_daily_sleep(records, output_dir)
    build_daily_activity(records, output_dir)
    build_daily_heart(records, output_dir)
    build_daily_body(records, output_dir)
    build_daily_respiratory(records, output_dir)
    process_workouts(input_dir, output_dir)
    process_activity_summary(input_dir, output_dir)
