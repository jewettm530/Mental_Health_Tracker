"""Stoic import and processing helpers.

This module imports Stoic JSON tables and builds clean daily mental-health
feature tables from the custom evening check-in questions.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from utils.cleaning import basic_clean, clean_column_name
from utils.date_utils import standardize_date
from utils.file_utils import clear_folder, ensure_dir, save_csv, save_json
from utils.paths import STOIC_CLEAN_DIR, STOIC_IMPORTED_DIR, STOIC_RAW

# Stable question UUIDs from the current Stoic export.
LOWEST_MOOD_QID = "945E2B63-3B52-42E3-810A-4519A57F4631"
LOWEST_MOOD_DURATION_QID = "88E82466-17EC-4BE5-A1B8-4CCF0759A4E2"
RELATIONSHIP_SECURITY_QID = "80ED4DE6-79DB-43B6-A722-57CC9F74FD75"
AUTOMATIC_THOUGHTS_QID = "65FD9D3F-F13E-4E54-8456-63744357BA8A"
TRIGGERS_QID = "EBE0ACAD-F23B-4298-8404-481CD76FC6CE"
SYMPTOMS_QID = "2D474CB9-8ABA-410C-B63B-FFF6F412AD21"
RECOVERY_QID = "599599C9-9927-45C1-AC94-14961B64C175"

TRIGGER_CODE_MAP = {
    "1": "Nothing obvious",
    "2": "Loneliness",
    "3": "Relationship uncertainty",
    "4": "Work",
    "5": "School",
    "6": "Family/Friends",
    "7": "Health",
    "8": "Other",
}

SYMPTOM_CODE_MAP = {
    "1": "Lack of interest",
    "2": "Worthlessness",
    "3": "Hopelessness",
    "4": "Low energy",
    "5": "Appetite change",
    "6": "Trouble concentrating",
    "7": "Passive self-harm thoughts",
    "8": "Active self-harm thoughts",
    "9": "None",
}

RECOVERY_CODE_MAP = {
    "1": "Time",
    "2": "Sleep",
    "3": "Talking about it",
    "4": "Just being social",
    "5": "Nature",
    "6": "Exercise",
    "7": "Hobby",
    "8": "Food",
    "9": "Therapy skill",
    "10": "Other",
    "11": "It didn’t",
}

LOWEST_MOOD_DURATION_LABELS = {
    0: "< 30 minutes",
    1: "30 minutes - 2 hours",
    2: "2-6 hours",
    3: "Most of the day",
    4: "Multiple days",
}


def _json_to_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, list):
        return pd.json_normalize(data)
    if isinstance(data, dict):
        for key in ["rows", "data", "items", "entries", "records"]:
            if isinstance(data.get(key), list):
                return pd.json_normalize(data[key])
        return pd.json_normalize(data)
    return pd.DataFrame({"value": [data]})


def import_stoic_zip(zip_path: Path = STOIC_RAW, output_dir: Path = STOIC_IMPORTED_DIR) -> None:
    """Convert all Stoic JSON files inside stoic.zip into imported CSVs."""
    ensure_dir(output_dir)
    clear_folder(output_dir)

    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        json_files = [name for name in z.namelist() if name.lower().endswith(".json")]
        for name in json_files:
            with z.open(name) as f:
                data = json.load(f)
            df = basic_clean(_json_to_dataframe(data))
            safe_name = clean_column_name(Path(name).stem) or "stoic_table"
            out_path = output_dir / f"{safe_name}.csv"
            save_csv(df, out_path)
            inventory.append({"source_file": name, "csv_file": out_path.name, "rows": len(df), "columns": list(df.columns)})

    save_json(inventory, output_dir / "stoic_import_inventory.json")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _to_date_from_ms(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="ms", errors="coerce").dt.date.astype("string")


def _parse_stoic_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce").dt.date.astype("string")


def _extract_int_codes(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return re.findall(r"\b\d+\b", str(value))


def _split_automatic_thoughts(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    raw = str(text).strip()
    if not raw:
        return []
    parts = re.split(r"[\n;]+", raw)
    if len(parts) == 1 and raw.count(",") >= 2:
        parts = raw.split(",")
    cleaned = []
    for part in parts:
        item = re.sub(r"^[-*•\d\.\)\s]+", "", str(part)).strip()
        item = re.sub(r"\s+", " ", item)
        if item:
            cleaned.append(item)
    return cleaned


def _clean_indicator_label(label: str) -> str:
    return clean_column_name(label).replace("family_friends", "family_or_friends").replace("it_didn_t", "it_did_not")


def _add_code_features(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    count_col: str,
    prefix: str,
    code_map: dict[str, str],
    exclude_from_count: set[str] | None = None,
) -> pd.DataFrame:
    """Add labels, counts, and binary indicator columns for coded text answers."""
    exclude_from_count = exclude_from_count or set()
    out = df.copy()
    if text_col not in out.columns:
        return out

    def mapped_labels(value: Any) -> list[str]:
        return [code_map.get(code, code) for code in _extract_int_codes(value)]

    out[label_col] = out[text_col].apply(lambda x: "; ".join(mapped_labels(x)))
    out[count_col] = out[text_col].apply(lambda x: sum(1 for code in _extract_int_codes(x) if code not in exclude_from_count))

    for code, label in code_map.items():
        col = f"{prefix}_{_clean_indicator_label(label)}"
        out[col] = out[text_col].apply(lambda x, code=code: int(code in _extract_int_codes(x)))

    return out


def _load_core_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _read_csv(input_dir / "answers.csv"), _read_csv(input_dir / "questions.csv"), _read_csv(input_dir / "routines.csv")


def build_stoic_answers_long(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    """Build a joined long table with one answer per row and readable question text."""
    ensure_dir(output_dir)
    answers, questions, routines = _load_core_tables(input_dir)

    if answers.empty:
        out = pd.DataFrame(columns=["date", "question_id", "question", "answer", "routine", "context"])
        save_csv(out, output_dir / "stoic_answers_long.csv")
        return out

    answers = answers.copy()
    question_map = {}
    if not questions.empty and {"uuid", "text"}.issubset(questions.columns):
        question_map = dict(zip(questions["uuid"], questions["text"]))

    routine_date_map = {}
    if not routines.empty and {"uuid", "date"}.issubset(routines.columns):
        routine_dates = routines[["uuid", "date"]].copy()
        routine_dates["date"] = _parse_stoic_date(routine_dates["date"])
        routine_date_map = dict(zip(routine_dates["uuid"], routine_dates["date"]))

    answers["date"] = answers.get("routine", pd.Series(index=answers.index, dtype="object")).map(routine_date_map)
    if "timestamp" in answers.columns:
        missing = answers["date"].isna()
        answers.loc[missing, "date"] = _to_date_from_ms(answers.loc[missing, "timestamp"])

    answers["question_id"] = answers.get("question")
    answers["question"] = answers["question_id"].map(question_map).fillna(answers["question_id"])
    answers["answer"] = answers.get("text")

    keep = [c for c in ["date", "question_id", "question", "answer", "routine", "context", "uuid", "timestamp"] if c in answers.columns]
    out = standardize_date(answers[keep].copy(), "date")
    save_csv(out, output_dir / "stoic_answers_long.csv")
    return out


def build_stoic_mental_health_daily(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    """Create one row per date with useful mental-health features."""
    long_df = build_stoic_answers_long(input_dir, output_dir)
    if long_df.empty:
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "stoic_mental_health_daily.csv")
        return out

    wanted = {
        LOWEST_MOOD_QID,
        LOWEST_MOOD_DURATION_QID,
        RELATIONSHIP_SECURITY_QID,
        AUTOMATIC_THOUGHTS_QID,
        TRIGGERS_QID,
        SYMPTOMS_QID,
        RECOVERY_QID,
    }
    custom = long_df[long_df["question_id"].isin(wanted)].copy()
    if custom.empty:
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "stoic_mental_health_daily.csv")
        return out

    if "timestamp" in custom.columns:
        custom = custom.sort_values(["date", "question_id", "timestamp"])
    custom = custom.dropna(subset=["date"]).drop_duplicates(["date", "question_id"], keep="last")

    wide = custom.pivot(index="date", columns="question_id", values="answer").reset_index()
    wide = wide.rename(
        columns={
            LOWEST_MOOD_QID: "lowest_mood_score",
            LOWEST_MOOD_DURATION_QID: "lowest_mood_duration_score",
            RELATIONSHIP_SECURITY_QID: "relationship_security_score",
            AUTOMATIC_THOUGHTS_QID: "automatic_thoughts_text",
            TRIGGERS_QID: "triggers_text",
            SYMPTOMS_QID: "symptoms_text",
            RECOVERY_QID: "recovery_methods_text",
        }
    )

    for col in ["lowest_mood_score", "lowest_mood_duration_score", "relationship_security_score"]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    if "lowest_mood_duration_score" in wide.columns:
        rounded = wide["lowest_mood_duration_score"].round().astype("Int64")
        wide["lowest_mood_duration_label"] = rounded.map(LOWEST_MOOD_DURATION_LABELS)

    wide = _add_code_features(wide, "triggers_text", "trigger_labels", "trigger_count", "trigger", TRIGGER_CODE_MAP)
    wide = _add_code_features(wide, "symptoms_text", "symptom_labels", "symptom_count", "symptom", SYMPTOM_CODE_MAP, exclude_from_count={"9"})
    wide = _add_code_features(wide, "recovery_methods_text", "recovery_labels", "recovery_count", "recovery", RECOVERY_CODE_MAP)

    if "automatic_thoughts_text" in wide.columns:
        wide["automatic_thought_count"] = wide["automatic_thoughts_text"].apply(lambda x: len(_split_automatic_thoughts(x)))
        wide["automatic_thought_any"] = (wide["automatic_thought_count"] > 0).astype(int)

    wide = standardize_date(wide, "date").sort_values("date")
    save_csv(wide, output_dir / "stoic_mental_health_daily.csv")
    return wide


def build_stoic_triggers_long(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    daily = build_stoic_mental_health_daily(input_dir, output_dir)
    rows = []
    for _, row in daily.iterrows():
        for code in _extract_int_codes(row.get("triggers_text")):
            rows.append({"date": row.get("date"), "trigger_code": code, "trigger": TRIGGER_CODE_MAP.get(code, code), "lowest_mood_score": row.get("lowest_mood_score"), "lowest_mood_duration_score": row.get("lowest_mood_duration_score"), "relationship_security_score": row.get("relationship_security_score")})
    out = pd.DataFrame(rows)
    save_csv(out, output_dir / "stoic_triggers_long.csv")
    return out


def build_stoic_symptoms_long(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    daily = build_stoic_mental_health_daily(input_dir, output_dir)
    rows = []
    for _, row in daily.iterrows():
        for code in _extract_int_codes(row.get("symptoms_text")):
            rows.append({"date": row.get("date"), "symptom_code": code, "symptom": SYMPTOM_CODE_MAP.get(code, code), "lowest_mood_score": row.get("lowest_mood_score"), "lowest_mood_duration_score": row.get("lowest_mood_duration_score"), "relationship_security_score": row.get("relationship_security_score")})
    out = pd.DataFrame(rows)
    save_csv(out, output_dir / "stoic_symptoms_long.csv")
    return out


def build_stoic_recovery_long(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    daily = build_stoic_mental_health_daily(input_dir, output_dir)
    rows = []
    for _, row in daily.iterrows():
        for code in _extract_int_codes(row.get("recovery_methods_text")):
            rows.append({"date": row.get("date"), "recovery_code": code, "recovery_method": RECOVERY_CODE_MAP.get(code, code), "lowest_mood_score": row.get("lowest_mood_score"), "lowest_mood_duration_score": row.get("lowest_mood_duration_score"), "relationship_security_score": row.get("relationship_security_score")})
    out = pd.DataFrame(rows)
    save_csv(out, output_dir / "stoic_recovery_long.csv")
    return out


def build_stoic_automatic_thoughts_long(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    daily = build_stoic_mental_health_daily(input_dir, output_dir)
    rows = []
    for _, row in daily.iterrows():
        for thought in _split_automatic_thoughts(row.get("automatic_thoughts_text")):
            rows.append({"date": row.get("date"), "automatic_thought": thought, "lowest_mood_score": row.get("lowest_mood_score"), "lowest_mood_duration_score": row.get("lowest_mood_duration_score"), "relationship_security_score": row.get("relationship_security_score")})
    out = pd.DataFrame(rows)
    save_csv(out, output_dir / "stoic_automatic_thoughts_long.csv")
    return out


def build_stoic_trigger_thought_pairs(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    daily = build_stoic_mental_health_daily(input_dir, output_dir)
    rows = []
    for _, row in daily.iterrows():
        triggers = [TRIGGER_CODE_MAP.get(code, code) for code in _extract_int_codes(row.get("triggers_text"))]
        thoughts = _split_automatic_thoughts(row.get("automatic_thoughts_text"))
        for trigger in triggers:
            for thought in thoughts:
                rows.append({"date": row.get("date"), "trigger": trigger, "automatic_thought": thought, "lowest_mood_score": row.get("lowest_mood_score")})
    out = pd.DataFrame(rows)
    save_csv(out, output_dir / "stoic_trigger_thought_pairs.csv")
    return out


def build_stoic_daily_mood(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    out = build_stoic_mental_health_daily(input_dir, output_dir)
    save_csv(out, output_dir / "stoic_daily_mood.csv")
    return out


def build_stoic_daily_wide(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    daily = build_stoic_mental_health_daily(input_dir, output_dir)
    if daily.empty:
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "stoic_daily_wide.csv")
        return out
    numeric_cols = [c for c in daily.columns if c == "date" or pd.api.types.is_numeric_dtype(daily[c])]
    out = daily[numeric_cols].copy()
    save_csv(out, output_dir / "stoic_daily_wide.csv")
    return out


def process_stoic(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> None:
    ensure_dir(output_dir)
    build_stoic_answers_long(input_dir, output_dir)
    build_stoic_mental_health_daily(input_dir, output_dir)
    build_stoic_daily_mood(input_dir, output_dir)
    build_stoic_daily_wide(input_dir, output_dir)
    build_stoic_triggers_long(input_dir, output_dir)
    build_stoic_symptoms_long(input_dir, output_dir)
    build_stoic_recovery_long(input_dir, output_dir)
    build_stoic_automatic_thoughts_long(input_dir, output_dir)
    build_stoic_trigger_thought_pairs(input_dir, output_dir)
