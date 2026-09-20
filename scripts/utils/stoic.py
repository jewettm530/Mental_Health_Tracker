"""Stoic import and processing helpers.

The project uses both Stoic exports when available:
- ``stoic.zip`` remains the canonical structured export (UUIDs, timestamps,
  contexts, and raw values).
- ``stoic.txt`` supplies human-readable question text/choice labels and the
  displayed 1-5 ratings that are not reliably represented in the ZIP metadata.

Missing/skipped questions are kept as missing values. They are never silently
converted to 0/"not selected" because that would bias correlation and group
comparison results.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from utils.cleaning import basic_clean, clean_column_name
from utils.date_utils import standardize_date
from utils.file_utils import clear_folder, ensure_dir, save_csv, save_json
from utils.paths import STOIC_CLEAN_DIR, STOIC_IMPORTED_DIR, STOIC_RAW, STOIC_TXT_RAW

# Stable question UUIDs decoded by comparing the ZIP and TXT exports.
QUICK_MOOD_QID = "14E2EE8A-6BA0-4429-AD65-B930658A367C"
ENERGY_QID = "01AE8FFB-6F03-4826-A8E1-76B0F18049B1"
STRESS_QID = "80D50DFF-9377-4235-9326-83ED7DDACF41"
EMOTIONS_QID = "BFFC3BA9-4376-451C-A4E0-53F446E8244D"
DAILY_SUMMARY_QID = "9F1DC8FB-6267-4A98-A650-11CC0C9166A3"
PRODUCTIVITY_QID = "C0448E93-7AC1-4481-BC17-AC21589ACE35"
CONNECTEDNESS_QID = "B1736386-20E7-43E6-9C61-715028F9D519"
SUBJECTIVE_SLEEP_QID = "F896B5DF-08A0-44F5-AEC8-36FAFCC4C9C2"
DAILY_PLAN_QID = "D3B76F2C-2A25-4A66-AB31-CA3A5290AAA0"
MAIN_FOCUS_QID = "FFEC5C36-D6F0-40AC-AE9A-3AD08F7D69DB"
MOTIVATION_QID = "DD8B7043-BAAD-46F2-A6CA-FB64852EAAE8"
DAILY_GOALS_QID = "2291AFEC-48A4-4261-8AC7-9368439A7B9D"

LOWEST_MOOD_QID = "945E2B63-3B52-42E3-810A-4519A57F4631"
LOWEST_MOOD_DURATION_QID = "88E82466-17EC-4BE5-A1B8-4CCF0759A4E2"
RELATIONSHIP_SECURITY_QID = "80ED4DE6-79DB-43B6-A722-57CC9F74FD75"
AUTOMATIC_THOUGHTS_QID = "65FD9D3F-F13E-4E54-8456-63744357BA8A"
TRIGGERS_QID = "EBE0ACAD-F23B-4298-8404-481CD76FC6CE"
SYMPTOMS_QID = "2D474CB9-8ABA-410C-B63B-FFF6F412AD21"
RECOVERY_QID = "599599C9-9927-45C1-AC94-14961B64C175"
INFLUENCES_QID = "989914F1-7FDC-4FBB-94B2-1083BF753C43"

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

# TXT export is explicitly 1-5. ZIP slider values are internally 0-4, so ZIP
# fallbacks are shifted by +1 before use.
LOWEST_MOOD_DURATION_LABELS = {
    1: "< 30 minutes",
    2: "30 minutes - 2 hours",
    3: "2-6 hours",
    4: "Most of the day",
    5: "Multiple days",
}

TXT_QUESTION_TO_COLUMN = {
    "How much energy do you feel?": "energy_score",
    "How stressed do you feel?": "stress_score",
    "How productive do you feel?": "productivity_score",
    "How connected to others do you feel?": "connectedness_score",
    "How well did you sleep?": "subjective_sleep_score",
    "How motivated do you feel?": "motivation_score",
    "Lowest mood today": "lowest_mood_score",
    "How long did the lowest mood last?": "lowest_mood_duration_score",
    "How secure did my important relationships feel?": "relationship_security_score",
    "How would you describe how you’re feeling today?": "emotions_text",
    "Which of these things is influencing your feelings today?": "influences_text",
    "Write a short summary of your day.": "daily_summary_text",
    "What’s your main focus for today?": "main_focus_text",
    "What do you plan to do today?": "daily_plan_text",
    "What are your goals for today?": "daily_goals_text",
    "What seemed to trigger it?": "triggers_text",
    "During my lowest mood I experienced:": "symptoms_text",
    "How did it improve?": "recovery_methods_text",
    "What recurring automatic thoughts did I experience?": "automatic_thoughts_text",
    "How are you feeling?": "routine_mood_label",
}

RATING_COLUMNS = {
    "energy_score",
    "stress_score",
    "productivity_score",
    "connectedness_score",
    "subjective_sleep_score",
    "motivation_score",
    "lowest_mood_score",
    "lowest_mood_duration_score",
    "relationship_security_score",
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
            inventory.append(
                {
                    "source_file": name,
                    "csv_file": out_path.name,
                    "rows": len(df),
                    "columns": list(df.columns),
                }
            )

    save_json(inventory, output_dir / "stoic_import_inventory.json")


def _parse_stoic_txt_file(txt_path: Path) -> pd.DataFrame:
    """Parse Stoic's human-readable history export into one Q/A row per record."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    date_re = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    rows: list[dict[str, Any]] = []
    current_date: str | None = None
    current_section: str | None = None
    i = 0

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if date_re.fullmatch(stripped):
            current_date = stripped
            current_section = None
            i += 1
            continue

        if stripped == "----------------------" or not stripped:
            i += 1
            continue

        if current_date and not stripped.startswith(("Q:", "A:")):
            # Because Q/A blocks are consumed below, a standalone line encountered
            # here is a section heading (e.g., Morning Preparation / Evening Reflection).
            current_section = stripped
            i += 1
            continue

        if current_date and stripped.startswith("Q:"):
            question_lines = [stripped[2:].strip()]
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith("A:"):
                    break
                if s.startswith("Q:") or date_re.fullmatch(s) or s == "----------------------":
                    break
                if s:
                    question_lines.append(s)
                j += 1

            answer_lines: list[str] = []
            if j < len(lines) and lines[j].strip().startswith("A:"):
                answer_lines.append(lines[j].strip()[2:].strip())
                k = j + 1
                while k < len(lines):
                    s = lines[k].strip()
                    if s.startswith("Q:") or date_re.fullmatch(s) or s == "----------------------":
                        break
                    # Section headings occur between completed Q/A blocks.
                    if s in {"Morning Preparation", "Evening Reflection"}:
                        break
                    if s:
                        answer_lines.append(lines[k].rstrip())
                    k += 1
                i = k
            else:
                i = j

            question = "\n".join(q for q in question_lines if q).strip()
            answer = "\n".join(a for a in answer_lines if a).strip()
            rows.append(
                {
                    "date": current_date,
                    "section": current_section,
                    "question": question,
                    "answer": answer if answer else pd.NA,
                }
            )
            continue

        i += 1

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "section", "question", "answer"])
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce").dt.date.astype("string")
    return out


def import_stoic_txt(txt_path: Path = STOIC_TXT_RAW, output_dir: Path = STOIC_IMPORTED_DIR) -> pd.DataFrame:
    """Import stoic.txt when present; missing TXT is allowed for backward compatibility."""
    ensure_dir(output_dir)
    out_path = output_dir / "stoic_txt_answers.csv"
    if not txt_path.exists():
        out = pd.DataFrame(columns=["date", "section", "question", "answer"])
        save_csv(out, out_path)
        return out

    out = _parse_stoic_txt_file(txt_path)
    save_csv(out, out_path)
    return out


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
    """Split one Stoic response into individual automatic thoughts.

    The free-text field is often entered as several short thoughts separated by
    periods, commas, semicolons, or new lines. Treat those as separate thoughts
    so one multi-thought entry does not count as a single observation.
    """
    if pd.isna(text):
        return []
    raw = str(text).strip()
    if not raw:
        return []

    # Split on new lines/semicolons, commas, and sentence-ending periods.
    # Decimal periods are protected by requiring period + whitespace/end.
    parts = re.split(r"(?:[\n;]+|,\s*|\.(?=\s|$))", raw)
    cleaned: list[str] = []
    for part in parts:
        item = re.sub(r"^[-*•\d\.\)\s]+", "", str(part)).strip(" .,-\t")
        item = re.sub(r"\s+", " ", item)
        if item:
            cleaned.append(item)
    return cleaned


def _split_named_choices(value: Any) -> list[str]:
    """Split human-readable multi-select answers such as 'Work, Pets, and Cleaning'."""
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    raw = re.sub(r",\s+and\s+", ", ", raw)
    raw = re.sub(r"\s+and\s+", ", ", raw)
    return [re.sub(r"\s+", " ", item).strip() for item in raw.split(",") if item.strip()]


def _clean_indicator_label(label: str) -> str:
    return clean_column_name(label).replace("family_friends", "family_or_friends").replace("it_didn_t", "it_did_not")


def _question_key(question: Any) -> str | None:
    """Map a TXT question to a stable canonical key using its first line/prefix."""
    if pd.isna(question):
        return None
    first = str(question).splitlines()[0].strip()
    if first.startswith("During my lowest mood I experienced:"):
        return "During my lowest mood I experienced:"
    if first.startswith("What recurring automatic thoughts did I experience?"):
        return "What recurring automatic thoughts did I experience?"
    return first


def _parse_rating_1_to_5(value: Any) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*5", str(value))
    if not match:
        return None
    return float(match.group(1))


def _zip_slider_to_1_to_5(series: pd.Series) -> pd.Series:
    """Stoic ZIP sliders are stored on an internal 0-4 continuum."""
    return pd.to_numeric(series, errors="coerce") + 1.0


def _add_code_features(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    count_col: str,
    prefix: str,
    code_map: dict[str, str],
    exclude_from_count: set[str] | None = None,
) -> pd.DataFrame:
    """Add labels/counts/indicators while preserving skipped questions as NULL.

    If a question was answered, unselected choices are 0. If the entire question
    was skipped/missing, every derived feature remains ``pd.NA`` so that the day
    is excluded from with-vs-without comparisons for that question.
    """
    exclude_from_count = exclude_from_count or set()
    out = df.copy()
    if text_col not in out.columns:
        return out

    answered = out[text_col].notna() & out[text_col].astype("string").str.strip().ne("")

    def mapped_labels(value: Any) -> Any:
        if pd.isna(value) or not str(value).strip():
            return pd.NA
        return "; ".join(code_map.get(code, code) for code in _extract_int_codes(value))

    out[label_col] = out[text_col].apply(mapped_labels).astype("string")

    counts = pd.Series(pd.NA, index=out.index, dtype="Int64")
    counts.loc[answered] = out.loc[answered, text_col].apply(
        lambda x: sum(1 for code in _extract_int_codes(x) if code not in exclude_from_count)
    ).astype("Int64")
    out[count_col] = counts

    for code, label in code_map.items():
        col = f"{prefix}_{_clean_indicator_label(label)}"
        vals = pd.Series(pd.NA, index=out.index, dtype="Int64")
        vals.loc[answered] = out.loc[answered, text_col].apply(
            lambda x, code=code: int(code in _extract_int_codes(x))
        ).astype("Int64")
        out[col] = vals

    return out


def _add_named_choice_features(df: pd.DataFrame, text_col: str, prefix: str) -> pd.DataFrame:
    """Create NULL-aware one-hot columns from human-readable TXT multi-selects."""
    out = df.copy()
    if text_col not in out.columns:
        return out

    answered = out[text_col].notna() & out[text_col].astype("string").str.strip().ne("")
    observed: set[str] = set()
    for value in out.loc[answered, text_col]:
        observed.update(_split_named_choices(value))

    count_col = f"{prefix}_count"
    counts = pd.Series(pd.NA, index=out.index, dtype="Int64")
    counts.loc[answered] = out.loc[answered, text_col].apply(lambda x: len(_split_named_choices(x))).astype("Int64")
    out[count_col] = counts

    for label in sorted(observed):
        col = f"{prefix}_{_clean_indicator_label(label)}"
        vals = pd.Series(pd.NA, index=out.index, dtype="Int64")
        vals.loc[answered] = out.loc[answered, text_col].apply(
            lambda x, label=label: int(label in _split_named_choices(x))
        ).astype("Int64")
        out[col] = vals
    return out




def _combine_focus_influence_context(df: pd.DataFrame) -> pd.DataFrame:
    """Create one Daily Context feature set from duplicate focus/influence choices.

    The original ``focus_*`` and ``influence_*`` columns are retained for audit
    purposes, but analysis/dashboard code can use ``context_*`` so items such as
    Birds, Work, Nature, or Partner are represented once. Missing source questions
    remain missing rather than being treated as an explicit 0.
    """
    out = df.copy()
    source_cols = [c for c in out.columns if c.startswith(("focus_", "influence_")) and not c.endswith("_count")]
    labels = sorted({c.split("_", 1)[1] for c in source_cols})
    if not labels:
        return out

    context_cols: list[str] = []
    for label in labels:
        candidates = [c for c in (f"focus_{label}", f"influence_{label}") if c in out.columns]
        values = out[candidates].apply(pd.to_numeric, errors="coerce")
        combined = pd.Series(pd.NA, index=out.index, dtype="Int64")
        observed = values.notna().any(axis=1)
        selected = values.eq(1).any(axis=1)
        combined.loc[observed] = selected.loc[observed].astype("Int64")
        col = f"context_{label}"
        out[col] = combined
        context_cols.append(col)

    observed_any = out[context_cols].notna().any(axis=1)
    counts = pd.Series(pd.NA, index=out.index, dtype="Int64")
    counts.loc[observed_any] = out.loc[observed_any, context_cols].fillna(0).sum(axis=1).astype("Int64")
    out["context_count"] = counts
    return out


def _load_core_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _read_csv(input_dir / "answers.csv"), _read_csv(input_dir / "questions.csv"), _read_csv(input_dir / "routines.csv")


def build_stoic_answers_long(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    """Build a joined ZIP long table with one answer per row."""
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


def build_stoic_txt_answers_long(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    txt = _read_csv(input_dir / "stoic_txt_answers.csv")
    if txt.empty:
        txt = pd.DataFrame(columns=["date", "section", "question", "answer"])
    elif "date" in txt.columns:
        txt = standardize_date(txt, "date")
    save_csv(txt, output_dir / "stoic_txt_answers_long.csv")
    return txt


def build_stoic_txt_daily(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    """Build human-readable daily features from stoic.txt."""
    txt = build_stoic_txt_answers_long(input_dir, output_dir)
    if txt.empty:
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "stoic_txt_daily.csv")
        return out

    txt = txt.dropna(subset=["date", "question"]).copy()
    txt["question_key"] = txt["question"].apply(_question_key)
    txt["column"] = txt["question_key"].map(TXT_QUESTION_TO_COLUMN)
    txt = txt.dropna(subset=["column"])
    if txt.empty:
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "stoic_txt_daily.csv")
        return out

    # If the same question appears more than once in a day, the last exported
    # response is the daily value used here. ZIP timestamps remain available in
    # stoic_answers_long.csv for more detailed within-day analyses.
    txt = txt.drop_duplicates(["date", "column"], keep="last")
    wide = txt.pivot(index="date", columns="column", values="answer").reset_index()

    for col in RATING_COLUMNS:
        if col in wide.columns:
            wide[col] = wide[col].apply(_parse_rating_1_to_5).astype("Float64")

    wide = standardize_date(wide, "date").sort_values("date")
    save_csv(wide, output_dir / "stoic_txt_daily.csv")
    return wide


def _build_zip_custom_daily(long_df: pd.DataFrame) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["date"])

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
            wide[col] = _zip_slider_to_1_to_5(wide[col]).astype("Float64")
    return wide


def _build_zip_quick_mood_daily(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the ZIP-only quick mood check-ins (same QID as 'How are you feeling?')."""
    mood = long_df[long_df["question_id"] == QUICK_MOOD_QID].copy()
    if mood.empty:
        return pd.DataFrame(columns=["date"])
    mood = mood.dropna(subset=["date"])
    mood["score"] = _zip_slider_to_1_to_5(mood["answer"])
    mood = mood.dropna(subset=["score"])
    if mood.empty:
        return pd.DataFrame(columns=["date"])

    if "timestamp" in mood.columns:
        mood = mood.sort_values(["date", "timestamp"])
    mood["rounded_score"] = mood["score"].round().clip(1, 5).astype("Int64")
    grouped = mood.groupby("date", as_index=False).agg(
        mood_checkin_mean_score=("score", "mean"),
        mood_checkin_min_score=("score", "min"),
        mood_checkin_max_score=("score", "max"),
        mood_checkin_last_score=("score", "last"),
        mood_checkin_sd_score=("score", lambda x: float(x.std(ddof=0)) if len(x) else np.nan),
        mood_checkin_count=("score", "count"),
    )
    grouped["mood_checkin_count"] = grouped["mood_checkin_count"].astype("Int64")

    # Preserve the rating distribution so monthly stability is not reduced to a
    # single mean. Stoic stores the quick-mood slider continuously, so ratings
    # are rounded to the nearest 1–5 bucket for these readable counts.
    rating_counts = (
        mood.pivot_table(index="date", columns="rounded_score", values="score", aggfunc="count", fill_value=0)
        .reindex(columns=range(1, 6), fill_value=0)
        .rename(columns=lambda c: f"mood_checkin_rating_{int(c)}_count")
        .reset_index()
    )
    for col in [c for c in rating_counts.columns if c.endswith("_count")]:
        rating_counts[col] = rating_counts[col].astype("Int64")
    return grouped.merge(rating_counts, on="date", how="left")


def _combine_prefer_txt(zip_df: pd.DataFrame, txt_df: pd.DataFrame) -> pd.DataFrame:
    if zip_df.empty:
        return txt_df.copy()
    if txt_df.empty:
        return zip_df.copy()

    merged = zip_df.merge(txt_df, on="date", how="outer", suffixes=("__zip", "__txt"))
    bases: set[str] = set()
    for col in merged.columns:
        if col.endswith("__zip"):
            bases.add(col[:-5])
        elif col.endswith("__txt"):
            bases.add(col[:-5])

    for base in bases:
        zcol = f"{base}__zip"
        tcol = f"{base}__txt"
        if tcol in merged.columns and zcol in merged.columns:
            merged[base] = merged[tcol].combine_first(merged[zcol])
        elif tcol in merged.columns:
            merged[base] = merged[tcol]
        else:
            merged[base] = merged[zcol]

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith(("__zip", "__txt"))])
    return merged


def build_stoic_mental_health_daily(input_dir: Path = STOIC_IMPORTED_DIR, output_dir: Path = STOIC_CLEAN_DIR) -> pd.DataFrame:
    """Create one row per date using ZIP structure plus TXT-readable enrichment."""
    long_df = build_stoic_answers_long(input_dir, output_dir)
    txt_daily = build_stoic_txt_daily(input_dir, output_dir)

    zip_custom = _build_zip_custom_daily(long_df) if not long_df.empty else pd.DataFrame(columns=["date"])
    wide = _combine_prefer_txt(zip_custom, txt_daily)

    quick_mood = _build_zip_quick_mood_daily(long_df) if not long_df.empty else pd.DataFrame(columns=["date"])
    if wide.empty:
        wide = quick_mood.copy()
    elif not quick_mood.empty:
        wide = wide.merge(quick_mood, on="date", how="outer")

    if wide.empty:
        out = pd.DataFrame(columns=["date"])
        save_csv(out, output_dir / "stoic_mental_health_daily.csv")
        return out

    if "lowest_mood_duration_score" in wide.columns:
        rounded = pd.to_numeric(wide["lowest_mood_duration_score"], errors="coerce").round().astype("Int64")
        wide["lowest_mood_duration_label"] = rounded.map(LOWEST_MOOD_DURATION_LABELS).astype("string")

    # Coded custom questions: skipped question => NULL, not 0.
    wide = _add_code_features(wide, "triggers_text", "trigger_labels", "trigger_count", "trigger", TRIGGER_CODE_MAP)
    wide = _add_code_features(
        wide,
        "symptoms_text",
        "symptom_labels",
        "symptom_count",
        "symptom",
        SYMPTOM_CODE_MAP,
        exclude_from_count={"9"},
    )
    wide = _add_code_features(wide, "recovery_methods_text", "recovery_labels", "recovery_count", "recovery", RECOVERY_CODE_MAP)

    # Human-readable multi-select fields available only/most reliably in TXT.
    wide = _add_named_choice_features(wide, "emotions_text", "emotion")
    wide = _add_named_choice_features(wide, "influences_text", "influence")
    wide = _add_named_choice_features(wide, "main_focus_text", "focus")
    wide = _combine_focus_influence_context(wide)

    if "automatic_thoughts_text" in wide.columns:
        answered = wide["automatic_thoughts_text"].notna() & wide["automatic_thoughts_text"].astype("string").str.strip().ne("")
        counts = pd.Series(pd.NA, index=wide.index, dtype="Int64")
        counts.loc[answered] = wide.loc[answered, "automatic_thoughts_text"].apply(lambda x: len(_split_automatic_thoughts(x))).astype("Int64")
        wide["automatic_thought_count"] = counts
        any_vals = pd.Series(pd.NA, index=wide.index, dtype="Int64")
        any_vals.loc[answered] = (counts.loc[answered] > 0).astype("Int64")
        wide["automatic_thought_any"] = any_vals

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
    build_stoic_txt_answers_long(input_dir, output_dir)
    build_stoic_txt_daily(input_dir, output_dir)
    build_stoic_mental_health_daily(input_dir, output_dir)
    build_stoic_daily_mood(input_dir, output_dir)
    build_stoic_daily_wide(input_dir, output_dir)
    build_stoic_triggers_long(input_dir, output_dir)
    build_stoic_symptoms_long(input_dir, output_dir)
    build_stoic_recovery_long(input_dir, output_dir)
    build_stoic_automatic_thoughts_long(input_dir, output_dir)
    build_stoic_trigger_thought_pairs(input_dir, output_dir)
