"""Extended insight analysis for the Health Tracker project.

Creates readable, question-focused outputs used by files 4–6:
- summary cards with plain-language interpretations and confidence ratings
- factors associated with best and worst mood days
- recovery effectiveness
- sleep-quality ranges
- lagged/next-day effects
- consistency and variability
- personalized baselines
- things-to-watch summaries
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

from utils.paths import MERGED_DIR, OUTPUT_DIR
from utils.file_utils import ensure_dir

INSIGHTS_DIR = OUTPUT_DIR / "analysis" / "insights"
PREDICTION_DIR = OUTPUT_DIR / "analysis" / "predictions"
RECOVERY_DIR = OUTPUT_DIR / "analysis" / "recovery"
SLEEP_DIR = OUTPUT_DIR / "analysis" / "sleep"
LAGGED_DIR = OUTPUT_DIR / "analysis" / "lagged"
CONSISTENCY_DIR = OUTPUT_DIR / "analysis" / "consistency"
BASELINES_DIR = OUTPUT_DIR / "analysis" / "baselines"
WATCH_DIR = OUTPUT_DIR / "analysis" / "things_to_watch"
TEMPORAL_DIR = OUTPUT_DIR / "analysis" / "temporal"

COUNT_COLUMNS = {"trigger_count", "symptom_count", "automatic_thought_count", "recovery_count"}
MOOD_COL = "lowest_mood_score"

VARIABLE_UNITS = {
    "sleep_hours_asleep": "hours",
    "activity_steps": "steps",
    "activity_exercise_minutes": "minutes",
    "activity_walking_running_distance": "miles",
    "activity_active_energy": "kcal",
    "heart_hrv_sdnn": "ms",
    "heart_resting_heart_rate": "bpm",
    "heart_rate": "bpm",
    "lowest_mood_score": "points (1–5)",
    "relationship_security_score": "points (1–5)",
    "stress_score": "points (1–5)",
    "energy_score": "points (1–5)",
    "productivity_score": "points (1–5)",
    "connectedness_score": "points (1–5)",
    "activity_time_in_daylight_minutes": "minutes",
    "daylight_exposure_pct_of_available": "%",
    "weather_daylight_hours": "hours",
    "weather_sunshine_hours": "hours",
    "weather_sunshine_fraction_pct": "%",
    "weather_cloud_cover_mean_pct": "%",
    "weather_precipitation_mm": "mm",
    "weather_precipitation_hours": "hours",
    "weather_temperature_mean_f": "°F",
    "weather_apparent_temperature_mean_f": "°F",
    "weather_shortwave_radiation_mj_m2": "MJ/m²",
    "screen_time_total_minutes": "minutes",
    "screen_time_late_night_minutes": "minutes",
}


def variable_unit(col: str) -> str:
    if col in VARIABLE_UNITS:
        return VARIABLE_UNITS[col]
    lowered = str(col).lower()
    if "minute" in lowered or "duration" in lowered and "score" not in lowered:
        return "minutes"
    if "distance" in lowered:
        return "distance units"
    if "score" in lowered:
        return "points"
    return "units"


def variable_source(col: str) -> str:
    c = str(col).lower()
    if col == "activity_time_in_daylight_minutes":
        return "Apple Watch"
    if col == "daylight_exposure_pct_of_available":
        return "Derived: Apple Watch + local weather"
    if c.startswith("weather_"):
        return "Local weather history (Open-Meteo)"
    if c.startswith("screen_time_"):
        return "Optional external screen-time file"
    if c.startswith(("trigger_", "symptom_", "recovery_", "automatic_thought_", "context_")):
        return "Stoic"
    if any(token in c for token in ["mood", "stress", "energy", "productivity", "connectedness", "motivation", "security", "subjective_sleep"]):
        return "Stoic"
    if c.startswith(("activity_", "heart_", "resp_", "body_", "ring_", "sleep_")) or c == "workout_count":
        return "Apple Health export"
    return "—"
BINARY_PREFIXES = ("trigger_", "symptom_", "automatic_thought_", "recovery_", "context_")


def ensure_extended_folders() -> None:
    for folder in [INSIGHTS_DIR, PREDICTION_DIR, RECOVERY_DIR, SLEEP_DIR, LAGGED_DIR, CONSISTENCY_DIR, BASELINES_DIR, WATCH_DIR, TEMPORAL_DIR]:
        ensure_dir(folder)


def pretty_label(name: str) -> str:
    overrides = {
        "activity_time_in_daylight_minutes": "Time in Daylight (Apple Watch)",
        "daylight_exposure_pct_of_available": "Personal Daylight Exposure (% of Available Daylight)",
        "weather_daylight_hours": "Available Daylight",
        "weather_sunshine_hours": "Sunshine Duration",
        "weather_sunshine_fraction_pct": "Sunny Share of Daylight",
        "weather_cloud_cover_mean_pct": "Average Cloud Cover",
        "weather_precipitation_mm": "Precipitation",
        "weather_precipitation_hours": "Hours with Precipitation",
        "weather_temperature_mean_f": "Average Temperature",
        "weather_apparent_temperature_mean_f": "Average Feels-Like Temperature",
        "weather_shortwave_radiation_mj_m2": "Solar Radiation",
        "screen_time_total_minutes": "Total Screen Time",
        "screen_time_late_night_minutes": "Late-Night Screen Time",
        "screen_time_social_minutes": "Social Screen Time",
        "screen_time_entertainment_minutes": "Entertainment Screen Time",
        "screen_time_productivity_minutes": "Productivity Screen Time",
    }
    if name in overrides:
        return overrides[name]
    if str(name).startswith("context_"):
        return "Daily Context: " + str(name)[len("context_"):].replace("_", " ").strip().title()
    text = str(name).replace("_", " ").strip().title()
    for old, new in {"Hrv": "HRV", "Sdnn": "SDNN", "Rem": "REM"}.items():
        text = text.replace(old, new)
    return text


def load_daily() -> pd.DataFrame:
    for filename in ["correlation_ready_daily_enriched.csv", "correlation_ready_daily.csv", "master_daily.csv"]:
        path = MERGED_DIR / filename
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
    raise FileNotFoundError("Run scripts/03_merge_data.py before scripts/04_analysis.py")


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def is_binary(s: pd.Series) -> bool:
    vals = set(pd.to_numeric(s, errors="coerce").dropna().unique())
    return bool(vals) and vals.issubset({0, 1})


def confidence_rating(n_total: int, n_with: int | None = None, n_without: int | None = None) -> tuple[str, str]:
    smallest = min(v for v in [n_with, n_without] if v is not None) if n_with is not None and n_without is not None else n_total
    if n_total < 10 or smallest < 3:
        return "Very low", "Treat this as an early clue, not a conclusion."
    if n_total < 20 or smallest < 5:
        return "Preliminary", "The pattern is worth watching, but more tracked days are needed."
    if n_total < 50 or smallest < 10:
        return "Moderate", "The pattern has useful support, though it may still shift as more data accumulates."
    return "High", "The pattern is supported by a relatively large number of tracked days."


def correlation_strength(r: float | None) -> str:
    if r is None or pd.isna(r):
        return "not available"
    a = abs(r)
    strength = "very weak" if a < .1 else "weak" if a < .3 else "moderate" if a < .5 else "strong" if a < .7 else "very strong"
    direction = "positive" if r > 0 else "negative" if r < 0 else "no"
    return f"{strength} {direction} relationship"


def pearson(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    temp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(temp) < 3 or temp.x.nunique() < 2 or temp.y.nunique() < 2:
        return None, None, len(temp)
    if stats is None:
        return float(temp.x.corr(temp.y)), None, len(temp)
    r, p = stats.pearsonr(temp.x, temp.y)
    return float(r), float(p), len(temp)


def difference_ci(a: pd.Series, b: pd.Series) -> tuple[float | None, float | None]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None, None
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if not np.isfinite(se) or se == 0:
        return None, None
    return diff - 1.96 * se, diff + 1.96 * se


def binary_factors(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in COUNT_COLUMNS or not col.startswith(BINARY_PREFIXES):
            continue
        if is_binary(df[col]):
            cols.append(col)
    return cols


def build_readable_factor_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if MOOD_COL not in df.columns:
        return pd.DataFrame()
    rows = []
    mood = numeric_series(df, MOOD_COL)
    for factor in binary_factors(df):
        binary = numeric_series(df, factor)
        temp = pd.DataFrame({"factor": binary, "mood": mood}).dropna()
        if temp.factor.nunique() < 2:
            continue
        with_vals = temp.loc[temp.factor == 1, "mood"]
        without_vals = temp.loc[temp.factor == 0, "mood"]
        if with_vals.empty or without_vals.empty:
            continue
        diff = with_vals.mean() - without_vals.mean()
        r, p, n = pearson(temp.factor, temp.mood)
        ci_low, ci_high = difference_ci(with_vals, without_vals)
        conf, note = confidence_rating(n, len(with_vals), len(without_vals))
        direction = "lower" if diff < 0 else "higher"
        meaning = (
            f"On days when {pretty_label(factor).lower()} was present, your lowest mood score was "
            f"{abs(diff):.2f} points {direction} on average than on days when it was absent."
        )
        corr_meaning = (
            f"The correlation suggests a {correlation_strength(r)} between this factor and lowest mood."
            if r is not None else "A stable correlation could not be calculated yet."
        )
        rows.append({
            "factor": factor,
            "factor_label": pretty_label(factor),
            "days_with": len(with_vals),
            "average_mood_with": with_vals.mean(),
            "days_without": len(without_vals),
            "average_mood_without": without_vals.mean(),
            "mood_difference": diff,
            "correlation": r,
            "correlation_interpretation": correlation_strength(r),
            "p_value": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": conf,
            "confidence_note": note,
            "plain_language_meaning": meaning,
            "correlation_meaning": corr_meaning,
            "n_total": n,
        })
    return pd.DataFrame(rows).sort_values("mood_difference") if rows else pd.DataFrame()


def build_predictions(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible = summary.copy()
    eligible["abs_effect"] = eligible["mood_difference"].abs()
    worst = eligible[eligible.mood_difference < 0].sort_values(["abs_effect", "n_total"], ascending=[False, False])
    best = eligible[eligible.mood_difference > 0].sort_values(["abs_effect", "n_total"], ascending=[False, False])
    return worst, best


def build_recovery_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    recovery_cols = [c for c in df.columns if c.startswith("recovery_") and c not in COUNT_COLUMNS and is_binary(df[c])]
    duration_col = next((c for c in ["recovery_time_minutes", "lowest_mood_duration_minutes", "lowest_mood_duration_score"] if c in df.columns), None)
    if not recovery_cols or duration_col is None:
        return pd.DataFrame()
    rows = []
    outcome = numeric_series(df, duration_col)
    for col in recovery_cols:
        used = numeric_series(df, col)
        temp = pd.DataFrame({"used": used, "outcome": outcome}).dropna()
        vals = temp.loc[temp.used == 1, "outcome"]
        if vals.empty:
            continue
        rows.append({
            "recovery_method": col,
            "recovery_method_label": pretty_label(col),
            "times_used_with_duration": len(vals),
            "average_recovery_duration": vals.mean(),
            "median_recovery_duration": vals.median(),
            "duration_measure": duration_col,
            "confidence": confidence_rating(len(vals))[0],
        })
    return pd.DataFrame(rows).sort_values("average_recovery_duration") if rows else pd.DataFrame()


def build_sleep_quality(df: pd.DataFrame) -> pd.DataFrame:
    if "sleep_hours_asleep" not in df.columns or MOOD_COL not in df.columns:
        return pd.DataFrame()
    temp = pd.DataFrame({"sleep": numeric_series(df, "sleep_hours_asleep"), "mood": numeric_series(df, MOOD_COL)}).dropna()
    if temp.empty:
        return pd.DataFrame()
    bins = [-np.inf, 6, 7, 8, 9, np.inf]
    labels = ["Under 6 hours", "6–7 hours", "7–8 hours", "8–9 hours", "9+ hours"]
    temp["sleep_range"] = pd.cut(temp.sleep, bins=bins, labels=labels, right=False)
    return temp.groupby("sleep_range", observed=False).agg(
        n_days=("mood", "count"),
        average_lowest_mood=("mood", "mean"),
        median_lowest_mood=("mood", "median"),
        average_sleep_hours=("sleep", "mean"),
    ).reset_index()



TEMPORAL_CANDIDATES = [
    "sleep_hours_asleep", "subjective_sleep_score", "mood_checkin_mean_score",
    "relationship_security_score", "stress_score", "energy_score", "productivity_score",
    "connectedness_score", "activity_steps", "activity_exercise_minutes",
    "activity_walking_running_distance", "activity_active_energy", "activity_time_in_daylight_minutes",
    "daylight_exposure_pct_of_available",
    "heart_hrv_sdnn", "heart_resting_heart_rate", "heart_rate",
    "weather_daylight_hours", "weather_sunshine_hours", "weather_sunshine_fraction_pct",
    "weather_cloud_cover_mean_pct", "weather_precipitation_mm", "weather_precipitation_hours",
    "weather_temperature_mean_f", "weather_apparent_temperature_mean_f",
    "weather_shortwave_radiation_mj_m2", "screen_time_total_minutes",
    "screen_time_late_night_minutes", "screen_time_social_minutes",
    "screen_time_entertainment_minutes", "screen_time_productivity_minutes",
    "screen_time_pickups", "screen_time_notifications",
]


def _temporal_meaning(kind: str, col: str, r: float | None) -> str:
    label = pretty_label(col).lower()
    if r is None or pd.isna(r):
        return "Not enough paired days yet."
    if abs(r) < .20:
        return f"No clear linear pattern is apparent for {label} in this timing window."
    if kind == "before":
        direction = "better" if r > 0 else "worse"
        return f"Higher {label} before the mood day tended to be followed by {direction} lowest-mood scores."
    if kind == "same":
        direction = "better" if r > 0 else "worse"
        return f"Higher {label} tended to occur on the same days as {direction} lowest-mood scores."
    direction = "higher" if r > 0 else "lower"
    return f"Better lowest-mood scores today tended to be followed by {direction} {label} afterward."


def build_temporal_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Separate relationships by temporal direction.

    Categories intentionally avoid causal language:
    - before: previous-day factors (plus previous night's sleep) vs today's mood
    - same: values measured/aggregated on the same calendar day as mood
    - after: today's mood vs tomorrow's values; recovery selections are handled
      elsewhere because they are explicitly responses to the low-mood episode.
    """
    if "date" not in df.columns or MOOD_COL not in df.columns:
        return pd.DataFrame()
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    mood = numeric_series(data, MOOD_COL)
    rows: list[dict] = []

    candidates = [c for c in TEMPORAL_CANDIDATES if c in data.columns and c != MOOD_COL]

    # Same-day co-occurrence.
    same_day_candidates = [c for c in candidates if c not in {"sleep_hours_asleep", "subjective_sleep_score"}]
    for col in same_day_candidates:
        r, p, n = pearson(numeric_series(data, col), mood)
        if r is None:
            continue
        rows.append({
            "timing_group": "Same-day / current",
            "timing_key": "same",
            "factor": col,
            "factor_label": pretty_label(col),
            "data_source": variable_source(col),
            "relationship": r,
            "p_value": p,
            "n_days": n,
            "data_support": confidence_rating(n)[0],
            "plain_language_meaning": _temporal_meaning("same", col, r),
        })

    # Previous night's sleep is already attributed to the wake date by the
    # Apple Health processor. Do not shift it an additional day.
    for col in [c for c in ["sleep_hours_asleep", "subjective_sleep_score"] if c in data.columns]:
        r, p, n = pearson(numeric_series(data, col), mood)
        if r is None:
            continue
        rows.append({
            "timing_group": "Before mood / potential influences",
            "timing_key": "before",
            "factor": col,
            "factor_label": "Previous Night's Sleep" if col == "sleep_hours_asleep" else "Morning Sleep Quality",
            "data_source": variable_source(col),
            "relationship": r,
            "p_value": p,
            "n_days": n,
            "data_support": confidence_rating(n)[0],
            "plain_language_meaning": _temporal_meaning("before", col, r),
        })

    # Prior calendar day -> today's mood. Use a date-based self-merge so gaps in
    # tracking are not accidentally treated as consecutive days.
    base = data[["date", MOOD_COL]].copy()
    base[MOOD_COL] = pd.to_numeric(base[MOOD_COL], errors="coerce")
    for col in [c for c in candidates if c not in {"sleep_hours_asleep", "subjective_sleep_score"}]:
        prior = data[["date", col]].copy()
        prior[col] = pd.to_numeric(prior[col], errors="coerce")
        prior["date"] = prior["date"] + pd.Timedelta(days=1)
        paired = base.merge(prior, on="date", how="inner")
        r, p, n = pearson(paired[col], paired[MOOD_COL])
        if r is None:
            continue
        rows.append({
            "timing_group": "Before mood / potential influences",
            "timing_key": "before",
            "factor": col,
            "factor_label": f"Previous Day: {pretty_label(col)}",
            "data_source": variable_source(col),
            "relationship": r,
            "p_value": p,
            "n_days": n,
            "data_support": confidence_rating(n)[0],
            "plain_language_meaning": _temporal_meaning("before", col, r),
        })

    # Today's mood -> next calendar day's measurements.
    mood_today = data[["date", MOOD_COL]].copy()
    mood_today[MOOD_COL] = pd.to_numeric(mood_today[MOOD_COL], errors="coerce")
    mood_today["date"] = mood_today["date"] + pd.Timedelta(days=1)
    for col in candidates:
        future = data[["date", col]].copy()
        future[col] = pd.to_numeric(future[col], errors="coerce")
        paired = future.merge(mood_today, on="date", how="inner")
        r, p, n = pearson(paired[MOOD_COL], paired[col])
        if r is None:
            continue
        rows.append({
            "timing_group": "After mood / possible consequences",
            "timing_key": "after",
            "factor": col,
            "factor_label": f"Next Day: {pretty_label(col)}",
            "data_source": variable_source(col),
            "relationship": r,
            "p_value": p,
            "n_days": n,
            "data_support": confidence_rating(n)[0],
            "plain_language_meaning": _temporal_meaning("after", col, r),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["absolute_relationship"] = out["relationship"].abs()
    return out.sort_values(["timing_key", "absolute_relationship"], ascending=[True, False]).reset_index(drop=True)


def build_lagged_effects(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or MOOD_COL not in df.columns:
        return pd.DataFrame()
    data = df.sort_values("date").copy()
    candidates = [c for c in ["sleep_hours_asleep", "activity_steps", "activity_exercise_minutes", "heart_hrv_sdnn", "heart_resting_heart_rate", "relationship_security_score", "mood_checkin_mean_score", "stress_score", "automatic_thought_count"] if c in data.columns]
    rows = []
    tomorrow_mood = numeric_series(data, MOOD_COL).shift(-1)
    for col in candidates:
        r, p, n = pearson(numeric_series(data, col), tomorrow_mood)
        rows.append({
            "predictor": col,
            "predictor_label": pretty_label(col),
            "outcome": "next_day_lowest_mood_score",
            "correlation": r,
            "correlation_interpretation": correlation_strength(r),
            "p_value": p,
            "n_days": n,
            "confidence": confidence_rating(n)[0],
            "plain_language_meaning": (
                "Not enough paired days to describe a next-day pattern yet."
                if r is None else
                f"No clear next-day relationship is apparent between {pretty_label(col).lower()} and tomorrow's lowest mood."
                if abs(r) < 0.20 else
                f"Higher {pretty_label(col).lower()} today tended to be associated with "
                f"{'higher' if r > 0 else 'lower'} lowest mood scores tomorrow."
            ),
        })
    return pd.DataFrame(rows).sort_values("correlation") if rows else pd.DataFrame()


MIN_MOOD_DAYS_PER_MONTH = 3


def build_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly average + variability and change from prior periods.

    The displayed rows remain limited to months with enough mood tracking, but
    month-over-month and prior-year comparisons use the full history available
    for each measure (important for Apple Health variables that predate Stoic).
    """
    if "date" not in df.columns or MOOD_COL not in df.columns:
        return pd.DataFrame()

    full = df.copy()
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full.dropna(subset=["date"])
    full["month_period"] = full["date"].dt.to_period("M")

    mood_counts = (
        full.assign(_mood=numeric_series(full, MOOD_COL))
        .groupby("month_period")["_mood"]
        .count()
    )
    eligible_months = mood_counts[mood_counts >= MIN_MOOD_DAYS_PER_MONTH].index
    display_data = full[full["month_period"].isin(eligible_months)]

    candidates = [
        c for c in [
            MOOD_COL, "mood_checkin_mean_score", "relationship_security_score",
            "stress_score", "sleep_hours_asleep", "activity_steps",
            "activity_exercise_minutes", "activity_time_in_daylight_minutes",
            "daylight_exposure_pct_of_available", "heart_hrv_sdnn",
        ] if c in full.columns
    ]

    rows = []
    for month, group in display_data.groupby("month_period"):
        mood_days = int(mood_counts.get(month, 0))
        for col in candidates:
            vals = numeric_series(group, col).dropna()
            if vals.empty:
                continue
            rows.append({
                "month": str(month),
                "variable": col,
                "variable_label": pretty_label(col),
                "mood_days_in_month": mood_days,
                "n_days": len(vals),
                "mean": vals.mean(),
                "standard_deviation": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                "range": vals.max() - vals.min(),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["monthly_change_pct"] = np.nan
    out["yearly_change_pct"] = np.nan

    for row_idx, row in out.iterrows():
        col = row["variable"]
        current_period = pd.Period(row["month"], freq="M")
        current_mean = row["mean"]

        prev_period = current_period - 1
        prev_vals = numeric_series(full.loc[full["month_period"] == prev_period], col).dropna()
        if not prev_vals.empty:
            prev_mean = prev_vals.mean()
            if prev_mean != 0:
                out.loc[row_idx, "monthly_change_pct"] = (current_mean - prev_mean) / abs(prev_mean) * 100

        previous_year = current_period.year - 1
        prior_year_vals = numeric_series(full.loc[full["date"].dt.year == previous_year], col).dropna()
        if not prior_year_vals.empty:
            prior_year_avg = prior_year_vals.mean()
            if prior_year_avg != 0:
                out.loc[row_idx, "yearly_change_pct"] = (current_mean - prior_year_avg) / abs(prior_year_avg) * 100

    return out.sort_values(["month", "variable_label"], ascending=[False, True]).reset_index(drop=True)


def build_personal_baselines(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [c for c in ["sleep_hours_asleep", "activity_steps", "activity_exercise_minutes", "activity_time_in_daylight_minutes", "daylight_exposure_pct_of_available", "heart_hrv_sdnn", "heart_resting_heart_rate", MOOD_COL, "relationship_security_score"] if c in df.columns]
    rows = []
    dated = df.sort_values("date") if "date" in df.columns else df
    for col in candidates:
        vals = numeric_series(dated, col).dropna()
        if vals.empty:
            continue
        recent = vals.tail(7)
        baseline = vals.mean()
        recent_mean = recent.mean()
        rows.append({
            "variable": col,
            "variable_label": pretty_label(col),
            "data_source": variable_source(col),
            "all_time_baseline": baseline,
            "recent_7_day_average": recent_mean,
            "difference_from_baseline": recent_mean - baseline,
            "percent_difference_from_baseline": ((recent_mean - baseline) / baseline * 100) if baseline != 0 else np.nan,
            "n_total_days": len(vals),
            "n_recent_days": len(recent),
        })
    return pd.DataFrame(rows)


def build_things_to_watch(summary: pd.DataFrame, temporal: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not summary.empty:
        usable = summary[summary["factor"] != "symptom_none"].copy()
        for _, row in usable.assign(abs_effect=usable.mood_difference.abs()).sort_values("abs_effect", ascending=False).head(4).iterrows():
            rows.append({
                "category": "Same-day association",
                "priority": abs(row.mood_difference),
                "headline": row.factor_label,
                "data_source": variable_source(str(row.factor)),
                "message": row.plain_language_meaning,
                "confidence": row.confidence,
                "relationship_percent": abs(row.correlation) * 100 if pd.notna(row.correlation) else np.nan,
            })
    if not temporal.empty:
        usable_temporal = temporal.dropna(subset=["relationship"]).copy()
        # Prioritize timing that can answer directionality questions rather than
        # duplicating the same-day association cards above.
        usable_temporal = usable_temporal[usable_temporal["timing_key"].isin(["before", "after"])]
        usable_temporal = usable_temporal[usable_temporal["absolute_relationship"] >= .20]
        for _, row in usable_temporal.sort_values("absolute_relationship", ascending=False).head(4).iterrows():
            category = "Potential preceding factor" if row.timing_key == "before" else "Possible after-mood pattern"
            rows.append({
                "category": category,
                "priority": abs(row.relationship),
                "headline": row.factor_label,
                "data_source": variable_source(str(row.factor)),
                "message": row.plain_language_meaning,
                "confidence": row.data_support,
                "relationship_percent": abs(row.relationship) * 100,
            })
    if not baselines.empty:
        for _, row in baselines.assign(abs_pct=lambda x: x.percent_difference_from_baseline.abs()).sort_values("abs_pct", ascending=False).head(3).iterrows():
            unit = variable_unit(str(row.variable))
            rows.append({
                "category": "Recent baseline change",
                "priority": abs(row.percent_difference_from_baseline) / 100 if pd.notna(row.percent_difference_from_baseline) else 0,
                "headline": row.variable_label,
                "data_source": row.get("data_source", variable_source(str(row.variable))),
                "message": f"Your recent 7-day average is {row.difference_from_baseline:+.2f} {unit} from your overall baseline ({row.percent_difference_from_baseline:+.1f}%).",
                "confidence": confidence_rating(int(row.n_total_days))[0],
                "relationship_percent": np.nan,
            })
    return pd.DataFrame(rows).sort_values("priority", ascending=False) if rows else pd.DataFrame()


def run_extended_analysis() -> dict[str, pd.DataFrame]:
    ensure_extended_folders()
    df = load_daily()
    summary = build_readable_factor_summaries(df)
    worst, best = build_predictions(summary)
    recovery = build_recovery_effectiveness(df)
    sleep = build_sleep_quality(df)
    lagged = build_lagged_effects(df)
    temporal = build_temporal_relationships(df)
    consistency = build_consistency(df)
    baselines = build_personal_baselines(df)
    watch = build_things_to_watch(summary, temporal, baselines)

    outputs = {
        INSIGHTS_DIR / "readable_mood_factor_summaries.csv": summary,
        PREDICTION_DIR / "what_predicts_worst_days.csv": worst,
        PREDICTION_DIR / "what_predicts_best_days.csv": best,
        RECOVERY_DIR / "recovery_effectiveness.csv": recovery,
        SLEEP_DIR / "sleep_quality_ranges.csv": sleep,
        LAGGED_DIR / "next_day_effects.csv": lagged,
        TEMPORAL_DIR / "timing_relationships.csv": temporal,
        CONSISTENCY_DIR / "monthly_consistency.csv": consistency,
        BASELINES_DIR / "personal_baselines.csv": baselines,
        WATCH_DIR / "things_to_watch.csv": watch,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
    return {path.stem: frame for path, frame in outputs.items()}
