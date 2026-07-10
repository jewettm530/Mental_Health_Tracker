"""utils.analysis

Useful statistical analysis for the Health Tracker project.

This module intentionally avoids one giant generic correlation dump. It creates
interpretable outputs that support questions like:
- What tends to be different on days with loneliness?
- Which health/activity factors are associated with better or worse mood?
- Which associations are worth visualizing later?

Design choices:
- Count columns are kept for descriptive context but excluded from correlation rankings.
- Trigger/symptom/thought/recovery binary features are not correlated with each other.
- Mood-focused outputs are prioritized over generic all-pairs correlations.
- Sleep is represented as total hours asleep, not individual sleep stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # scipy is optional but recommended
    stats = None

from utils.paths import MERGED_DIR, OUTPUT_DIR
from utils.file_utils import ensure_dir


# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

ANALYSIS_DIR = OUTPUT_DIR / "analysis"
DESCRIPTIVE_DIR = ANALYSIS_DIR / "descriptive"
ASSOCIATIONS_DIR = ANALYSIS_DIR / "associations"
GROUP_COMPARISONS_DIR = ANALYSIS_DIR / "group_comparisons"
PREDICTORS_DIR = ANALYSIS_DIR / "predictors"
SUMMARY_DIR = ANALYSIS_DIR / "summary"


def ensure_analysis_folders() -> None:
    for folder in (
        ANALYSIS_DIR,
        DESCRIPTIVE_DIR,
        ASSOCIATIONS_DIR,
        GROUP_COMPARISONS_DIR,
        PREDICTORS_DIR,
        SUMMARY_DIR,
    ):
        ensure_dir(folder)


# -----------------------------------------------------------------------------
# Column definitions
# -----------------------------------------------------------------------------

PRIMARY_MOOD_TARGET = "lowest_mood_score"

MOOD_TARGETS = [
    "lowest_mood_score",
    "relationship_security_score",
    "lowest_mood_duration_score",
]

COUNT_COLUMNS = {
    "trigger_count",
    "symptom_count",
    "automatic_thought_count",
    "recovery_count",
}

RING_EXCLUDE_PATTERNS = ("ring_", "goal", "unit")
SLEEP_STAGE_PATTERNS = ("awake", "rem", "core", "deep", "asleepcore", "asleepdeep", "asleeprem", "inbed")

HEALTH_PREFIXES = (
    "activity_",
    "heart_",
    "body_",
    "resp_",
    "workout_",
    "sleep_",
)

PREFERRED_HEALTH_FEATURES = [
    "sleep_hours_asleep",
    "activity_steps",
    "activity_exercise_minutes",
    "activity_active_energy",
    "activity_walking_running_distance",
    "heart_hrv_sdnn",
    "heart_resting_heart_rate",
    "heart_rate",
    "workout_count",
    "workout_duration_total",
]

PREFERRED_GROUP_OUTCOMES = [
    "lowest_mood_score",
    "relationship_security_score",
    "lowest_mood_duration_score",
    "sleep_hours_asleep",
    "activity_steps",
    "activity_exercise_minutes",
    "heart_hrv_sdnn",
    "heart_resting_heart_rate",
]

# These can still exist in the enriched dataset if you create them manually,
# but they are not useful as default “days with vs without” factors.
# HRV is more useful as a continuous signal, and recovery_time is better
# handled as a duration/outcome question rather than a binary factor.
DEFAULT_REPORT_EXCLUDE_FEATURES = {
    "low_hrv_day",
    "high_hrv_day",
    "recovery_time",
}

DEFAULT_REPORT_EXCLUDE_PREFIXES = (
    "low_hrv_",
    "high_hrv_",
)


# -----------------------------------------------------------------------------
# Loading and cleaning
# -----------------------------------------------------------------------------


def load_master_daily() -> pd.DataFrame:
    path = MERGED_DIR / "master_daily.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/03_merge_data.py before scripts/04_analysis.py."
        )
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def normalize_name(name: str) -> str:
    return str(name).lower().strip().replace(" ", "_")


def pretty_label(col: str) -> str:
    text = str(col).replace("_", " ").strip().title()
    replacements = {
        "Hrv": "HRV",
        "Sdnn": "SDNN",
        "Resp": "Resp",
        "Hr": "HR",
        "Vs": "vs",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def coerce_numeric_frame(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.columns)
    for col in cols:
        if col != "date" and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def numeric_columns(df: pd.DataFrame, min_non_null: int = 2) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col == "date":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= min_non_null:
            cols.append(col)
    return cols


def is_binary_series(s: pd.Series) -> bool:
    vals = set(pd.to_numeric(s, errors="coerce").dropna().unique())
    return len(vals) > 0 and vals.issubset({0, 1})


def feature_group(col: str) -> str:
    name = normalize_name(col)
    if name in MOOD_TARGETS or name in {"mood_score", "average_mood_score"}:
        return "mood"
    if name.startswith("trigger_"):
        return "trigger"
    if name.startswith("symptom_"):
        return "symptom"
    if name.startswith("recovery_"):
        return "recovery"
    if name.startswith("automatic_thought_") or name.startswith("thought_"):
        return "thought"
    if name in COUNT_COLUMNS or name.endswith("_count"):
        return "count"
    if name.startswith("sleep_") or name == "sleep_hours_asleep":
        return "sleep"
    if name.startswith("activity_") or name.startswith("workout_"):
        return "activity"
    if name.startswith("heart_") or name == "heart_rate":
        return "heart"
    if name.startswith("body_") or name.startswith("resp_") or name.startswith("respiratory_"):
        return "health"
    return "other"


def is_count_col(col: str) -> bool:
    name = normalize_name(col)
    return name in COUNT_COLUMNS or name.endswith("_count")


def is_apple_ring_noise(col: str) -> bool:
    name = normalize_name(col)
    return any(pattern in name for pattern in RING_EXCLUDE_PATTERNS)


def is_sleep_stage_col(col: str) -> bool:
    name = normalize_name(col)
    if not name.startswith("sleep_"):
        return False
    if name == "sleep_hours_asleep":
        return False
    return any(pattern in name for pattern in SLEEP_STAGE_PATTERNS)


def is_default_excluded_feature(col: str) -> bool:
    name = normalize_name(col)
    return name in DEFAULT_REPORT_EXCLUDE_FEATURES or any(name.startswith(prefix) for prefix in DEFAULT_REPORT_EXCLUDE_PREFIXES)


def detect_binary_features(df: pd.DataFrame) -> list[str]:
    features = []
    for col in numeric_columns(df, min_non_null=2):
        if col in MOOD_TARGETS or is_count_col(col) or is_apple_ring_noise(col) or is_default_excluded_feature(col):
            continue
        if is_binary_series(df[col]):
            features.append(col)
    return features


def detect_mood_targets(df: pd.DataFrame, min_non_null: int = 2) -> list[str]:
    return [
        col for col in MOOD_TARGETS
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= min_non_null
    ]


def add_total_sleep_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single useful sleep variable: sleep_hours_asleep.

    Preference order:
    1. existing sleep_hours_asleep
    2. sleep_sleep_hours_asleep from merged/prefixed files
    3. sum Apple sleep categories containing asleep/core/deep/rem, excluding awake/inbed
    4. sleep_hours_total_tracked / sleep_sleep_hours_total_tracked as last resort
    """
    out = df.copy()
    if "sleep_hours_asleep" in out.columns:
        out["sleep_hours_asleep"] = pd.to_numeric(out["sleep_hours_asleep"], errors="coerce")
        return out

    if "sleep_sleep_hours_asleep" in out.columns:
        out["sleep_hours_asleep"] = pd.to_numeric(out["sleep_sleep_hours_asleep"], errors="coerce")
        return out

    stage_cols = []
    for col in out.columns:
        name = normalize_name(col)
        if not name.startswith("sleep"):
            continue
        if "awake" in name or "inbed" in name:
            continue
        if any(term in name for term in ("asleep", "core", "deep", "rem")):
            stage_cols.append(col)

    if stage_cols:
        temp = out[stage_cols].apply(pd.to_numeric, errors="coerce")
        out["sleep_hours_asleep"] = temp.sum(axis=1, min_count=1)
        return out

    for fallback in ["sleep_hours_total_tracked", "sleep_sleep_hours_total_tracked"]:
        if fallback in out.columns:
            out["sleep_hours_asleep"] = pd.to_numeric(out[fallback], errors="coerce")
            return out

    return out


def create_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique_values": int(s.nunique(dropna=True)),
            "feature_group": feature_group(col),
        })
    return pd.DataFrame(rows)


def create_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    num = coerce_numeric_frame(df, cols)[cols]
    summary = num.describe().T.reset_index().rename(columns={"index": "column"})
    summary["feature_group"] = summary["column"].map(feature_group)
    return summary


# -----------------------------------------------------------------------------
# Correlations
# -----------------------------------------------------------------------------


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    temp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    n = len(temp)
    if n < 3 or temp["x"].nunique() < 2 or temp["y"].nunique() < 2:
        return None, None, n
    r = float(temp["x"].corr(temp["y"]))
    p = None
    if stats is not None:
        try:
            _, p_val = stats.pearsonr(temp["x"], temp["y"])
            p = float(p_val)
        except Exception:
            p = None
    return r, p, n


def allowed_correlation_pair(a: str, b: str, allow_mood_binary: bool = True) -> bool:
    """Return False for technically true but unhelpful relationships."""
    a_name, b_name = normalize_name(a), normalize_name(b)
    a_group, b_group = feature_group(a), feature_group(b)

    if a_name == b_name or a_name.replace("stoic_", "") == b_name.replace("stoic_", ""):
        return False

    # Remove counts anywhere in correlations/rankings.
    if is_count_col(a) or is_count_col(b):
        return False

    # Remove Apple ring goals/units and sleep stages.
    if is_apple_ring_noise(a) or is_apple_ring_noise(b):
        return False
    if is_sleep_stage_col(a) or is_sleep_stage_col(b):
        return False

    binary_like = {"trigger", "symptom", "thought", "recovery"}

    # Do not correlate trigger/symptom/thought/recovery with each other.
    # That mostly creates co-occurrence noise rather than mood insight.
    if a_group in binary_like and b_group in binary_like:
        return False

    # Avoid generic health-health/activity-activity duplicate relationships.
    if a_group == b_group and a_group in {"activity", "heart", "sleep", "health"}:
        return False

    return True


def correlation_report(df: pd.DataFrame, min_n: int = 4) -> pd.DataFrame:
    """Filtered all-pairs correlations.

    This is intentionally conservative: count variables, sleep stages, Apple ring
    goals/units, binary-vs-binary co-occurrences, and same-group health duplicates
    are excluded.
    """
    cols = numeric_columns(df, min_non_null=3)
    df_num = coerce_numeric_frame(df, cols)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if not allowed_correlation_pair(a, b):
                continue
            r, p, n = safe_corr(df_num[a], df_num[b])
            if r is None or n < min_n:
                continue
            rows.append({
                "variable_1": a,
                "variable_2": b,
                "variable_1_label": pretty_label(a),
                "variable_2_label": pretty_label(b),
                "variable_1_group": feature_group(a),
                "variable_2_group": feature_group(b),
                "correlation": round(r, 4),
                "abs_correlation": round(abs(r), 4),
                "p_value": None if p is None else round(p, 6),
                "n_days": n,
                "interpretation_note": "exploratory" if n < 15 else "more stable",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_correlation", ascending=False)
    return out


def mood_focused_correlation_report(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    targets = detect_mood_targets(df, min_non_null=2)
    cols = numeric_columns(df, min_non_null=2)
    df_num = coerce_numeric_frame(df, cols)
    rows = []
    for target in targets:
        for feature in cols:
            if feature == target:
                continue
            if is_count_col(feature) or is_apple_ring_noise(feature) or is_sleep_stage_col(feature):
                continue
            if not allowed_correlation_pair(target, feature):
                continue
            r, p, n = safe_corr(df_num[feature], df_num[target])
            if r is None or n < min_n:
                continue
            rows.append({
                "target": target,
                "target_label": pretty_label(target),
                "feature": feature,
                "feature_label": pretty_label(feature),
                "feature_group": feature_group(feature),
                "correlation": round(r, 4),
                "abs_correlation": round(abs(r), 4),
                "p_value": None if p is None else round(p, 6),
                "n_days": n,
                "interpretation_note": "exploratory" if n < 15 else "more stable",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "abs_correlation"], ascending=[True, False])
    return out


# -----------------------------------------------------------------------------
# Group comparisons: days with vs without
# -----------------------------------------------------------------------------


def confidence_interval_mean_diff(with_values: pd.Series, without_values: pd.Series) -> tuple[float | None, float | None]:
    x = pd.to_numeric(with_values, errors="coerce").dropna()
    y = pd.to_numeric(without_values, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return None, None
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    if se == 0 or np.isnan(se):
        return None, None
    diff = x.mean() - y.mean()
    return float(diff - 1.96 * se), float(diff + 1.96 * se)


def t_test_p_value(with_values: pd.Series, without_values: pd.Series) -> float | None:
    if stats is None:
        return None
    x = pd.to_numeric(with_values, errors="coerce").dropna()
    y = pd.to_numeric(without_values, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None
    try:
        _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        return None


def group_comparison_report(
    df: pd.DataFrame,
    binary_features: list[str] | None = None,
    outcomes: list[str] | None = None,
) -> pd.DataFrame:
    df_num = coerce_numeric_frame(df)
    binary_features = binary_features or detect_binary_features(df_num)
    outcomes = outcomes or [c for c in PREFERRED_GROUP_OUTCOMES if c in df_num.columns]

    rows = []
    for feature in binary_features:
        f_group = feature_group(feature)
        if f_group == "count" or is_default_excluded_feature(feature) or not is_binary_series(df_num[feature]):
            continue

        feature_values = pd.to_numeric(df_num[feature], errors="coerce")
        with_mask = feature_values == 1
        without_mask = feature_values == 0

        for outcome in outcomes:
            if outcome == feature or outcome not in df_num.columns or is_count_col(outcome):
                continue
            outcome_values = pd.to_numeric(df_num[outcome], errors="coerce")
            valid_with = with_mask & outcome_values.notna()
            valid_without = without_mask & outcome_values.notna()
            n_with = int(valid_with.sum())
            n_without = int(valid_without.sum())
            if n_with < 1 or n_without < 1:
                continue

            with_values = outcome_values.loc[valid_with]
            without_values = outcome_values.loc[valid_without]
            with_mean = float(with_values.mean())
            without_mean = float(without_values.mean())
            diff = with_mean - without_mean
            r, corr_p, corr_n = safe_corr(feature_values, outcome_values)
            ci_low, ci_high = confidence_interval_mean_diff(with_values, without_values)
            p = t_test_p_value(with_values, without_values)

            rows.append({
                "feature": feature,
                "feature_label": pretty_label(feature),
                "feature_group": f_group,
                "outcome": outcome,
                "outcome_label": pretty_label(outcome),
                "n_days_with": n_with,
                "n_days_without": n_without,
                "mean_with": round(with_mean, 4),
                "mean_without": round(without_mean, 4),
                "difference_with_minus_without": round(diff, 4),
                "abs_difference": round(abs(diff), 4),
                "correlation": None if r is None else round(r, 4),
                "correlation_p_value": None if corr_p is None else round(corr_p, 6),
                "correlation_n_days": corr_n,
                "t_test_p_value": None if p is None else round(p, 6),
                "ci_95_low": None if ci_low is None else round(ci_low, 4),
                "ci_95_high": None if ci_high is None else round(ci_high, 4),
                "interpretation_note": "small sample clue" if min(n_with, n_without) < 5 else "more stable comparison",
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["outcome", "abs_difference"], ascending=[True, False])
    return out


# -----------------------------------------------------------------------------
# Threshold features for intuitive yes/no comparisons
# -----------------------------------------------------------------------------


def _add_quantile_flag(df: pd.DataFrame, source_col: str, flag_col: str, quantile: float, direction: str) -> None:
    if source_col not in df.columns:
        return
    s = pd.to_numeric(df[source_col], errors="coerce")
    valid = s.dropna()
    if len(valid) < 10 or valid.nunique() < 2:
        return
    cutoff = valid.quantile(quantile)
    if direction == "high":
        df[flag_col] = (s >= cutoff).astype("Int64")
    else:
        df[flag_col] = (s <= cutoff).astype("Int64")


def create_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_total_sleep_column(df)

    if "activity_exercise_minutes" in out.columns:
        s = pd.to_numeric(out["activity_exercise_minutes"], errors="coerce")
        out["exercised_day"] = (s > 0).astype("Int64")
    if "workout_count" in out.columns:
        s = pd.to_numeric(out["workout_count"], errors="coerce")
        out["workout_day"] = (s > 0).astype("Int64")

    # Interpretable high/low health/activity flags.
    _add_quantile_flag(out, "activity_steps", "high_steps_day", 0.75, "high")
    _add_quantile_flag(out, "activity_steps", "low_steps_day", 0.25, "low")
    _add_quantile_flag(out, "activity_exercise_minutes", "high_exercise_day", 0.75, "high")
    _add_quantile_flag(out, "sleep_hours_asleep", "high_sleep_day", 0.75, "high")
    _add_quantile_flag(out, "sleep_hours_asleep", "low_sleep_day", 0.25, "low")
    # Do not create high/low HRV day flags by default. HRV is more useful
    # as a continuous trend and scatter relationship with mood/security.
    _add_quantile_flag(out, "heart_resting_heart_rate", "high_resting_hr_day", 0.75, "high")
    _add_quantile_flag(out, "heart_resting_heart_rate", "low_resting_hr_day", 0.25, "low")

    return out


# -----------------------------------------------------------------------------
# Useful association summaries
# -----------------------------------------------------------------------------


def create_top_positive_negative_associations(group_report: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_report.empty:
        cols = ["feature", "feature_label", "outcome", "difference_with_minus_without", "n_days_with", "n_days_without"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    mood = group_report[group_report["outcome"] == PRIMARY_MOOD_TARGET].copy()
    if mood.empty:
        return pd.DataFrame(), pd.DataFrame()

    # For lowest mood score, higher means better. Positive difference = better days when feature is present.
    positive = mood[mood["difference_with_minus_without"] > 0].sort_values("difference_with_minus_without", ascending=False)
    negative = mood[mood["difference_with_minus_without"] < 0].sort_values("difference_with_minus_without", ascending=True)
    return positive, negative


def create_plain_language_report(group_report: pd.DataFrame, max_sections: int = 20) -> str:
    if group_report.empty:
        return "# Useful Mood Comparison Report\n\nNo group comparisons could be calculated yet."

    mood = group_report[group_report["outcome"] == PRIMARY_MOOD_TARGET].copy()
    if not mood.empty:
        mood = mood[~mood["feature"].map(is_default_excluded_feature)].copy()
    if mood.empty:
        return "# Useful Mood Comparison Report\n\nNo lowest-mood comparisons could be calculated yet."

    mood = mood.sort_values("abs_difference", ascending=False).head(max_sections)
    lines = ["# Useful Mood Comparison Report", ""]
    lines.append("These are exploratory patterns, not proof of cause. The report compares days when a factor was present to days when it was not present.")
    lines.append("")

    for _, row in mood.iterrows():
        label = row["feature_label"]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"Days with {label}: {int(row['n_days_with'])}")
        lines.append(f"Average lowest mood = {row['mean_with']:.2f}")
        lines.append("")
        lines.append(f"Days without {label}: {int(row['n_days_without'])}")
        lines.append(f"Average lowest mood = {row['mean_without']:.2f}")
        lines.append("")
        lines.append(f"Difference = {row['difference_with_minus_without']:.2f}")
        if pd.notna(row.get("correlation")):
            lines.append(f"Correlation = {row['correlation']:.2f}")
        if pd.notna(row.get("t_test_p_value")):
            lines.append(f"p-value = {row['t_test_p_value']:.4f}")
        if pd.notna(row.get("ci_95_low")) and pd.notna(row.get("ci_95_high")):
            lines.append(f"95% CI for difference = [{row['ci_95_low']:.2f}, {row['ci_95_high']:.2f}]")
        lines.append(f"Note = {row['interpretation_note']}")
        lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def run_useful_analysis() -> None:
    ensure_analysis_folders()

    master = load_master_daily()
    enriched = create_threshold_features(master)

    # Save enriched daily file for visualization/dashboard use.
    enriched_path = MERGED_DIR / "correlation_ready_daily_enriched.csv"
    enriched.to_csv(enriched_path, index=False)

    dataset_summary = create_dataset_summary(enriched)
    numeric_summary = create_numeric_summary(enriched)
    dataset_summary.to_csv(DESCRIPTIVE_DIR / "dataset_summary.csv", index=False)
    numeric_summary.to_csv(DESCRIPTIVE_DIR / "numeric_summary.csv", index=False)

    filtered_corr = correlation_report(enriched)
    mood_corr = mood_focused_correlation_report(enriched)
    filtered_corr.to_csv(ASSOCIATIONS_DIR / "filtered_correlations.csv", index=False)
    mood_corr.to_csv(ASSOCIATIONS_DIR / "mood_focused_correlations.csv", index=False)

    group_report = group_comparison_report(enriched)
    group_report.to_csv(GROUP_COMPARISONS_DIR / "days_with_vs_without.csv", index=False)
    if not group_report.empty:
        for outcome in group_report["outcome"].dropna().unique():
            subset = group_report[group_report["outcome"] == outcome].sort_values("abs_difference", ascending=False)
            subset.to_csv(GROUP_COMPARISONS_DIR / f"days_with_vs_without__{outcome}.csv", index=False)

    positive, negative = create_top_positive_negative_associations(group_report)
    positive.to_csv(PREDICTORS_DIR / "top_positive_associations_lowest_mood.csv", index=False)
    negative.to_csv(PREDICTORS_DIR / "top_negative_associations_lowest_mood.csv", index=False)

    # Backward-compatible names in case older scripts look for these.
    positive.to_csv(PREDICTORS_DIR / "factors_associated_with_better_lowest_mood.csv", index=False)
    negative.to_csv(PREDICTORS_DIR / "factors_associated_with_worse_lowest_mood.csv", index=False)

    report_text = create_plain_language_report(group_report)
    (SUMMARY_DIR / "useful_mood_comparison_report.md").write_text(report_text, encoding="utf-8")

    inventory = pd.DataFrame([
        {"output": "descriptive/dataset_summary.csv", "rows": len(dataset_summary)},
        {"output": "descriptive/numeric_summary.csv", "rows": len(numeric_summary)},
        {"output": "associations/filtered_correlations.csv", "rows": len(filtered_corr)},
        {"output": "associations/mood_focused_correlations.csv", "rows": len(mood_corr)},
        {"output": "group_comparisons/days_with_vs_without.csv", "rows": len(group_report)},
        {"output": "predictors/top_positive_associations_lowest_mood.csv", "rows": len(positive)},
        {"output": "predictors/top_negative_associations_lowest_mood.csv", "rows": len(negative)},
        {"output": "summary/useful_mood_comparison_report.md", "rows": None},
        {"output": "processed/merged/correlation_ready_daily_enriched.csv", "rows": len(enriched)},
    ])
    inventory.to_csv(ANALYSIS_DIR / "analysis_inventory.csv", index=False)
    print(f"Analysis outputs saved to: {ANALYSIS_DIR}")
