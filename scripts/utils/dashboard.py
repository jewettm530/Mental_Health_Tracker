"""Utility functions for the Streamlit dashboard.

These helpers keep 06_dashboard.py focused on layout while this module handles
loading data, selecting variables, computing summaries, and making reusable
matplotlib figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:  # pragma: no cover - dashboard still works without scipy
    stats = None


TEXT_LIKE_HINTS = (
    "text",
    "note",
    "journal",
    "source_table",
    "answer",
    "question",
    "unit",
    "calendar",
)

# Columns that are technically numeric but usually clutter the variable explorer
# without answering a meaningful mental-health question. Keep the actual values
# like steps, HRV, sleep, and stand hours; hide goals/units/internal duplicates.
EXPLORER_EXCLUDE_HINTS = (
    "goal",
    "unit",
    "ring_activeenergyburned",  # duplicate of activity_active_energy
)

COUNT_COLUMNS = {
    "trigger_count",
    "symptom_count",
    "automatic_thought_count",
    "recovery_count",
    "workout_count",
}

MENTAL_HEALTH_TARGETS = [
    "lowest_mood_score",
    "lowest_mood_duration_score",
    "relationship_security_score",
]


@dataclass
class PairSummary:
    """Container for the selected-variable analysis result."""

    summary_markdown: str
    table: pd.DataFrame
    figure: plt.Figure | None


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV safely, returning an empty dataframe if unavailable."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_dashboard_data(merged_dir: Path, output_dir: Path, stoic_clean_dir: Path, apple_clean_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the dashboard's main datasets."""
    analysis_dir = output_dir / "analysis"
    return {
        "master": read_csv_if_exists(merged_dir / "master_daily.csv"),
        "enriched": read_csv_if_exists(merged_dir / "correlation_ready_daily_enriched.csv"),
        "corr_ready": read_csv_if_exists(merged_dir / "correlation_ready_daily.csv"),
        "stoic_daily": read_csv_if_exists(stoic_clean_dir / "stoic_mental_health_daily.csv"),
        "triggers_long": read_csv_if_exists(stoic_clean_dir / "stoic_triggers_long.csv"),
        "thoughts_long": read_csv_if_exists(stoic_clean_dir / "stoic_automatic_thoughts_long.csv"),
        "trigger_thought_pairs": read_csv_if_exists(stoic_clean_dir / "stoic_trigger_thought_pairs.csv"),
        "apple_sleep": read_csv_if_exists(apple_clean_dir / "apple_daily_sleep.csv"),
        "apple_activity": read_csv_if_exists(apple_clean_dir / "apple_daily_activity.csv"),
        "apple_heart": read_csv_if_exists(apple_clean_dir / "apple_daily_heart.csv"),
        "apple_workouts": read_csv_if_exists(apple_clean_dir / "apple_daily_workouts.csv"),
        "group_comparisons": read_csv_if_exists(analysis_dir / "group_comparisons" / "days_with_vs_without.csv"),
        "mood_correlations": read_csv_if_exists(analysis_dir / "associations" / "mood_focused_correlations.csv"),
        "top_positive": read_csv_if_exists(analysis_dir / "predictors" / "top_positive_associations_lowest_mood.csv"),
        "top_negative": read_csv_if_exists(analysis_dir / "predictors" / "top_negative_associations_lowest_mood.csv"),
        "readable_summaries": read_csv_if_exists(analysis_dir / "insights" / "readable_mood_factor_summaries.csv"),
        "worst_day_predictors": read_csv_if_exists(analysis_dir / "predictions" / "what_predicts_worst_days.csv"),
        "best_day_predictors": read_csv_if_exists(analysis_dir / "predictions" / "what_predicts_best_days.csv"),
        "recovery_effectiveness": read_csv_if_exists(analysis_dir / "recovery" / "recovery_effectiveness.csv"),
        "sleep_quality": read_csv_if_exists(analysis_dir / "sleep" / "sleep_quality_ranges.csv"),
        "lagged_effects": read_csv_if_exists(analysis_dir / "lagged" / "next_day_effects.csv"),
        "consistency": read_csv_if_exists(analysis_dir / "consistency" / "monthly_consistency.csv"),
        "personal_baselines": read_csv_if_exists(analysis_dir / "baselines" / "personal_baselines.csv"),
        "things_to_watch": read_csv_if_exists(analysis_dir / "things_to_watch" / "things_to_watch.csv"),
    }


def choose_daily_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Choose the richest daily dataset available."""
    for key in ("enriched", "corr_ready", "master"):
        df = datasets.get(key, pd.DataFrame())
        if not df.empty:
            return df.copy()
    return pd.DataFrame()


def split_camel_case(text: str) -> str:
    """Add spaces inside camelCase/PascalCase labels."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return text


def pretty_label(name: str) -> str:
    """Convert snake_case/camelCase/export names into readable labels.

    This intentionally handles Apple Health's mixed naming style, including
    columns like ring_appleStandHours and resp_respiratory_rate.
    """
    if not isinstance(name, str):
        return str(name)

    cleaned = name.replace("HKQuantityTypeIdentifier", "").replace("HKCategoryTypeIdentifier", "")
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = split_camel_case(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Normalize common source prefixes and repeated fragments.
    replacements = {
        "resp respiratory": "respiratory",
        "Resp Respiratory": "Respiratory",
        "hrv sdnn": "HRV SDNN",
        "HRV Sdnn": "HRV SDNN",
        "hrv": "HRV",
        "sdnn": "SDNN",
        "apple": "Apple",
        "active Energy": "Active Energy",
        "basal Energy": "Basal Energy",
        "walking Running": "Walking/Running",
        "stand Hours": "Stand Hours",
        "exercise Time": "Exercise Time",
        "move Time": "Move Time",
    }
    for old, new in replacements.items():
        cleaned = re.sub(old, new, cleaned, flags=re.IGNORECASE)

    words = []
    for word in cleaned.split():
        upper_word = word.upper()
        if upper_word in {"HRV", "SDNN", "REM"}:
            words.append(upper_word)
        elif word.lower() == "ios":
            words.append("iOS")
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def is_text_like_column(col: str) -> bool:
    lowered = col.lower()
    return any(hint in lowered for hint in TEXT_LIKE_HINTS)


def is_binary_series(series: pd.Series) -> bool:
    vals = pd.to_numeric(series.dropna(), errors="coerce").dropna().unique()
    if len(vals) == 0:
        return False
    return set(vals).issubset({0, 1, 0.0, 1.0})


def is_numeric_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().sum() > 0


def variable_group(col: str) -> str:
    """Assign a practical category for dashboard filtering."""
    c = col.lower()
    if col == "date":
        return "date"
    if c in COUNT_COLUMNS or c.endswith("_count"):
        return "count"
    if c.startswith("trigger_"):
        return "trigger"
    if c.startswith("symptom_"):
        return "symptom"
    if c.startswith("recovery_"):
        return "recovery"
    if c.startswith("automatic_thought_"):
        return "thought"
    if "mood" in c or "security" in c or "duration" in c:
        return "mood/relationship"
    if "sleep" in c or "asleep" in c or "awake" in c:
        return "sleep"
    if "heart" in c or "hrv" in c:
        return "heart"
    if any(word in c for word in ["step", "exercise", "distance", "energy", "flight", "stand", "workout"]):
        return "activity"
    if any(word in c for word in ["body", "mass", "weight", "oxygen", "respiratory"]):
        return "body/respiratory"
    return "other"


def is_explorer_clutter_column(col: str) -> bool:
    """Hide low-value/internal columns from the variable explorer."""
    lowered = col.lower()
    return any(hint in lowered for hint in EXPLORER_EXCLUDE_HINTS)


def get_variable_options(df: pd.DataFrame, include_counts: bool = False) -> list[str]:
    """Return useful variables for pair exploration."""
    if df.empty:
        return []
    options = []
    for col in df.columns:
        if col == "date" or is_text_like_column(col) or is_explorer_clutter_column(col):
            continue
        if not include_counts and (col in COUNT_COLUMNS or col.endswith("_count")):
            continue
        if is_numeric_series(df[col]):
            options.append(col)
    return sorted(options, key=lambda c: (variable_group(c), pretty_label(c)))


def format_option(col: str) -> str:
    return f"{pretty_label(col)}  [{variable_group(col)}]"


def clean_pair_data(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    temp = df[[x_col, y_col]].copy()
    temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    return temp.dropna()


def pearson_stats(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    """Return Pearson r and p-value where possible."""
    temp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(temp) < 3 or temp["x"].nunique() < 2 or temp["y"].nunique() < 2:
        return None, None
    if stats is None:
        return float(temp["x"].corr(temp["y"])), None
    r, p = stats.pearsonr(temp["x"], temp["y"])
    return float(r), float(p)


def make_scatter_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str | None = None) -> plt.Figure | None:
    temp = clean_pair_data(df, x_col, y_col)
    if len(temp) < 3:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(temp[x_col], temp[y_col], alpha=0.75)
    if temp[x_col].nunique() >= 2 and temp[y_col].nunique() >= 2:
        try:
            slope, intercept = np.polyfit(temp[x_col], temp[y_col], 1)
            x_vals = np.linspace(temp[x_col].min(), temp[x_col].max(), 100)
            ax.plot(x_vals, slope * x_vals + intercept, linewidth=2)
        except Exception:
            pass
    ax.set_title(title or f"{pretty_label(x_col)} vs {pretty_label(y_col)}")
    ax.set_xlabel(pretty_label(x_col))
    ax.set_ylabel(pretty_label(y_col))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def make_binary_numeric_figure(df: pd.DataFrame, binary_col: str, numeric_col: str) -> plt.Figure | None:
    temp = clean_pair_data(df, binary_col, numeric_col)
    if temp.empty or temp[binary_col].nunique() < 2:
        return None
    without_vals = temp.loc[temp[binary_col] == 0, numeric_col]
    with_vals = temp.loc[temp[binary_col] == 1, numeric_col]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([without_vals, with_vals], labels=[f"Without\n{pretty_label(binary_col)}", f"With\n{pretty_label(binary_col)}"])
    ax.set_title(f"{pretty_label(numeric_col)} with vs without {pretty_label(binary_col)}")
    ax.set_ylabel(pretty_label(numeric_col))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def make_binary_binary_figure(df: pd.DataFrame, col_a: str, col_b: str) -> tuple[pd.DataFrame, plt.Figure | None]:
    temp = clean_pair_data(df, col_a, col_b)
    if temp.empty:
        return pd.DataFrame(), None
    table = pd.crosstab(temp[col_a].astype(int), temp[col_b].astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(table.values, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    ax.set_xlabel(pretty_label(col_b))
    ax.set_ylabel(pretty_label(col_a))
    ax.set_title(f"Co-occurrence: {pretty_label(col_a)} and {pretty_label(col_b)}")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, int(table.iloc[i, j]), ha="center", va="center")
    fig.tight_layout()
    return table.reset_index(), fig


def summarize_binary_vs_numeric(df: pd.DataFrame, binary_col: str, numeric_col: str) -> PairSummary:
    temp = clean_pair_data(df, binary_col, numeric_col)
    if temp.empty or temp[binary_col].nunique() < 2:
        return PairSummary("Not enough usable data for a with-vs-without comparison.", pd.DataFrame(), None)

    without_vals = temp.loc[temp[binary_col] == 0, numeric_col]
    with_vals = temp.loc[temp[binary_col] == 1, numeric_col]
    diff = with_vals.mean() - without_vals.mean()
    r, p_corr = pearson_stats(temp[binary_col], temp[numeric_col])

    p_ttest = None
    if stats is not None and len(with_vals) >= 2 and len(without_vals) >= 2 and with_vals.nunique() > 1 and without_vals.nunique() > 1:
        try:
            _, p_ttest = stats.ttest_ind(with_vals, without_vals, equal_var=False, nan_policy="omit")
            p_ttest = float(p_ttest)
        except Exception:
            p_ttest = None

    table = pd.DataFrame(
        [
            {"group": f"Days with {pretty_label(binary_col)}", "n_days": len(with_vals), f"average_{numeric_col}": with_vals.mean()},
            {"group": f"Days without {pretty_label(binary_col)}", "n_days": len(without_vals), f"average_{numeric_col}": without_vals.mean()},
            {"group": "Difference: with - without", "n_days": len(temp), f"average_{numeric_col}": diff},
        ]
    )

    lines = [
        f"### {pretty_label(binary_col)} vs {pretty_label(numeric_col)}",
        f"Days with {pretty_label(binary_col)}: **{len(with_vals)}**",
        f"Average {pretty_label(numeric_col)} = **{with_vals.mean():.2f}**",
        "",
        f"Days without {pretty_label(binary_col)}: **{len(without_vals)}**",
        f"Average {pretty_label(numeric_col)} = **{without_vals.mean():.2f}**",
        "",
        f"Difference = **{diff:+.2f}** {pretty_label(numeric_col)} points/units",
    ]
    if r is not None:
        lines.append(f"Pearson r = **{r:.2f}**" + (f", p = **{p_corr:.4f}**" if p_corr is not None else ""))
    if p_ttest is not None:
        lines.append(f"Welch t-test p-value = **{p_ttest:.4f}**")
    if len(temp) < 10:
        lines.append("\nSmall sample note: treat this as a clue, not a conclusion yet.")

    return PairSummary("\n".join(lines), table, make_binary_numeric_figure(df, binary_col, numeric_col))


def summarize_numeric_vs_numeric(df: pd.DataFrame, x_col: str, y_col: str) -> PairSummary:
    temp = clean_pair_data(df, x_col, y_col)
    if len(temp) < 3:
        return PairSummary("Not enough paired numeric data yet.", pd.DataFrame(), None)
    r, p = pearson_stats(temp[x_col], temp[y_col])
    table = pd.DataFrame(
        [
            {"metric": "paired_days", "value": len(temp)},
            {"metric": f"average_{x_col}", "value": temp[x_col].mean()},
            {"metric": f"average_{y_col}", "value": temp[y_col].mean()},
            {"metric": "pearson_r", "value": r},
            {"metric": "p_value", "value": p},
        ]
    )
    lines = [
        f"### {pretty_label(x_col)} vs {pretty_label(y_col)}",
        f"Paired days = **{len(temp)}**",
    ]
    if r is not None:
        lines.append(f"Pearson r = **{r:.2f}**" + (f", p = **{p:.4f}**" if p is not None else ""))
        if abs(r) < 0.2:
            lines.append("This is a weak linear relationship so far.")
        elif abs(r) < 0.5:
            lines.append("This is a moderate linear relationship so far.")
        else:
            lines.append("This is a stronger linear relationship so far, but it still does not prove causation.")
    return PairSummary("\n".join(lines), table, make_scatter_figure(df, x_col, y_col))


def summarize_binary_vs_binary(df: pd.DataFrame, col_a: str, col_b: str) -> PairSummary:
    temp = clean_pair_data(df, col_a, col_b)
    if temp.empty:
        return PairSummary("Not enough data for a binary co-occurrence comparison.", pd.DataFrame(), None)
    r, p = pearson_stats(temp[col_a], temp[col_b])
    table, fig = make_binary_binary_figure(df, col_a, col_b)
    lines = [
        f"### {pretty_label(col_a)} and {pretty_label(col_b)}",
        f"Paired days = **{len(temp)}**",
        "This is a co-occurrence comparison between two yes/no factors.",
    ]
    if r is not None:
        lines.append(f"Phi/Pearson correlation = **{r:.2f}**" + (f", p = **{p:.4f}**" if p is not None else ""))
    if variable_group(col_a) in {"trigger", "symptom", "thought", "recovery"} and variable_group(col_b) in {"trigger", "symptom", "thought", "recovery"}:
        lines.append("\nNote: this can show co-occurrence, but it is usually less actionable than comparing either factor against mood, sleep, activity, or relationship security.")
    return PairSummary("\n".join(lines), table, fig)


def summarize_selected_pair(df: pd.DataFrame, var_a: str, var_b: str) -> PairSummary:
    """Produce a useful summary/table/plot for two selected variables."""
    if df.empty or var_a not in df.columns or var_b not in df.columns:
        return PairSummary("Selected variables were not found in the daily dataset.", pd.DataFrame(), None)
    if var_a == var_b:
        return PairSummary("Choose two different variables.", pd.DataFrame(), None)

    a_binary = is_binary_series(df[var_a])
    b_binary = is_binary_series(df[var_b])

    if a_binary and not b_binary:
        return summarize_binary_vs_numeric(df, var_a, var_b)
    if b_binary and not a_binary:
        return summarize_binary_vs_numeric(df, var_b, var_a)
    if a_binary and b_binary:
        return summarize_binary_vs_binary(df, var_a, var_b)
    return summarize_numeric_vs_numeric(df, var_a, var_b)


def list_plot_files(plots_dir: Path, topic: str | None = None) -> list[Path]:
    """List generated plot image files."""
    base = plots_dir / topic if topic else plots_dir
    if not base.exists():
        return []
    return sorted(base.glob("*.png"))


def top_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(n).copy()


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Small dashboard inventory table."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "label": pretty_label(col),
                "group": variable_group(col),
                "non_null": int(df[col].notna().sum()),
                "unique": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "label"])
