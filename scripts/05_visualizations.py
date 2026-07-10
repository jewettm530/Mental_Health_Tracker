"""05_visualizations.py

Question-based visualizations for the Health Tracker project.

This file intentionally stops making generic "top correlations" plots. Instead,
each function answers one practical question using a plot, table, text summary,
or a combination.

Run order:
    python3 scripts/01_import_data.py
    python3 scripts/02_process_data.py
    python3 scripts/03_merge_data.py
    python3 scripts/04_analysis.py
    python3 scripts/05_visualizations.py
"""

from __future__ import annotations

from utils.extended_visualizations import run_extended_visualizations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.file_utils import ensure_dir, save_csv
from utils.paths import APPLE_CLEAN_DIR, MERGED_DIR, OUTPUT_DIR, PLOTS_DIR, STOIC_CLEAN_DIR, ensure_project_folders
from utils.plotting import (
    build_targeted_correlation_matrix,
    plot_association_bars,
    plot_bar_counts,
    plot_box_by_category,
    plot_heatmap,
    plot_scatter_with_trend,
    plot_timeline,
    plot_dual_rolling_timeline,
    plot_category_mean_bar,
    pretty_label,
    summarize_binary_vs_outcome,
)
from utils.reports import write_available_columns_report, write_plot_inventory


ANALYSIS_DIR = OUTPUT_DIR / "analysis"
VIS_SUMMARY_DIR = OUTPUT_DIR / "reports"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def make_plot_folders() -> dict[str, Path]:
    folders = {
        "mood": PLOTS_DIR / "mood",
        "triggers": PLOTS_DIR / "triggers",
        "thoughts": PLOTS_DIR / "thoughts",
        "health": PLOTS_DIR / "health",
        "activity": PLOTS_DIR / "activity",
        "relationships": PLOTS_DIR / "relationships",
        "recovery": PLOTS_DIR / "recovery",
        "associations": PLOTS_DIR / "associations",
        "overview": PLOTS_DIR / "overview",
        "debug": PLOTS_DIR / "debug",
    }
    for folder in folders.values():
        ensure_dir(folder)
    ensure_dir(VIS_SUMMARY_DIR)
    return folders


def load_visualization_data() -> dict[str, pd.DataFrame]:
    return {
        "master": read_csv_if_exists(MERGED_DIR / "master_daily.csv"),
        "enriched": read_csv_if_exists(MERGED_DIR / "correlation_ready_daily_enriched.csv"),
        "stoic_daily": read_csv_if_exists(STOIC_CLEAN_DIR / "stoic_mental_health_daily.csv"),
        "triggers_long": read_csv_if_exists(STOIC_CLEAN_DIR / "stoic_triggers_long.csv"),
        "thoughts_long": read_csv_if_exists(STOIC_CLEAN_DIR / "stoic_automatic_thoughts_long.csv"),
        "trigger_thought_pairs": read_csv_if_exists(STOIC_CLEAN_DIR / "stoic_trigger_thought_pairs.csv"),
        "apple_sleep": read_csv_if_exists(APPLE_CLEAN_DIR / "apple_daily_sleep.csv"),
        "apple_activity": read_csv_if_exists(APPLE_CLEAN_DIR / "apple_daily_activity.csv"),
        "apple_workouts": read_csv_if_exists(APPLE_CLEAN_DIR / "apple_daily_workouts.csv"),
        "group_comparisons": read_csv_if_exists(ANALYSIS_DIR / "group_comparisons" / "days_with_vs_without.csv"),
        "mood_correlations": read_csv_if_exists(ANALYSIS_DIR / "associations" / "mood_focused_correlations.csv"),
        "top_positive": read_csv_if_exists(ANALYSIS_DIR / "predictors" / "top_positive_associations_lowest_mood.csv"),
        "top_negative": read_csv_if_exists(ANALYSIS_DIR / "predictors" / "top_negative_associations_lowest_mood.csv"),
    }


def _choose_daily(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    enriched = datasets.get("enriched", pd.DataFrame())
    if not enriched.empty:
        return enriched
    return datasets.get("master", pd.DataFrame())


def _summary_line_for_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return f"{pretty_label(x_col)} vs {pretty_label(y_col)}: no usable data yet."
    temp = df[[x_col, y_col]].copy()
    temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    temp = temp.dropna()
    if len(temp) < 3 or temp[x_col].nunique() < 2 or temp[y_col].nunique() < 2:
        return f"{pretty_label(x_col)} vs {pretty_label(y_col)}: not enough usable paired days yet."
    r = temp[x_col].corr(temp[y_col])
    return f"{pretty_label(x_col)} vs {pretty_label(y_col)}: Pearson r = {r:.2f} across {len(temp)} paired days."


def question_mood_timeline(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: How has my lowest mood changed over time?"""
    daily = datasets["stoic_daily"] if not datasets["stoic_daily"].empty else _choose_daily(datasets)
    made = plot_timeline(
        daily,
        "date",
        "lowest_mood_score",
        "Mood timeline: lowest mood over time",
        "Lowest mood score",
        folders["mood"] / "lowest_mood_timeline.png",
    )
    return "Created lowest mood timeline." if made else "Could not create mood timeline yet."


def question_trigger_frequency(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: Which triggers show up most often?"""
    triggers = datasets["triggers_long"]
    if triggers.empty or "trigger" not in triggers.columns:
        return "No trigger-frequency plot yet because trigger labels were not found."
    counts = triggers["trigger"].value_counts()
    save_csv(counts.reset_index().rename(columns={"index": "trigger", "trigger": "count"}), folders["triggers"] / "trigger_frequency_table.csv")
    plot_bar_counts(counts, "Trigger frequency", "Count", "Trigger", folders["triggers"] / "trigger_frequency.png")
    return f"Created trigger-frequency plot using {int(counts.sum())} trigger entries."




def question_trigger_vs_lowest_mood(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: Which triggers are associated with lower or higher lowest mood?"""
    triggers = datasets["triggers_long"]
    if triggers.empty or not {"trigger", "lowest_mood_score"}.issubset(triggers.columns):
        return "No trigger vs lowest-mood plot yet because trigger/mood rows were not found."

    temp = triggers[["trigger", "lowest_mood_score"]].copy().dropna()
    temp["lowest_mood_score"] = pd.to_numeric(temp["lowest_mood_score"], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return "No trigger vs lowest-mood plot yet because there are no usable mood scores by trigger."

    summary = (
        temp.groupby("trigger")["lowest_mood_score"]
        .agg(n_days="count", average_lowest_mood="mean", median_lowest_mood="median")
        .reset_index()
        .sort_values("average_lowest_mood")
    )
    save_csv(summary, folders["triggers"] / "trigger_vs_lowest_mood_summary.csv")

    made_box = plot_box_by_category(
        temp,
        "trigger",
        "lowest_mood_score",
        "Lowest mood distribution by trigger",
        folders["triggers"] / "trigger_vs_lowest_mood_boxplot.png",
    )
    made_bar = plot_category_mean_bar(
        temp,
        "trigger",
        "lowest_mood_score",
        "Average lowest mood by trigger",
        folders["triggers"] / "trigger_vs_lowest_mood_average.png",
    )
    if made_box or made_bar:
        return f"Created trigger vs lowest-mood plots/table for {summary.shape[0]} triggers."
    return "Created trigger vs lowest-mood summary table; not enough data for plot yet."

def question_trigger_vs_automatic_thoughts(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: Which automatic thoughts tend to appear with which triggers?"""
    pairs = datasets["trigger_thought_pairs"]
    if pairs.empty or not {"trigger", "automatic_thought"}.issubset(pairs.columns):
        return "No trigger vs automatic-thought table yet because paired trigger/thought rows were not found."

    table = pd.crosstab(pairs["trigger"], pairs["automatic_thought"])
    save_csv(table.reset_index(), folders["triggers"] / "trigger_vs_automatic_thoughts_table.csv")
    if not table.empty:
        # Co-occurrence heatmap uses counts, not correlations.
        import matplotlib.pyplot as plt

        sub = table.loc[
            table.sum(axis=1).sort_values(ascending=False).head(12).index,
            table.sum(axis=0).sort_values(ascending=False).head(12).index,
        ]
        plt.figure(figsize=(max(8, 0.6 * sub.shape[1]), max(5, 0.55 * sub.shape[0])))
        im = plt.imshow(sub.values, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Trigger vs automatic thoughts")
        plt.xticks(range(sub.shape[1]), [pretty_label(c) for c in sub.columns], rotation=45, ha="right")
        plt.yticks(range(sub.shape[0]), [pretty_label(i) for i in sub.index])
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                val = int(sub.iloc[i, j])
                if val:
                    plt.text(j, i, str(val), ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(folders["triggers"] / "trigger_vs_automatic_thoughts_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
    return "Created trigger vs automatic-thought co-occurrence table/heatmap."


def question_automatic_thoughts_and_mood(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: Which automatic thoughts are most common, and how do they relate to lowest mood?"""
    thoughts = datasets["thoughts_long"]
    if thoughts.empty or "automatic_thought" not in thoughts.columns:
        return "No automatic-thought plots yet because automatic-thought labels were not found."
    counts = thoughts["automatic_thought"].value_counts()
    save_csv(counts.reset_index().rename(columns={"index": "automatic_thought", "automatic_thought": "count"}), folders["thoughts"] / "automatic_thought_frequency_table.csv")
    plot_bar_counts(counts, "Automatic thoughts frequency", "Count", "Automatic thought", folders["thoughts"] / "automatic_thought_frequency.png")
    made_box = plot_box_by_category(
        thoughts,
        "automatic_thought",
        "lowest_mood_score",
        "Lowest mood by automatic thought",
        folders["thoughts"] / "automatic_thoughts_vs_lowest_mood.png",
    )
    return "Created automatic-thought frequency and mood plot." if made_box else "Created automatic-thought frequency plot; not enough mood data for boxplot yet."


def question_activity_vs_mood(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: Are more active days associated with better lowest mood?"""
    daily = _choose_daily(datasets)
    outputs = []
    for x_col, name in [
        ("activity_steps", "steps_vs_lowest_mood.png"),
        ("activity_active_energy", "active_energy_vs_lowest_mood.png"),
        ("activity_walking_running_distance", "walking_distance_vs_lowest_mood.png"),
    ]:
        made = plot_scatter_with_trend(
            daily,
            x_col,
            "lowest_mood_score",
            f"{pretty_label(x_col)} vs lowest mood",
            folders["activity"] / name,
        )
        if made:
            outputs.append(_summary_line_for_scatter(daily, x_col, "lowest_mood_score"))
    return "\n".join(outputs) if outputs else "No activity vs mood plots yet because there are not enough paired activity/mood days."


def question_sleep_vs_mood(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: How do total hours asleep relate to lowest mood?"""
    daily = _choose_daily(datasets)
    made = plot_scatter_with_trend(
        daily,
        "sleep_hours_asleep",
        "lowest_mood_score",
        "Total hours asleep vs lowest mood",
        folders["health"] / "sleep_hours_asleep_vs_lowest_mood.png",
        x_label="Total hours asleep",
        y_label="Lowest mood score",
    )
    return _summary_line_for_scatter(daily, "sleep_hours_asleep", "lowest_mood_score") if made else "No sleep vs mood plot yet. Run 02/03/04 again so sleep_hours_asleep exists, then rerun file 5."




def question_hrv_patterns(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: How does HRV relate to mood, security, and longer-term trends?"""
    daily = _choose_daily(datasets)
    outputs = []

    made = plot_scatter_with_trend(
        daily,
        "heart_hrv_sdnn",
        "lowest_mood_score",
        "HRV vs lowest mood",
        folders["health"] / "hrv_vs_lowest_mood.png",
        x_label="HRV SDNN",
        y_label="Lowest mood score",
    )
    if made:
        outputs.append(_summary_line_for_scatter(daily, "heart_hrv_sdnn", "lowest_mood_score"))

    made = plot_scatter_with_trend(
        daily,
        "heart_hrv_sdnn",
        "relationship_security_score",
        "HRV vs relationship security",
        folders["health"] / "hrv_vs_relationship_security.png",
        x_label="HRV SDNN",
        y_label="Relationship security score",
    )
    if made:
        outputs.append(_summary_line_for_scatter(daily, "heart_hrv_sdnn", "relationship_security_score"))

    made = plot_timeline(
        daily,
        "date",
        "heart_hrv_sdnn",
        "HRV trend over time",
        "HRV SDNN",
        folders["health"] / "hrv_trend_over_time.png",
        rolling=7,
    )
    if made:
        outputs.append("Created HRV trend over time with a 7-day rolling average.")

    made = plot_dual_rolling_timeline(
        daily,
        "date",
        "heart_hrv_sdnn",
        "lowest_mood_score",
        "7-day rolling HRV vs 7-day rolling lowest mood",
        folders["health"] / "rolling_hrv_vs_rolling_lowest_mood.png",
        rolling=7,
        y1_label="HRV SDNN",
        y2_label="Lowest mood score",
    )
    if made:
        outputs.append("Created 7-day rolling HRV vs 7-day rolling mood plot.")

    return "\n".join(outputs) if outputs else "No HRV plots yet because there are not enough paired HRV/mood days."

def question_exercise_vs_mood(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: Are exercise minutes or workout days associated with better lowest mood?"""
    daily = _choose_daily(datasets)
    made = plot_scatter_with_trend(
        daily,
        "activity_exercise_minutes",
        "lowest_mood_score",
        "Exercise minutes vs lowest mood",
        folders["activity"] / "exercise_minutes_vs_lowest_mood.png",
    )

    group = datasets["group_comparisons"]
    summary = ""
    if not group.empty:
        subset = group[(group["feature"].isin(["exercised_day", "workout_day", "high_exercise_day"])) & (group["outcome"] == "lowest_mood_score")]
        if not subset.empty:
            save_csv(subset, folders["activity"] / "exercise_day_mood_summary.csv")
            summary = f" Exercise comparison table saved with {len(subset)} rows."
    return (_summary_line_for_scatter(daily, "activity_exercise_minutes", "lowest_mood_score") if made else "No exercise scatter plot yet.") + summary


def question_relationship_security_vs_mood(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: How does relationship security relate to lowest mood?"""
    daily = _choose_daily(datasets)
    plot_timeline(
        daily,
        "date",
        "relationship_security_score",
        "Relationship security over time",
        "Relationship security score",
        folders["relationships"] / "relationship_security_timeline.png",
    )
    made = plot_scatter_with_trend(
        daily,
        "relationship_security_score",
        "lowest_mood_score",
        "Relationship security vs lowest mood",
        folders["relationships"] / "relationship_security_vs_lowest_mood.png",
    )
    return _summary_line_for_scatter(daily, "relationship_security_score", "lowest_mood_score") if made else "No relationship-security mood plot yet."


def question_recovery_vs_duration(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: Which recovery methods are linked with shorter/longer low-mood duration?"""
    group = datasets["group_comparisons"]
    if group.empty:
        return "No recovery-duration summary yet because group comparisons were not found."
    subset = group[(group["feature_group"] == "recovery") & (group["outcome"] == "lowest_mood_duration_score")].copy()
    if subset.empty:
        return "No recovery-duration summary yet because recovery method features were not found."
    subset = subset.sort_values("abs_difference", ascending=False)
    save_csv(subset, folders["recovery"] / "recovery_method_vs_lowest_mood_duration.csv")
    plot_association_bars(
        subset,
        "Recovery method vs lowest-mood duration",
        folders["recovery"] / "recovery_method_vs_lowest_mood_duration.png",
        value_col="difference_with_minus_without",
    )
    return f"Created recovery-duration summary using {len(subset)} recovery comparisons."


def question_top_positive_negative_associations(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: What factors are associated with better or worse lowest mood?"""
    positive = datasets["top_positive"]
    negative = datasets["top_negative"]
    messages = []
    if not positive.empty:
        save_csv(positive, folders["associations"] / "top_positive_associations_lowest_mood.csv")
        plot_association_bars(
            positive,
            "Top positive associations with lowest mood",
            folders["associations"] / "top_positive_associations_lowest_mood.png",
        )
        messages.append(f"Top positive associations: {len(positive)} rows.")
    if not negative.empty:
        save_csv(negative, folders["associations"] / "top_negative_associations_lowest_mood.csv")
        plot_association_bars(
            negative,
            "Top negative associations with lowest mood",
            folders["associations"] / "top_negative_associations_lowest_mood.png",
        )
        messages.append(f"Top negative associations: {len(negative)} rows.")
    return " ".join(messages) if messages else "No positive/negative association plots yet. Run file 4 first."


def question_targeted_heatmap(datasets: dict[str, pd.DataFrame], folders: dict[str, Path]) -> str:
    """Question: How do key health/trigger/thought variables relate to mood outcomes?"""
    daily = _choose_daily(datasets)
    if daily.empty:
        return "No targeted heatmap yet because daily data was not found."

    row_vars = [c for c in [
        "lowest_mood_score",
        "relationship_security_score",
        "lowest_mood_duration_score",
    ] if c in daily.columns]

    # Purposefully selected columns. No count variables, no sleep stages, no Apple ring goals.
    candidate_cols = [
        "sleep_hours_asleep",
        "activity_steps",
        "activity_exercise_minutes",
        "activity_walking_running_distance",
        "heart_hrv_sdnn",
        "heart_resting_heart_rate",
        "trigger_loneliness",
        "trigger_relationship_uncertainty",
        "trigger_nothing_obvious",
        "symptom_low_energy",
        "symptom_hopelessness",
        "symptom_lack_of_interest",
        "symptom_trouble_concentrating",
        "symptom_appetite_change",
        "automatic_thought_any",
    ]
    col_vars = [c for c in candidate_cols if c in daily.columns]
    matrix = build_targeted_correlation_matrix(daily, row_vars=row_vars, col_vars=col_vars, min_n=3)
    if matrix.empty:
        return "No targeted heatmap yet because there are not enough paired data points."
    save_csv(matrix.reset_index(), folders["associations"] / "targeted_mood_association_matrix.csv")
    plot_heatmap(matrix, "Targeted mood association heatmap", folders["associations"] / "targeted_mood_association_heatmap.png")
    return "Created targeted heatmap focused on mood outcomes vs selected health/trigger/thought variables."


def write_visualization_summary(sections: list[tuple[str, str]]) -> None:
    lines = ["# Visualization Summary", ""]
    lines.append("Each section answers one practical question. These are exploratory and should be interpreted as patterns, not proof of cause.")
    lines.append("")
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    (VIS_SUMMARY_DIR / "visualization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_project_folders()
    folders = make_plot_folders()
    print("=== 05 Question-Based Visualizations ===")

    datasets = load_visualization_data()
    write_available_columns_report(datasets, folders["debug"] / "available_columns.csv")

    sections = [
        ("Mood timeline", question_mood_timeline(datasets, folders)),
        ("Trigger frequency", question_trigger_frequency(datasets, folders)),
        ("Trigger vs lowest mood", question_trigger_vs_lowest_mood(datasets, folders)),
        ("Trigger vs automatic thoughts", question_trigger_vs_automatic_thoughts(datasets, folders)),
        ("Automatic thoughts and mood", question_automatic_thoughts_and_mood(datasets, folders)),
        ("Activity vs mood", question_activity_vs_mood(datasets, folders)),
        ("Sleep vs mood", question_sleep_vs_mood(datasets, folders)),
        ("HRV patterns", question_hrv_patterns(datasets, folders)),
        ("Exercise vs mood", question_exercise_vs_mood(datasets, folders)),
        ("Relationship security vs mood", question_relationship_security_vs_mood(datasets, folders)),
        ("Recovery vs mood duration", question_recovery_vs_duration(datasets, folders)),
        ("Top positive and negative associations", question_top_positive_negative_associations(datasets, folders)),
        ("Targeted mood association heatmap", question_targeted_heatmap(datasets, folders)),
    ]

    # Overview duplicates for quick access.
    daily = _choose_daily(datasets)
    plot_timeline(daily, "date", "lowest_mood_score", "Overview: lowest mood", "Lowest mood score", folders["overview"] / "overview_lowest_mood.png")
    triggers = datasets["triggers_long"]
    if not triggers.empty and "trigger" in triggers.columns:
        plot_bar_counts(triggers["trigger"].value_counts(), "Overview: trigger frequency", "Count", "Trigger", folders["overview"] / "overview_trigger_frequency.png")

    write_visualization_summary(sections)
    write_plot_inventory(PLOTS_DIR, folders["debug"] / "plot_inventory.csv")
    print(f"Visualizations saved to: {PLOTS_DIR}")
    print(f"Summary saved to: {VIS_SUMMARY_DIR / 'visualization_report.md'}")
    run_extended_visualizations()
    print("Finished 05_visualizations.py")


if __name__ == "__main__":
    main()

