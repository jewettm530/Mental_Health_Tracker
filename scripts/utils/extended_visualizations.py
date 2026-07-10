"""Plots for extended Health Tracker insights."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils.paths import OUTPUT_DIR, PLOTS_DIR
from utils.file_utils import ensure_dir

ANALYSIS = OUTPUT_DIR / "analysis"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def _save_bar(df: pd.DataFrame, label: str, value: str, title: str, xlabel: str, path: Path, n: int = 10) -> bool:
    if df.empty or label not in df.columns or value not in df.columns:
        return False
    plot_df = df[[label, value]].dropna().head(n).sort_values(value)
    if plot_df.empty:
        return False
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(9, max(4, .45 * len(plot_df))))
    ax.barh(plot_df[label], plot_df[value])
    ax.axvline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_predictions() -> list[str]:
    out = []
    pred_dir = PLOTS_DIR / "predictions"
    worst = _read(ANALYSIS / "predictions" / "what_predicts_worst_days.csv")
    best = _read(ANALYSIS / "predictions" / "what_predicts_best_days.csv")
    if _save_bar(worst, "factor_label", "mood_difference", "Factors associated with worse lowest mood", "Mood points lower/higher than days without factor", pred_dir / "worst_day_predictors.png"):
        out.append("worst_day_predictors.png")
    if _save_bar(best, "factor_label", "mood_difference", "Factors associated with better lowest mood", "Mood points lower/higher than days without factor", pred_dir / "best_day_predictors.png"):
        out.append("best_day_predictors.png")
    return out


def plot_recovery_effectiveness() -> list[str]:
    df = _read(ANALYSIS / "recovery" / "recovery_effectiveness.csv")
    path = PLOTS_DIR / "recovery" / "recovery_effectiveness.png"
    ok = _save_bar(df.sort_values("average_recovery_duration") if not df.empty else df, "recovery_method_label", "average_recovery_duration", "Average recovery duration by method", "Average duration (uses your available duration measure)", path, 15)
    return [path.name] if ok else []


def plot_sleep_quality() -> list[str]:
    df = _read(ANALYSIS / "sleep" / "sleep_quality_ranges.csv")
    if df.empty:
        return []
    path = PLOTS_DIR / "health" / "sleep_quality_ranges_vs_mood.png"
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["sleep_range"].astype(str), df["average_lowest_mood"])
    ax.set_title("Average lowest mood by sleep range")
    ax.set_xlabel("Total hours asleep")
    ax.set_ylabel("Average lowest mood score")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def plot_lagged_effects() -> list[str]:
    df = _read(ANALYSIS / "lagged" / "next_day_effects.csv")
    if df.empty:
        return []
    df = df.dropna(subset=["correlation"]).copy()
    path = PLOTS_DIR / "lagged" / "next_day_mood_associations.png"
    ok = _save_bar(df.reindex(df.correlation.abs().sort_values(ascending=False).index), "predictor_label", "correlation", "Today’s factors vs tomorrow’s lowest mood", "Correlation with next-day lowest mood", path, 15)
    return [path.name] if ok else []


def plot_consistency() -> list[str]:
    """Create focused variability trends instead of one crowded multi-scale plot."""
    df = _read(ANALYSIS / "consistency" / "monthly_consistency.csv")
    if df.empty:
        return []

    df = df.copy()
    df["month_date"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["month_date", "standard_deviation"])
    if df.empty:
        return []

    plot_specs = [
        ("lowest_mood_score", "Mood variability over time", "mood_variability.png"),
        ("relationship_security_score", "Relationship security variability over time", "relationship_security_variability.png"),
        ("sleep_hours_asleep", "Sleep-duration variability over time", "sleep_variability.png"),
        ("heart_hrv_sdnn", "HRV variability over time", "hrv_variability.png"),
    ]

    out = []
    folder = PLOTS_DIR / "consistency"
    ensure_dir(folder)
    for variable, title, filename in plot_specs:
        group = df[df["variable"] == variable].sort_values("month_date")
        if group.empty:
            continue
        path = folder / filename
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(group["month_date"], group["standard_deviation"], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly standard deviation")
        ax.grid(True, alpha=.25)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path.name)
    return out


def plot_personal_baselines() -> list[str]:
    df = _read(ANALYSIS / "baselines" / "personal_baselines.csv")
    path = PLOTS_DIR / "baselines" / "recent_vs_personal_baseline.png"
    ok = _save_bar(df, "variable_label", "percent_difference_from_baseline", "Recent 7-day average vs personal baseline", "Percent above/below personal baseline", path, 20)
    return [path.name] if ok else []


def run_extended_visualizations() -> dict[str, list[str]]:
    return {
        "predictions": plot_predictions(),
        "recovery": plot_recovery_effectiveness(),
        "sleep": plot_sleep_quality(),
        "lagged": plot_lagged_effects(),
        "consistency": plot_consistency(),
        "baselines": plot_personal_baselines(),
    }
