"""Reusable plotting helpers for Health_Tracker.

The visualization layer should answer specific questions, not create generic plots
just because columns exist. These helpers stay simple and defensive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.file_utils import ensure_dir


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def pretty_label(col: str) -> str:
    text = str(col).replace("_", " ").strip().title()
    text = text.replace("Hrv", "HRV").replace("Sdnn", "SDNN")
    text = text.replace("Hr", "HR")
    return text


def save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _prep_xy(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()
    temp = df[[x_col, y_col]].copy()
    temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    return temp.dropna()


def correlation_text(temp: pd.DataFrame, x_col: str, y_col: str) -> str:
    if len(temp) < 3 or temp[x_col].nunique() < 2 or temp[y_col].nunique() < 2:
        return "not enough data"
    return f"Pearson r = {temp[x_col].corr(temp[y_col]):.2f}, n = {len(temp)}"


# -----------------------------------------------------------------------------
# Plot types
# -----------------------------------------------------------------------------


def plot_bar_counts(counts: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path, top_n: int = 25) -> bool:
    counts = counts.dropna()
    counts = counts[counts > 0].sort_values(ascending=False).head(top_n)
    if counts.empty:
        return False

    plt.figure(figsize=(10, max(4, 0.35 * len(counts))))
    y = np.arange(len(counts))
    plt.barh(y, counts.values)
    plt.yticks(y, [pretty_label(x) for x in counts.index])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(output_path)
    return True


def plot_timeline(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    rolling: int | None = 7,
) -> bool:
    if df.empty or date_col not in df.columns or y_col not in df.columns:
        return False
    temp = df[[date_col, y_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    temp = temp.dropna().sort_values(date_col)
    if temp.empty:
        return False

    plt.figure(figsize=(12, 5))
    plt.plot(temp[date_col], temp[y_col], marker="o", linewidth=1, label=ylabel)
    if rolling and len(temp) >= rolling:
        temp[f"{y_col}_rolling"] = temp[y_col].rolling(rolling, min_periods=2).mean()
        plt.plot(temp[date_col], temp[f"{y_col}_rolling"], linewidth=2, label=f"{rolling}-day rolling average")
        plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    save_fig(output_path)
    return True


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    x_label: str | None = None,
    y_label: str | None = None,
) -> bool:
    temp = _prep_xy(df, x_col, y_col)
    if len(temp) < 3:
        return False

    plt.figure(figsize=(7, 5))
    plt.scatter(temp[x_col], temp[y_col], alpha=0.75)
    if temp[x_col].nunique() > 1:
        m, b = np.polyfit(temp[x_col], temp[y_col], 1)
        xs = np.linspace(temp[x_col].min(), temp[x_col].max(), 100)
        plt.plot(xs, m * xs + b, linewidth=2)

    plt.title(f"{title}\n{correlation_text(temp, x_col, y_col)}")
    plt.xlabel(x_label or pretty_label(x_col))
    plt.ylabel(y_label or pretty_label(y_col))
    save_fig(output_path)
    return True


def plot_box_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> bool:
    if df.empty or category_col not in df.columns or value_col not in df.columns:
        return False
    temp = df[[category_col, value_col]].copy().dropna()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return False

    top_categories = temp[category_col].value_counts().head(top_n).index
    temp = temp[temp[category_col].isin(top_categories)]
    grouped = list(temp.groupby(category_col))
    if not grouped:
        return False

    labels = [name for name, _ in grouped]
    values = [grp[value_col].values for _, grp in grouped]

    plt.figure(figsize=(max(8, 0.6 * len(labels)), 5))
    plt.boxplot(values, labels=[pretty_label(x) for x in labels], vert=True)
    plt.title(title)
    plt.xlabel(pretty_label(category_col))
    plt.ylabel(pretty_label(value_col))
    plt.xticks(rotation=45, ha="right")
    save_fig(output_path)
    return True


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path, annotate: bool = True) -> bool:
    if matrix.empty:
        return False
    matrix = matrix.copy().dropna(how="all").dropna(axis=1, how="all")
    if matrix.empty:
        return False

    plt.figure(figsize=(max(8, 0.55 * matrix.shape[1]), max(5, 0.55 * matrix.shape[0])))
    im = plt.imshow(matrix.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xticks(range(matrix.shape[1]), [pretty_label(c) for c in matrix.columns], rotation=45, ha="right")
    plt.yticks(range(matrix.shape[0]), [pretty_label(i) for i in matrix.index])

    if annotate and matrix.shape[0] <= 12 and matrix.shape[1] <= 15:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                if pd.notna(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    save_fig(output_path)
    return True


def plot_association_bars(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    value_col: str = "difference_with_minus_without",
    label_col: str = "feature_label",
    top_n: int = 12,
) -> bool:
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        return False
    temp = df[[label_col, value_col]].copy().dropna()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna().head(top_n)
    if temp.empty:
        return False

    temp = temp.sort_values(value_col)
    plt.figure(figsize=(10, max(4, 0.4 * len(temp))))
    y = np.arange(len(temp))
    plt.barh(y, temp[value_col].values)
    plt.yticks(y, temp[label_col].astype(str).tolist())
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Average mood difference: days with factor minus days without")
    plt.ylabel("Factor")
    save_fig(output_path)
    return True


# -----------------------------------------------------------------------------
# Tables/summaries for question-based outputs
# -----------------------------------------------------------------------------


def summarize_binary_vs_outcome(df: pd.DataFrame, feature: str, outcome: str) -> pd.DataFrame:
    if df.empty or feature not in df.columns or outcome not in df.columns:
        return pd.DataFrame()
    temp = df[[feature, outcome]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp[outcome] = pd.to_numeric(temp[outcome], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return pd.DataFrame()

    rows = []
    for value, label in [(1, "with"), (0, "without")]:
        vals = temp.loc[temp[feature] == value, outcome]
        if len(vals) == 0:
            continue
        rows.append({
            "group": label,
            "feature": feature,
            "outcome": outcome,
            "n_days": int(len(vals)),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(vals.median()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        })
    return pd.DataFrame(rows)


def build_targeted_correlation_matrix(df: pd.DataFrame, row_vars: list[str], col_vars: list[str], min_n: int = 3) -> pd.DataFrame:
    rows = []
    for r in row_vars:
        row = {"outcome": r}
        if r not in df.columns:
            continue
        for c in col_vars:
            if c not in df.columns or c == r:
                row[c] = np.nan
                continue
            temp = _prep_xy(df, c, r)
            if len(temp) < min_n or temp[c].nunique() < 2 or temp[r].nunique() < 2:
                row[c] = np.nan
            else:
                row[c] = temp[c].corr(temp[r])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("outcome")


def plot_dual_rolling_timeline(
    df: pd.DataFrame,
    date_col: str,
    y1_col: str,
    y2_col: str,
    title: str,
    output_path: Path,
    rolling: int = 7,
    y1_label: str | None = None,
    y2_label: str | None = None,
) -> bool:
    """Plot two rolling-average time series on separate y-axes.

    This is useful when units differ, such as HRV in ms and mood score.
    """
    if df.empty or date_col not in df.columns or y1_col not in df.columns or y2_col not in df.columns:
        return False
    temp = df[[date_col, y1_col, y2_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[y1_col] = pd.to_numeric(temp[y1_col], errors="coerce")
    temp[y2_col] = pd.to_numeric(temp[y2_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).sort_values(date_col)
    if temp[[y1_col, y2_col]].dropna(how="all").empty:
        return False

    temp[f"{y1_col}_rolling"] = temp[y1_col].rolling(rolling, min_periods=2).mean()
    temp[f"{y2_col}_rolling"] = temp[y2_col].rolling(rolling, min_periods=2).mean()
    if temp[[f"{y1_col}_rolling", f"{y2_col}_rolling"]].dropna(how="all").empty:
        return False

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(temp[date_col], temp[f"{y1_col}_rolling"], linewidth=2, label=y1_label or pretty_label(y1_col))
    ax1.set_xlabel("Date")
    ax1.set_ylabel(y1_label or pretty_label(y1_col))
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(temp[date_col], temp[f"{y2_col}_rolling"], linewidth=2, linestyle="--", label=y2_label or pretty_label(y2_col))
    ax2.set_ylabel(y2_label or pretty_label(y2_col))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(title)
    save_fig(output_path)
    return True


def plot_category_mean_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> bool:
    """Plot average value by category, with n shown in the label."""
    if df.empty or category_col not in df.columns or value_col not in df.columns:
        return False
    temp = df[[category_col, value_col]].copy().dropna()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return False

    summary = (
        temp.groupby(category_col)[value_col]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
    )
    summary = summary.loc[summary["count"] > 0].head(top_n)
    if summary.empty:
        return False

    labels = [f"{pretty_label(idx)} (n={int(row['count'])})" for idx, row in summary.iterrows()]
    plt.figure(figsize=(10, max(4, 0.4 * len(summary))))
    y = np.arange(len(summary))
    plt.barh(y, summary["mean"].values)
    plt.yticks(y, labels)
    plt.title(title)
    plt.xlabel(f"Average {pretty_label(value_col)}")
    plt.ylabel(pretty_label(category_col))
    save_fig(output_path)
    return True
