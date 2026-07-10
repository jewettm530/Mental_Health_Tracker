"""Small report/inventory helpers for plot and summary generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.file_utils import ensure_dir, save_csv


def write_available_columns_report(datasets: dict[str, pd.DataFrame], output_path: Path) -> None:
    rows = []
    for name, df in datasets.items():
        if df is None or df.empty:
            rows.append({"dataset": name, "column": "<empty>", "non_null": 0})
            continue
        for col in df.columns:
            rows.append({"dataset": name, "column": col, "non_null": int(df[col].notna().sum())})
    save_csv(pd.DataFrame(rows), output_path)


def write_plot_inventory(plot_root: Path, output_path: Path) -> None:
    rows = []
    for path in sorted(plot_root.rglob("*.png")):
        rows.append({
            "folder": str(path.parent.relative_to(plot_root)),
            "file": path.name,
            "path": str(path),
        })
    save_csv(pd.DataFrame(rows), output_path)


def write_markdown_report(title: str, sections: list[tuple[str, str]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip() if body else "No usable data yet.", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "No usable data yet."
    return df.head(max_rows).to_markdown(index=False)
