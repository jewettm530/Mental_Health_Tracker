"""06_dashboard.py

Streamlit dashboard for the Health Tracker project.

Run from the project root:
    streamlit run scripts/06_dashboard.py

This dashboard reads files produced by:
    01_import_data.py
    02_process_data.py
    03_merge_data.py
    04_analysis.py
    05_visualizations.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.paths import APPLE_CLEAN_DIR, MERGED_DIR, OUTPUT_DIR, PLOTS_DIR, STOIC_CLEAN_DIR, ensure_project_folders
from utils.dashboard import (
    choose_daily_dataset,
    describe_dataset,
    format_option,
    get_variable_options,
    list_plot_files,
    load_dashboard_data,
    pretty_label,
    summarize_selected_pair,
    top_table,
    variable_group,
)
from utils.dashboard_style import (
    apply_dashboard_style,
    confidence_badge,
    render_hero,
    render_watch_card,
    section_header,
)


st.set_page_config(
    page_title="Health Tracker Dashboard",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data_cached() -> dict[str, pd.DataFrame]:
    return load_dashboard_data(MERGED_DIR, OUTPUT_DIR, STOIC_CLEAN_DIR, APPLE_CLEAN_DIR)


def show_png_folder(topic: str, caption_prefix: str | None = None) -> None:
    files = list_plot_files(PLOTS_DIR, topic)
    if not files:
        st.info(f"No generated plots found in data/outputs/plots/{topic}/ yet. Run `python3 scripts/05_visualizations.py` first.")
        return
    for i in range(0, len(files), 2):
        cols = st.columns(2, gap="large")
        for col, file_path in zip(cols, files[i : i + 2]):
            with col:
                with st.container(border=True):
                    caption = caption_prefix or pretty_label(file_path.stem)
                    st.markdown(f"**{caption}**")
                    st.image(str(file_path), width="stretch")


def show_dataframe_if_available(df: pd.DataFrame, label: str, n: int = 20) -> None:
    if df.empty:
        st.info(f"No {label} data available yet.")
    else:
        st.dataframe(top_table(df, n), width="stretch")


def render_overview(datasets: dict[str, pd.DataFrame], daily: pd.DataFrame) -> None:
    section_header("Overview", "A quick view of your data coverage and strongest current patterns.")
    if daily.empty:
        st.warning("No merged daily data found. Run files 1–3 first.")
        return

    min_date = pd.to_datetime(daily["date"], errors="coerce").min() if "date" in daily.columns else None
    max_date = pd.to_datetime(daily["date"], errors="coerce").max() if "date" in daily.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily rows", f"{len(daily):,}")
    c2.metric("Variables", f"{daily.shape[1]:,}")
    c3.metric("Start date", "—" if pd.isna(min_date) else min_date.date().isoformat())
    c4.metric("End date", "—" if pd.isna(max_date) else max_date.date().isoformat())

    st.markdown("### Quick tables")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top positive associations with lowest mood**")
        show_dataframe_if_available(datasets.get("top_positive", pd.DataFrame()), "positive association", n=10)
    with c2:
        st.markdown("**Top negative associations with lowest mood**")
        show_dataframe_if_available(datasets.get("top_negative", pd.DataFrame()), "negative association", n=10)

    st.markdown("### Overview plots")
    show_png_folder("overview")


def render_topic_tab(topic: str, title: str, description: str | None = None) -> None:
    section_header(title, description)
    show_png_folder(topic)


def render_triggers_thoughts(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Triggers & thoughts", "Trigger frequency, trigger vs lowest mood, and trigger/thought co-occurrence.")
    show_png_folder("triggers")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Triggers long table**")
        show_dataframe_if_available(datasets.get("triggers_long", pd.DataFrame()), "trigger", n=25)
    with c2:
        st.markdown("**Automatic thoughts long table**")
        show_dataframe_if_available(datasets.get("thoughts_long", pd.DataFrame()), "automatic thought", n=25)

    st.markdown("### Thought plots")
    show_png_folder("thoughts")


def render_association_explorer(daily: pd.DataFrame) -> None:
    section_header("Variable explorer", "Choose two variables and let the dashboard select the clearest comparison method.")
    st.markdown(
        "Select any two daily variables. The dashboard will choose the most useful summary: "
        "correlation/scatter for numeric pairs, with-vs-without summaries for yes/no factors, "
        "or co-occurrence for two binary factors."
    )

    if daily.empty:
        st.warning("No merged daily data found.")
        return

    include_counts = st.checkbox("Include count variables", value=False, help="Count variables can be useful sometimes, but they are often less insightful than specific triggers/symptoms or health metrics.")
    variables = get_variable_options(daily, include_counts=include_counts)
    if len(variables) < 2:
        st.warning("Not enough usable variables found for comparison.")
        return

    default_a = "sleep_hours_asleep" if "sleep_hours_asleep" in variables else variables[0]
    default_b = "lowest_mood_score" if "lowest_mood_score" in variables else variables[min(1, len(variables) - 1)]

    c1, c2 = st.columns(2)
    with c1:
        var_a = st.selectbox("Variable 1", variables, index=variables.index(default_a), format_func=format_option)
    with c2:
        var_b = st.selectbox("Variable 2", variables, index=variables.index(default_b), format_func=format_option)

    result = summarize_selected_pair(daily, var_a, var_b)

    st.markdown(result.summary_markdown)
    if result.figure is not None:
        st.pyplot(result.figure, clear_figure=True)
    if not result.table.empty:
        st.dataframe(result.table, width="stretch")

    with st.expander("Variable details"):
        st.write(
            {
                "variable_1": var_a,
                "variable_1_group": variable_group(var_a),
                "variable_2": var_b,
                "variable_2_group": variable_group(var_b),
            }
        )



def _fmt(value, digits=2):
    if pd.isna(value):
        return "Not available"
    return f"{value:.{digits}f}"


def render_summary_tab(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Readable mood summaries", "Each card explains what the statistics mean and how much confidence to place in the result.")
    df = datasets.get("readable_summaries", pd.DataFrame())
    if df.empty:
        st.info("No readable summaries yet. Run file 4 again after collecting check-in data.")
        return
    confidence_filter = st.multiselect("Confidence levels", sorted(df["confidence"].dropna().unique()), default=sorted(df["confidence"].dropna().unique()))
    shown = df[df["confidence"].isin(confidence_filter)] if confidence_filter else df
    for _, row in shown.iterrows():
        with st.expander(str(row["factor_label"]), expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Days with factor", int(row["days_with"]))
            c2.metric("Mood with factor", _fmt(row["average_mood_with"]))
            c3.metric("Mood without factor", _fmt(row["average_mood_without"]))
            st.markdown(f"**What this means:** {row['plain_language_meaning']}")
            st.markdown(f"**Correlation:** {_fmt(row['correlation'])} — {row['correlation_meaning']}")
            if pd.notna(row.get("p_value")):
                st.markdown(f"**p-value:** {_fmt(row['p_value'], 4)}")
            if pd.notna(row.get("ci_low")) and pd.notna(row.get("ci_high")):
                st.markdown(f"**95% confidence interval for the difference:** [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]")
            st.markdown(f'{confidence_badge(str(row["confidence"]))} <span class="ht-muted">{row["confidence_note"]}</span>', unsafe_allow_html=True)


def render_predictions_tab(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("What predicts my best and worst days?", "These are associations, not proof that one factor caused the mood change.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Factors associated with worse days")
        show_dataframe_if_available(datasets.get("worst_day_predictors", pd.DataFrame()), "worst-day predictor", 15)
    with c2:
        st.markdown("### Factors associated with better days")
        show_dataframe_if_available(datasets.get("best_day_predictors", pd.DataFrame()), "best-day predictor", 15)
    show_png_folder("predictions")


def render_recovery_insights(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Recovery effectiveness", "Compare recovery methods using the duration information available in your check-ins.")
    df = datasets.get("recovery_effectiveness", pd.DataFrame())
    if df.empty:
        st.info("Recovery effectiveness needs recovery methods plus a usable duration measure.")
    else:
        st.dataframe(df, width="stretch")
    show_png_folder("recovery")


def render_sleep_quality(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Sleep quality", "Compares average lowest mood across total-hours-asleep ranges.")
    show_dataframe_if_available(datasets.get("sleep_quality", pd.DataFrame()), "sleep quality", 20)
    show_png_folder("health")


def render_lagged_effects(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Lagged effects", "Looks at whether today’s sleep, activity, HRV, or security is associated with tomorrow’s lowest mood.")
    df = datasets.get("lagged_effects", pd.DataFrame())
    if df.empty:
        st.info("Not enough consecutive tracked days yet for lagged analysis.")
    else:
        st.dataframe(df, width="stretch")
        for _, row in df.dropna(subset=["correlation"]).iterrows():
            st.markdown(f"- **{row['predictor_label']}**: {row['plain_language_meaning']} Correlation = {_fmt(row['correlation'])}; confidence = {row['confidence']}.")
    show_png_folder("lagged")


def render_consistency(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Consistency and variability", "Track whether mood, sleep, HRV, and relationship security are becoming more or less stable over time.")
    st.caption(
        "This page shows the newest months first and includes only months with at least "
        "three recorded lowest-mood scores. A lower standard deviation means the variable "
        "was more stable during that month."
    )

    df = datasets.get("consistency", pd.DataFrame()).copy()
    if df.empty:
        st.info(
            "No months currently meet the minimum mood-tracking requirement. "
            "This section will populate after at least three mood check-ins occur in one month."
        )
        return

    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        df = df.sort_values(["month", "variable_label"], ascending=[False, True])
        df["month"] = df["month"].dt.strftime("%b %Y")

    preferred = [
        "month", "variable_label", "mood_days_in_month", "n_days",
        "mean", "standard_deviation", "range",
    ]
    shown_cols = [c for c in preferred if c in df.columns]
    st.dataframe(df[shown_cols], width="stretch", hide_index=True)

    st.markdown("### Variability trends")
    st.caption(
        "These plots are separated by topic because mood, sleep, HRV, and relationship "
        "security use different scales. The direction over time matters more than comparing "
        "their raw standard-deviation values to one another."
    )
    show_png_folder("consistency")


def render_baselines(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Personalized baselines", "Compares your most recent seven available days with your own overall average—not a generic population cutoff.")
    show_dataframe_if_available(datasets.get("personal_baselines", pd.DataFrame()), "personal baseline", 30)
    show_png_folder("baselines")


def render_things_to_watch(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Things to watch", "The strongest current clues, next-day patterns, and recent departures from your baseline.")
    df = datasets.get("things_to_watch", pd.DataFrame())
    if df.empty:
        st.info("No watch items yet. They will appear as more mental-health days are tracked.")
        return
    for _, row in df.iterrows():
        render_watch_card(
            headline=str(row["headline"]),
            message=str(row["message"]),
            category=str(row["category"]),
            confidence=str(row["confidence"]),
        )

def render_data_inventory(daily: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Data inventory", "See what is available, how complete each variable is, and which source it belongs to.")
    if daily.empty:
        st.warning("No daily data found.")
        return
    inventory = describe_dataset(daily)

    group_filter = st.multiselect(
        "Filter by variable group",
        sorted(inventory["group"].dropna().unique()),
        default=sorted(inventory["group"].dropna().unique()),
    )
    filtered = inventory[inventory["group"].isin(group_filter)] if group_filter else inventory
    st.dataframe(filtered, width="stretch")

    st.markdown("### Loaded dataset shapes")
    shapes = pd.DataFrame(
        [{"dataset": key, "rows": df.shape[0], "columns": df.shape[1]} for key, df in datasets.items()]
    ).sort_values("dataset")
    st.dataframe(shapes, width="stretch")


def main() -> None:
    ensure_project_folders()
    apply_dashboard_style()
    render_hero(
        "Health Tracker",
        "A private, personal view of mood, sleep, activity, relationships, triggers, recovery, and physiological patterns.",
        eyebrow="Mental health & wellbeing",
    )

    with st.sidebar:
        st.markdown("## Health Tracker")
        st.caption("Dashboard controls and pipeline status")
        if st.button("↻ Refresh dashboard data", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        with st.expander("Pipeline run order"):
            st.code(
                "python3 scripts/01_import_data.py\n"
                "python3 scripts/02_process_data.py\n"
                "python3 scripts/03_merge_data.py\n"
                "python3 scripts/04_analysis.py\n"
                "python3 scripts/05_visualizations.py"
            )
        st.markdown("---")
        st.caption("Your data stays on your computer when you run this dashboard locally.")

    datasets = load_data_cached()
    daily = choose_daily_dataset(datasets)

    tabs = st.tabs(
        [
            "⌂ Overview", "✦ Summary", "↗ Predictions", "◉ Mood", "♡ Health", "☾ Sleep Quality",
            "◌ Activity", "∞ Relationships", "◇ Triggers & Thoughts", "↺ Recovery",
            "→ Lagged Effects", "≈ Consistency", "— Personal Baselines", "! Things to Watch",
            "⌕ Associations Explorer", "▦ Data Inventory",
        ]
    )

    with tabs[0]: render_overview(datasets, daily)
    with tabs[1]: render_summary_tab(datasets)
    with tabs[2]: render_predictions_tab(datasets)
    with tabs[3]: render_topic_tab("mood", "Mood", "Mood timeline, lowest mood patterns, and mood-focused summaries.")
    with tabs[4]: render_topic_tab("health", "Health", "HRV, heart, respiratory, and general health plots.")
    with tabs[5]: render_sleep_quality(datasets)
    with tabs[6]: render_topic_tab("activity", "Activity", "Steps, exercise, workouts, and activity versus mood.")
    with tabs[7]: render_topic_tab("relationships", "Relationships", "Relationship security and mood-related relationship patterns.")
    with tabs[8]: render_triggers_thoughts(datasets)
    with tabs[9]: render_recovery_insights(datasets)
    with tabs[10]: render_lagged_effects(datasets)
    with tabs[11]: render_consistency(datasets)
    with tabs[12]: render_baselines(datasets)
    with tabs[13]: render_things_to_watch(datasets)
    with tabs[14]: render_association_explorer(daily)
    with tabs[15]: render_data_inventory(daily, datasets)


if __name__ == "__main__":
    main()
