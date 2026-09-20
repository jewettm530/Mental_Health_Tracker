"""Streamlit dashboard for the Health Tracker project.

Run from the project root:
    streamlit run scripts/06_dashboard.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.paths import APPLE_CLEAN_DIR, MERGED_DIR, OUTPUT_DIR, PLOTS_DIR, STOIC_CLEAN_DIR, ensure_project_folders
from utils.dashboard import (
    choose_daily_dataset,
    describe_dataset,
    format_option,
    get_variable_options,
    load_dashboard_data,
    pretty_label,
    relationship_percent,
    relationship_strength,
    summarize_selected_pair,
    variable_description,
    variable_group,
    variable_source,
    variable_unit,
)
from utils.dashboard_style import apply_dashboard_style, confidence_badge, render_hero, render_watch_card, section_header

st.set_page_config(page_title="Health Tracker Dashboard", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def load_data_cached() -> dict[str, pd.DataFrame]:
    return load_dashboard_data(MERGED_DIR, OUTPUT_DIR, STOIC_CLEAN_DIR, APPLE_CLEAN_DIR)


def _fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):.{digits}f}"


def _pct(value, digits: int = 0, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}f}%"


def _corr(x: pd.Series, y: pd.Series) -> tuple[float | None, int]:
    temp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(temp) < 3 or temp.x.nunique() < 2 or temp.y.nunique() < 2:
        return None, len(temp)
    return float(temp.x.corr(temp.y)), len(temp)


def _data_support(n: int) -> str:
    if n < 10:
        return "Very low"
    if n < 20:
        return "Preliminary"
    if n < 50:
        return "Moderate"
    return "High"


def _relationship_meaning(r: float | None, feature_label: str, outcome_label: str = "lowest mood") -> str:
    if r is None or pd.isna(r):
        return "Not enough paired days yet."
    if abs(r) < .10:
        return f"No meaningful linear relationship is apparent between {feature_label.lower()} and {outcome_label}."
    direction = "better" if r > 0 else "worse"
    return f"Higher {feature_label.lower()} tended to occur with {direction} {outcome_label} scores."


def _association_display(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df.head(n).iterrows():
        factor = row.get("factor", row.get("feature", ""))
        label = row.get("factor_label", row.get("feature_label", pretty_label(str(factor))))
        days_with = row.get("days_with", row.get("n_days_with", np.nan))
        days_without = row.get("days_without", row.get("n_days_without", np.nan))
        total = (0 if pd.isna(days_with) else days_with) + (0 if pd.isna(days_without) else days_without)
        present_pct = days_with / total * 100 if total else np.nan
        diff = row.get("mood_difference", row.get("difference_with_minus_without", np.nan))
        corr = row.get("correlation", np.nan)
        support = row.get("confidence")
        if not support:
            support = _data_support(int(total)) if total else "Very low"
        rows.append({
            "Factor": str(label).replace("Context ", "Daily Context: "),
            "Days present": int(days_with) if pd.notna(days_with) else "—",
            "Days absent": int(days_without) if pd.notna(days_without) else "—",
            "Present on answered days": _pct(present_pct),
            "Mood difference": f"{float(diff):+.2f} pts" if pd.notna(diff) else "—",
            "Relationship": relationship_percent(corr),
            "Data support": str(support),
        })
    return pd.DataFrame(rows)


def _technical_stats(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df.head(n).iterrows():
        label = row.get("factor_label", row.get("feature_label", ""))
        p = row.get("p_value", row.get("correlation_p_value", np.nan))
        ci_low = row.get("ci_low", row.get("ci_95_low", np.nan))
        ci_high = row.get("ci_high", row.get("ci_95_high", np.nan))
        rows.append({
            "Factor": label,
            "Correlation p-value": _fmt(p, 4) if pd.notna(p) else "—",
            "95% CI for mood difference": f"[{float(ci_low):.2f}, {float(ci_high):.2f}]" if pd.notna(ci_low) and pd.notna(ci_high) else "—",
        })
    return pd.DataFrame(rows)


def _association_key() -> None:
    st.caption(
        "**How to read these tables:** “Present on answered days” uses only days when that question was actually answered. "
        "Mood difference is the average change in your 1–5 lowest-mood score when the factor was present; positive means a better lowest mood and negative means a worse one. "
        "Relationship is the correlation × 100 (for example, −53% = r −0.53); it describes direction/strength, not probability or percent causation. "
        "Data support reflects how many usable days are behind the pattern."
    )


def _stats_key() -> None:
    st.caption(
        "**Optional statistics:** A p-value asks how surprising the pattern would be if there were truly no relationship; smaller values provide stronger evidence against 'no relationship.' "
        "A 95% confidence interval is the plausible range for the average difference under the model. With small samples both can be unstable, so they are kept as supporting details rather than headline results."
    )


def _mood_daily_figure(daily: pd.DataFrame) -> plt.Figure | None:
    if "date" not in daily or "lowest_mood_score" not in daily:
        return None
    temp = daily[["date", "lowest_mood_score"]].copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp["lowest_mood_score"] = pd.to_numeric(temp["lowest_mood_score"], errors="coerce")
    temp = temp.dropna().sort_values("date")
    if temp.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(temp.date, temp.lowest_mood_score, marker="o", linewidth=1)
    ax.set_title("Lowest mood recorded each day")
    ax.set_ylabel("Lowest mood (1 = worst, 5 = better)")
    ax.set_xlabel("Date")
    ax.set_ylim(.7, 5.3)
    ax.grid(True, alpha=.25)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    return fig


def _mood_trend_figure(daily: pd.DataFrame) -> plt.Figure | None:
    if "date" not in daily or "lowest_mood_score" not in daily:
        return None
    temp = daily[["date", "lowest_mood_score"]].copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp["lowest_mood_score"] = pd.to_numeric(temp["lowest_mood_score"], errors="coerce")
    temp = temp.dropna().sort_values("date").set_index("date")
    if len(temp) < 2:
        return None
    trend = temp["lowest_mood_score"].rolling("7D", min_periods=2).mean().dropna()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(trend.index, trend.values, linewidth=2)
    ax.set_title("7-day lowest-mood trend")
    ax.set_ylabel("7-day average (1 = worst, 5 = better)")
    ax.set_xlabel("Date")
    ax.set_ylim(.7, 5.3)
    ax.grid(True, alpha=.25)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    return fig


def _scatter_figure(daily: pd.DataFrame, x_col: str, y_col: str, title: str) -> plt.Figure | None:
    if x_col not in daily or y_col not in daily:
        return None
    temp = daily[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(temp) < 3:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(temp[x_col], temp[y_col], alpha=.7)
    if temp[x_col].nunique() > 1:
        slope, intercept = np.polyfit(temp[x_col], temp[y_col], 1)
        xs = np.linspace(temp[x_col].min(), temp[x_col].max(), 100)
        ax.plot(xs, slope * xs + intercept, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(pretty_label(x_col))
    ax.set_ylabel(pretty_label(y_col))
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    return fig


def _correlation_summary(daily: pd.DataFrame, variables: list[str], outcome: str = "lowest_mood_score") -> pd.DataFrame:
    if outcome not in daily:
        return pd.DataFrame()
    rows = []
    for col in variables:
        if col not in daily or col == outcome:
            continue
        r, n = _corr(daily[col], daily[outcome])
        if r is None:
            continue
        rows.append({
            "Measure": pretty_label(col),
            "Unit": variable_unit(col) or "—",
            "Data source": variable_source(col),
            "Paired days": n,
            "Relationship": relationship_percent(r),
            "Strength": relationship_strength(r),
            "What it suggests": _relationship_meaning(r, pretty_label(col)),
            "_abs": abs(r),
            "_variable": col,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_abs", ascending=False).reset_index(drop=True)


def _recent_baseline_summary(daily: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    if "date" not in daily:
        return pd.DataFrame()
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    rows = []
    for col in variables:
        if col not in data:
            continue
        temp = data[["date", col]].copy()
        temp[col] = pd.to_numeric(temp[col], errors="coerce")
        temp = temp.dropna().sort_values("date")
        if temp.empty:
            continue
        baseline = temp[col].mean()
        recent = temp[col].tail(7).mean()
        pct = (recent - baseline) / abs(baseline) * 100 if baseline != 0 else np.nan
        rows.append({
            "Measure": pretty_label(col),
            "Data source": variable_source(col),
            "Overall average": f"{baseline:.2f} {variable_unit(col)}".strip(),
            "Recent 7-day average": f"{recent:.2f} {variable_unit(col)}".strip(),
            "Change from baseline": _pct(pct, 1, signed=True),
        })
    return pd.DataFrame(rows)


def render_overview(datasets: dict[str, pd.DataFrame], daily: pd.DataFrame) -> None:
    section_header("Overview", "A readable snapshot of your coverage and strongest current patterns.")
    if daily.empty:
        st.warning("No merged daily data found. Run files 1–3 first.")
        return
    dates = pd.to_datetime(daily.get("date"), errors="coerce")
    mood_days = int(pd.to_numeric(daily.get("lowest_mood_score"), errors="coerce").notna().sum()) if "lowest_mood_score" in daily else 0
    quick_days = int(pd.to_numeric(daily.get("mood_checkin_mean_score"), errors="coerce").notna().sum()) if "mood_checkin_mean_score" in daily else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days in merged dataset", f"{len(daily):,}")
    c2.metric("Detailed mood-check-in days", f"{mood_days:,}")
    c3.metric("Quick mood-check-in days", f"{quick_days:,}")
    c4.metric("Latest date", "—" if dates.isna().all() else dates.max().date().isoformat())

    st.markdown("### Patterns linked with lowest mood")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Factors that show up with better lowest-mood scores**")
        table = _association_display(datasets.get("best_day_predictors", pd.DataFrame()), 10)
        if not table.empty:
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info("No usable patterns yet.")
    with c2:
        st.markdown("**Factors that show up with worse lowest-mood scores**")
        table = _association_display(datasets.get("worst_day_predictors", pd.DataFrame()), 10)
        if not table.empty:
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info("No usable patterns yet.")
    _association_key()
    with st.expander("Statistical details (optional)"):
        st.markdown("**Better-score table statistics**")
        st.dataframe(_technical_stats(datasets.get("best_day_predictors", pd.DataFrame()), 10), width="stretch", hide_index=True)
        st.markdown("**Worse-score table statistics**")
        st.dataframe(_technical_stats(datasets.get("worst_day_predictors", pd.DataFrame()), 10), width="stretch", hide_index=True)
        _stats_key()

    st.markdown("### Mood at a glance")
    fig = _mood_daily_figure(daily)
    if fig:
        st.pyplot(fig, clear_figure=True)
        st.caption("Each point is one day. This chart intentionally does **not** mix the daily values with a rolling average; the smoother trend is shown separately on the Mood page.")


def render_summary_tab(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Readable mood summaries", "Plain-language summaries first; technical statistical details are optional.")
    df = datasets.get("readable_summaries", pd.DataFrame())
    if df.empty:
        st.info("No readable summaries yet. Run file 4 again after collecting check-in data.")
        return
    levels = sorted(df["confidence"].dropna().unique())
    confidence_filter = st.multiselect("Data support levels", levels, default=levels)
    shown = df[df["confidence"].isin(confidence_filter)] if confidence_filter else df
    for _, row in shown.iterrows():
        with st.expander(str(row["factor_label"]), expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Days with factor", int(row["days_with"]))
            c2.metric("Mood with factor", _fmt(row["average_mood_with"]))
            c3.metric("Mood without factor", _fmt(row["average_mood_without"]))
            st.markdown(f"**What this means:** {row['plain_language_meaning']}")
            st.markdown(f"**Relationship:** {relationship_percent(row['correlation'])} — {row['correlation_meaning']}")
            st.markdown(f'{confidence_badge(str(row["confidence"]))} <span class="ht-muted">{row["confidence_note"]}</span>', unsafe_allow_html=True)
            with st.expander("Statistical details"):
                if pd.notna(row.get("p_value")):
                    st.markdown(f"**p-value:** {_fmt(row['p_value'], 4)}")
                if pd.notna(row.get("ci_low")) and pd.notna(row.get("ci_high")):
                    st.markdown(f"**95% confidence interval for the mood difference:** [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]")
                _stats_key()
    st.caption("Relationship percentages are correlation coefficients shown on a −100% to +100% scale for readability; they are not probabilities.")


def render_predictions_tab(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Best & worst day patterns", "Which tracked factors tend to appear on better or worse lowest-mood days. These are associations, not proof of cause.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Worse lowest-mood days")
        df = _association_display(datasets.get("worst_day_predictors", pd.DataFrame()), 15)
        if not df.empty:
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No usable patterns yet.")
    with c2:
        st.markdown("### Better lowest-mood days")
        df = _association_display(datasets.get("best_day_predictors", pd.DataFrame()), 15)
        if not df.empty:
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No usable patterns yet.")
    _association_key()
    with st.expander("Statistical details (optional)"):
        st.dataframe(_technical_stats(datasets.get("worst_day_predictors", pd.DataFrame()), 15), width="stretch", hide_index=True)
        st.dataframe(_technical_stats(datasets.get("best_day_predictors", pd.DataFrame()), 15), width="stretch", hide_index=True)
        _stats_key()


def render_mood_tab(daily: pd.DataFrame) -> None:
    section_header("Mood & relationships", "Daily mood, longer-term trends, quick check-in stability, and relationship-security patterns in one place.")
    if daily.empty:
        st.info("No daily data available.")
        return
    mood = pd.to_numeric(daily.get("lowest_mood_score"), errors="coerce") if "lowest_mood_score" in daily else pd.Series(dtype=float)
    quick = pd.to_numeric(daily.get("mood_checkin_mean_score"), errors="coerce") if "mood_checkin_mean_score" in daily else pd.Series(dtype=float)
    security = pd.to_numeric(daily.get("relationship_security_score"), errors="coerce") if "relationship_security_score" in daily else pd.Series(dtype=float)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest-mood days tracked", int(mood.notna().sum()))
    c2.metric("Average lowest mood", _fmt(mood.mean()))
    c3.metric("Quick mood days", int(quick.notna().sum()))
    c4.metric("Average relationship security", _fmt(security.mean()))
    st.caption("For mood and relationship-security scores, **higher = better**. Lowest mood means the lowest point reached that day, not your average mood for the whole day.")

    st.markdown("### Daily lowest mood")
    fig = _mood_daily_figure(daily)
    if fig:
        st.pyplot(fig, clear_figure=True)
    st.caption("Use this chart to see individual good/bad days and sudden changes.")

    st.markdown("### 7-day mood trend")
    fig = _mood_trend_figure(daily)
    if fig:
        st.pyplot(fig, clear_figure=True)
    st.caption("This is a separate smoothed trend so it is easier to see whether your recent baseline is moving up or down without mixing it with the daily points.")

    if "mood_checkin_mean_score" in daily:
        st.markdown("### Quick mood check-in distribution")
        temp = daily[["date", "mood_checkin_mean_score"]].copy()
        temp["mood_checkin_mean_score"] = pd.to_numeric(temp["mood_checkin_mean_score"], errors="coerce")
        temp = temp.dropna(subset=["mood_checkin_mean_score"])
        if not temp.empty:
            rating_cols = [f"mood_checkin_rating_{i}_count" for i in range(1, 6)]
            if all(c in daily.columns for c in rating_cols):
                counts = pd.DataFrame({
                    "Mood rating": [f"{i}/5" for i in range(1, 6)],
                    "Check-ins": [int(pd.to_numeric(daily[c], errors="coerce").fillna(0).sum()) for c in rating_cols],
                })
                value_col = "Check-ins"
                note = "Counts individual quick mood check-ins, so multiple check-ins on the same day are preserved instead of being hidden inside the daily mean."
            else:
                temp["Rounded rating"] = temp["mood_checkin_mean_score"].round().clip(1, 5).astype(int)
                counts = temp["Rounded rating"].value_counts().reindex(range(1, 6), fill_value=0).rename_axis("Mood rating").reset_index(name="Days")
                counts["Mood rating"] = counts["Mood rating"].astype(str) + "/5"
                value_col = "Days"
                note = "Counts days by the rounded daily quick-mood average."
            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.dataframe(counts, width="stretch", hide_index=True)
            with c2:
                st.bar_chart(counts.set_index("Mood rating")[value_col])
            st.caption("Stoic stores the quick-mood slider continuously, so values are rounded to the nearest 1–5 bucket for readability. " + note)

    mental_vars = ["mood_checkin_mean_score", "relationship_security_score", "stress_score", "energy_score", "productivity_score", "connectedness_score", "subjective_sleep_score", "motivation_score"]
    summary = _correlation_summary(daily, mental_vars)
    if not summary.empty:
        st.markdown("### Which mental-health ratings move with lowest mood?")
        st.dataframe(summary.drop(columns=["_abs", "_variable"]), width="stretch", hide_index=True)
        st.caption("This table helps separate related concepts: for example, relationship security may move with mood even when stress does not. Relationship % is correlation ×100, not causation.")

    if "relationship_security_score" in daily:
        st.markdown("### Relationship security")
        rel = daily[["date", "relationship_security_score"]].copy()
        rel["date"] = pd.to_datetime(rel["date"], errors="coerce")
        rel["relationship_security_score"] = pd.to_numeric(rel["relationship_security_score"], errors="coerce")
        rel = rel.dropna().sort_values("date")
        if not rel.empty:
            st.line_chart(rel.set_index("date")["relationship_security_score"])
        r, n = _corr(daily["relationship_security_score"], daily["lowest_mood_score"]) if "lowest_mood_score" in daily else (None, 0)
        if r is not None:
            st.markdown(f"**Relationship security vs lowest mood:** {relationship_percent(r)} ({relationship_strength(r).lower()}), based on {n} paired days.")
            fig = _scatter_figure(daily, "relationship_security_score", "lowest_mood_score", "Relationship security vs lowest mood")
            if fig:
                st.pyplot(fig, clear_figure=True)


def _health_variables(daily: pd.DataFrame) -> list[str]:
    preferred = [
        "sleep_hours_asleep", "subjective_sleep_score", "activity_steps", "activity_exercise_minutes",
        "activity_walking_running_distance", "activity_active_energy", "activity_stand_minutes", "activity_flights_climbed", "activity_time_in_daylight_minutes",
        "daylight_exposure_pct_of_available",
        "heart_hrv_sdnn", "heart_resting_heart_rate", "heart_rate", "resp_respiratory_rate",
        "body_oxygen_saturation", "body_blood_oxygen_saturation",
    ]
    return [c for c in preferred if c in daily]


def render_health_tab(datasets: dict[str, pd.DataFrame], daily: pd.DataFrame) -> None:
    section_header("Health", "A single summary page for sleep, activity, HRV, heart data, and other physical-health measures in relation to mood.")
    variables = _health_variables(daily)
    summary = _correlation_summary(daily, variables)
    if summary.empty:
        st.info("Not enough paired health and mood data yet.")
        return

    st.markdown("### Health measures vs lowest mood")
    st.dataframe(summary.drop(columns=["_abs", "_variable"]), width="stretch", hide_index=True)
    st.caption("Sorted by the strongest current relationship. A positive relationship means higher values tend to occur with better lowest-mood scores; negative means higher values tend to occur with worse scores.")

    st.markdown("### Recent health compared with your usual baseline")
    baseline = _recent_baseline_summary(daily, variables)
    if not baseline.empty:
        st.dataframe(baseline, width="stretch", hide_index=True)

    st.markdown("### Explore one health measure")
    selectable = summary["_variable"].tolist()
    selected = st.selectbox("Health measure", selectable, format_func=pretty_label, key="health_metric")
    row = summary[summary["_variable"] == selected].iloc[0]
    st.markdown(f"**{row['Relationship']} relationship with lowest mood** — {row['What it suggests']} ({int(row['Paired days'])} paired days)")
    fig = _scatter_figure(daily, selected, "lowest_mood_score", f"{pretty_label(selected)} vs lowest mood")
    if fig:
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Sleep")
    sleep = datasets.get("sleep_quality", pd.DataFrame()).copy()
    if not sleep.empty:
        sleep = sleep.rename(columns={"sleep_range": "Hours asleep", "n_days": "Mood days", "average_lowest_mood": "Average lowest mood", "average_sleep_hours": "Average hours asleep"})
        cols = [c for c in ["Hours asleep", "Mood days", "Average lowest mood", "Average hours asleep"] if c in sleep]
        st.dataframe(sleep[cols], width="stretch", hide_index=True)
        st.caption("This groups sleep duration into ranges so you can compare mood without needing to interpret a scatterplot.")
    sleep_vars = [c for c in ["sleep_hours_asleep", "subjective_sleep_score"] if c in daily]
    sleep_summary = _correlation_summary(daily, sleep_vars)
    if not sleep_summary.empty:
        st.dataframe(sleep_summary.drop(columns=["_abs", "_variable"]), width="stretch", hide_index=True)

    st.markdown("### Heart & HRV")
    heart_vars = [c for c in ["heart_hrv_sdnn", "heart_resting_heart_rate", "heart_rate", "resp_respiratory_rate"] if c in daily]
    heart_summary = _correlation_summary(daily, heart_vars)
    if not heart_summary.empty:
        st.dataframe(heart_summary.drop(columns=["_abs", "_variable"]), width="stretch", hide_index=True)

    st.markdown("### Weather & daylight context")
    context_vars = [c for c in [
        "activity_time_in_daylight_minutes", "daylight_exposure_pct_of_available", "weather_daylight_hours", "weather_sunshine_hours",
        "weather_sunshine_fraction_pct", "weather_cloud_cover_mean_pct", "weather_precipitation_mm",
        "weather_precipitation_hours", "weather_temperature_mean_f", "weather_apparent_temperature_mean_f",
        "weather_shortwave_radiation_mj_m2",
    ] if c in daily]
    context_summary = _correlation_summary(daily, context_vars)
    if context_summary.empty:
        st.caption("No weather/daylight context has been merged yet. Local weather history becomes automatic after config/context.json is set; Apple Watch Time in Daylight comes from the Apple Health export.")
    else:
        st.dataframe(context_summary.drop(columns=["_abs", "_variable"]), width="stretch", hide_index=True)
        st.caption(
            "**Data source key:** Apple Watch = your recorded personal daylight exposure; "
            "Local weather history (Open-Meteo) = environmental conditions/opportunity at the configured location; "
            "Derived: Apple Watch + local weather = a measure calculated from both sources."
        )


def render_activity_tab(daily: pd.DataFrame) -> None:
    section_header("Activity", "Steps, exercise, distance, workouts, recent trends, and their current relationship with mood.")
    variables = [c for c in ["activity_steps", "activity_exercise_minutes", "activity_walking_running_distance", "activity_active_energy", "activity_stand_minutes", "activity_flights_climbed", "activity_time_in_daylight_minutes", "daylight_exposure_pct_of_available", "workout_count"] if c in daily]
    if not variables:
        st.info("No activity data available.")
        return
    summary = _correlation_summary(daily, variables)
    if not summary.empty:
        st.markdown("### Activity measures vs lowest mood")
        st.dataframe(summary.drop(columns=["_abs", "_variable"]), width="stretch", hide_index=True)

    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    latest = data.date.max()
    recent = data[data.date >= latest - pd.Timedelta(days=90)].copy() if pd.notna(latest) else data
    st.markdown("### Recent activity trends")
    if "activity_steps" in recent:
        steps = recent[["date", "activity_steps"]].copy()
        steps["Steps"] = pd.to_numeric(steps["activity_steps"], errors="coerce")
        steps = steps.set_index("date")["Steps"].rolling("7D", min_periods=1).mean()
        st.markdown("**7-day average steps**")
        st.line_chart(steps)
    if "activity_exercise_minutes" in recent:
        ex = recent[["date", "activity_exercise_minutes"]].copy()
        ex["Exercise minutes"] = pd.to_numeric(ex["activity_exercise_minutes"], errors="coerce")
        ex = ex.set_index("date")["Exercise minutes"].rolling("7D", min_periods=1).mean()
        st.markdown("**7-day average exercise minutes**")
        st.line_chart(ex)
    st.caption("The rolling activity charts show trends only; the table above is where activity is compared directly with mood.")


def _frequency_table(df: pd.DataFrame, item_col: str, label: str) -> pd.DataFrame:
    if df.empty or item_col not in df:
        return pd.DataFrame()
    temp = df.copy()
    temp["date"] = pd.to_datetime(temp.get("date"), errors="coerce")
    temp = temp.dropna(subset=[item_col])
    if temp.empty:
        return pd.DataFrame()
    total_dates = max(temp["date"].nunique(), 1)
    grouped = temp.groupby(item_col).agg(Times=(item_col, "size"), Days=("date", "nunique")).reset_index()
    grouped["% of check-in days"] = grouped["Days"] / total_dates * 100
    grouped = grouped.sort_values(["Days", "Times"], ascending=False)
    grouped = grouped.rename(columns={item_col: label})
    grouped["% of check-in days"] = grouped["% of check-in days"].map(lambda x: f"{x:.0f}%")
    return grouped


def render_triggers_thoughts(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Triggers & thoughts", "Overall frequencies plus recent entries. Multi-thought responses are split into separate thoughts.")
    triggers = datasets.get("triggers_long", pd.DataFrame()).copy()
    thoughts = datasets.get("thoughts_long", pd.DataFrame()).copy()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Trigger frequency")
        freq = _frequency_table(triggers, "trigger", "Trigger")
        if not freq.empty:
            st.dataframe(freq, width="stretch", hide_index=True)
        else:
            st.info("No trigger entries yet.")
    with c2:
        st.markdown("### Automatic-thought frequency")
        freq = _frequency_table(thoughts, "automatic_thought", "Automatic thought")
        if not freq.empty:
            st.dataframe(freq.head(30), width="stretch", hide_index=True)
        else:
            st.info("No thought entries yet.")
    st.caption("The percentage is based only on check-in dates where at least one trigger/thought was recorded, not all calendar days.")

    st.markdown("### Most recent entries")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Recent triggers**")
        if not triggers.empty:
            triggers["date"] = pd.to_datetime(triggers["date"], errors="coerce")
            cols = [c for c in ["date", "trigger", "lowest_mood_score"] if c in triggers]
            st.dataframe(triggers.sort_values("date", ascending=False)[cols].head(25), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Recent automatic thoughts**")
        if not thoughts.empty:
            thoughts["date"] = pd.to_datetime(thoughts["date"], errors="coerce")
            cols = [c for c in ["date", "automatic_thought", "lowest_mood_score"] if c in thoughts]
            st.dataframe(thoughts.sort_values("date", ascending=False)[cols].head(25), width="stretch", hide_index=True)


def render_recovery_insights(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Recovery", "How often recovery methods were tracked and how long the low-mood period tended to last when they were used.")
    df = datasets.get("recovery_effectiveness", pd.DataFrame()).copy()
    if df.empty:
        st.info("Recovery effectiveness needs recovery methods plus a usable duration measure.")
        return
    duration_labels = {1: "< 30 minutes", 2: "30 min–2 hours", 3: "2–6 hours", 4: "Most of the day", 5: "Multiple days"}
    shown = pd.DataFrame({
        "Recovery method": df["recovery_method_label"].str.replace("Recovery ", "", regex=False),
        "Times tracked": df["times_used_with_duration"].astype(int),
        "Average low-mood duration score": df["average_recovery_duration"].map(lambda x: f"{x:.2f}/5"),
        "Typical duration": df["median_recovery_duration"].round().astype("Int64").map(duration_labels),
        "Data support": df["confidence"],
    })
    st.dataframe(shown, width="stretch", hide_index=True)
    st.caption("Duration score: 1 = under 30 minutes, 2 = 30 min–2 hours, 3 = 2–6 hours, 4 = most of the day, 5 = multiple days. Lower is a shorter low-mood episode.")


def render_temporal_relationships(datasets: dict[str, pd.DataFrame]) -> None:
    section_header(
        "Timing & direction",
        "Separate what happened before mood, what occurred on the same day, and what followed mood. These are temporal associations, not proof of cause.",
    )
    df = datasets.get("temporal_relationships", pd.DataFrame()).copy()
    if df.empty:
        st.info("Not enough date-aligned data yet for timing analysis. Run 04_analysis.py after rebuilding the merged dataset.")
        return
    df = df.dropna(subset=["relationship"])
    if df.empty:
        st.info("Not enough paired days yet for timing analysis.")
        return

    sections = [
        (
            "Before mood / potential influences",
            "Things measured before the mood outcome: previous night's sleep plus prior-day variables. These are the most useful patterns for asking what might precede a mood change.",
        ),
        (
            "Same-day / current",
            "Things measured on the same calendar day as mood. They may move together, but this view cannot tell which one came first.",
        ),
        (
            "After mood / possible consequences",
            "Today's mood compared with the next day's measurements. These can show whether mood is followed by changes in activity, sleep, physiology, screen use, or other behavior.",
        ),
    ]

    for group_name, explanation in sections:
        st.markdown(f"### {group_name}")
        st.caption(explanation)
        group = df[df["timing_group"] == group_name].copy()
        if group.empty:
            st.info("No usable variables in this timing group yet.")
            continue
        group = group.sort_values("absolute_relationship", ascending=False)
        shown = pd.DataFrame({
            "Measure": group["factor_label"],
            "Data source": group.get("data_source", group["factor"].map(variable_source)),
            "Relationship": group["relationship"].map(relationship_percent),
            "Strength": group["relationship"].map(relationship_strength),
            "Paired days": group["n_days"].astype(int),
            "Data support": group["data_support"],
            "What it suggests": group["plain_language_meaning"],
        })
        st.dataframe(shown, width="stretch", hide_index=True)

        no_assoc = group[group["relationship"].abs() < .20]
        if not no_assoc.empty:
            names = ", ".join(no_assoc["factor_label"].head(8).tolist())
            st.caption(f"Little/no current linear relationship (under 20%): {names}.")

    st.caption(
        "A relationship shown here is correlation ×100 for readability. 'Before' gives stronger timing information than same-day association, "
        "but none of these tables by themselves prove that one factor causes the mood change."
    )
    with st.expander("Statistical details (optional)"):
        detail_cols = [c for c in ["timing_group", "factor_label", "data_source", "relationship", "p_value", "n_days"] if c in df.columns]
        detail = df[detail_cols].copy()
        detail["relationship"] = detail["relationship"].map(relationship_percent)
        detail = detail.rename(columns={
            "timing_group": "Timing",
            "factor_label": "Measure",
            "data_source": "Data source",
            "relationship": "Relationship",
            "p_value": "p-value",
            "n_days": "Paired days",
        })
        st.dataframe(detail, width="stretch", hide_index=True)
        _stats_key()

def _monthly_mood_distribution(daily: pd.DataFrame) -> pd.DataFrame:
    if "date" not in daily or "mood_checkin_mean_score" not in daily:
        return pd.DataFrame()
    temp = daily.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp["score"] = pd.to_numeric(temp["mood_checkin_mean_score"], errors="coerce")
    temp = temp.dropna(subset=["date"])
    temp["month"] = temp["date"].dt.to_period("M").astype(str)
    rating_cols = [f"mood_checkin_rating_{i}_count" for i in range(1, 6)]

    stats = temp.groupby("month")["score"].agg(["mean", "std", "count"])
    if all(c in temp.columns for c in rating_cols):
        for c in rating_cols:
            temp[c] = pd.to_numeric(temp[c], errors="coerce").fillna(0)
        counts = temp.groupby("month")[rating_cols].sum()
        counts.columns = [f"{i}/5 check-ins" for i in range(1, 6)]
    else:
        scored = temp.dropna(subset=["score"]).copy()
        scored["rating"] = scored["score"].round().clip(1, 5).astype(int)
        counts = scored.pivot_table(index="month", columns="rating", values="date", aggfunc="count", fill_value=0)
        counts = counts.reindex(columns=range(1, 6), fill_value=0)
        counts.columns = [f"{c}/5 days" for c in counts.columns]

    out = counts.join(stats).reset_index().sort_values("month", ascending=False)
    return out.rename(columns={"month": "Month", "mean": "Average rating", "std": "Variability", "count": "Tracked days"})


def _consistency_figure(df: pd.DataFrame, variable: str) -> plt.Figure | None:
    group = df[df["variable"] == variable].copy()
    if group.empty:
        return None
    group["month_date"] = pd.to_datetime(group["month"] + "-01", errors="coerce")
    group = group.dropna(subset=["month_date", "mean"]).sort_values("month_date")
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    yerr = pd.to_numeric(group["standard_deviation"], errors="coerce").fillna(0)
    ax.errorbar(group["month_date"], group["mean"], yerr=yerr, marker="o", capsize=4)
    ax.set_title(f"{pretty_label(variable)}: monthly average with variability")
    ax.set_ylabel(pretty_label(variable))
    ax.set_xlabel("Month")
    ax.grid(True, alpha=.25)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    return fig


def render_consistency(datasets: dict[str, pd.DataFrame], daily: pd.DataFrame) -> None:
    section_header("Consistency & stability", "Monthly averages and variability are shown together so 'stable' is not confused with 'good.'")
    df = datasets.get("consistency", pd.DataFrame()).copy()
    if df.empty:
        st.info("No months currently meet the minimum mood-tracking requirement.")
        return
    shown = df.copy()
    shown["Month"] = pd.to_datetime(shown["month"], format="%Y-%m", errors="coerce").dt.strftime("%b %Y")
    shown["Measure"] = shown["variable"].map(pretty_label)
    shown["Data source"] = shown["variable"].map(variable_source)
    shown["Monthly average"] = shown["mean"].map(lambda x: f"{x:.2f}")
    shown["Variability (SD)"] = shown["standard_deviation"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    shown["Change vs prior month"] = shown.get("monthly_change_pct", pd.Series(np.nan, index=shown.index)).map(lambda x: _pct(x, 1, True))
    shown["Change vs previous year's average"] = shown.get("yearly_change_pct", pd.Series(np.nan, index=shown.index)).map(lambda x: _pct(x, 1, True))
    table = shown[["Month", "Measure", "Data source", "n_days", "Monthly average", "Variability (SD)", "Change vs prior month", "Change vs previous year's average"]].rename(columns={"n_days": "Days tracked"})
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Variability is the monthly standard deviation: lower means more stable, but the monthly average tells you **where** that stable value sits. Monthly change compares with the prior tracked month; yearly change compares with the previous calendar year's average when available.")

    st.markdown("### Average and variability on the same graph")
    variables = df["variable"].dropna().unique().tolist()
    selected = st.selectbox("Measure", variables, format_func=pretty_label, key="consistency_metric")
    fig = _consistency_figure(df, selected)
    if fig:
        st.pyplot(fig, clear_figure=True)
    st.caption("The point is the monthly average; the vertical bar shows ±1 standard deviation. This makes a stable 1/5 visibly different from a stable 4/5.")

    dist = _monthly_mood_distribution(daily)
    if not dist.empty:
        st.markdown("### Quick mood rating stability")
        st.dataframe(dist, width="stretch", hide_index=True)
        st.caption("This preserves the rating distribution, so two months with the same average no longer look identical when one stayed near 3/5 and another swung between 1/5 and 5/5. Rating columns count individual quick check-ins; Tracked days counts days with at least one quick mood value.")


def render_baselines(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Personal baselines", "Your recent seven available days compared with your own historical average.")
    df = datasets.get("personal_baselines", pd.DataFrame()).copy()
    if df.empty:
        st.info("No personal baselines available yet.")
        return
    shown = pd.DataFrame({
        "Measure": df["variable"].map(pretty_label),
        "Data source": df.get("data_source", df["variable"].map(variable_source)),
        "Overall baseline": df.apply(lambda r: f"{r['all_time_baseline']:.2f} {variable_unit(r['variable'])}".strip(), axis=1),
        "Recent 7-day average": df.apply(lambda r: f"{r['recent_7_day_average']:.2f} {variable_unit(r['variable'])}".strip(), axis=1),
        "Difference": df.apply(lambda r: f"{r['difference_from_baseline']:+.2f} {variable_unit(r['variable'])}".strip(), axis=1),
        "% difference from baseline": df["percent_difference_from_baseline"].map(lambda x: _pct(x, 2, True)),
        "Total days": df["n_total_days"].astype(int),
    })
    st.dataframe(shown, width="stretch", hide_index=True)
    st.caption("The percent column is already calculated as a true percentage: 8.32 means **8.32%**, not 0.0832.")


def render_things_to_watch(datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Things to watch", "Strong same-day clues, before/after timing patterns, and recent departures from your own baseline.")
    df = datasets.get("things_to_watch", pd.DataFrame()).copy()
    if df.empty:
        st.info("No watch items yet.")
        return
    if "headline" in df:
        df = df[df["headline"].str.lower().ne("symptom none")]
    for _, row in df.iterrows():
        rp = row.get("relationship_percent") if "relationship_percent" in df.columns else None
        render_watch_card(
            headline=str(row["headline"]),
            message=str(row["message"]),
            category=str(row["category"]),
            confidence=str(row["confidence"]),
            relationship_percent=None if pd.isna(rp) else float(rp),
            source=None if "data_source" not in df.columns or pd.isna(row.get("data_source")) else str(row.get("data_source")),
        )
    st.caption("“Data support” is based on how many usable observations exist. “Relationship strength” is the absolute correlation ×100 when a correlation is available. They are intentionally shown separately rather than inventing a statistical 'confidence percentage.'")


def render_association_explorer(daily: pd.DataFrame) -> None:
    section_header("Associations explorer", "Choose two understandable daily measures and see how they move together.")
    if daily.empty:
        st.warning("No merged daily data found.")
        return
    include_counts = st.checkbox("Include count variables", value=False)
    variables = get_variable_options(daily, include_counts=include_counts)
    if len(variables) < 2:
        st.warning("Not enough usable variables found.")
        return
    default_a = "sleep_hours_asleep" if "sleep_hours_asleep" in variables else variables[0]
    default_b = "lowest_mood_score" if "lowest_mood_score" in variables else variables[min(1, len(variables)-1)]
    c1, c2 = st.columns(2)
    with c1:
        var_a = st.selectbox("Variable 1", variables, index=variables.index(default_a), format_func=format_option)
        st.caption(variable_description(var_a))
    with c2:
        var_b = st.selectbox("Variable 2", variables, index=variables.index(default_b), format_func=format_option)
        st.caption(variable_description(var_b))
    result = summarize_selected_pair(daily, var_a, var_b)
    st.markdown(result.summary_markdown)
    if result.figure is not None:
        st.pyplot(result.figure, clear_figure=True)
    if not result.table.empty:
        st.dataframe(result.table, width="stretch", hide_index=True)
    with st.expander("Variable details"):
        st.markdown(f"**{pretty_label(var_a)}:** {variable_description(var_a)}  \n**Data source:** {variable_source(var_a)}")
        st.markdown(f"**{pretty_label(var_b)}:** {variable_description(var_b)}  \n**Data source:** {variable_source(var_b)}")
        st.caption("Internal variable names are available in Data Inventory if you need them for coding/debugging.")


def render_data_inventory(daily: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> None:
    section_header("Data inventory", "Technical reference for what is available and how complete it is.")
    if daily.empty:
        st.warning("No daily data found.")
        return
    inventory = describe_dataset(daily)
    group_filter = st.multiselect("Filter by variable group", sorted(inventory["group"].dropna().unique()), default=sorted(inventory["group"].dropna().unique()))
    filtered = inventory[inventory["group"].isin(group_filter)] if group_filter else inventory
    st.dataframe(filtered, width="stretch", hide_index=True)
    with st.expander("Loaded dataset shapes"):
        shapes = pd.DataFrame([{"dataset": key, "rows": df.shape[0], "columns": df.shape[1]} for key, df in datasets.items()]).sort_values("dataset")
        st.dataframe(shapes, width="stretch", hide_index=True)


def main() -> None:
    ensure_project_folders()
    apply_dashboard_style()
    render_hero("Health Tracker", "A private, readable view of mood, sleep, activity, relationships, triggers, recovery, and physiological patterns.", eyebrow="Mental health & wellbeing")
    with st.sidebar:
        st.markdown("## Health Tracker")
        st.caption("Dashboard controls and pipeline status")
        if st.button("↻ Refresh dashboard data", width="stretch"):
            st.cache_data.clear(); st.rerun()
        with st.expander("Pipeline run order"):
            st.code("python3 scripts/01_import_data.py\npython3 scripts/02_process_data.py\npython3 scripts/03_merge_data.py\npython3 scripts/04_analysis.py\npython3 scripts/05_visualizations.py")
        st.markdown("---")
        st.caption("Your data stays on your computer when you run this dashboard locally.")

    datasets = load_data_cached()
    daily = choose_daily_dataset(datasets)
    tabs = st.tabs([
        "⌂ Overview", "✦ Summary", "↗ Best & Worst Days", "◉ Mood & Relationships", "♡ Health",
        "◌ Activity", "◇ Triggers & Thoughts", "↺ Recovery", "→ Timing & Direction", "≈ Consistency",
        "— Personal Baselines", "! Things to Watch", "⌕ Associations Explorer", "▦ Data Inventory",
    ])
    with tabs[0]: render_overview(datasets, daily)
    with tabs[1]: render_summary_tab(datasets)
    with tabs[2]: render_predictions_tab(datasets)
    with tabs[3]: render_mood_tab(daily)
    with tabs[4]: render_health_tab(datasets, daily)
    with tabs[5]: render_activity_tab(daily)
    with tabs[6]: render_triggers_thoughts(datasets)
    with tabs[7]: render_recovery_insights(datasets)
    with tabs[8]: render_temporal_relationships(datasets)
    with tabs[9]: render_consistency(datasets, daily)
    with tabs[10]: render_baselines(datasets)
    with tabs[11]: render_things_to_watch(datasets)
    with tabs[12]: render_association_explorer(daily)
    with tabs[13]: render_data_inventory(daily, datasets)


if __name__ == "__main__":
    main()
