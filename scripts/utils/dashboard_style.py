"""Visual styling helpers for the Streamlit dashboard.

This module changes presentation only. It does not alter any calculations,
filters, datasets, or statistical results.
"""

from __future__ import annotations

import html

import streamlit as st


APP_CSS = r"""
<style>
:root {
    --ht-bg: #f5f7f6;
    --ht-card: #ffffff;
    --ht-card-soft: #eef4f1;
    --ht-text: #22302b;
    --ht-muted: #64736d;
    --ht-accent: #3f7162;
    --ht-accent-soft: #dceae4;
    --ht-border: #dce5e1;
    --ht-warm: #a66a4c;
}

.stApp {
    background: linear-gradient(180deg, #f4f7f5 0%, #fbfcfb 40%, #f6f8f7 100%);
    color: var(--ht-text);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--ht-text);
    letter-spacing: -0.02em;
}

[data-testid="stSidebar"] {
    background: #edf3f0;
    border-right: 1px solid var(--ht-border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
}

.ht-hero {
    padding: 1.45rem 1.55rem;
    border: 1px solid var(--ht-border);
    border-radius: 22px;
    background: linear-gradient(135deg, #ffffff 0%, #e8f1ed 100%);
    box-shadow: 0 8px 28px rgba(45, 83, 70, 0.07);
    margin-bottom: 1.2rem;
}

.ht-eyebrow {
    color: var(--ht-accent);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.ht-hero h1 {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.1rem);
    line-height: 1.05;
}

.ht-hero p {
    color: var(--ht-muted);
    max-width: 850px;
    margin: 0.75rem 0 0;
    font-size: 1.03rem;
}

.ht-section {
    margin: 0.15rem 0 1rem;
}

.ht-section-title {
    font-size: 1.5rem;
    font-weight: 760;
    margin: 0;
}

.ht-section-subtitle {
    color: var(--ht-muted);
    margin-top: 0.28rem;
    line-height: 1.5;
}

.ht-card {
    border: 1px solid var(--ht-border);
    border-radius: 18px;
    background: rgba(255,255,255,0.92);
    padding: 1rem 1.05rem;
    box-shadow: 0 5px 18px rgba(47, 76, 66, 0.045);
}

.ht-insight-card {
    border: 1px solid var(--ht-border);
    border-left: 5px solid var(--ht-accent);
    border-radius: 16px;
    background: #ffffff;
    padding: 1rem 1.1rem;
    margin: 0.75rem 0;
}

.ht-watch-card {
    border: 1px solid var(--ht-border);
    border-radius: 16px;
    background: linear-gradient(135deg, #ffffff, #f0f5f2);
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
}

.ht-watch-card h3 {
    margin: 0 0 0.4rem;
    font-size: 1.05rem;
}

.ht-muted {
    color: var(--ht-muted);
}

.ht-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    font-size: 0.76rem;
    font-weight: 700;
    border: 1px solid transparent;
}

.ht-badge-high, .ht-badge-moderate {
    color: #245843;
    background: #dff0e7;
    border-color: #b9dccb;
}

.ht-badge-preliminary {
    color: #765321;
    background: #f8edd6;
    border-color: #ead5a8;
}

.ht-badge-low, .ht-badge-very-low {
    color: #844b43;
    background: #f6e3df;
    border-color: #e7c2bb;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid var(--ht-border);
    border-radius: 16px;
    padding: 0.85rem 1rem;
    box-shadow: 0 4px 14px rgba(45, 75, 64, 0.035);
}

[data-testid="stMetricLabel"] {
    color: var(--ht-muted);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.35rem;
    overflow-x: auto;
    scrollbar-width: thin;
    padding: 0.35rem;
    border-radius: 14px;
    background: #e9efec;
}

[data-testid="stTabs"] button[role="tab"] {
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    color: #52625c;
    white-space: nowrap;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #ffffff;
    color: var(--ht-accent);
    box-shadow: 0 2px 8px rgba(42, 71, 61, 0.08);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--ht-border);
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    border: 1px solid var(--ht-border);
    border-radius: 14px;
    background: rgba(255,255,255,0.86);
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid #bcd0c7;
    background: #ffffff;
    color: var(--ht-accent);
    font-weight: 650;
}

.stButton > button:hover {
    border-color: var(--ht-accent);
    background: #f4f8f6;
    color: #28594a;
}

div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    border-radius: 12px;
    border-color: var(--ht-border);
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: var(--ht-border);
}


/* ------------------------------------------------------------------
   Keep the dashboard readable regardless of the user's OS/browser
   dark-mode preference. The visual design is intentionally light.
   ------------------------------------------------------------------ */
html, body, [data-testid="stAppViewContainer"], .stApp {
    color-scheme: light !important;
    color: var(--ht-text) !important;
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    color: var(--ht-text) !important;
}

/* Normal Streamlit text */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stText"],
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: var(--ht-text) !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: var(--ht-muted) !important;
}

/* Our custom HTML blocks should never inherit white/dark-theme text. */
.ht-hero,
.ht-card,
.ht-insight-card,
.ht-watch-card,
.ht-section,
.ht-section-title,
.ht-watch-card h3,
.ht-watch-card div {
    color: var(--ht-text) !important;
}

.ht-muted,
.ht-section-subtitle,
.ht-hero p {
    color: var(--ht-muted) !important;
}

/* Sidebar text */
[data-testid="stSidebar"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] summary,
[data-testid="stSidebar"] summary p {
    color: var(--ht-text) !important;
}

/* Inputs/selects. BaseWeb may otherwise inherit dark-theme foregrounds. */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"],
[data-baseweb="textarea"] > div,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #ffffff !important;
    color: var(--ht-text) !important;
}

[data-baseweb="select"] span,
[data-baseweb="select"] input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    color: var(--ht-text) !important;
    -webkit-text-fill-color: var(--ht-text) !important;
}

/* Dropdown menus render in a portal outside the main app container. */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background: #ffffff !important;
    color: var(--ht-text) !important;
}

[role="option"] * {
    color: var(--ht-text) !important;
}

/* Tabs and expanders */
[data-testid="stTabs"] button[role="tab"],
[data-testid="stTabs"] button[role="tab"] p {
    color: #52625c !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
    color: var(--ht-accent) !important;
}

[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    background: transparent !important;
    color: var(--ht-text) !important;
}

/* Alerts */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: var(--ht-text) !important;
}

/* Buttons */
.stButton > button,
.stButton > button p,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondary"] p {
    color: var(--ht-accent) !important;
}

/* Dataframes use Streamlit's theme for the grid itself. This surrounding
   rule keeps any fallback/table text legible too. */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] * {
    color-scheme: light !important;
}

/* Links */
a, [data-testid="stMarkdownContainer"] a {
    color: #2f6757 !important;
}

@media (max-width: 800px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .ht-hero { padding: 1.15rem; }
}
</style>
"""


def apply_dashboard_style() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, eyebrow: str = "Personal insights dashboard") -> None:
    st.markdown(
        f"""
        <div class="ht-hero">
            <div class="ht-eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="ht-section-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ht-section"><div class="ht-section-title">{html.escape(title)}</div>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def confidence_badge(confidence: str) -> str:
    raw = str(confidence or "Unknown")
    css_key = raw.strip().lower().replace(" ", "-")
    return f'<span class="ht-badge ht-badge-{html.escape(css_key)}">{html.escape(raw)} data support</span>'


def render_watch_card(
    headline: str,
    message: str,
    category: str,
    confidence: str,
    relationship_percent: float | None = None,
    source: str | None = None,
) -> None:
    """Render a watch item using native Streamlit components.

    This intentionally avoids raw HTML for dynamic text. Multiline raw-HTML
    blocks can occasionally be split by Streamlit's Markdown parser, which
    causes closing ``<span>``/``</div>`` fragments to appear as visible text.
    """
    details = [f"{str(confidence or 'Unknown')} data support"]

    if relationship_percent is not None:
        try:
            value = float(relationship_percent)
            if value == value:  # NaN-safe check
                details.append(f"Relationship strength: {abs(value):.0f}%")
        except (TypeError, ValueError):
            pass

    if category:
        details.append(str(category))

    if source and str(source).strip() not in {"", "—", "nan", "None"}:
        details.append(f"Source: {str(source)}")

    with st.container(border=True):
        st.markdown(f"### {str(headline)}")
        st.write(str(message))
        st.caption(" · ".join(details))
