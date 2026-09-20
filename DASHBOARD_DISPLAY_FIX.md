# Dashboard display fix

This revision fixes two presentation/runtime problems in the Streamlit dashboard.

## 1. DeltaGenerator / Streamlit internals appearing on pages

Several table renders in `scripts/06_dashboard.py` were written as conditional expression statements, for example:

```python
st.dataframe(table, ...) if not table.empty else st.info(...)
```

Streamlit's magic renderer can treat the return value of this expression as displayable output. Since `st.dataframe()` returns a `DeltaGenerator`, the app could render Streamlit's internal `DeltaGenerator` representation/documentation on the page.

All six occurrences were replaced with ordinary `if/else` blocks.

## 2. White text on white backgrounds

The dashboard styling uses light cards and backgrounds, but Streamlit could still inherit a dark theme from the browser/system/app settings. This could make Streamlit-native text remain light while the custom dashboard background was also light.

The project now includes `.streamlit/config.toml` with an explicit light theme. `scripts/utils/dashboard_style.py` also contains fallback text-color rules for headings, Markdown, captions, metrics, tabs, expanders, inputs, sidebar text, buttons, alerts, and dropdowns.

## Run the dashboard

Run Streamlit from the **project root** so the `.streamlit/config.toml` theme is loaded:

```bash
streamlit run scripts/06_dashboard.py
```

Your existing `data/` directory can stay unchanged.
