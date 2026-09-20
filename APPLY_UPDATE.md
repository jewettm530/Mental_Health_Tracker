# Apply the Passive Context + Timing & Direction + Daylight Exposure Update

This package contains the full code/config/documentation update. It does **not** include your raw health or Stoic exports.

## 1. Back up the current project

Keep a copy of your current Health_Tracker folder before replacing code.

## 2. Copy this update over the project

Replace the matching project files/folders with the ones in this package. Keep your existing `data/raw/` folder, especially:

- `data/raw/health.zip`
- `data/raw/stoic.zip`
- `data/raw/stoic.txt`

## 3. Weather setup (optional but recommended)

Copy:

`config/context.example.json`

to:

`config/context.json`

Then set the location used for historical weather/daylight opportunity. Do not use this as proof of where you personally were on a day; Apple Watch Time in Daylight is the personal exposure source.

## 4. Rebuild from the raw exports

From the project root run:

```bash
python scripts/01_import_data.py
python scripts/02_process_data.py
python scripts/03_merge_data.py
python scripts/04_analysis.py
python scripts/05_visualizations.py
```

Then launch:

```bash
streamlit run scripts/06_dashboard.py
```

## What to expect

The Apple Health import will now include `HKQuantityTypeIdentifierTimeInDaylight` and aggregate it to daily `activity_time_in_daylight_minutes`.

The supplied export was verified to contain 10,598 Time in Daylight records from Apple Watch only, producing 691 recorded daily totals from 2024-04-05 through 2026-09-18.

If weather context is configured, the merged daily data will also contain `daylight_exposure_pct_of_available`.

## Weather/daylight source labels

This build adds a visible **Data source** field for passive daylight/weather variables:
- **Apple Watch** — personal Time in Daylight.
- **Local weather history (Open-Meteo)** — environmental daylight/weather conditions for the configured location.
- **Derived: Apple Watch + local weather** — Personal Daylight Exposure %.

After replacing the code, rerun `02_process_data.py`, `03_merge_data.py`, and `04_analysis.py` before launching the dashboard so the regenerated Timing / Things to Watch outputs also carry source metadata.
