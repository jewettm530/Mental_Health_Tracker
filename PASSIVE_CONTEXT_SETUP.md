# Passive Context Setup

## 1. Weather and daylight

Copy:

`config/context.example.json` -> `config/context.json`

Then enter a city/location name or latitude/longitude. No weather API key is required.

Run the normal pipeline:

```bash
python scripts/01_import_data.py
python scripts/02_process_data.py
python scripts/03_merge_data.py
python scripts/04_analysis.py
streamlit run scripts/06_dashboard.py
```

`02_process_data.py` will automatically fetch historical weather/daylight context for the date range already represented by Stoic/Apple Health data.

Environmental fields include:
- available daylight hours
- sunshine hours
- sunny share of daylight
- mean cloud cover
- precipitation and precipitation hours
- average / feels-like temperature
- solar radiation

These describe outdoor conditions, not personal exposure.

## 2. Apple Watch Time in Daylight

This project has now been verified against the user's Apple Health export. The export contains `HKQuantityTypeIdentifierTimeInDaylight` records from the Apple Watch with values in minutes. No extra integration is required.

The importer sums those Watch records by local calendar date into:

- `activity_time_in_daylight_minutes` — personal daylight exposure recorded by Apple Watch

When weather context is also available, the merge step derives:

- `daylight_exposure_pct_of_available` — Apple Watch daylight minutes ÷ available daylight minutes × 100

The percentage is a behavioral exposure ratio, **not** a biological sunlight-dose estimate. A missing Watch record remains NULL; it is not automatically treated as 0 minutes. The current export has only one Time in Daylight source (Apple Watch), so direct daily summation does not require source de-duplication.

## 3. Screen time

The project now accepts an optional file:

`data/raw/screen_time_daily.csv`

Canonical supported columns are:
- `date`
- `screen_time_total_minutes`
- `screen_time_late_night_minutes`
- `screen_time_social_minutes`
- `screen_time_entertainment_minutes`
- `screen_time_productivity_minutes`
- `screen_time_pickups`
- `screen_time_notifications`

You do **not** need this file for the rest of the project to run. It is a future integration point for an external tracker/exporter because Apple's built-in Screen Time history is not directly readable by this Python project.

## 4. How timing is interpreted

### Before mood
Previous-day values and previous-night sleep are compared with today's lowest mood. This is the most useful group for asking what may precede a mood change.

### Same-day
The variable and mood happened on the same calendar day. Association can be detected, but temporal order is unknown.

### After mood
Today's mood is compared with the following day's values. This is useful for asking whether mood is followed by changes in activity, sleep, physiology, screen use, or other behavior.

None of these categories alone establish causation.
