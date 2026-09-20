# Daylight Exposure Update

This version is verified against the supplied Apple Health export.

## Verified source

- HealthKit type: `HKQuantityTypeIdentifierTimeInDaylight`
- Device source: Apple Watch only
- Export unit: minutes
- Raw daylight records found: **10,598**
- Days with daylight records: **691**
- Verified range: **2024-04-05 through 2026-09-18**

The Watch records are short minute-valued intervals. The importer sums them by local calendar date into:

- `activity_time_in_daylight_minutes`

Missing daylight records remain NULL rather than being assumed to mean 0 minutes.

## Weather + personal exposure

When historical weather/daylight context is configured, the merge step also calculates:

- `daylight_exposure_pct_of_available`

Formula:

`Apple Watch Time in Daylight minutes / (available daylight hours × 60) × 100`

This is a behavioral exposure ratio, not a biological sunlight-dose estimate. It can be affected by travel because weather opportunity is based on the configured location.

## Analysis placement

Both daylight measures are available to:

- Health
- Activity
- Associations Explorer
- Consistency
- Personal Baselines / Things to Watch
- Timing & Direction: Before mood, Same-day/current, and After mood
