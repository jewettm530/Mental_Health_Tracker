# Health Tracker
A personal mental-health and health analytics project that combines Stoic mood-tracking data with Apple Health data.

The project imports and cleans raw exports, creates daily datasets, analyzes meaningful patterns, generates question-focused visualizations, and displays the results in an interactive Streamlit dashboard.

The goal is not to diagnose or predict mental-health conditions. It is to identify personal patterns involving mood, triggers, symptoms, automatic thoughts, recovery methods, sleep, activity, heart metrics, and relationship security.

## Privacy
This project contains highly sensitive personal health information.
The following directories are excluded from Git:
- `data/raw/`
- `data/processed/`
- `data/outputs/`

Do not commit Apple Health exports, Stoic backups, processed health data, reports, or dashboard outputs to a public repository.

## Project folder structure
Health_Tracker/
├── data/
│   ├── raw/
│   │   ├── stoic.zip
│   │   ├── stoic.txt
│   │   └── health.zip
│   ├── processed/
│   │   ├── stoic/
│   │   ├── apple_health/
│   │   └── merged/
│   └── outputs/
│       ├── analysis/
│       └── plots/
│
├── scripts/
│   ├── 01_import_data.py
│   ├── 02_process_data.py
│   ├── 03_merge_data.py
│   ├── 04_analysis.py
│   ├── 05_visualizations.py
│   ├── 06_dashboard.py
│   └── utils/
│       ├── analysis.py
│       ├── apple_health.py
│       ├── cleaning.py
│       ├── dashboard.py
│       ├── dashboard_style.py
│       ├── date_utils.py
│       ├── file_utils.py
│       ├── merge.py
│       ├── paths.py
│       ├── plotting.py
│       ├── reports.py
│       └── stoic.py
│
├── README.md
├── data_dictionary.csv
├── requirements.txt
└── .gitignore

## Raw data files
data/raw/stoic.zip
data/raw/stoic.txt
data/raw/health.zip

`stoic.zip` and `health.zip` are required. `stoic.txt` is strongly recommended: it adds human-readable Stoic question/choice labels, displayed 1–5 ratings, and can contain newer entries than the ZIP. The pipeline still runs without it for backward compatibility.

### Pipeline
### 1. Import
`01_import_data.py` reads:
- `data/raw/stoic.zip`
- `data/raw/stoic.txt` (when available)
- `data/raw/health.zip`

The Stoic ZIP remains the structured source for UUIDs, timestamps, contexts, and repeated check-ins. The TXT export is parsed as a complementary human-readable source and is used to decode labels and displayed ratings without double-counting overlapping observations.

### 2. Process
`02_process_data.py` converts the imported files into clean, separated daily datasets for:
- Mood and relationship measures
- Energy, stress, productivity, connectedness, motivation, and subjective sleep ratings
- Repeated quick mood check-ins from the structured ZIP
- Daily emotion and influence selections
- Triggers
- Symptoms
- Automatic thoughts
- Recovery methods
- Sleep
- Activity
- Heart metrics
- Respiratory and body metrics
- Workouts

### 3. Merge
`03_merge_data.py` creates:
- `master_daily.csv`
- `correlation_ready_daily.csv`
The master file preserves useful text and categorical fields. The correlation-ready file contains numeric analysis features.

### 4. Analyze
`04_analysis.py` produces:
- Readable mood-factor comparisons
- Best-day and worst-day associations
- Sleep-range summaries
- Recovery effectiveness
- Timing-aware relationships: before mood, same-day, and after mood
- Monthly consistency measures
- Personalized baselines
- Things-to-watch summaries

### 5. Visualize
`05_visualizations.py` generates question-focused plots organized by topic.

### 6. Dashboard
`06_dashboard.py` launches an interactive Streamlit dashboard with topic tabs, readable summaries, confidence ratings, and a two-variable association explorer.

## Running the project
From the project root:
bash
python3 scripts/01_import_data.py
python3 scripts/02_process_data.py
python3 scripts/03_merge_data.py
python3 scripts/04_analysis.py
python3 scripts/05_visualizations.py
streamlit run scripts/06_dashboard.py

## Main merged datasets
### `master_daily.csv`
This is the full daily merged dataset. It keeps all available merged columns, including text or category-like fields when present.

Use this when you want the most complete daily view.

### `correlation_ready_daily.csv`
This keeps only:
- `date`
- numeric columns
- columns that contain at least one usable numeric value
Use this later for correlation checks, lag analysis, and graphs.

## Notes on import logs
Full import logs are optional for now. The current setup already creates some lightweight inventory files, including:
- data/processed/merged/merge_inventory.json
- data/processed/merged/correlation_ready_daily_enriched.csv
- data/processed/merged/correlation_ready_daily.csv
- data/processed/merged/master_daily.csv

## Dashboard sections
- Overview
- Summary
- Best & Worst Days
- Mood & Relationships
- Health
- Activity
- Triggers & Thoughts
- Recovery
- Timing & Direction
- Consistency
- Personal Baselines
- Things to Watch
- Associations Explorer
- Data Inventory

## Confidence ratings
Results are labeled according to the amount of supporting data.
- Very low: only a few observations; treat as an early clue
- Preliminary: a possible pattern that needs more data
- Moderate: supported by a more useful sample
- High: supported by a comparatively large number of observations
A strong correlation based on very few days should not be treated as a reliable conclusion.

## Interpreting results
- Average mood difference: the average lowest-mood score on days when a factor was present minus the average on days when it was absent.
- Positive difference: the lowest mood was higher on days with the factor.
- Negative difference: the lowest mood was lower on days with the factor.
- Correlation: how strongly two numeric variables tend to move together.
- Confidence interval: a range of plausible values for the estimated difference.
- p-value: a statistical measure included for context; it should not be interpreted without considering sample size and effect size.
- Standard deviation: the amount of variability in a measure. Lower monthly mood standard deviation means more stable mood scores.

## Limitations
- Results show associations, not causation.
- Early findings may be based on very small samples.
- Apple Health data may be missing when devices were not worn or measurements were unavailable.
- Same-day associations do not establish which factor occurred first.
- Binary variables indicate that a factor was recorded, not its intensity.
- A skipped Stoic multi-select question is stored as NULL, not 0. A 0 means the question was answered and that specific option was not selected. This prevents missing answers from being treated as evidence that a symptom, trigger, or recovery method was absent.
- Derived Apple Health flags also remain NULL when the underlying metric is unavailable.
- Stoic 1–5 ratings use the displayed app scale. Structured ZIP slider fallbacks are converted from the internal 0–4 representation to 1–5.
- The project is for personal reflection and is not a diagnostic or medical tool.
## Passive context: weather, daylight, and screen time

The project can now add optional passive context without making the Stoic journal longer.

### Weather/daylight opportunity

Copy `config/context.example.json` to `config/context.json` and set either:

```json
{
  "location_name": "CITY, STATE/REGION, COUNTRY",
  "timezone": "auto"
}
```

or explicit latitude/longitude. When `scripts/02_process_data.py` runs, it fetches daily historical context from Open-Meteo for the dates already present in the project. The resulting variables include available daylight, sunshine duration, sunny share of daylight, average cloud cover, precipitation, temperature, and solar radiation.

These are **environmental opportunity** measures. They do not prove how long you personally spent outside.

### Personal daylight exposure

The Apple Health importer is verified to read `HKQuantityTypeIdentifierTimeInDaylight` from this project's Apple Watch export (10,598 records across 691 recorded days in the supplied export). It becomes `activity_time_in_daylight_minutes` after the daily merge. When weather context is configured, the project also derives `daylight_exposure_pct_of_available` (personal daylight minutes ÷ available daylight minutes × 100). Personal exposure remains separate from weather-based sunshine/daylight because they answer different questions.

### Screen time

Apple's built-in Screen Time history is not directly available to this Python project. The project is ready for a future external source: if `data/raw/screen_time_daily.csv` exists, `02_process_data.py` will normalize and merge recognized daily fields such as total minutes, late-night minutes, category minutes, pickups, and notifications.

### Timing & Direction analysis

The former Next-Day Patterns page is now **Timing & Direction** and separates relationships into:

1. **Before mood / potential influences** — previous night's sleep plus previous-day measures vs today's mood.
2. **Same-day / current** — variables that occur on the same calendar day as mood; useful for association, but temporal direction is unknown.
3. **After mood / possible consequences** — today's mood vs the following day's measurements.

These remain correlations, not causal conclusions. The separation is meant to stop a same-day association from being described as though it necessarily caused the mood change.

### Sleep date alignment

Overnight Apple Health sleep is now assigned to the **wake date** (the date the sleep period ends). This makes the intended relationship `previous night's sleep -> today's mood` line up correctly.


### Passive-data provenance

Weather/daylight variables now show a **Data source** label in the dashboard:
- **Apple Watch** for personal Time in Daylight.
- **Local weather history (Open-Meteo)** for environmental conditions.
- **Derived: Apple Watch + local weather** for Personal Daylight Exposure %.

See `WEATHER_DATA_SOURCE_UPDATE.md` for details.
