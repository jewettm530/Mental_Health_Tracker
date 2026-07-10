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
data/raw/health.zip
The scripts expect those exact filenames.

### Pipeline
### 1. Import
`01_import_data.py` reads:
- `data/raw/stoic.zip`
- `data/raw/health.zip`

It extracts and normalizes the source files into imported datasets.

### 2. Process
`02_process_data.py` converts the imported files into clean, separated daily datasets for:
- Mood and relationship measures
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
- Lagged and next-day effects
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
- Summary
- Mood
- Health
- Sleep
- Activity
- Heart
- Relationships
- Triggers
- Symptoms
- Automatic Thoughts
- Recovery
- Predictions
- Lagged Effects
- Consistency
- Personal Baselines
- Things to Watch
- Associations Explorer

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
- The project is for personal reflection and is not a diagnostic or medical tool.