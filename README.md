# Health Tracker

This project imports and prepares personal mood tracking data from Stoic and health data from Apple Health so the data can later be analyzed and visualized.

The main goal is to understand patterns in mood, sleep, activity, heart metrics, and other health signals without turning the project into a manual tracking burden.

## Project folder structure

```text
Health_Tracker/
├── data/
│   ├── raw/
│   │   ├── health.zip
│   │   └── stoic.zip
│   │
│   ├── processed/
│   │   ├── apple_health/
│   │   │   ├── imported/
│   │   │   └── clean/
│   │   ├── stoic/
│   │   │   ├── imported/
│   │   │   └── clean/
│   │   └── merged/
│   │
│   └── outputs/
│       ├── plots/
│       ├── reports/
│       └── dashboards/
│
├── scripts/
│   ├── 01_import_data.py
│   ├── 02_process_data.py
│   ├── 03_merge_data.py
│   └── utils/
│
├── README.md
├── requirements.txt
├── data_dictionary.csv
└── .gitignore
```

## Raw data files

Place the two raw export zip files here:

```text
data/raw/stoic.zip
data/raw/health.zip
```

The scripts expect those exact filenames.

## Run order

From the project root:

```bash
cd /Users/maddiemac/My_Projects/Health_Tracker
python scripts/01_import_data.py
python scripts/02_process_data.py
python scripts/03_merge_data.py
```

## Script responsibilities

### `01_import_data.py`

Reads the raw zip files and converts them into imported CSV files.

Outputs:

```text
data/processed/stoic/imported/
data/processed/apple_health/imported/
```

### `02_process_data.py`

Cleans and separates the imported data into daily feature tables.

Main outputs:

```text
data/processed/stoic/clean/stoic_daily_mood.csv
data/processed/stoic/clean/stoic_answers_long.csv
data/processed/stoic/clean/stoic_daily_wide.csv

data/processed/apple_health/clean/apple_daily_sleep.csv
data/processed/apple_health/clean/apple_daily_activity.csv
data/processed/apple_health/clean/apple_daily_heart.csv
data/processed/apple_health/clean/apple_daily_body.csv
data/processed/apple_health/clean/apple_daily_respiratory.csv
data/processed/apple_health/clean/apple_daily_workouts.csv
data/processed/apple_health/clean/apple_activity_summary.csv
```

### `03_merge_data.py`

Combines the separated daily datasets into merged daily files.

Outputs:

```text
data/processed/merged/master_daily.csv
data/processed/merged/correlation_ready_daily.csv
data/processed/merged/merge_inventory.json
```

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

Full import logs are optional for now. The current setup already creates lightweight inventory files, especially:

```text
data/processed/merged/merge_inventory.json
```

That is enough at this stage because the project is still focused on setup, not auditing every import run.
