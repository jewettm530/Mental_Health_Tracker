# Personal Health & Mood Analytics

<p align="center">
  <strong>A privacy-conscious Python pipeline for exploring relationships between self-tracking and Apple Health data</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Apple%20Health-EA4C89?style=flat-square&logo=apple&logoColor=white">
  <img src="https://img.shields.io/badge/Privacy-Local%20Data-success?style=flat-square">
</p>

## Overview

This project is a **personal-informatics analytics pipeline** combining self-tracking data exported from Stoic with health and activity data exported from Apple Health.

The pipeline:

* imports and cleans heterogeneous data exports;
* creates standardized daily datasets;
* integrates health and self-tracking variables;
* explores statistical associations;
* analyzes lagged and next-day relationships;
* creates visualizations;
* presents results in an interactive Streamlit dashboard.

The project focuses on **longitudinal data engineering, exploratory analysis, visualization, and privacy-conscious software design**.

---

## Data Domains

Depending on the available exports, the pipeline can analyze:

### Self-Tracking

* mood;
* triggers;
* symptoms;
* automatic thoughts;
* recovery strategies;
* relationship-related measures.

### Apple Health

* sleep;
* activity;
* workouts;
* heart-related measurements;
* respiratory/body measurements;
* other available wellness metrics.

---

## Analysis Features

The pipeline can explore:

* mood-factor differences;
* sleep and mood relationships;
* activity patterns;
* recovery-method associations;
* same-day correlations;
* next-day and lagged relationships;
* monthly consistency;
* individualized baselines;
* variables that may warrant additional tracking.

Results include qualitative confidence labels so that visually strong patterns based on very small amounts of data are not presented as equally reliable.

---

## Pipeline

```text
Stoic Export ───────┐
                    ▼
Apple Health ──► Import & Clean
                    │
                    ▼
                Daily Merge
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Analysis   Visualizations  Lagged Effects
        └───────────┼───────────┘
                    ▼
             Streamlit Dashboard
```

---

## Project Structure

```text
Mental_Health_Tracker/
├── data/
│   ├── raw/              # Private — ignored by Git
│   ├── processed/        # Private — ignored by Git
│   └── outputs/          # Private — ignored by Git
│
├── scripts/
│   ├── 01_import_data.py
│   ├── 02_process_data.py
│   ├── 03_merge_data.py
│   ├── 04_analysis.py
│   ├── 05_visualizations.py
│   ├── 06_dashboard.py
│   └── utils/
│
├── data_dictionary.csv
├── requirements.txt
└── README.md
```

---

## Quick Start

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place local exports at:

```text
data/raw/stoic.zip
data/raw/health.zip
```

Run:

```bash
python3 scripts/01_import_data.py
python3 scripts/02_process_data.py
python3 scripts/03_merge_data.py
python3 scripts/04_analysis.py
python3 scripts/05_visualizations.py
```

Launch the dashboard:

```bash
streamlit run scripts/06_dashboard.py
```

---

## Privacy

Health and journal exports can contain highly sensitive information.

The repository excludes:

```text
data/raw/
data/processed/
data/outputs/
```

Raw exports, processed personal datasets, journal content, and generated private reports should **never be committed publicly**.

Any public screenshots or demonstrations should use synthetic or appropriately anonymized data.

---

## Interpretation & Limitations

* Results describe associations, not causation.
* Same-day relationships do not establish temporal direction.
* Lagged relationships improve temporal interpretation but still do not prove causality.
* Early findings may rely on small numbers of observations.
* Missing wearable data may reflect device usage rather than physiological change.
* Self-tracking data can contain reporting and selection biases.

---

## Responsible Use

This project is intended for **personal informatics, longitudinal-data analysis, and software-development exploration**.

It is not a diagnostic system or medical device and should not be used for automated medical decisions.

---

**Skills:** `Python` · `pandas` · `Streamlit` · `Longitudinal Data` · `Apple Health` · `Statistical Analysis` · `Data Visualization` · `Time-Series Analysis` · `Privacy-Conscious Data Engineering`

---

<details>
<summary><h1>Technical Details</h1></summary>

### Processing Stages

#### 1. Import

```text
01_import_data.py
```

reads the Stoic and Apple Health archives and normalizes the source exports.

#### 2. Process

```text
02_process_data.py
```

creates cleaned daily datasets for available domains such as:

* mood and relationship measures;
* triggers;
* symptoms;
* automatic thoughts;
* recovery methods;
* sleep;
* activity;
* heart metrics;
* respiratory/body measures;
* workouts.

#### 3. Merge

```text
03_merge_data.py
```

creates the primary integrated datasets.

### Main Datasets

#### `master_daily.csv`

The broadest daily merged dataset.

It retains useful numeric, categorical, and text-like information where appropriate.

#### `correlation_ready_daily.csv`

A simplified numeric dataset intended for:

* correlations;
* lag analysis;
* numeric visualizations;
* exploratory associations.

### Analysis

```text
04_analysis.py
```

supports outputs such as:

* mood-factor comparisons;
* best/worst-day associations;
* sleep-range summaries;
* recovery-method comparisons;
* lagged relationships;
* monthly consistency;
* individualized baselines.

### Visualization

```text
05_visualizations.py
```

generates question-focused plots organized by topic.

### Dashboard

```text
06_dashboard.py
```

provides an interactive Streamlit interface with sections for areas such as:

* mood;
* sleep;
* activity;
* heart-related metrics;
* triggers;
* recovery;
* lagged effects;
* consistency;
* personal baselines;
* association exploration.

### Confidence Labels

Exploratory findings are labeled according to the amount of supporting data.

Conceptually:

* **Very low** — very few usable observations
* **Preliminary** — possible pattern requiring more data
* **Moderate** — more useful supporting sample
* **High** — comparatively stronger observational support

These labels help prevent a large effect or correlation based on only a few observations from being treated as a robust finding.

### Statistical Interpretation

Correlations and average differences are treated as exploratory associations.

Interpretation should consider:

* effect size;
* number of observations;
* confidence intervals;
* missingness;
* temporal ordering;
* repeated testing.

A statistically interesting association should not automatically be interpreted as practically meaningful or causal.

</details>
